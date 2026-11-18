// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// PTX -> Metal Shading Language translator.
//
// Model
//  * Every PTX virtual register becomes an MSL local of its declared type; instructions convert
//    operands to the instruction type (PTX "operand type follows the instruction" rule).
//  * Global/generic addresses are 64-bit Metal GPU virtual addresses (MTLBuffer.gpuAddress based),
//    so CUDA pointer arithmetic is preserved bit-for-bit and dereferenced via reinterpret_cast.
//  * Kernel parameters are a packed byte block bound with setBytes at buffer(0), laid out with
//    CUDA ABI alignment (offsets reported back to libcuda).
//  * All .shared variables (static and .extern dynamic) live in a single threadgroup buffer at
//    threadgroup(0); static vars first, the dynamic region at align16(static size).
//  * .local becomes a per-thread array. Generic pointers carry a tag in bits 63:62
//    (2 = shared, 1 = local, 0 = global) produced by cvta and dispatched at runtime.
//  * MSL has no goto, so control flow is lowered to a basic-block state machine
//    (`for(;;) switch(_pc)`), which LLVM's jump threading turns back into a direct CFG.
//  * FP64: Apple GPUs have no FP64 ALU. Default is df64 (double-float pairs, ~48-bit mantissa);
//    with lowerFP64, f64 math runs in fp32. Both keep IEEE double *storage* in memory exact.

import Foundation

public struct PTXTranslateOptions: Sendable {
    public enum FP64Mode: Sendable { case df64, fp32 }
    public var fp64: FP64Mode
    public init(lowerFP64: Bool = false) { fp64 = lowerFP64 ? .fp32 : .df64 }
}

public struct KernelParam: Sendable {
    public let name: String
    public let offset: UInt32
    public let size: UInt32
    public let align: UInt32
}

public struct TranslatedKernel: Sendable {
    public let entry: String
    public let mslName: String
    public let msl: String
    public let params: [KernelParam]
    public let paramBytes: UInt32
    public let staticSharedBytes: UInt32
    public let usesSharedMemory: Bool
    public let usesFP64: Bool
    public let fp64Lowered: Bool
    public let warnings: [String]
}

public enum PTXToMSLTranslator {
    public static func entries(ptx: String) throws -> [String] { try PTXParser.parse(ptx).entryOrder }

    public static func translate(ptx: String, entry: String, options: PTXTranslateOptions = .init()) throws -> TranslatedKernel {
        let module = try PTXParser.parse(ptx)
        guard let fn = module.functions[entry], fn.isEntry else {
            throw PTXError(line: 0, message: "entry '\(entry)' not found (entries: \(module.entryOrder.joined(separator: ", ")))")
        }
        var g = FunctionGen(module: module, fn: fn, options: options)
        return try g.generate()
    }

    /// Translates every entry of a module (used by tests and `cuModuleLoad` validation).
    public static func translateAll(ptx: String, options: PTXTranslateOptions = .init()) throws -> [TranslatedKernel] {
        try entries(ptx: ptx).map { try translate(ptx: ptx, entry: $0, options: options) }
    }
}

struct FunctionGen {
    let module: PTXModule
    let fn: PTXFunction
    let options: PTXTranslateOptions

    var regs: [String: PT] = [:]
    var sharedOffset: [String: Int] = [:]
    var sharedSize = 0
    var localOffset: [String: Int] = [:]
    var localSize = 0
    var paramOffset: [String: (Int, ParamDecl)] = [:]
    var paramBytes = 0
    var labels: [String: Int] = [:]
    var usesShared = false
    var usesLocal = false
    var usesFP64 = false
    var usesCarry = false
    var usesSimd = false
    var convergent = false
    var blockID: [Int: Int] = [:]
    var warnings: [String] = []
    var tmp = 0

    init(module: PTXModule, fn: PTXFunction, options: PTXTranslateOptions) {
        self.module = module
        self.fn = fn
        self.options = options
    }

    static func ident(_ s: String) -> String {
        var out = ""
        for c in s.unicodeScalars {
            if CharacterSet.alphanumerics.contains(c) && c.isASCII || c == "_" { out.unicodeScalars.append(c) } else { out += "_" }
        }
        return out
    }

    func regName(_ r: String) -> String { "R_" + Self.ident(String(r.dropFirst())) }

    mutating func warn(_ s: String) { if !warnings.contains(s) { warnings.append(s) } }

    func fail(_ inst: Inst, _ msg: String) -> PTXError { PTXError(line: inst.line, message: "\(msg): `\(inst.text)`") }

    // MARK: Types

    var tc: TypeConv { TypeConv(fp64: options.fp64) }
    func msl(_ t: PT) -> String { tc.msl(t) }
    func uint(_ bits: Int) -> String { tc.uint(bits) }
    func rawType(_ t: PT) -> PT { t == .f64 ? .b64 : t }
    func convert(_ e: String, from: PT, to: PT) -> String { tc.convert(e, from: from, to: to) }

    // MARK: Operands

    func intLiteral(_ v: Int64, _ t: PT) -> String {
        switch (t.k, t.bits) {
        case (.s, 64): return v == Int64.min ? "long(0x8000000000000000ul)" : "\(v)l"
        case (.u, 64), (.b, 64): return String(format: "0x%llxul", UInt64(bitPattern: v))
        case (.s, _): return "\(msl(t))(\(Int32(truncatingIfNeeded: v)))"
        default:
            let m = t.bits >= 32 ? UInt64(UInt32(truncatingIfNeeded: v)) : UInt64(bitPattern: v) & ((1 << UInt64(t.bits)) - 1)
            return t.bits == 32 ? String(format: "0x%llxu", m) : "\(msl(t))(\(m)u)"
        }
    }

    func floatLiteral(_ d: Double, _ t: PT) -> String {
        if t == .f64 {
            if options.fp64 == .df64 {
                let hi = Float(d)
                let lo = d.isFinite ? Float(d - Double(hi)) : 0
                return String(format: "_df(as_type<float>(0x%08xu), as_type<float>(0x%08xu))", hi.bitPattern, lo.bitPattern)
            }
            return String(format: "as_type<float>(0x%08xu)", Float(d).bitPattern)
        }
        if t.bits == 32 { return String(format: "as_type<float>(0x%08xu)", Float(d).bitPattern) }
        return "\(msl(t))(as_type<float>(0x\(String(Float(d).bitPattern, radix: 16))u))"
    }

    func special(_ s: String, _ inst: Inst) throws -> String {
        let parts = s.split(separator: ".")
        let base = String(parts[0])
        let comp = parts.count > 1 ? "." + parts[1] : ""
        switch base {
        case "%tid": return "_tid\(comp)"
        case "%ntid": return "_ntid\(comp)"
        case "%ctaid": return "_ctaid\(comp)"
        case "%nctaid": return "_nctaid\(comp)"
        case "%laneid": return "_lane"
        case "%warpid": return "_warp"
        case "%nwarpid": return "_nwarp"
        case "%lanemask_eq": return "(1u << _lane)"
        case "%lanemask_le": return "((2u << _lane) - 1u)"
        case "%lanemask_lt": return "((1u << _lane) - 1u)"
        case "%lanemask_ge": return "(~((1u << _lane) - 1u))"
        case "%lanemask_gt": return "(~((2u << _lane) - 1u))"
        default: throw fail(inst, "special register \(s) has no Metal equivalent")
        }
    }

    mutating func val(_ o: Operand, _ t: PT, _ inst: Inst) throws -> String {
        switch o {
        case .reg(let r):
            guard let rt = regs[r] else { throw fail(inst, "undeclared register \(r)") }
            return convert(regName(r), from: rt, to: t)
        case .special(let s):
            return convert(try special(s, inst), from: .u32, to: t)
        case .imm(let v):
            if t.k == .pred { return v != 0 ? "true" : "false" }
            if t.isFloat { return floatLiteral(Double(v), t) }
            return intLiteral(v, t)
        case .fimm32(let b):
            if t.isFloat {
                if t.bits == 32 { return String(format: "as_type<float>(0x%08xu)", b) }
                return floatLiteral(Double(Float(bitPattern: b)), t)
            }
            return intLiteral(Int64(b), t)
        case .fimm64(let b):
            if t.isFloat { return floatLiteral(Double(bitPattern: b), t) }
            return intLiteral(Int64(bitPattern: b), t)
        case .not(let inner):
            return "(!\(try val(inner, .pred, inst)))"
        case .sym(let s):
            if let off = sharedOffset[s] { usesShared = true; return intLiteral(Int64(off), t) }
            if let off = localOffset[s] { usesLocal = true; return intLiteral(Int64(off), t) }
            throw fail(inst, "address of '\(s)' is not supported (module-scope .global/.const variables or param addresses)")
        default:
            throw fail(inst, "unexpected operand \(o)")
        }
    }

    func destType(_ o: Operand, _ inst: Inst) throws -> PT {
        guard case .reg(let r) = o, let t = regs[r] else { throw fail(inst, "destination must be a register") }
        return t
    }

    func set(_ o: Operand, _ expr: String, _ t: PT, _ inst: Inst) throws -> String {
        switch o {
        case .sink: return "(void)(\(expr));"
        case .reg(let r):
            guard let rt = regs[r] else { throw fail(inst, "undeclared register \(r)") }
            return "\(regName(r)) = \(convert(expr, from: t, to: rt));"
        default: throw fail(inst, "bad destination \(o)")
        }
    }

    // MARK: Memory

    enum Space { case global, shared, local, param, generic }

    func space(_ inst: Inst) throws -> Space {
        if inst.has("global") { return .global }
        if inst.has("shared") || inst.mods.contains(where: { $0.hasPrefix("shared::") }) { return .shared }
        if inst.has("local") { return .local }
        if inst.has("param") { return .param }
        if inst.has("const") { throw fail(inst, ".const space is not supported") }
        return .generic
    }

    /// Address expression: ulong for global/generic, uint offset for shared/local/param.
    mutating func address(_ o: Operand, _ sp: Space, _ inst: Inst) throws -> String {
        guard case .mem(let base, let off) = o else { throw fail(inst, "expected memory operand") }
        let wide = sp == .global || sp == .generic
        var b: String
        switch base {
        case nil: b = wide ? "0ul" : "0u"
        case .some(.sym(let s)):
            if sp == .param {
                guard let (po, _) = paramOffset[s] else { throw fail(inst, "unknown parameter \(s)") }
                b = "\(po)u"
            } else if let so = sharedOffset[s] {
                usesShared = true
                b = wide ? "(\(so)ul | 0x8000000000000000ul)" : "\(so)u"
            } else if let lo = localOffset[s] {
                usesLocal = true
                b = wide ? "(\(lo)ul | 0x4000000000000000ul)" : "\(lo)u"
            } else {
                throw fail(inst, "unknown symbol \(s)")
            }
        case .some(let r):
            b = try val(r, wide ? .u64 : .u32, inst)
        }
        if off == 0 { return b }
        let mag = off < 0 ? UInt64(bitPattern: 0 &- off) : UInt64(off)
        return "(\(b) \(off < 0 ? "-" : "+") \(mag)\(wide ? "ul" : "u"))"
    }

    func lvalue(_ sp: Space, _ addr: String, _ raw: PT, volatile: Bool) -> String {
        let ty = msl(raw)
        let v = volatile ? "volatile " : ""
        switch sp {
        case .global, .generic: return "(*reinterpret_cast<\(v)device \(ty)*>(\(addr)))"
        case .shared: return "(*reinterpret_cast<\(v)threadgroup \(ty)*>(_smem + (\(addr))))"
        case .local: return "(*reinterpret_cast<thread \(ty)*>(_local + (\(addr))))"
        case .param: return "(*reinterpret_cast<constant \(ty)*>(_params + (\(addr))))"
        }
    }

    /// Emits `body(lvalue)` with runtime dispatch for generic addresses.
    mutating func access(_ sp: Space, _ addr: String, _ raw: PT, volatile: Bool, _ body: (String) -> String) -> [String] {
        guard sp == .generic else { return [body(lvalue(sp, addr, raw, volatile: volatile))] }
        let hasShared = !sharedOffset.isEmpty || !module.externShared.isEmpty
        let hasLocal = !localOffset.isEmpty
        if !hasShared && !hasLocal { return [body(lvalue(.global, addr, raw, volatile: volatile))] }
        tmp += 1
        let a = "_ga\(tmp)"
        var out = ["{ ulong \(a) = \(addr);"]
        var chain = ""
        if hasShared {
            usesShared = true
            out.append("  \(chain)if ((\(a) >> 62) == 2ul) { \(body(lvalue(.shared, "uint(\(a))", raw, volatile: volatile))) }")
            chain = "else "
        }
        if hasLocal {
            usesLocal = true
            out.append("  \(chain)if ((\(a) >> 62) == 1ul) { \(body(lvalue(.local, "uint(\(a))", raw, volatile: volatile))) }")
            chain = "else "
        }
        out.append("  else { \(body(lvalue(.global, a, raw, volatile: volatile))) } }")
        return out
    }

    // MARK: Generation

    mutating func generate() throws -> TranslatedKernel {
        // Parameters (CUDA ABI: natural/declared alignment, packed in declaration order).
        var params = [KernelParam]()
        for p in fn.params {
            paramBytes = (paramBytes + p.align - 1) / p.align * p.align
            paramOffset[p.name] = (paramBytes, p)
            params.append(KernelParam(name: p.name, offset: UInt32(paramBytes), size: UInt32(p.size), align: UInt32(p.align)))
            paramBytes += p.size
        }
        // Shared memory layout: module-scope statics, function statics, then dynamic (.extern).
        var staticVars = module.moduleShared
        var blocks: [[BodyStmt]] = [[]]
        for s in fn.body {
            switch s {
            case .reg(let t, let names, let line):
                for n in names {
                    if let old = regs[n], old != t { throw PTXError(line: line, message: "register \(n) redeclared with a different type") }
                    regs[n] = t
                    if t == .f64 { usesFP64 = true }
                }
            case .shared(let v): staticVars.append(v)
            case .local(let v):
                localSize = (localSize + v.align - 1) / v.align * v.align
                localOffset[v.name] = localSize
                localSize += v.size
            case .label(let l):
                labels[l] = blocks.count
                blocks.append([])
            case .inst: break
            }
            if case .label = s { continue }
            blocks[blocks.count - 1].append(s)
        }
        for v in staticVars {
            sharedSize = (sharedSize + v.align - 1) / v.align * v.align
            sharedOffset[v.name] = sharedSize
            sharedSize += v.size
        }
        let dynamicBase = (sharedSize + 15) / 16 * 16
        for v in module.externShared { sharedOffset[v.name] = dynamicBase }

        // Body.
        var body = [String]()
        let structured = labels.count > 0
        // Convergence-sensitive kernels (barriers / SIMD-group ops) need SIMT reconvergence at join
        // points: blocks are numbered in reverse post-order and each SIMD group executes only its
        // minimum pending block ("min-PC" scheduling), so divergent lanes re-join before a barrier.
        convergent = structured && fn.body.contains {
            if case .inst(let i) = $0 { return ["bar", "barrier", "shfl", "vote", "redux", "activemask", "match"].contains(i.op) }
            return false
        }
        var order = Array(blocks.indices)
        if convergent {
            func terminates(_ blk: [BodyStmt]) -> Bool {
                guard let last = blk.last(where: { if case .inst = $0 { return true }; return false }), case .inst(let i) = last else { return false }
                return i.pred == nil && ["bra", "ret", "exit", "trap"].contains(i.op)
            }
            var succ = [[Int]](repeating: [], count: blocks.count)
            for (bi, blk) in blocks.enumerated() {
                for st in blk {
                    if case .inst(let i) = st, i.op == "bra", case .sym(let l)? = i.operands.first, let t = labels[l] { succ[bi].append(t) }
                }
                if !terminates(blk), bi + 1 < blocks.count { succ[bi].append(bi + 1) }
            }
            var seen = Set<Int>()
            var post = [Int]()
            func dfs(_ b: Int) {
                seen.insert(b)
                for t in succ[b] where !seen.contains(t) { dfs(t) }
                post.append(b)
            }
            dfs(0)
            order = post.reversed()
            var rpo = [Int: Int]()
            for (k, b) in order.enumerated() { rpo[b] = k }
            blockID = rpo
            for (l, b) in labels { labels[l] = rpo[b] ?? Int(Int32.max) }
        }
        for bi in order {
            let blk = blocks[bi]
            if structured { body.append("case \(convergent ? blockID[bi]! : bi): {") }
            var terminated = false
            for s in blk {
                guard case .inst(let inst) = s else { continue }
                var lines = try emit(inst)
                if let p = inst.pred {
                    guard let pt = regs[p.name], pt == .pred else { throw fail(inst, "guard \(p.name) is not a predicate") }
                    lines = ["if (\(p.negated ? "!" : "")\(regName(p.name))) {"] + lines.map { "  " + $0 } + ["}"]
                } else if ["bra", "ret", "exit", "trap"].contains(inst.op) {
                    terminated = true
                }
                body.append(contentsOf: lines.map { "  " + $0 })
            }
            if convergent && !terminated {
                // Explicit fall-through to the next textual block.
                if bi + 1 < blocks.count, let next = blockID[bi + 1] { body.append("  _pc = \(next)u; break;") } else {
                    body.append("  _pc = 0xffffffffu; break;")
                }
            }
            if structured { body.append("}") }
        }

        // Assemble.
        if usesFP64 {
            warn(options.fp64 == .df64
                ? "FP64: Apple GPUs have no FP64 ALU; .f64 math is emulated with df64 (double-float, ~48-bit mantissa, float exponent range). Memory keeps exact IEEE-754 doubles."
                : "FP64: .f64 math lowered to fp32 (MPT_CF_FP64_LOWER). Memory keeps exact IEEE-754 doubles; precision is ~7 digits.")
        }
        let name = "ptx_" + Self.ident(fn.name)
        var sig = ["constant uchar* _params [[buffer(0)]]"]
        let needSmem = usesShared || !staticVars.isEmpty || !module.externShared.isEmpty
        if needSmem { sig.append("threadgroup uchar* _smem [[threadgroup(0)]]") }
        sig += [
            "uint3 _tid [[thread_position_in_threadgroup]]", "uint3 _ntid [[threads_per_threadgroup]]",
            "uint3 _ctaid [[threadgroup_position_in_grid]]", "uint3 _nctaid [[threadgroups_per_grid]]",
            "uint _lane [[thread_index_in_simdgroup]]", "uint _warp [[simdgroup_index_in_threadgroup]]",
            "uint _nwarp [[simdgroups_per_threadgroup]]",
        ]
        var src = MSLPrelude.source
        src += "// Translated by MetalPassthrough ptx2msl from PTX \(module.version) \(module.target), entry \(fn.name)\n"
        src += "kernel void \(name)(\n    " + sig.joined(separator: ",\n    ") + ")\n{\n"
        if usesLocal || localSize > 0 {
            src += "  thread uint4 _localv[\(max(1, (localSize + 15) / 16))];\n"
            src += "  thread uchar* _local = reinterpret_cast<thread uchar*>(_localv);\n"
        }
        for (r, t) in regs.sorted(by: { $0.key < $1.key }) {
            let initv: String
            switch (t.k, t.bits) {
            case (.pred, _): initv = "false"
            case (.f, 64) where options.fp64 == .df64: initv = "_df(0.0f, 0.0f)"
            default: initv = "0"
            }
            src += "  \(msl(t)) \(regName(r)) = \(initv);\n"
        }
        if usesCarry { src += "  uint _cc = 0u;\n" }
        src += "  (void)_lane; (void)_warp; (void)_nwarp; (void)_nctaid; (void)_params;\n"
        if convergent {
            src += "  uint _pc = 0u;\n  for (;;) {\n  uint _cur = simd_min(_pc);\n  if (_cur == 0xffffffffu) return;\n"
            src += "  if (_pc == _cur) {\n  switch (_cur) {\n"
            src += body.joined(separator: "\n") + "\n"
            src += "  default: _pc = 0xffffffffu; break;\n  }\n  }\n  }\n}\n"
        } else if structured {
            src += "  uint _pc = 0u;\n  for (;;) {\n  switch (_pc) {\n"
            src += body.joined(separator: "\n") + "\n"
            src += "  default: return;\n  }\n  }\n}\n"
        } else {
            src += body.joined(separator: "\n") + "\n}\n"
        }
        return TranslatedKernel(
            entry: fn.name, mslName: name, msl: src, params: params, paramBytes: UInt32(paramBytes),
            staticSharedBytes: UInt32(module.externShared.isEmpty ? sharedSize : dynamicBase),
            usesSharedMemory: needSmem, usesFP64: usesFP64, fp64Lowered: usesFP64 && options.fp64 == .fp32,
            warnings: warnings)
    }

    func isDF(_ t: PT) -> Bool { t == .f64 && options.fp64 == .df64 }

    // MARK: Instructions

    mutating func emit(_ i: Inst) throws -> [String] {
        let ops = i.operands
        func need(_ n: Int) throws { if ops.count < n { throw fail(i, "expected \(n) operands") } }
        let types = i.types
        let t = types.last ?? .b32
        if [Rnd.rz, .rm, .rp].contains(where: { i.has($0.rawValue) }) && t.isFloat && i.op != "cvt" {
            warn("directed rounding modes (.rz/.rm/.rp) are not available in Metal; round-to-nearest is used")
        }

        switch i.op {
        // ---------------------------------------------------------------- data movement
        case "ld", "ldu":
            try need(2)
            let sp = try space(i)
            let vol = i.has("volatile")
            let addr = try address(ops[1], sp, i)
            let raw = rawType(t)
            if case .vec(let elems) = ops[0] {
                var out = [String]()
                for (k, e) in elems.enumerated() {
                    if case .sink = e { continue }
                    let a = k == 0 ? addr : "(\(addr) + \(k * t.bytes)\(sp == .global || sp == .generic ? "ul" : "u"))"
                    let dt = try destType(e, i)
                    let r = regName(regNameOf(e))
                    let tc = self.tc
                    out += access(sp, a, raw, volatile: vol) { "\(r) = \(tc.convert($0, from: raw, to: dt));" }
                }
                return out
            }
            let dt = try destType(ops[0], i)
            let r = regName(regNameOf(ops[0]))
            let tc = self.tc
            return access(sp, addr, raw, volatile: vol) { "\(r) = \(tc.convert($0, from: raw, to: dt));" }

        case "st":
            try need(2)
            let sp = try space(i)
            guard sp != .param else { throw fail(i, "st.param (device function calls) is not supported") }
            let vol = i.has("volatile")
            let addr = try address(ops[0], sp, i)
            let raw = rawType(t)
            if case .vec(let elems) = ops[1] {
                var out = [String]()
                for (k, e) in elems.enumerated() {
                    let a = k == 0 ? addr : "(\(addr) + \(k * t.bytes)\(sp == .global || sp == .generic ? "ul" : "u"))"
                    let v = try val(e, raw, i)
                    out += access(sp, a, raw, volatile: vol) { "\($0) = \(v);" }
                }
                return out
            }
            let v = try val(ops[1], raw, i)
            return access(sp, addr, raw, volatile: vol) { "\($0) = \(v);" }

        case "mov":
            try need(2)
            if case .vec(let elems) = ops[1] {  // pack
                let et = PT(k: .b, bits: t.bits / elems.count)
                let vt = "\(uint(et.bits))\(elems.count)"
                let parts = try elems.map { try val($0, et, i) }
                return [try set(ops[0], "as_type<\(uint(t.bits))>(\(vt)(\(parts.joined(separator: ", "))))", PT(k: .b, bits: t.bits), i)]
            }
            if case .vec(let elems) = ops[0] {  // unpack
                let et = PT(k: .b, bits: t.bits / elems.count)
                let vt = "\(uint(et.bits))\(elems.count)"
                tmp += 1
                let v = "_v\(tmp)"
                var out = ["{ \(vt) \(v) = as_type<\(vt)>(\(try val(ops[1], PT(k: .b, bits: t.bits), i)));"]
                let comps = ["x", "y", "z", "w"]
                for (k, e) in elems.enumerated() {
                    if case .sink = e { continue }
                    out.append("  " + (try set(e, "\(v).\(comps[k])", et, i)))
                }
                out.append("}")
                return out
            }
            return [try set(ops[0], try val(ops[1], t, i), t, i)]

        case "cvta":
            try need(2)
            let a = try val(ops[1], .u64, i)
            let toGeneric = !i.has("to")
            var e: String
            if i.has("shared") {
                usesShared = true
                e = toGeneric ? "(\(a) | 0x8000000000000000ul)" : "(\(a) & 0xfffffffful)"
            } else if i.has("local") {
                usesLocal = true
                e = toGeneric ? "(\(a) | 0x4000000000000000ul)" : "(\(a) & 0xfffffffful)"
            } else if i.has("global") {
                e = a
            } else {
                throw fail(i, "cvta for this state space is not supported")
            }
            return [try set(ops[0], e, .u64, i)]

        // ---------------------------------------------------------------- integer / float arithmetic
        case "add", "sub":
            try need(3)
            let isAdd = i.op == "add"
            if t.isFloat {
                let a = try val(ops[1], t, i), b = try val(ops[2], t, i)
                var e = isDF(t) ? "_df_\(isAdd ? "add" : "sub")(\(a), \(b))" : "(\(a) \(isAdd ? "+" : "-") \(b))"
                if i.has("sat"), !isDF(t) { e = "saturate(\(e))" }
                return [try set(ops[0], e, t, i)]
            }
            if i.has("cc") || ["addc", "subc"].contains(i.op) { return try carry(i) }
            if i.has("sat"), t == .s32 {
                let a = try val(ops[1], .s32, i), b = try val(ops[2], .s32, i)
                return [try set(ops[0], "int(clamp(long(\(a)) \(isAdd ? "+" : "-") long(\(b)), -2147483648l, 2147483647l))", .s32, i)]
            }
            let u = t.unsigned
            let w = t.bits <= 32 ? "uint" : "ulong"
            let a = try val(ops[1], u, i), b = try val(ops[2], u, i)
            return [try set(ops[0], "\(msl(u))(\(w)(\(a)) \(isAdd ? "+" : "-") \(w)(\(b)))", u, i)]

        case "addc", "subc":
            return try carry(i)

        case "mul":
            try need(3)
            if t.isFloat {
                let a = try val(ops[1], t, i), b = try val(ops[2], t, i)
                var e = isDF(t) ? "_df_mul(\(a), \(b))" : "(\(a) * \(b))"
                if i.has("sat"), !isDF(t) { e = "saturate(\(e))" }
                return [try set(ops[0], e, t, i)]
            }
            let (prod, pt) = try intMul(i, t, ops[1], ops[2])
            return [try set(ops[0], prod, pt, i)]

        case "mad", "fma":
            try need(4)
            if t.isFloat {
                if i.op == "mad" && !i.mods.contains(where: { ["rn", "rz", "rm", "rp"].contains($0) }) && module.target.contains("sm_1") {
                    warn("mad.f32 without rounding (sm_1x) is executed as fused multiply-add")
                }
                let a = try val(ops[1], t, i), b = try val(ops[2], t, i), c = try val(ops[3], t, i)
                var e = isDF(t) ? "_df_fma(\(a), \(b), \(c))" : "fma(\(a), \(b), \(c))"
                if i.has("sat"), !isDF(t) { e = "saturate(\(e))" }
                return [try set(ops[0], e, t, i)]
            }
            let (prod, pt) = try intMul(i, t, ops[1], ops[2])
            let c = try val(ops[3], pt.unsigned, i)
            let w = pt.bits <= 32 ? "uint" : "ulong"
            return [try set(ops[0], "\(msl(pt.unsigned))(\(w)(\(prod)) + \(w)(\(c)))", pt.unsigned, i)]

        case "mul24", "mad24":
            try need(i.op == "mad24" ? 4 : 3)
            let a = try val(ops[1], t, i), b = try val(ops[2], t, i)
            let ext = t.isSigned ? "(int(uint(\(a)) << 8) >> 8)" : "(\(a) & 0xffffffu)"
            let ext2 = t.isSigned ? "(int(uint(\(b)) << 8) >> 8)" : "(\(b) & 0xffffffu)"
            var e = i.has("hi")
                ? "uint((\(t.isSigned ? "long" : "ulong")(\(ext)) * \(t.isSigned ? "long" : "ulong")(\(ext2))) >> 16)"
                : "uint(\(ext)) * uint(\(ext2))"
            if i.op == "mad24" { e = "(\(e) + uint(\(try val(ops[3], t, i))))" }
            return [try set(ops[0], e, .u32, i)]

        case "sad":
            try need(4)
            let a = try val(ops[1], t, i), b = try val(ops[2], t, i), c = try val(ops[3], t, i)
            return [try set(ops[0], "\(msl(t))(absdiff(\(a), \(b)) + \(c))", t, i)]

        case "div":
            try need(3)
            let a = try val(ops[1], t, i), b = try val(ops[2], t, i)
            if t.isFloat {
                if isDF(t) { return [try set(ops[0], "_df_div(\(a), \(b))", t, i)] }
                let fast = i.has("approx") || i.has("full")
                return [try set(ops[0], fast ? "fast::divide(\(a), \(b))" : "precise::divide(\(a), \(b))", t, i)]
            }
            return [try set(ops[0], "(\(b) == 0 ? \(msl(t))(\(t.isSigned ? "-1" : "~0u")) : \(msl(t))(\(a) / \(b)))", t, i)]

        case "rem":
            try need(3)
            let a = try val(ops[1], t, i), b = try val(ops[2], t, i)
            return [try set(ops[0], "(\(b) == 0 ? \(a) : \(msl(t))(\(a) % \(b)))", t, i)]

        case "min", "max":
            try need(3)
            let a = try val(ops[1], t, i), b = try val(ops[2], t, i)
            if isDF(t) { return [try set(ops[0], "_df_\(i.op)(\(a), \(b))", t, i)] }
            if t.isFloat { return [try set(ops[0], "f\(i.op)(\(a), \(b))", t, i)] }
            let ct = t.k == .b ? t.unsigned : t
            return [try set(ops[0], "\(i.op)(\(try val(ops[1], ct, i)), \(try val(ops[2], ct, i)))", ct, i)]

        case "abs":
            try need(2)
            let a = try val(ops[1], t, i)
            if isDF(t) { return [try set(ops[0], "_df_abs(\(a))", t, i)] }
            if t.isFloat { return [try set(ops[0], "fabs(\(a))", t, i)] }
            let st = t.signed
            let sa = try val(ops[1], st, i)
            return [try set(ops[0], "\(msl(st))(\(sa) < 0 ? \(msl(t.unsigned))(0) - \(msl(t.unsigned))(\(sa)) : \(msl(t.unsigned))(\(sa)))", st, i)]

        case "neg":
            try need(2)
            if isDF(t) { return [try set(ops[0], "_df_neg(\(try val(ops[1], t, i)))", t, i)] }
            if t.isFloat { return [try set(ops[0], "(-\(try val(ops[1], t, i)))", t, i)] }
            let u = t.unsigned
            return [try set(ops[0], "\(msl(u))(\(t.bits <= 32 ? "0u" : "0ul") - \(try val(ops[1], u, i)))", u, i)]

        case "and", "or", "xor":
            try need(3)
            if t.k == .pred {
                let sym = i.op == "and" ? "&&" : (i.op == "or" ? "||" : "!=")
                return [try set(ops[0], "(\(try val(ops[1], .pred, i)) \(sym) \(try val(ops[2], .pred, i)))", .pred, i)]
            }
            let u = t.unsigned
            let sym = i.op == "and" ? "&" : (i.op == "or" ? "|" : "^")
            return [try set(ops[0], "\(msl(u))(\(try val(ops[1], u, i)) \(sym) \(try val(ops[2], u, i)))", u, i)]

        case "not":
            try need(2)
            if t.k == .pred { return [try set(ops[0], "(!\(try val(ops[1], .pred, i)))", .pred, i)] }
            return [try set(ops[0], "\(msl(t.unsigned))(~\(try val(ops[1], t.unsigned, i)))", t.unsigned, i)]

        case "cnot":
            try need(2)
            return [try set(ops[0], "\(msl(t.unsigned))(\(try val(ops[1], t.unsigned, i)) == 0 ? 1 : 0)", t.unsigned, i)]

        case "shl", "shr":
            try need(3)
            let n = try val(ops[2], .u32, i)
            let bits = t.bits
            if i.op == "shl" || !t.isSigned {
                let u = t.unsigned
                let a = try val(ops[1], u, i)
                let sh = i.op == "shl" ? "<<" : ">>"
                if case .imm(let k) = ops[2], k >= 0, k < bits { return [try set(ops[0], "\(msl(u))(\(a) \(sh) \(k)u)", u, i)] }
                return [try set(ops[0], "(\(n) >= \(bits)u ? \(msl(u))(0) : \(msl(u))(\(a) \(sh) \(n)))", u, i)]
            }
            let a = try val(ops[1], t, i)
            return [try set(ops[0], "\(msl(t))(\(a) >> min(\(n), \(bits - 1)u))", t, i)]

        case "shf":
            try need(4)
            let a = try val(ops[1], .u32, i), b = try val(ops[2], .u32, i), c = try val(ops[3], .u32, i)
            let wrap = i.has("wrap") ? "true" : "false"
            return [try set(ops[0], "_shf_\(i.has("l") ? "l" : "r")(\(a), \(b), \(c), \(wrap))", .u32, i)]

        case "popc":
            try need(2)
            return [try set(ops[0], "uint(popcount(\(try val(ops[1], t.unsigned, i))))", .u32, i)]
        case "clz":
            try need(2)
            return [try set(ops[0], "uint(clz(\(try val(ops[1], t.unsigned, i))))", .u32, i)]
        case "brev":
            try need(2)
            return [try set(ops[0], "reverse_bits(\(try val(ops[1], t.unsigned, i)))", t.unsigned, i)]
        case "bfind":
            try need(2)
            let f = t.bits == 64 ? "_bfind64" : "_bfind32"
            return [try set(ops[0], "\(f)(\(try val(ops[1], t.unsigned, i)), \(t.isSigned), \(i.has("shiftamt")))", .u32, i)]
        case "bfe":
            try need(4)
            let f = t.bits == 64 ? "_bfe64" : "_bfe32"
            let e = "\(f)(\(try val(ops[1], t.unsigned, i)), \(try val(ops[2], .u32, i)), \(try val(ops[3], .u32, i)), \(t.isSigned))"
            return [try set(ops[0], e, t.unsigned, i)]
        case "bfi":
            try need(5)
            let f = t.bits == 64 ? "_bfi64" : "_bfi32"
            let e = "\(f)(\(try val(ops[1], t.unsigned, i)), \(try val(ops[2], t.unsigned, i)), \(try val(ops[3], .u32, i)), \(try val(ops[4], .u32, i)))"
            return [try set(ops[0], e, t.unsigned, i)]
        case "prmt":
            try need(4)
            if i.mods.contains(where: { ["f4e", "b4e", "rc8", "ecl", "ecr", "rc16"].contains($0) }) {
                throw fail(i, "prmt special modes are not supported")
            }
            return [try set(ops[0], "_prmt(\(try val(ops[1], .u32, i)), \(try val(ops[2], .u32, i)), \(try val(ops[3], .u32, i)))", .u32, i)]
        case "lop3":
            try need(5)
            guard case .imm(let lut) = ops[4] else { throw fail(i, "lop3 needs an immediate LUT") }
            let u = t.unsigned
            let a = try val(ops[1], u, i), b = try val(ops[2], u, i), c = try val(ops[3], u, i)
            var terms = [String]()
            for k in 0..<8 where (lut >> k) & 1 == 1 {
                terms.append("(\(k & 4 != 0 ? a : "~\(a)") & \(k & 2 != 0 ? b : "~\(b)") & \(k & 1 != 0 ? c : "~\(c)"))")
            }
            let e = terms.isEmpty ? "0u" : (terms.count == 8 ? "0xffffffffu" : terms.joined(separator: " | "))
            return [try set(ops[0], "\(msl(u))(\(e))", u, i)]

        // ---------------------------------------------------------------- floating point specials
        case "sqrt", "rsqrt", "rcp", "sin", "cos", "ex2", "lg2", "tanh":
            try need(2)
            let a = try val(ops[1], t, i)
            if isDF(t) {
                switch i.op {
                case "sqrt": return [try set(ops[0], "_df_sqrt(\(a))", t, i)]
                case "rcp": return [try set(ops[0], "_df_rcp(\(a))", t, i)]
                case "rsqrt": return [try set(ops[0], "_df_rcp(_df_sqrt(\(a)))", t, i)]
                default: throw fail(i, "\(i.op).f64 is not supported")
                }
            }
            let approx = i.has("approx")
            let e: String
            switch i.op {
            case "sqrt": e = approx ? "fast::sqrt(\(a))" : "precise::sqrt(\(a))"
            case "rsqrt": e = approx ? "fast::rsqrt(\(a))" : "precise::rsqrt(\(a))"
            case "rcp": e = approx ? "fast::divide(1.0f, \(a))" : "precise::divide(1.0f, \(a))"
            case "sin": e = "fast::sin(\(a))"
            case "cos": e = "fast::cos(\(a))"
            case "ex2": e = approx ? "fast::exp2(\(a))" : "precise::exp2(\(a))"
            case "lg2": e = "fast::log2(\(a))"
            default: e = "fast::tanh(\(a))"
            }
            let f = t.bits == 16 ? "half(\(e.replacingOccurrences(of: a, with: "float(\(a))")))" : e
            return [try set(ops[0], f, t, i)]

        case "copysign":
            try need(3)
            if isDF(t) {
                let a = try val(ops[1], t, i), b = try val(ops[2], t, i)
                return [try set(ops[0], "((\(a).hi < 0.0f) != (\(b).hi < 0.0f) ? _df_neg(\(b)) : \(b))", t, i)]
            }
            return [try set(ops[0], "copysign(\(try val(ops[2], t, i)), \(try val(ops[1], t, i)))", t, i)]

        case "testp":
            try need(2)
            let a = isDF(t) ? "\(try val(ops[1], t, i)).hi" : try val(ops[1], t, i)
            let e: String
            if i.has("finite") { e = "isfinite(\(a))" } else if i.has("infinite") { e = "isinf(\(a))" } else if i.has("number") {
                e = "(!isnan(\(a)))"
            } else if i.has("notanumber") { e = "isnan(\(a))" } else if i.has("normal") { e = "isnormal(\(a))" } else if i.has("subnormal") {
                e = "(\(a) != 0 && !isnormal(\(a)) && isfinite(\(a)))"
            } else { throw fail(i, "unknown testp") }
            return [try set(ops[0], e, .pred, i)]

        // ---------------------------------------------------------------- comparison / select
        case "setp", "set":
            try need(3)
            let cmpOps = ["eq", "ne", "lt", "le", "gt", "ge", "lo", "ls", "hi", "hs", "equ", "neu", "ltu", "leu", "gtu", "geu", "num", "nan"]
            guard let c = i.mods.first(where: { cmpOps.contains($0) }) else { throw fail(i, "missing comparison") }
            let boolOp = i.mods.first(where: { ["and", "or", "xor"].contains($0) })
            let st = i.op == "set" ? (types.count > 1 ? types[1] : t) : t  // set.CmpOp.dtype.stype
            let ct = st.k == .b ? st.unsigned : st
            let a = try val(ops[1], ct, i), b = try val(ops[2], ct, i)
            var e = compare(c, a, b, ct)
            var ne = "(!\(e))"
            if let bo = boolOp {
                try need(4)
                let p = try val(ops[3], .pred, i)
                let sym = bo == "and" ? "&&" : (bo == "or" ? "||" : "!=")
                e = "(\(e) \(sym) \(p))"
                ne = "(\(ne) \(sym) \(p))"
            }
            if i.op == "set" {
                let dt = types.first ?? .u32
                let tv = dt.isFloat ? "1.0f" : "0xffffffffu"
                return [try set(ops[0], "(\(e) ? \(tv) : 0)", dt.isFloat ? .f32 : .u32, i)]
            }
            if case .pair(let p, let q) = ops[0] {
                return [try set(p, e, .pred, i), try set(q, ne, .pred, i)]
            }
            return [try set(ops[0], e, .pred, i)]

        case "selp":
            try need(4)
            let a = try val(ops[1], t, i), b = try val(ops[2], t, i), p = try val(ops[3], .pred, i)
            return [try set(ops[0], "(\(p) ? \(a) : \(b))", t, i)]

        case "slct":
            try need(4)
            let ct = types.count > 1 ? types[1] : .s32
            let a = try val(ops[1], types[0], i), b = try val(ops[2], types[0], i), c = try val(ops[3], ct, i)
            return [try set(ops[0], "(\(c) >= 0 ? \(a) : \(b))", types[0], i)]

        // ---------------------------------------------------------------- conversion
        case "cvt":
            try need(2)
            return [try cvt(i)]

        // ---------------------------------------------------------------- control flow
        case "bra":
            try need(1)
            guard case .sym(let l) = ops[0], let target = labels[l] else { throw fail(i, "unknown branch target") }
            return [convergent ? "{ _pc = \(target)u; break; }" : "{ _pc = \(target)u; continue; }"]
        case "ret", "exit":
            return [convergent ? "{ _pc = 0xffffffffu; break; }" : "return;"]
        case "trap":
            warn("trap is lowered to an early return (Metal has no device-side abort)")
            return [convergent ? "{ _pc = 0xffffffffu; break; }" : "return;"]
        case "brkpt", "nanosleep", "pmevent", "prefetch", "prefetchu", "discard", "griddepcontrol", "applypriority", "createpolicy":
            return []
        case "call":
            throw fail(i, "non-inlined device function calls (incl. printf/vprintf) are not supported")

        // ---------------------------------------------------------------- synchronization
        case "bar", "barrier":
            if i.has("warp") { usesSimd = true; return ["simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"] }
            if i.has("arrive") || i.has("red") { throw fail(i, "bar.arrive / bar.red are not supported") }
            if ops.count > 1 { warn("bar.sync with a thread count is executed as a full threadgroup barrier") }
            return ["threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"]
        case "membar", "fence":
            if i.has("cta") {
                return ["atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup);"]
            }
            return ["atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device);"]

        case "atom", "red":
            return try atomic(i)

        // ---------------------------------------------------------------- warp (SIMD-group) ops
        case "shfl":
            return try shfl(i)
        case "vote":
            usesSimd = true
            let p = try val(ops[1], .pred, i)
            if i.has("ballot") {
                return [try set(ops[0], "uint(static_cast<simd_vote::vote_t>(simd_ballot(\(p))))", .u32, i)]
            }
            let e: String
            if i.has("all") { e = "simd_all(\(p))" } else if i.has("any") { e = "simd_any(\(p))" } else if i.has("uni") {
                e = "(simd_all(\(p)) || !simd_any(\(p)))"
            } else { throw fail(i, "unknown vote mode") }
            return [try set(ops[0], e, .pred, i)]
        case "activemask":
            usesSimd = true
            return [try set(ops[0], "uint(static_cast<simd_vote::vote_t>(simd_active_threads_mask()))", .u32, i)]
        case "redux":
            usesSimd = true
            try need(2)
            let a = try val(ops[1], t, i)
            let f: String
            if i.has("add") { f = "simd_sum" } else if i.has("min") { f = "simd_min" } else if i.has("max") { f = "simd_max" } else if i.has("and") {
                f = "simd_and"
            } else if i.has("or") { f = "simd_or" } else if i.has("xor") { f = "simd_xor" } else { throw fail(i, "unknown redux op") }
            return [try set(ops[0], "\(f)(\(a))", t, i)]

        case "tex", "tld4", "txq", "suld", "sust", "sured", "suq", "mma", "wmma", "ldmatrix", "stmatrix", "cp", "wgmma", "tcgen05", "mbarrier",
            "cvt.pack", "elect", "setmaxnreg", "stackrestore", "stacksave", "alloca", "match", "dp4a", "dp2a":
            throw fail(i, "'\(i.op)' (textures / surfaces / tensor cores / async copies / clusters) has no MetalPassthrough lowering yet")
        default:
            throw fail(i, "unsupported instruction '\(i.op)'")
        }
    }

    enum Rnd: String { case rn, rz, rm, rp }

    func regNameOf(_ o: Operand) -> String {
        if case .reg(let r) = o { return r }
        return "_"
    }

    /// Integer multiply; returns (expression, result type).
    mutating func intMul(_ i: Inst, _ t: PT, _ x: Operand, _ y: Operand) throws -> (String, PT) {
        if i.has("wide") {
            let wt = PT(k: t.k, bits: t.bits * 2)
            let wm = msl(wt)
            return ("\(wm)(\(wm)(\(try val(x, t, i))) * \(wm)(\(try val(y, t, i))))", wt)
        }
        if i.has("hi") {
            if t.bits == 16 {
                let w = t.isSigned ? "int" : "uint"
                return ("\(msl(t))((\(w)(\(try val(x, t, i))) * \(w)(\(try val(y, t, i)))) >> 16)", t)
            }
            return ("mulhi(\(try val(x, t, i)), \(try val(y, t, i)))", t)
        }
        let u = t.unsigned
        let w = t.bits <= 32 ? "uint" : "ulong"
        return ("\(msl(u))(\(w)(\(try val(x, u, i))) * \(w)(\(try val(y, u, i))))", u)
    }

    mutating func carry(_ i: Inst) throws -> [String] {
        usesCarry = true
        let t = i.types.last ?? .u32
        guard t.bits == 32 else { throw fail(i, "64-bit carry chains are not supported") }
        let a = try val(i.operands[1], .u32, i), b = try val(i.operands[2], .u32, i)
        let isAdd = i.op.hasPrefix("add")
        let withCarry = i.op == "addc" || i.op == "subc"
        let writes = i.has("cc")
        tmp += 1
        let s = "_c\(tmp)"
        var out = [String]()
        if isAdd {
            out.append("{ ulong \(s) = ulong(\(a)) + ulong(\(b))\(withCarry ? " + ulong(_cc)" : "");")
            if writes { out.append("  _cc = uint(\(s) >> 32);") }
        } else {
            out.append("{ ulong \(s) = ulong(\(a)) - ulong(\(b))\(withCarry ? " - ulong(_cc)" : "");")
            if writes { out.append("  _cc = uint((\(s) >> 32) & 1ul);") }
        }
        out.append("  " + (try set(i.operands[0], "uint(\(s))", .u32, i)) + " }")
        return out
    }

    func compare(_ c: String, _ a: String, _ b: String, _ t: PT) -> String {
        if isDF(t) {
            let k = "_df_cmp(\(a), \(b))"
            switch c {
            case "eq": return "(\(k) == 0)"
            case "ne": return "(abs(\(k)) == 1)"
            case "lt": return "(\(k) == -1)"
            case "le": return "(\(k) == -1 || \(k) == 0)"
            case "gt": return "(\(k) == 1)"
            case "ge": return "(\(k) == 1 || \(k) == 0)"
            case "equ": return "(\(k) == 0 || \(k) == 2)"
            case "neu": return "(\(k) != 0)"
            case "ltu": return "(\(k) == -1 || \(k) == 2)"
            case "leu": return "(\(k) != 1)"
            case "gtu": return "(\(k) == 1 || \(k) == 2)"
            case "geu": return "(\(k) != -1)"
            case "num": return "(\(k) != 2)"
            default: return "(\(k) == 2)"
            }
        }
        if t.isFloat {
            switch c {
            case "eq": return "(\(a) == \(b))"
            case "ne": return "(\(a) < \(b) || \(a) > \(b))"
            case "lt": return "(\(a) < \(b))"
            case "le": return "(\(a) <= \(b))"
            case "gt": return "(\(a) > \(b))"
            case "ge": return "(\(a) >= \(b))"
            case "equ": return "(!(\(a) < \(b) || \(a) > \(b)))"
            case "neu": return "(\(a) != \(b))"
            case "ltu": return "(!(\(a) >= \(b)))"
            case "leu": return "(!(\(a) > \(b)))"
            case "gtu": return "(!(\(a) <= \(b)))"
            case "geu": return "(!(\(a) < \(b)))"
            case "num": return "(!isnan(\(a)) && !isnan(\(b)))"
            default: return "(isnan(\(a)) || isnan(\(b)))"
            }
        }
        switch c {
        case "eq": return "(\(a) == \(b))"
        case "ne": return "(\(a) != \(b))"
        case "lt", "lo": return "(\(a) < \(b))"
        case "le", "ls": return "(\(a) <= \(b))"
        case "gt", "hi": return "(\(a) > \(b))"
        default: return "(\(a) >= \(b))"
        }
    }

    mutating func cvt(_ i: Inst) throws -> String {
        let ts = i.types
        guard ts.count >= 2 else { throw fail(i, "cvt needs destination and source types") }
        let dt = ts[0], at = ts[1]
        let a = try val(i.operands[1], at, i)
        let sat = i.has("sat")
        let modeOf: () -> (String, Int) = {
            if i.has("rni") || i.has("rn") { return ("rint", 0) }
            if i.has("rmi") || i.has("rm") { return ("floor", 2) }
            if i.has("rpi") || i.has("rp") { return ("ceil", 3) }
            return ("trunc", 1)
        }
        // int -> int
        if dt.isInt && at.isInt {
            var e = "\(msl(dt))(\(a))"
            if sat, dt.bits < 64 {
                let lo = dt.isSigned ? -(Int64(1) << (dt.bits - 1)) : 0
                let hi = dt.isSigned ? (Int64(1) << (dt.bits - 1)) - 1 : (Int64(1) << dt.bits) - 1
                e = "\(msl(dt))(clamp(long(\(a)), \(lo)l, \(hi)l))"
            }
            return try set(i.operands[0], e, dt, i)
        }
        // int -> float
        if dt.isFloat && at.isInt {
            if i.has("rz") || i.has("rm") || i.has("rp") { warn("cvt int->float directed rounding is executed as round-to-nearest") }
            if isDF(dt) { return try set(i.operands[0], at.isSigned ? "_df_from_s64(long(\(a)))" : "_df_from_u64(ulong(\(a)))", dt, i) }
            let ft = dt == .f64 ? "float" : msl(dt)
            return try set(i.operands[0], "\(ft)(\(a))", dt, i)
        }
        // float -> int
        if dt.isInt && at.isFloat {
            let (rf, mode) = modeOf()
            let r: String
            if isDF(at) {
                r = "_df_to_s64(\(a), \(mode))"
                let lo = dt.isSigned ? -(Int64(1) << (dt.bits - 1)) : 0
                let e = dt.bits == 64 ? "\(msl(dt))(\(r))" : "\(msl(dt))(clamp(\(r), \(lo)l, \(dt.isSigned ? (Int64(1) << (dt.bits - 1)) - 1 : (Int64(1) << dt.bits) - 1)l))"
                return try set(i.operands[0], e, dt, i)
            }
            let f = at.bits == 32 || at == .f64 ? "\(rf)(\(a))" : "\(rf)(float(\(a)))"
            switch (dt.isSigned, dt.bits) {
            case (true, 32): r = "_f2s32(\(f))"
            case (false, 32): r = "_f2u32(\(f))"
            case (true, 64): r = "_f2s64(\(f))"
            case (false, 64): r = "_f2u64(\(f))"
            case (true, let b):
                r = "\(msl(dt))(clamp(_f2s32(\(f)), \(-(1 << (b - 1))), \((1 << (b - 1)) - 1)))"
            case (false, let b):
                r = "\(msl(dt))(min(_f2u32(\(f)), \((1 << b) - 1)u))"
            }
            return try set(i.operands[0], r, dt, i)
        }
        // float -> float
        if dt == at {
            let (rf, mode) = modeOf()
            let roundsToInt = i.has("rni") || i.has("rzi") || i.has("rmi") || i.has("rpi")
            if isDF(dt) {
                return try set(i.operands[0], roundsToInt ? "_df_round(\(a), \(mode))" : a, dt, i)
            }
            var e = roundsToInt ? "\(rf)(\(a))" : a
            if sat { e = "saturate(\(e))" }
            return try set(i.operands[0], e, dt, i)
        }
        var e = convert(a, from: at, to: dt)
        if sat, !isDF(dt) { e = "saturate(\(e))" }
        return try set(i.operands[0], e, dt, i)
    }

    mutating func atomic(_ i: Inst) throws -> [String] {
        let isRed = i.op == "red"
        let ops = i.operands
        let t = i.types.last ?? .u32
        let sp = try space(i)
        guard sp == .global || sp == .shared || sp == .generic else { throw fail(i, "atomics on this state space are not supported") }
        let opNames = ["add", "min", "max", "inc", "dec", "and", "or", "xor", "exch", "cas"]
        guard let op = i.mods.first(where: { opNames.contains($0) }) else { throw fail(i, "unknown atomic op") }
        guard t.bits == 32 else {
            throw fail(i, "\(t.bits)-bit atomics are not supported by Metal (only 32-bit integer/float atomics)")
        }
        if i.mods.contains(where: { ["acquire", "release", "acq_rel"].contains($0) }) {
            warn("atomic memory-order qualifiers are executed as relaxed atomics plus device fences")
        }
        let addrOp = isRed ? ops[0] : ops[1]
        let vOps = isRed ? Array(ops[1...]) : Array(ops[2...])
        let addr = try address(addrOp, sp, i)
        let isFloat = t == .f32
        if isFloat && !["add", "exch"].contains(op) { throw fail(i, "atomic \(op).f32 is not supported") }
        let vt: PT = isFloat ? .f32 : (t.isSigned ? .s32 : .u32)
        let v = try val(vOps[0], vt, i)
        let c = op == "cas" ? try val(vOps[1], vt, i) : ""

        func expr(_ s: Space, _ a: String) -> String {
            let dev = s == .global
            let qual = dev ? "device" : "threadgroup"
            let ptr: String = dev
                ? "reinterpret_cast<device \(isFloat ? "atomic_float" : (t.isSigned ? "atomic_int" : "atomic_uint"))*>(\(a))"
                : "reinterpret_cast<threadgroup \(isFloat ? "atomic_uint" : (t.isSigned ? "atomic_int" : "atomic_uint"))*>(_smem + (\(a)))"
            let uptr: String = dev ? "reinterpret_cast<device atomic_uint*>(\(a))" : "reinterpret_cast<threadgroup atomic_uint*>(_smem + (\(a)))"
            let sfx = dev ? "d" : "t"
            _ = qual
            switch op {
            case "add":
                if isFloat && !dev { return "_addf_t(\(uptr), \(v))" }
                return "atomic_fetch_add_explicit(\(ptr), \(v), memory_order_relaxed)"
            case "min", "max", "and", "or", "xor":
                return "atomic_fetch_\(op)_explicit(\(ptr), \(v), memory_order_relaxed)"
            case "exch":
                if isFloat && !dev { return "as_type<float>(atomic_exchange_explicit(\(uptr), as_type<uint>(\(v)), memory_order_relaxed))" }
                return "atomic_exchange_explicit(\(ptr), \(v), memory_order_relaxed)"
            case "cas":
                if isFloat { return "as_type<float>(_cas_\(sfx)(\(uptr), as_type<uint>(\(v)), as_type<uint>(\(c))))" }
                return "\(tc.msl(vt))(_cas_\(sfx)(\(uptr), uint(\(v)), uint(\(c))))"
            case "inc": return "_inc_\(sfx)(\(uptr), uint(\(v)))"
            default: return "_dec_\(sfx)(\(uptr), uint(\(v)))"
            }
        }
        var destInfo: (String, PT)?
        if !isRed, case .reg(let r) = ops[0], let rt = regs[r] { destInfo = (regName(r), rt) }
        let tc = self.tc
        let assign: (String) -> String = { e in
            if let (n, rt) = destInfo { return "\(n) = \(tc.convert(e, from: vt, to: rt));" }
            return "(void)(\(e));"
        }
        switch sp {
        case .global: return [assign(expr(.global, addr))]
        case .shared:
            usesShared = true
            return [assign(expr(.shared, addr))]
        default:
            tmp += 1
            let a = "_aa\(tmp)"
            if sharedOffset.isEmpty && module.externShared.isEmpty { return [assign(expr(.global, addr))] }
            usesShared = true
            return [
                "{ ulong \(a) = \(addr);", "  if ((\(a) >> 62) == 2ul) { \(assign(expr(.shared, "uint(\(a))"))) }",
                "  else { \(assign(expr(.global, a))) } }",
            ]
        }
    }

    mutating func shfl(_ i: Inst) throws -> [String] {
        usesSimd = true
        let ops = i.operands
        let sync = i.has("sync")
        guard ops.count >= (sync ? 5 : 4) else { throw fail(i, "shfl needs 4-5 operands") }
        let a = try val(ops[1], .u32, i), b = try val(ops[2], .u32, i), c = try val(ops[3], .u32, i)
        tmp += 1
        let n = tmp
        var out = [
            "{ uint _b\(n) = (\(b)) & 31u, _cv\(n) = (\(c)) & 31u, _seg\(n) = ((\(c)) >> 8) & 31u;",
            "  uint _max\(n) = (_lane & _seg\(n)) | (_cv\(n) & ~_seg\(n)); uint _min\(n) = _lane & _seg\(n); int _j\(n); bool _ok\(n);",
        ]
        if i.has("up") {
            out.append("  _j\(n) = int(_lane) - int(_b\(n)); _ok\(n) = _j\(n) >= int(_max\(n));")
        } else if i.has("down") {
            out.append("  _j\(n) = int(_lane + _b\(n)); _ok\(n) = _j\(n) <= int(_max\(n));")
        } else if i.has("bfly") {
            out.append("  _j\(n) = int(_lane ^ _b\(n)); _ok\(n) = _j\(n) <= int(_max\(n));")
        } else if i.has("idx") {
            out.append("  _j\(n) = int(_min\(n) | (_b\(n) & ~_seg\(n))); _ok\(n) = _j\(n) <= int(_max\(n));")
        } else {
            throw fail(i, "unknown shfl mode")
        }
        out.append("  if (!_ok\(n)) _j\(n) = int(_lane); (void)_min\(n);")
        out.append("  uint _s\(n) = simd_shuffle(\(a), ushort(_j\(n)));")
        if case .pair(let d, let p) = ops[0] {
            out.append("  " + (try set(d, "_s\(n)", .u32, i)))
            out.append("  " + (try set(p, "_ok\(n)", .pred, i)))
        } else {
            out.append("  " + (try set(ops[0], "_s\(n)", .u32, i)))
        }
        out.append("}")
        return out
    }
}


/// Pure type mapping / register reinterpretation logic (captured by codegen closures).
struct TypeConv {
    let fp64: PTXTranslateOptions.FP64Mode


    func msl(_ t: PT) -> String {
        switch t.k {
        case .pred: return "bool"
        case .f:
            switch t.bits {
            case 16: return "half"
            case 64: return fp64 == .df64 ? "df64" : "float"
            default: return "float"
            }
        case .bf: return "bfloat"
        case .s:
            switch t.bits { case 8: return "char"; case 16: return "short"; case 64: return "long"; default: return "int" }
        case .u, .b:
            switch t.bits { case 8: return "uchar"; case 16: return "ushort"; case 64: return "ulong"; default: return "uint" }
        }
    }

    func uint(_ bits: Int) -> String {
        switch bits { case 8: return "uchar"; case 16: return "ushort"; case 64: return "ulong"; default: return "uint" }
    }

    /// Register-level reinterpretation / resize (not a numeric `cvt`).
    func convert(_ e: String, from: PT, to: PT) -> String {
        if from == to || msl(from) == msl(to) && from.isFloat == to.isFloat && from.bits == to.bits { return e }
        if to.k == .pred { return "((\(e)) != 0)" }
        if from.k == .pred { return "\(msl(to))((\(e)) ? 1 : 0)" }
        if from.isInt && to.isInt { return "\(msl(to))(\(e))" }
        if from.isFloat && to.isFloat {
            if from == .f64 {
                let f = fp64 == .df64 ? "_df_to_f32(\(e))" : "(\(e))"
                return to.bits == 32 ? f : "\(msl(to))(\(f))"
            }
            if to == .f64 {
                let f = from.bits == 32 ? e : "float(\(e))"
                return fp64 == .df64 ? "_df_from_f32(\(f))" : f
            }
            return "\(msl(to))(\(e))"
        }
        if from.isInt && to.isFloat {
            if to == .f64 {
                let b = from.bits == 64 ? "ulong(\(e))" : "ulong(\(uint(from.bits))(\(e)))"
                return fp64 == .df64 ? "_df_from_bits(\(b))" : "_f64bits_to_f32(\(b))"
            }
            let u = uint(to.bits)
            let inner = (from.bits == to.bits && !from.isSigned) ? e : "\(u)(\(e))"
            return "as_type<\(msl(to))>(\(inner))"
        }
        // float -> int bits
        if from == .f64 {
            let b = fp64 == .df64 ? "_df_to_bits(\(e))" : "_f32_to_f64bits(\(e))"
            return to.bits == 64 && !to.isSigned ? b : "\(msl(to))(\(b))"
        }
        let bitsExpr = "as_type<\(uint(from.bits))>(\(e))"
        return (to.bits == from.bits && !to.isSigned) ? bitsExpr : "\(msl(to))(\(bitsExpr))"
    }

}
