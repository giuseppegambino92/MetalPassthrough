// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// PTX front end: comment stripping, statement splitting, module / entry / instruction parsing.
// Targets the PTX that NVIDIA nvcc emits (ISA 6.x - 9.x, .address_size 64).

import Foundation

public struct PTXError: Error, CustomStringConvertible, Sendable {
    public let line: Int
    public let message: String
    public var description: String { line > 0 ? "PTX line \(line): \(message)" : message }
}

/// PTX scalar type: kind + bit width.
struct PT: Hashable, CustomStringConvertible {
    enum K: Hashable { case u, s, b, f, bf, pred }
    let k: K
    let bits: Int

    static let pred = PT(k: .pred, bits: 1)
    static let u8 = PT(k: .u, bits: 8), u16 = PT(k: .u, bits: 16), u32 = PT(k: .u, bits: 32), u64 = PT(k: .u, bits: 64)
    static let s8 = PT(k: .s, bits: 8), s16 = PT(k: .s, bits: 16), s32 = PT(k: .s, bits: 32), s64 = PT(k: .s, bits: 64)
    static let b16 = PT(k: .b, bits: 16), b32 = PT(k: .b, bits: 32), b64 = PT(k: .b, bits: 64)
    static let f16 = PT(k: .f, bits: 16), f32 = PT(k: .f, bits: 32), f64 = PT(k: .f, bits: 64)

    static func parse(_ s: String) -> PT? {
        switch s {
        case "pred": return .pred
        case "u8": return .u8
        case "u16": return .u16
        case "u32": return .u32
        case "u64": return .u64
        case "s8": return .s8
        case "s16": return .s16
        case "s32": return .s32
        case "s64": return .s64
        case "b8": return PT(k: .b, bits: 8)
        case "b16": return .b16
        case "b32": return .b32
        case "b64": return .b64
        case "b128": return PT(k: .b, bits: 128)
        case "f16": return .f16
        case "bf16": return PT(k: .bf, bits: 16)
        case "f32": return .f32
        case "f64": return .f64
        default: return nil
        }
    }

    var isInt: Bool { k == .u || k == .s || k == .b }
    var isFloat: Bool { k == .f || k == .bf }
    var isSigned: Bool { k == .s }
    var bytes: Int { max(1, bits / 8) }
    var unsigned: PT { PT(k: .u, bits: bits) }
    var signed: PT { PT(k: .s, bits: bits) }
    var description: String { k == .pred ? "pred" : "\(k)\(bits)" }
}

indirect enum Operand: CustomStringConvertible {
    case reg(String)
    case special(String)
    case imm(Int64)
    case fimm32(UInt32)
    case fimm64(UInt64)
    case sym(String)
    case mem(base: Operand?, offset: Int64)
    case vec([Operand])
    case pair(Operand, Operand)
    case not(Operand)
    case sink

    var description: String {
        switch self {
        case .reg(let r): return r
        case .special(let s): return s
        case .imm(let i): return "\(i)"
        case .fimm32(let b): return String(format: "0f%08X", b)
        case .fimm64(let b): return String(format: "0d%016llX", b)
        case .sym(let s): return s
        case .mem(let b, let o): return "[\(b.map { "\($0)" } ?? "")+\(o)]"
        case .vec(let v): return "{" + v.map { "\($0)" }.joined(separator: ",") + "}"
        case .pair(let a, let b): return "\(a)|\(b)"
        case .not(let o): return "!\(o)"
        case .sink: return "_"
        }
    }
}

struct Inst {
    let line: Int
    let pred: (name: String, negated: Bool)?
    let op: String
    let mods: [String]
    let operands: [Operand]
    let text: String

    var types: [PT] { mods.compactMap(PT.parse) }
    func has(_ m: String) -> Bool { mods.contains(m) }
}

struct VarDecl {
    let name: String
    let align: Int
    let size: Int
    let line: Int
}

struct ParamDecl {
    let name: String
    let type: PT?  // nil for .b8 arrays (structs)
    let size: Int
    let align: Int
}

enum BodyStmt {
    case reg(PT, [String], Int)
    case shared(VarDecl)
    case local(VarDecl)
    case label(String)
    case inst(Inst)
}

struct PTXFunction {
    let name: String
    let isEntry: Bool
    let params: [ParamDecl]
    let body: [BodyStmt]
    let line: Int
}

struct PTXModule {
    var version = ""
    var target = ""
    var addressSize = 64
    var externShared: [VarDecl] = []
    var moduleShared: [VarDecl] = []
    var functions: [String: PTXFunction] = [:]
    var entryOrder: [String] = []
}

enum PTXParser {
    // MARK: Lexing

    /// Removes // and /* */ comments while preserving newlines (for line numbers) and strings.
    static func stripComments(_ src: String) -> [UInt8] {
        let s = Array(src.utf8)
        var out = [UInt8]()
        out.reserveCapacity(s.count)
        var i = 0
        var inString = false
        while i < s.count {
            let c = s[i]
            if inString {
                out.append(c)
                if c == UInt8(ascii: "\"") { inString = false }
                i += 1
                continue
            }
            if c == UInt8(ascii: "\"") { inString = true; out.append(c); i += 1; continue }
            if c == UInt8(ascii: "/"), i + 1 < s.count, s[i + 1] == UInt8(ascii: "/") {
                while i < s.count, s[i] != UInt8(ascii: "\n") { i += 1 }
                continue
            }
            if c == UInt8(ascii: "/"), i + 1 < s.count, s[i + 1] == UInt8(ascii: "*") {
                i += 2
                while i + 1 < s.count, !(s[i] == UInt8(ascii: "*") && s[i + 1] == UInt8(ascii: "/")) {
                    if s[i] == UInt8(ascii: "\n") { out.append(s[i]) }
                    i += 1
                }
                i += 2
                continue
            }
            out.append(c)
            i += 1
        }
        return out
    }

    enum Tok {
        case stmt(String, Int)
        case open(Int)
        case close(Int)
    }

    static func isStructuralBrace(_ buffer: String) -> Bool {
        var t = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        // Drop leading labels.
        while let r = t.range(of: #"^[A-Za-z_$][A-Za-z0-9_$]*\s*:"#, options: .regularExpression) {
            t = String(t[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if t.isEmpty { return true }
        if t.contains("=") { return false }  // initializer list
        return t.hasPrefix(".") && (t.contains(".entry") || t.contains(".func"))
    }

    static func tokenize(_ src: String) -> [Tok] {
        let bytes = stripComments(src)
        var toks = [Tok]()
        var buf = [UInt8]()
        var line = 1
        var bufLine = 1
        var vecDepth = 0
        var lineStart = 0  // index in buf where the current line starts
        func flush() {
            let s = String(decoding: buf, as: UTF8.self)
            if !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { toks.append(.stmt(s, bufLine)) }
            buf.removeAll(keepingCapacity: true)
            bufLine = line
            lineStart = 0
        }
        // Directives terminated by end-of-line rather than ';'. They are handled per *line*: LLVM-based
        // producers (Triton) put `.loc` lines between a label and its instruction, so checking the
        // whole buffer would glue the directive to (and then drop) the following instruction.
        let dropLines = [".loc", ".file", ".pragma"]
        let ownStatements = [".version", ".target", ".address_size", ".section"]
        for c in bytes {
            if c == UInt8(ascii: "\n") {
                if vecDepth == 0 {
                    let cur = String(decoding: buf[lineStart...], as: UTF8.self).trimmingCharacters(in: .whitespaces)
                    if dropLines.contains(where: { cur.hasPrefix($0) }) {
                        buf.removeSubrange(lineStart...)
                    } else if ownStatements.contains(where: { cur.hasPrefix($0) }) {
                        let before = Array(buf[..<lineStart])
                        buf = before
                        flush()  // whatever precedes (labels) stays a statement of its own
                        buf = Array(cur.utf8)
                        flush()
                    }
                }
                line += 1
                buf.append(c)
                lineStart = buf.count
                continue
            }
            if buf.allSatisfy({ $0 == 32 || $0 == 9 || $0 == 10 || $0 == 13 }) { bufLine = line }
            switch c {
            case UInt8(ascii: ";"):
                flush()
                vecDepth = 0
            case UInt8(ascii: "{"):
                if vecDepth == 0, isStructuralBrace(String(decoding: buf, as: UTF8.self)) {
                    flush()
                    toks.append(.open(line))
                } else {
                    vecDepth += 1
                    buf.append(c)
                }
            case UInt8(ascii: "}"):
                if vecDepth > 0 {
                    vecDepth -= 1
                    buf.append(c)
                } else {
                    flush()
                    toks.append(.close(line))
                }
            default:
                buf.append(c)
            }
        }
        flush()
        return toks
    }

    // MARK: Module

    static func parse(_ src: String) throws -> PTXModule {
        var m = PTXModule()
        let toks = tokenize(src)
        var i = 0
        while i < toks.count {
            guard case .stmt(let raw, let line) = toks[i] else {
                if case .open = toks[i] {
                    // Module-scope brace blocks are DWARF payloads (`.section .debug_* { ... }`, emitted
                    // by Triton/LLVM with line info): skip to the matching close.
                    var depth = 0
                    repeat {
                        if case .open = toks[i] { depth += 1 } else if case .close = toks[i] { depth -= 1 }
                        i += 1
                    } while depth > 0 && i < toks.count
                    continue
                }
                i += 1
                continue
            }
            let s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            let words = s.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" || $0 == "\r" }).map(String.init)
            if s.contains(".entry") || s.contains(".func") {
                let isEntry = s.contains(".entry")
                let (name, params) = try parseHeader(s, line: line)
                // A declaration (prototype) has no body.
                if i + 1 < toks.count, case .open = toks[i + 1] {
                    var depth = 0
                    var j = i + 1
                    var bodyToks = [Tok]()
                    repeat {
                        switch toks[j] {
                        case .open: depth += 1
                        case .close: depth -= 1
                        case .stmt: break
                        }
                        if depth > 0 || { if case .close = toks[j] { return false }; return true }() { bodyToks.append(toks[j]) }
                        j += 1
                    } while depth > 0 && j < toks.count
                    let body = try parseBody(bodyToks)
                    m.functions[name] = PTXFunction(name: name, isEntry: isEntry, params: params, body: body, line: line)
                    if isEntry { m.entryOrder.append(name) }
                    i = j
                } else {
                    i += 1
                }
                continue
            }
            switch words.first ?? "" {
            case ".version": m.version = words.count > 1 ? words[1] : ""
            case ".target": m.target = words.dropFirst().joined(separator: " ")
            case ".address_size": m.addressSize = Int(words.count > 1 ? words[1] : "64") ?? 64
            case ".file", ".section", ".loc", ".pragma": break
            default:
                if words.contains(".shared") {
                    let v = try parseVar(words, line: line)
                    if words.contains(".extern") { m.externShared.append(v) } else { m.moduleShared.append(v) }
                } else if words.contains(".global") || words.contains(".const") {
                    // Module-scope device variables need cuModuleGetGlobal + program-scope mutable
                    // storage, which Metal does not offer. Rejected at use; recorded nowhere.
                    break
                }
            }
            i += 1
        }
        guard m.addressSize == 64 else { throw PTXError(line: 0, message: ".address_size \(m.addressSize) is not supported (64 only)") }
        return m
    }

    static func parseHeader(_ s: String, line: Int) throws -> (String, [ParamDecl]) {
        guard let open = s.firstIndex(of: "(") else {
            // .entry with no parameter list
            let words = s.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            guard let k = words.firstIndex(where: { $0 == ".entry" || $0 == ".func" }), k + 1 < words.count else {
                throw PTXError(line: line, message: "malformed function header")
            }
            return (words[k + 1], [])
        }
        let before = s[..<open].split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard let name = before.last else { throw PTXError(line: line, message: "malformed function header") }
        // .func may have a return parameter list first: .func (.param .b32 r) name (...)
        if before.last == ".func" || name.hasPrefix(".") {
            throw PTXError(line: line, message: ".func with return values is not supported")
        }
        guard let close = s[open...].firstIndex(of: ")") else { throw PTXError(line: line, message: "unterminated parameter list") }
        let inner = s[s.index(after: open)..<close]
        var params = [ParamDecl]()
        for p in inner.split(separator: ",") {
            let w = p.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            guard !w.isEmpty else { continue }
            var align = 0
            var type: PT?
            var rawName = w.last!
            var count = 1
            // `.param .u64 .ptr .global .align 1 p`: that .align describes the pointee, not the param.
            if let k = w.firstIndex(of: ".align"), k + 1 < w.count, !w.contains(".ptr") { align = Int(w[k + 1]) ?? 0 }
            for t in w where t.hasPrefix(".") { if let pt = PT.parse(String(t.dropFirst())) { type = pt } }
            if let lb = rawName.firstIndex(of: "[") {
                count = Int(rawName[rawName.index(after: lb)...].dropLast()) ?? 1
                rawName = String(rawName[..<lb])
            }
            guard let t = type else { throw PTXError(line: line, message: "parameter '\(rawName)' has no type") }
            let size = t.bytes * count
            params.append(
                ParamDecl(name: rawName, type: count == 1 ? t : nil, size: size, align: align > 0 ? align : t.bytes))
        }
        return (name, params)
    }

    static func parseVar(_ words: [String], line: Int) throws -> VarDecl {
        var align = 1
        var elem = 1
        if let k = words.firstIndex(of: ".align"), k + 1 < words.count { align = Int(words[k + 1]) ?? 1 }
        for t in words where t.hasPrefix(".") { if let pt = PT.parse(String(t.dropFirst())) { elem = pt.bytes } }
        guard var name = words.last else { throw PTXError(line: line, message: "malformed variable") }
        var count = 1
        if let lb = name.firstIndex(of: "[") {
            let inner = name[name.index(after: lb)...].dropLast()
            count = inner.isEmpty ? 0 : (Int(inner) ?? 0)
            name = String(name[..<lb])
        }
        return VarDecl(name: name, align: max(align, elem), size: elem * count, line: line)
    }

    // MARK: Body

    static func parseBody(_ toks: [Tok]) throws -> [BodyStmt] {
        var out = [BodyStmt]()
        for t in toks {
            guard case .stmt(let raw, let line) = t else { continue }  // nested scopes are flattened
            var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            while let r = s.range(of: #"^[A-Za-z_$][A-Za-z0-9_$]*\s*:"#, options: .regularExpression) {
                let label = s[r].dropLast().trimmingCharacters(in: .whitespaces)
                out.append(.label(label))
                s = String(s[r.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if s.isEmpty { continue }
            let words = s.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            switch words[0] {
            case ".reg":
                guard words.count >= 3, let t = PT.parse(String(words[1].dropFirst())) else {
                    if words.count >= 3, words[1].hasPrefix(".v") {
                        throw PTXError(line: line, message: "vector registers are not supported: \(s)")
                    }
                    throw PTXError(line: line, message: "bad .reg: \(s)")
                }
                var names = [String]()
                for part in words[2...].joined(separator: " ").split(separator: ",") {
                    let n = part.trimmingCharacters(in: .whitespaces)
                    if let lt = n.firstIndex(of: "<") {
                        let base = String(n[..<lt])
                        let cnt = Int(n[n.index(after: lt)...].dropLast()) ?? 0
                        for k in 0..<cnt { names.append("\(base)\(k)") }
                    } else if !n.isEmpty {
                        names.append(n)
                    }
                }
                out.append(.reg(t, names, line))
            case ".shared":
                out.append(.shared(try parseVar(words, line: line)))
            case ".local":
                out.append(.local(try parseVar(words, line: line)))
            case ".pragma", ".loc", ".maxntid", ".reqntid", ".minnctapersm", ".maxnreg", ".file", ".noreturn":
                continue
            default:
                if words[0].hasPrefix(".") {
                    throw PTXError(line: line, message: "unsupported directive in function body: \(words[0])")
                }
                out.append(.inst(try parseInst(s, line: line)))
            }
        }
        return out
    }

    static func splitTopLevel(_ s: Substring) -> [String] {
        var parts = [String]()
        var depth = 0
        var cur = ""
        for c in s {
            if c == "{" || c == "[" || c == "(" { depth += 1 }
            if c == "}" || c == "]" || c == ")" { depth -= 1 }
            if c == ",", depth == 0 {
                parts.append(cur.trimmingCharacters(in: .whitespacesAndNewlines))
                cur = ""
            } else {
                cur.append(c)
            }
        }
        let last = cur.trimmingCharacters(in: .whitespacesAndNewlines)
        if !last.isEmpty { parts.append(last) }
        return parts
    }

    static func parseInst(_ s: String, line: Int) throws -> Inst {
        var rest = Substring(s)
        var pred: (String, Bool)?
        if rest.hasPrefix("@") {
            let end = rest.firstIndex(where: { $0.isWhitespace }) ?? rest.endIndex
            var p = rest[rest.index(after: rest.startIndex)..<end]
            var neg = false
            if p.hasPrefix("!") { neg = true; p = p.dropFirst() }
            pred = (String(p), neg)
            rest = rest[end...].drop(while: { $0.isWhitespace })
        }
        let opEnd = rest.firstIndex(where: { $0.isWhitespace }) ?? rest.endIndex
        let opcode = rest[..<opEnd]
        let operandText = rest[opEnd...]
        let parts = opcode.split(separator: ".").map(String.init)
        guard let op = parts.first else { throw PTXError(line: line, message: "empty instruction") }
        let operands = try splitTopLevel(operandText).map { try parseOperand($0, line: line) }
        return Inst(line: line, pred: pred, op: op, mods: Array(parts.dropFirst()), operands: operands, text: s)
    }

    static let specialNames: Set<String> = [
        "%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%warpid", "%nwarpid", "%lanemask_eq", "%lanemask_le",
        "%lanemask_lt", "%lanemask_ge", "%lanemask_gt", "%clock", "%clock64", "%globaltimer", "%smid", "%nsmid",
        "%gridid", "%clusterid", "%nclusterid", "%cluster_ctaid", "%cluster_nctaid",
    ]

    static func parseInt(_ t: String) -> Int64? {
        var s = t
        var neg = false
        if s.hasPrefix("-") { neg = true; s.removeFirst() }
        if s.hasSuffix("U") || s.hasSuffix("u") { s.removeLast() }
        var v: UInt64?
        if s.hasPrefix("0x") || s.hasPrefix("0X") {
            v = UInt64(s.dropFirst(2), radix: 16)
        } else if s.count > 1, s.hasPrefix("0"), s.allSatisfy({ ("0"..."7").contains($0) }) {
            v = UInt64(s.dropFirst(), radix: 8)
        } else if s.hasPrefix("0b") || s.hasPrefix("0B") {
            v = UInt64(s.dropFirst(2), radix: 2)
        } else {
            v = UInt64(s)
        }
        guard let u = v else { return nil }
        let i = Int64(bitPattern: u)
        return neg ? 0 &- i : i
    }

    static func parseOperand(_ raw: String, line: Int) throws -> Operand {
        let s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s == "_" { return .sink }
        if s.hasPrefix("[") {
            guard s.hasSuffix("]") else { throw PTXError(line: line, message: "bad address \(s)") }
            let inner = String(s.dropFirst().dropLast()).replacingOccurrences(of: " ", with: "")
            if let plus = inner.firstIndex(of: "+") {
                let base = String(inner[..<plus])
                guard let off = parseInt(String(inner[inner.index(after: plus)...])) else {
                    throw PTXError(line: line, message: "bad address offset \(s)")
                }
                return .mem(base: base.isEmpty ? nil : try parseOperand(base, line: line), offset: off)
            }
            if let minus = inner.dropFirst().firstIndex(of: "-") {
                let base = String(inner[..<minus])
                guard let off = parseInt(String(inner[minus...])) else { throw PTXError(line: line, message: "bad address \(s)") }
                return .mem(base: try parseOperand(base, line: line), offset: off)
            }
            if let v = parseInt(inner) { return .mem(base: nil, offset: v) }
            return .mem(base: try parseOperand(inner, line: line), offset: 0)
        }
        if s.hasPrefix("{") {
            guard s.hasSuffix("}") else { throw PTXError(line: line, message: "bad vector \(s)") }
            return .vec(try splitTopLevel(s.dropFirst().dropLast()).map { try parseOperand($0, line: line) })
        }
        if let bar = s.firstIndex(of: "|") {
            return .pair(try parseOperand(String(s[..<bar]), line: line), try parseOperand(String(s[s.index(after: bar)...]), line: line))
        }
        if s.hasPrefix("!") { return .not(try parseOperand(String(s.dropFirst()), line: line)) }
        if s.hasPrefix("%") {
            let base = s.split(separator: ".").first.map(String.init) ?? s
            if specialNames.contains(base) { return .special(s) }
            return .reg(s)
        }
        if s == "WARP_SZ" { return .imm(32) }
        if (s.hasPrefix("0f") || s.hasPrefix("0F")), s.count == 10, let b = UInt32(s.dropFirst(2), radix: 16) { return .fimm32(b) }
        if (s.hasPrefix("0d") || s.hasPrefix("0D")), s.count == 18, let b = UInt64(s.dropFirst(2), radix: 16) { return .fimm64(b) }
        if let v = parseInt(s) { return .imm(v) }
        if let d = Double(s), s.contains(".") || s.contains("e") || s.contains("E") { return .fimm64(d.bitPattern) }
        if let f = s.first, f.isLetter || f == "_" || f == "$" { return .sym(s) }
        throw PTXError(line: line, message: "cannot parse operand '\(s)'")
    }
}
