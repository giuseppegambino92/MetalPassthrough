// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// One dispatcher per guest session (= one guest process / CUDA context). It drains the lock-free
// submission queue that lives in guest memory, encodes Metal work into batched command buffers,
// and posts completions. The AF_VSOCK connection is only a doorbell: the worker spins briefly
// when idle and then parks on the socket; the guest rings it only if `host_sleeping` is set.

import MPTProtocol
import Darwin
import Foundation
import Metal

@available(macOS 27.0, *)
public final class MetalComputeDispatcher: @unchecked Sendable {
    public struct Options: Sendable {
        public var spinMicros: UInt64 = 50
        public var maxBatch: Int = 64
        public var verbose = false
        public init() {}
    }

    let sessionID: UInt32
    let device: MTLDevice
    let queue: MTLCommandQueue
    let arenas: MetalArenaAllocator
    let pipelines: PipelineCache
    let info: HostGPUInfo
    let stream: FDStream
    let options: Options
    let registry: SessionRegistry
    let arena0: GuestArena
    let ring: UnsafeMutablePointer<mpt_ring_header>
    let sq: UnsafeMutablePointer<mpt_sqe>
    let cq: UnsafeMutablePointer<mpt_cqe>

    private var handles: [UInt64: MTLComputePipelineState] = [:]
    private var nextHandle: UInt64 = 1
    private var cb: MTLCommandBuffer?
    private var computeEnc: MTLComputeCommandEncoder?
    private var blitEnc: MTLBlitCommandEncoder?
    private var batched = 0
    private var lastCommitted: MTLCommandBuffer?

    // Updated from command buffer completion handlers (Metal's completion thread).
    private let statLock = NSLock()
    private var stats = mpt_stats()
    private var gpuNsSinceSync: UInt64 = 0
    private var lastGPUStart: Double = 0
    private var lastGPUEnd: Double = 0
    private var firstError: Error?

    init(
        sessionID: UInt32, device: MTLDevice, arenas: MetalArenaAllocator, arena0: GuestArena, pipelines: PipelineCache,
        info: HostGPUInfo, stream: FDStream, registry: SessionRegistry, options: Options
    ) {
        self.sessionID = sessionID
        self.device = device
        self.queue = device.makeCommandQueue(maxCommandBufferCount: 256)!
        self.queue.label = "metalpassthrough.session.\(sessionID)"
        self.arenas = arenas
        self.arena0 = arena0
        self.pipelines = pipelines
        self.info = info
        self.stream = stream
        self.registry = registry
        self.options = options
        if let rs = arena0.residency { queue.addResidencySet(rs) }
        ring = arena0.base.advanced(by: Int(MPT_RING_OFFSET)).assumingMemoryBound(to: mpt_ring_header.self)
        sq = arena0.base.advanced(by: Int(MPT_SQ_OFFSET)).assumingMemoryBound(to: mpt_sqe.self)
        cq = arena0.base.advanced(by: Int(MPT_CQ_OFFSET)).assumingMemoryBound(to: mpt_cqe.self)
    }

    func log(_ s: @autoclosure () -> String) {
        if options.verbose { FileHandle.standardError.write("[metalpassthrough s\(sessionID)] \(s())\n".data(using: .utf8)!) }
    }

    // MARK: - Ring plumbing

    private var sqTailPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 128) }
    private var guestWaitingPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 132) }
    private var sqHeadPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 256) }
    private var hostSleepingPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 260) }
    private var cqTailPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 384) }
    private var cqHeadPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 512) }

    /// Main loop; returns when the guest closes the session or the connection drops.
    func run() {
        var head = mpt_load_relaxed_u32(sqHeadPtr)
        var buf = [UInt8](repeating: 0, count: 64)
        outer: while true {
            let tail = mpt_load_acquire_u32(sqTailPtr)
            if tail != head {
                while head != tail {
                    let sqe = sq[Int(head % MPT_SQ_DEPTH)]
                    let keepGoing = execute(sqe, slot: head % MPT_SQ_DEPTH)
                    head &+= 1
                    mpt_store_release_u32(sqHeadPtr, head)
                    if !keepGoing { break outer }
                }
                continue
            }
            // Idle: start the GPU on whatever is batched, then spin, then park on the doorbell.
            if batched > 0 { commit() }
            publish()
            let spinUntil = DispatchTime.now().uptimeNanoseconds + options.spinMicros * 1000
            var woke = false
            while DispatchTime.now().uptimeNanoseconds < spinUntil {
                if mpt_load_acquire_u32(sqTailPtr) != head { woke = true; break }
                mpt_cpu_relax()
            }
            if woke { continue }
            mpt_store_seqcst_u32(hostSleepingPtr, 1)
            if mpt_load_seqcst_u32(sqTailPtr) != head {
                mpt_store_seqcst_u32(hostSleepingPtr, 0)
                continue
            }
            var pfd = pollfd(fd: stream.fd, events: Int16(POLLIN), revents: 0)
            let r = poll(&pfd, 1, 1000)
            if r > 0 {
                let n = Darwin.read(stream.fd, &buf, buf.count)
                if n <= 0 { break }  // guest went away
                statLock.withLock { stats.doorbells &+= 1 }
            }
            mpt_store_seqcst_u32(hostSleepingPtr, 0)
        }
        // Teardown: never leave the GPU writing into guest memory after the session ends.
        if batched > 0 { commit() }
        lastCommitted?.waitUntilCompleted()
        for a in arenas.all {
            if let rs = a.residency { queue.removeResidencySet(rs) }
            _ = arenas.free(id: a.id)
        }
        log("session closed")
    }

    private func post(tag: UInt32, status: Int32, _ r0: UInt64 = 0, _ r1: UInt64 = 0, _ r2: UInt64 = 0) {
        let tail = mpt_load_relaxed_u32(cqTailPtr)
        while tail &- mpt_load_acquire_u32(cqHeadPtr) >= MPT_CQ_DEPTH { mpt_cpu_relax() }  // CQ full
        cq[Int(tail % MPT_CQ_DEPTH)] = mpt_cqe(tag: tag, status: status, r: (r0, r1, r2))
        mpt_store_release_u32(cqTailPtr, tail &+ 1)
        mpt_fence_seqcst()
        if mpt_load_seqcst_u32(guestWaitingPtr) != 0 {
            var b: UInt8 = 1
            _ = Darwin.write(stream.fd, &b, 1)
        }
    }

    // MARK: - Command buffers

    private func openCB() -> MTLCommandBuffer {
        if let cb { return cb }
        let c = queue.makeCommandBufferWithUnretainedReferences()!
        c.label = "metalpassthrough.batch"
        cb = c
        return c
    }

    private func compute() -> MTLComputeCommandEncoder {
        if let computeEnc { return computeEnc }
        blitEnc?.endEncoding()
        blitEnc = nil
        // Serial dispatch: every dispatch observes the results of the previous one (CUDA stream order).
        let e = openCB().makeComputeCommandEncoder(dispatchType: .serial)!
        computeEnc = e
        return e
    }

    private func blit() -> MTLBlitCommandEncoder {
        if let blitEnc { return blitEnc }
        computeEnc?.endEncoding()
        computeEnc = nil
        let e = openCB().makeBlitCommandEncoder()!
        blitEnc = e
        return e
    }

    private func endEncoders() {
        computeEnc?.endEncoding()
        computeEnc = nil
        blitEnc?.endEncoding()
        blitEnc = nil
    }

    @discardableResult
    private func commit(onComplete: (@Sendable (Double, Error?) -> Void)? = nil) -> MTLCommandBuffer {
        endEncoders()
        let c = openCB()
        let dispatches = batched
        // Keep the arenas alive until the GPU is done with them (unretained command buffer).
        let keep = arenas.all
        c.addCompletedHandler { [weak self] done in
            withExtendedLifetime(keep) {}
            guard let self else { return }
            let ns = done.gpuEndTime > done.gpuStartTime ? UInt64((done.gpuEndTime - done.gpuStartTime) * 1e9) : 0
            self.statLock.withLock {
                self.gpuNsSinceSync &+= ns
                self.stats.total_gpu_ns &+= ns
                self.stats.command_buffers &+= 1
                if dispatches > 0 { self.stats.last_kernel_gpu_ns = ns }
                if done.gpuStartTime > 0 { self.lastGPUStart = done.gpuStartTime }
                if done.gpuEndTime > 0 { self.lastGPUEnd = done.gpuEndTime }
                if let e = done.error, self.firstError == nil { self.firstError = e }
            }
            // Empty command buffers (e.g. an event recorded before any work) carry no GPU
            // timestamps: use the host mach clock, which is the same timebase as gpuEndTime.
            let stamp = done.gpuEndTime > 0
                ? done.gpuEndTime
                : max(self.statLock.withLock { self.lastGPUEnd }, Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e9)
            onComplete?(stamp, done.error)
        }
        c.commit()
        lastCommitted = c
        cb = nil
        batched = 0
        return c
    }

    // MARK: - Opcodes

    private func string(_ dev: UInt64, _ len: UInt64) throws -> String {
        let p = try arenas.pointer(dev, len)
        return String(decoding: UnsafeRawBufferPointer(start: p, count: Int(len)), as: UTF8.self)
    }

    private func writeLog(_ dev: UInt64, _ len: UInt64, _ text: String) {
        guard len > 0, let p = try? arenas.pointer(dev, len) else { return }
        let bytes = Array(text.utf8.prefix(Int(len) - 1))
        bytes.withUnsafeBytes { p.copyMemory(from: $0.baseAddress!, byteCount: bytes.count) }
        p.storeBytes(of: 0, toByteOffset: bytes.count, as: UInt8.self)
    }

    private func register(_ ps: MTLComputePipelineState) -> UInt64 {
        let h = nextHandle
        nextHandle += 1
        handles[h] = ps
        return h
    }

    /// Returns false when the session should end.
    private func execute(_ e: mpt_sqe, slot: UInt32) -> Bool {
        statLock.withLock { stats.commands &+= 1 }
        let quiet = e.flags & UInt16(MPT_F_NO_CQE) != 0
        func ok(_ r0: UInt64 = 0, _ r1: UInt64 = 0, _ r2: UInt64 = 0) {
            if !quiet { post(tag: e.tag, status: 0, r0, r1, r2) }
        }
        func fail(_ status: Int32, _ why: String) {
            log("op \(e.opcode) failed: \(why)")
            post(tag: e.tag, status: status)
        }
        do {
            switch Int(e.opcode) {
            case Int(MPT_OP_NOP):
                ok()
            case Int(MPT_OP_PING):
                ok(clock_gettime_nsec_np(CLOCK_UPTIME_RAW))
            case Int(MPT_OP_DEVICE_INFO):
                info.write(to: arena0.base.advanced(by: Int(MPT_INFO_OFFSET)))
                arena0.base.advanced(by: Int(MPT_INFO_OFFSET)).assumingMemoryBound(to: mpt_device_info.self).pointee
                    .current_allocated = UInt64(device.currentAllocatedSize)
                ok(UInt64(MemoryLayout<mpt_device_info>.size))
            case Int(MPT_OP_ALLOC_ARENA):
                var nonce = [UInt8](repeating: 0, count: 16)
                withUnsafeBytes(of: e.a.0) { nonce.replaceSubrange(0..<8, with: $0) }
                withUnsafeBytes(of: e.a.1) { nonce.replaceSubrange(8..<16, with: $0) }
                do {
                    let a = try arenas.map(nonce: nonce, count: Int(e.a.2))
                    if let rs = a.residency { queue.addResidencySet(rs) }
                    log("arena \(a.id): \(a.length >> 20) MiB @ gpu 0x\(String(a.gpuBase, radix: 16))")
                    ok(UInt64(a.id), a.gpuBase)
                } catch { fail(MPT_ENOTFOUND, "\(error)") }
            case Int(MPT_OP_FREE_ARENA):
                // Make sure in-flight GPU work no longer references it.
                if batched > 0 { commit() }
                lastCommitted?.waitUntilCompleted()
                if e.a.0 == 0 { fail(MPT_EINVAL, "arena 0 holds the ring") } else if arenas.free(id: UInt32(e.a.0)) { ok() } else {
                    fail(MPT_ENOENT, "no arena \(e.a.0)")
                }
            case Int(MPT_OP_COMPILE_MSL):
                let src = try string(e.a.0, e.a.1)
                let fn = try string(e.a.2, e.a.3)
                do {
                    let c = try pipelines.pipeline(source: src, function: fn, fastMath: e.a.4 & UInt64(MPT_CF_FAST_MATH) != 0)
                    let h = register(c.state)
                    writeLog(e.a.5, e.a.6, "")
                    ok(h, mpt_pack_tg(UInt32(c.state.maxTotalThreadsPerThreadgroup), UInt32(c.state.threadExecutionWidth)),
                       UInt64(c.state.staticThreadgroupMemoryLength))
                } catch {
                    writeLog(e.a.5, e.a.6, "\(error)")
                    fail(MPT_ECOMPILE, "\(error)")
                }
            case Int(MPT_OP_COMPILE_PTX):
                try compilePTX(e)
            case Int(MPT_OP_DISPATCH):
                try dispatch(e)
                if e.flags & UInt16(MPT_F_COMMIT) != 0 || batched >= options.maxBatch { commit() }
                ok()
            case Int(MPT_OP_COPY):
                let (da, doff) = try arenas.resolve(e.a.0, e.a.2)
                let (sa, soff) = try arenas.resolve(e.a.1, e.a.2)
                if e.a.2 > 0 {
                    blit().copy(from: sa.buffer, sourceOffset: sa.bufferOffset + soff, to: da.buffer, destinationOffset: da.bufferOffset + doff, size: Int(e.a.2))
                    batched += 1
                }
                if e.flags & UInt16(MPT_F_COMMIT) != 0 { commit() }
                ok()
            case Int(MPT_OP_FILL):
                try fill(e)
                if e.flags & UInt16(MPT_F_COMMIT) != 0 { commit() }
                ok()
            case Int(MPT_OP_FLUSH):
                if batched > 0 { commit() }
                ok()
            case Int(MPT_OP_SYNC):
                if batched > 0 || cb != nil { commit() }
                lastCommitted?.waitUntilCompleted()
                let (ns, s, en, err) = statLock.withLock { () -> (UInt64, Double, Double, Error?) in
                    defer { gpuNsSinceSync = 0; firstError = nil }
                    return (gpuNsSinceSync, lastGPUStart, lastGPUEnd, firstError)
                }
                if let err { fail(MPT_EGPU, "\(err)") } else { ok(ns, UInt64(s * 1e9), UInt64(en * 1e9)) }
            case Int(MPT_OP_EVENT_RECORD):
                let p = try arenas.pointer(e.a.0, UInt64(MemoryLayout<mpt_event>.size))
                let addr = UInt(bitPattern: p)
                _ = openCB()
                commit { end, err in
                    let ev = UnsafeMutableRawPointer(bitPattern: addr)!
                    ev.storeBytes(of: UInt64(end * 1e9), toByteOffset: 0, as: UInt64.self)
                    ev.storeBytes(of: err == nil ? 0 : MPT_EGPU, toByteOffset: 12, as: Int32.self)
                    mpt_store_release_u32(ev.advanced(by: 8), 1)
                }
                ok()
            case Int(MPT_OP_RELEASE_PIPE):
                handles.removeValue(forKey: e.a.0)
                ok()
            case Int(MPT_OP_STATS):
                writeStats()
                ok(UInt64(MemoryLayout<mpt_stats>.size))
            case Int(MPT_OP_CLOSE):
                if batched > 0 { commit() }
                lastCommitted?.waitUntilCompleted()
                ok()
                return false
            default:
                fail(MPT_EINVAL, "unknown opcode \(e.opcode)")
            }
        } catch let err as ArenaError {
            fail(MPT_EFAULT, "\(err)")
        } catch let err as CompileError {
            fail(MPT_ECOMPILE, "\(err)")
        } catch {
            fail(MPT_EPROTO, "\(error)")
        }
        return true
    }

    private func publish() {
        let st = statLock.withLock { stats }
        registry.update(sessionID, arena: arenas.totalBytes, dispatches: st.dispatches, gpuNs: st.total_gpu_ns,
                        lastKernel: st.last_kernel_gpu_ns)
    }

    private func writeStats() {
        publish()
        let p = arena0.base.advanced(by: Int(MPT_STATS_OFFSET)).assumingMemoryBound(to: mpt_stats.self)
        let pc = pipelines.stats
        var s = statLock.withLock { stats }
        s.pipeline_compiles = pc.compiles
        s.pipeline_cache_hits = pc.hits
        s.pipelines_resident = UInt64(pc.resident)
        s.arena_bytes = arenas.totalBytes
        s.arenas = UInt32(arenas.all.count)
        s.active_sessions_vm = registry.sessionsInThisVM
        s.active_containers = registry.activeContainers()
        let snap = registry.snapshot
        s.vm_last_kernel_gpu_ns = snap.lastKernel
        s.nprocs = UInt32(min(snap.procs.count, Int(MPT_MAX_PROCS)))
        withUnsafeMutableBytes(of: &s.procs) { raw in
            let stride = raw.count / Int(MPT_MAX_PROCS)
            for (i, pinfo) in snap.procs.prefix(Int(MPT_MAX_PROCS)).enumerated() {
                let base = raw.baseAddress!.advanced(by: i * stride)
                base.storeBytes(of: pinfo.pid, toByteOffset: 0, as: UInt32.self)
                base.storeBytes(of: pinfo.session, toByteOffset: 4, as: UInt32.self)
                base.storeBytes(of: pinfo.arenaBytes, toByteOffset: 8, as: UInt64.self)
                base.storeBytes(of: pinfo.dispatches, toByteOffset: 16, as: UInt64.self)
                base.storeBytes(of: pinfo.gpuNs, toByteOffset: 24, as: UInt64.self)
                let name = Array(pinfo.comm.utf8.prefix(15)) + [0]
                for (k, b) in name.enumerated() { base.storeBytes(of: b, toByteOffset: 32 + k, as: UInt8.self) }
                s.vm_arena_bytes &+= pinfo.arenaBytes
                s.vm_dispatches &+= pinfo.dispatches
                s.vm_total_gpu_ns &+= pinfo.gpuNs
            }
        }
        p.pointee = s
    }

    private func dispatch(_ e: mpt_sqe) throws {
        guard let ps = handles[e.a.0] else { throw ArenaError.outOfRange(e.a.0, 0) }
        let len = e.a.2
        guard len >= UInt64(MemoryLayout<mpt_dispatch_desc>.size) else { throw ArenaError.outOfRange(e.a.1, len) }
        let raw = try arenas.pointer(e.a.1, len)
        let d = raw.assumingMemoryBound(to: mpt_dispatch_desc.self).pointee
        let bindBase = raw.advanced(by: MemoryLayout<mpt_dispatch_desc>.size)
        guard UInt64(MemoryLayout<mpt_dispatch_desc>.size) + UInt64(d.nbindings) * 24 <= len else {
            throw ArenaError.outOfRange(e.a.1, len)
        }
        let enc = compute()
        enc.setComputePipelineState(ps)
        for i in 0..<Int(d.nbindings) {
            let b = bindBase.advanced(by: i * 24).assumingMemoryBound(to: mpt_binding.self).pointee
            switch Int(b.kind) {
            case Int(MPT_BIND_BUFFER):
                let (a, off) = try arenas.resolve(b.addr, max(b.len, 0))
                enc.setBuffer(a.buffer, offset: a.bufferOffset + off, index: Int(b.index))
            case Int(MPT_BIND_BYTES):
                guard b.addr <= len, b.len <= len - b.addr, b.len <= 4096 else { throw ArenaError.outOfRange(b.addr, b.len) }
                enc.setBytes(raw.advanced(by: Int(b.addr)), length: Int(b.len), index: Int(b.index))
            case Int(MPT_BIND_TGMEM):
                enc.setThreadgroupMemoryLength(Int((b.len + 15) & ~15), index: Int(b.index))
            default:
                throw ArenaError.outOfRange(UInt64(b.kind), 0)
            }
        }
        if d.use_all_arenas != 0 {
            for a in arenas.all { enc.useResource(a.buffer, usage: [.read, .write]) }
        }
        let grid = MTLSize(width: Int(d.grid.0), height: Int(max(d.grid.1, 1)), depth: Int(max(d.grid.2, 1)))
        let tg = MTLSize(width: Int(d.tg.0), height: Int(max(d.tg.1, 1)), depth: Int(max(d.tg.2, 1)))
        if d.mode == UInt32(MPT_DISPATCH_THREADS) {
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        } else {
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tg)
        }
        batched += 1
        statLock.withLock { stats.dispatches &+= 1 }
    }

    private func fill(_ e: mpt_sqe) throws {
        let (a, off) = try arenas.resolve(e.a.0, e.a.1)
        let width = Int(e.a.3)
        if width == 1 || (e.a.2 & 0xff) * 0x0101_0101 == e.a.2 & 0xffff_ffff {
            blit().fill(buffer: a.buffer, range: (a.bufferOffset + off)..<(a.bufferOffset + off + Int(e.a.1)), value: UInt8(e.a.2 & 0xff))
            batched += 1
            return
        }
        // Multi-byte patterns: a tiny kernel (blit fill is byte-only).
        let src = """
            #include <metal_stdlib>
            using namespace metal;
            kernel void mpt_fill16(device ushort* p [[buffer(0)]], constant uint& v [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) p[i] = ushort(v); }
            kernel void mpt_fill32(device uint* p [[buffer(0)]], constant uint& v [[buffer(1)]], constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) { if (i < n) p[i] = v; }
            """
        let ps = try pipelines.pipeline(source: src, function: width == 2 ? "mpt_fill16" : "mpt_fill32", fastMath: false).state
        var v = UInt32(truncatingIfNeeded: e.a.2)
        var n = UInt32(Int(e.a.1) / width)
        let enc = compute()
        enc.setComputePipelineState(ps)
        enc.setBuffer(a.buffer, offset: a.bufferOffset + off, index: 0)
        enc.setBytes(&v, length: 4, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        batched += 1
    }

    private func compilePTX(_ e: mpt_sqe) throws {
        let ptx = try string(e.a.0, e.a.1)
        let entry = try string(e.a.2, e.a.3)
        let flags = UInt32(truncatingIfNeeded: e.a.4)
        let infoSize = UInt64(MemoryLayout<mpt_kernel_info>.size)
        let outLen = e.a.6
        guard outLen >= infoSize + 1 else { throw ArenaError.outOfRange(e.a.5, outLen) }
        let out = try arenas.pointer(e.a.5, outLen)
        memset(out, 0, Int(infoSize))
        func logText(_ s: String) { writeLog(e.a.5 + infoSize, outLen - infoSize, s) }
        let kernel: TranslatedKernel
        do {
            kernel = try PTXToMSLTranslator.translate(
                ptx: ptx, entry: entry,
                options: .init(lowerFP64: flags & MPT_CF_FP64_LOWER != 0))
        } catch {
            logText("PTX translation failed: \(error)")
            post(tag: e.tag, status: MPT_ETRANSLATE)
            return
        }
        let ki = out.assumingMemoryBound(to: mpt_kernel_info.self)
        ki.pointee.nparams = UInt32(kernel.params.count)
        ki.pointee.params_bytes = kernel.paramBytes
        ki.pointee.shared_bytes = kernel.staticSharedBytes
        ki.pointee.flags = (kernel.usesSharedMemory ? MPT_KF_USES_SHARED : 0) | (kernel.usesFP64 ? MPT_KF_USES_FP64 : 0)
            | (kernel.fp64Lowered ? MPT_KF_FP64_LOWERED : 0)
        withUnsafeMutableBytes(of: &ki.pointee.param_offset) { offs in
            withUnsafeMutableBytes(of: &ki.pointee.param_size) { sizes in
                for (i, p) in kernel.params.prefix(Int(MPT_MAX_PARAMS)).enumerated() {
                    offs.storeBytes(of: p.offset, toByteOffset: i * 4, as: UInt32.self)
                    sizes.storeBytes(of: p.size, toByteOffset: i * 4, as: UInt32.self)
                }
            }
        }
        do {
            let c = try pipelines.pipeline(source: kernel.msl, function: kernel.mslName, fastMath: flags & MPT_CF_FAST_MATH != 0)
            let h = register(c.state)
            var text = kernel.warnings.joined(separator: "\n")
            if flags & MPT_CF_DUMP_MSL != 0 { text += "\n" + kernel.msl }
            logText(text)
            ok3(e, h, mpt_pack_tg(UInt32(c.state.maxTotalThreadsPerThreadgroup), UInt32(c.state.threadExecutionWidth)),
                UInt64(c.state.staticThreadgroupMemoryLength))
        } catch {
            logText("generated MSL failed to compile: \(error)\n--- MSL ---\n\(kernel.msl)")
            post(tag: e.tag, status: MPT_ECOMPILE)
        }
    }

    private func ok3(_ e: mpt_sqe, _ r0: UInt64, _ r1: UInt64, _ r2: UInt64) {
        if e.flags & UInt16(MPT_F_NO_CQE) == 0 { post(tag: e.tag, status: 0, r0, r1, r2) }
    }
}
