// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// One dispatcher per guest session (= one guest process / CUDA context). It drains the lock-free
// submission queue that lives in guest memory, encodes Metal work into batched command buffers,
// and posts completions. The AF_VSOCK connection is only a doorbell: the worker spins briefly
// when idle and then parks on the socket; the guest rings it only if `host_sleeping` is set.
//
// Streams: every CUDA-style stream is its own MTLCommandQueue, so independent streams really run
// concurrently. Cross-stream ordering uses MTLSharedEvents:
//   * each stream signals its own `progress` event at the end of every committed command buffer;
//   * stream 0 is the legacy default stream: work on it waits for everything already submitted to
//     blocking streams, and work on a blocking stream waits for everything submitted to stream 0;
//   * EVENT_RECORD signals a per-event MTLSharedEvent; STREAM_WAIT_EVENT encodes a GPU-side wait
//     for the event's most recent value (CUDA cuStreamWaitEvent semantics).
//
// Hardening: every field read from guest memory is untrusted. All lengths, indices, counts and
// enum values are validated before use (a Swift trap would take the whole VM's GPU service down).

import Darwin
import Foundation
import MPTProtocol
import Metal
import MetalPerformanceShaders

@available(macOS 27.0, *)
final class StreamState {
    let id: UInt32
    let queue: MTLCommandQueue
    let blocking: Bool
    let progress: MTLSharedEvent
    /// Orders encoders inside one command buffer (arena buffers are untracked: no implicit hazards).
    let fence: MTLFence
    var submitted: UInt64 = 0  // value `progress` reaches when everything committed so far is done
    var cb: MTLCommandBuffer?
    var computeEnc: MTLComputeCommandEncoder?
    var blitEnc: MTLBlitCommandEncoder?
    var batched = 0
    var lastCommitted: MTLCommandBuffer?
    var waitedInOpenCB: [UInt32: UInt64] = [:]  // legacy-stream dependencies already encoded in `cb`
    var gpuNsSinceSync: UInt64 = 0  // updated under the dispatcher's statLock
    var firstError: Error?
    /// Host-private fp32 staging for mixed-precision GEMM (A, B, C). Reused across this stream's
    /// GEMMs: each GEMM ends its command buffer, and the next one starts only after it completed.
    var gemmScratch: [MTLBuffer?] = [nil, nil, nil]

    init(id: UInt32, queue: MTLCommandQueue, blocking: Bool, progress: MTLSharedEvent, fence: MTLFence) {
        self.id = id
        self.queue = queue
        self.blocking = blocking
        self.progress = progress
        self.fence = fence
    }

    var hasOpenWork: Bool { cb != nil }
}

@available(macOS 27.0, *)
public final class MetalComputeDispatcher: @unchecked Sendable {
    public struct Options: Sendable {
        public var spinMicros: UInt64 = 50
        /// Dispatches per command buffer while the guest keeps the SQ non-empty. The worker also
        /// commits whenever the SQ drains, so this only bounds batches under sustained load; each
        /// command buffer has a fixed cost (queue slot, progress event, completion handler), and
        /// small batches throttled throughput on the queue's in-flight limit.
        public var maxBatch: Int = 2048
        public var verbose = false
        public init() {}
    }

    /// Upper bounds for guest-controlled sizes (defence in depth; real limits are lower).
    static let maxStringBytes: UInt64 = 64 << 20
    static let maxBindings: UInt32 = 64
    static let maxBindIndex: UInt32 = 30
    /// A committed command buffer cannot be cancelled, so absurd launch sizes are rejected up front
    /// (2^40 threads is far beyond any real workload). This does not prevent a guest from occupying
    /// the GPU with a kernel that never terminates.
    static let maxThreadsPerDispatch: UInt64 = 1 << 40

    let sessionID: UInt32
    let device: MTLDevice
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
    private var streams: [UInt32: StreamState] = [:]
    private var events: [UInt64: (event: MTLSharedEvent, value: UInt64)] = [:]
    private var residencySets: [MTLResidencySet] = []
    /// Snapshot of the session's arenas for the hot path (refreshed whenever arenas change).
    private var arenaList: [GuestArena] = []
    private var connectionLost = false

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
        self.arenas = arenas
        self.arena0 = arena0
        self.pipelines = pipelines
        self.info = info
        self.stream = stream
        self.registry = registry
        self.options = options
        ring = arena0.base.advanced(by: Int(MPT_RING_OFFSET)).assumingMemoryBound(to: mpt_ring_header.self)
        sq = arena0.base.advanced(by: Int(MPT_SQ_OFFSET)).assumingMemoryBound(to: mpt_sqe.self)
        cq = arena0.base.advanced(by: Int(MPT_CQ_OFFSET)).assumingMemoryBound(to: mpt_cqe.self)
        if let rs = arena0.residency { residencySets.append(rs) }
        arenaList = arenas.all
        streams[0] = makeStream(id: 0, blocking: true)
    }

    func log(_ s: @autoclosure () -> String) {
        if options.verbose { FileHandle.standardError.write("[metalpassthrough s\(sessionID)] \(s())\n".data(using: .utf8)!) }
    }

    private func makeStream(id: UInt32, blocking: Bool) -> StreamState {
        let q = device.makeCommandQueue(maxCommandBufferCount: 256)!
        q.label = "metalpassthrough.s\(sessionID).stream\(id)"
        for rs in residencySets { q.addResidencySet(rs) }
        return StreamState(id: id, queue: q, blocking: blocking, progress: device.makeSharedEvent()!, fence: device.makeFence()!)
    }

    // MARK: - Ring plumbing

    private var sqTailPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 128) }
    private var guestWaitingPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 132) }
    private var sqHeadPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 256) }
    private var hostSleepingPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 260) }
    private var cqTailPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 384) }
    private var cqHeadPtr: UnsafeMutableRawPointer { UnsafeMutableRawPointer(ring).advanced(by: 512) }

    /// Main loop; returns when the guest closes the session, the connection drops, or the guest
    /// violates the ring protocol.
    func run() {
        var head = mpt_load_relaxed_u32(sqHeadPtr)
        var buf = [UInt8](repeating: 0, count: 64)
        outer: while !connectionLost {
            let tail = mpt_load_acquire_u32(sqTailPtr)
            if tail != head {
                if tail &- head > MPT_SQ_DEPTH {
                    log("protocol violation: sq_tail \(tail) is \(tail &- head) entries ahead of head \(head)")
                    break
                }
                while head != tail {
                    let sqe = sq[Int(head % MPT_SQ_DEPTH)]
                    let keepGoing = execute(sqe)
                    head &+= 1
                    mpt_store_release_u32(sqHeadPtr, head)
                    if !keepGoing || connectionLost { break outer }
                }
                continue
            }
            // Idle: start the GPU on whatever is batched, then spin, then park on the doorbell.
            for s in streams.values where s.hasOpenWork { commit(s) }
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
        drainAll()
        for a in arenas.all {
            if let rs = a.residency { for s in streams.values { s.queue.removeResidencySet(rs) } }
            _ = arenas.free(id: a.id)
        }
        log("session closed")
    }

    /// Posts a completion. If the guest stops consuming the CQ, back off instead of burning a core,
    /// and give up once the connection is gone.
    private func post(tag: UInt32, status: Int32, _ r0: UInt64 = 0, _ r1: UInt64 = 0, _ r2: UInt64 = 0) {
        let tail = mpt_load_relaxed_u32(cqTailPtr)
        var spins = 0
        while tail &- mpt_load_acquire_u32(cqHeadPtr) >= MPT_CQ_DEPTH {
            spins += 1
            if spins < 2000 { mpt_cpu_relax(); continue }
            var pfd = pollfd(fd: stream.fd, events: Int16(POLLIN), revents: 0)
            if poll(&pfd, 1, 1) > 0, pfd.revents & Int16(POLLHUP | POLLERR) != 0 || peerClosed() {
                connectionLost = true
                return
            }
        }
        cq[Int(tail % MPT_CQ_DEPTH)] = mpt_cqe(tag: tag, status: status, r: (r0, r1, r2))
        mpt_store_release_u32(cqTailPtr, tail &+ 1)
        mpt_fence_seqcst()
        if mpt_load_seqcst_u32(guestWaitingPtr) != 0 {
            var b: UInt8 = 1
            _ = Darwin.write(stream.fd, &b, 1)
        }
    }

    private func peerClosed() -> Bool {
        var b: UInt8 = 0
        return recv(stream.fd, &b, 1, MSG_PEEK | MSG_DONTWAIT) == 0
    }

    // MARK: - Command buffers (per stream)

    private func openCB(_ s: StreamState) -> MTLCommandBuffer {
        if let cb = s.cb { return cb }
        let c = s.queue.makeCommandBufferWithUnretainedReferences()!
        c.label = "metalpassthrough.s\(sessionID).stream\(s.id)"
        // Arena buffers are untracked, so stream order across command buffers is explicit: each
        // command buffer starts after the previous one on this stream has fully completed.
        if s.submitted > 0 { c.encodeWaitForEvent(s.progress, value: s.submitted) }
        s.cb = c
        s.waitedInOpenCB = [:]
        return c
    }

    private func endEncoders(_ s: StreamState) {
        if let e = s.computeEnc { e.updateFence(s.fence); e.endEncoding() }
        s.computeEnc = nil
        if let e = s.blitEnc { e.updateFence(s.fence); e.endEncoding() }
        s.blitEnc = nil
    }

    /// Encodes a GPU-side wait for `event >= value` before any further work on `s`.
    private func encodeWait(_ s: StreamState, _ event: MTLSharedEvent, _ value: UInt64) {
        endEncoders(s)
        openCB(s).encodeWaitForEvent(event, value: value)
    }

    /// Applies CUDA legacy-default-stream ordering before new work is encoded on `s`.
    private func prepare(_ s: StreamState) {
        if s.id == 0 {
            for o in streams.values where o.id != 0 && o.blocking {
                if o.hasOpenWork { commit(o) }
                if o.submitted > (s.waitedInOpenCB[o.id] ?? 0) {
                    encodeWait(s, o.progress, o.submitted)
                    s.waitedInOpenCB[o.id] = o.submitted
                }
            }
        } else if s.blocking, let z = streams[0] {
            if z.hasOpenWork { commit(z) }
            if z.submitted > (s.waitedInOpenCB[0] ?? 0) {
                encodeWait(s, z.progress, z.submitted)
                s.waitedInOpenCB[0] = z.submitted
            }
        }
        _ = openCB(s)
    }

    private func compute(_ s: StreamState) -> MTLComputeCommandEncoder {
        prepare(s)
        if let e = s.computeEnc { return e }
        if let b = s.blitEnc { b.updateFence(s.fence); b.endEncoding() }
        s.blitEnc = nil
        // Serial dispatch: every dispatch observes the results of the previous one (stream order).
        let e = openCB(s).makeComputeCommandEncoder(dispatchType: .serial)!
        e.waitForFence(s.fence)
        s.computeEnc = e
        return e
    }

    private func blit(_ s: StreamState) -> MTLBlitCommandEncoder {
        prepare(s)
        if let e = s.blitEnc { return e }
        if let c = s.computeEnc { c.updateFence(s.fence); c.endEncoding() }
        s.computeEnc = nil
        let e = openCB(s).makeBlitCommandEncoder()!
        e.waitForFence(s.fence)
        s.blitEnc = e
        return e
    }

    @discardableResult
    private func commit(_ s: StreamState, onComplete: (@Sendable (Double, Error?) -> Void)? = nil) -> MTLCommandBuffer {
        endEncoders(s)
        let c = openCB(s)
        let dispatches = s.batched
        s.submitted += 1
        c.encodeSignalEvent(s.progress, value: s.submitted)
        // Keep the arenas alive until the GPU is done with them (unretained command buffer).
        let keep = arenaList
        c.addCompletedHandler { [weak self, weak s] done in
            withExtendedLifetime(keep) {}
            guard let self else { return }
            let ns = done.gpuEndTime > done.gpuStartTime ? UInt64((done.gpuEndTime - done.gpuStartTime) * 1e9) : 0
            self.statLock.withLock {
                self.gpuNsSinceSync &+= ns
                s?.gpuNsSinceSync &+= ns
                self.stats.total_gpu_ns &+= ns
                self.stats.command_buffers &+= 1
                if dispatches > 0 { self.stats.last_kernel_gpu_ns = ns }
                if done.gpuStartTime > 0 { self.lastGPUStart = done.gpuStartTime }
                if done.gpuEndTime > 0 { self.lastGPUEnd = max(self.lastGPUEnd, done.gpuEndTime) }
                if let e = done.error {
                    if self.firstError == nil { self.firstError = e }
                    if s?.firstError == nil { s?.firstError = e }
                }
            }
            // Empty command buffers (e.g. an event recorded before any work) carry no GPU
            // timestamps: use the host mach clock, which is the same timebase as gpuEndTime.
            let stamp = done.gpuEndTime > 0
                ? done.gpuEndTime
                : max(self.statLock.withLock { self.lastGPUEnd }, Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e9)
            onComplete?(stamp, done.error)
        }
        c.commit()
        s.lastCommitted = c
        s.cb = nil
        s.batched = 0
        s.waitedInOpenCB = [:]
        return c
    }

    /// Blocks the worker until `s` completed. Every stream's batched work is committed first: the
    /// worker cannot commit anything while it waits, so leaving other streams' command buffers
    /// open would serialize independent streams behind this one.
    private func drain(_ s: StreamState) {
        for o in streams.values where o.hasOpenWork { commit(o) }
        s.lastCommitted?.waitUntilCompleted()
    }

    private func drainAll() {
        for s in streams.values where s.hasOpenWork { commit(s) }
        for s in streams.values { s.lastCommitted?.waitUntilCompleted() }
    }

    // MARK: - Opcodes

    private func string(_ dev: UInt64, _ len: UInt64) throws -> String {
        guard len <= Self.maxStringBytes else { throw ArenaError.outOfRange(dev, len) }
        let p = try arenas.pointer(dev, len)
        return String(decoding: UnsafeRawBufferPointer(start: p, count: Int(len)), as: UTF8.self)
    }

    private func writeLog(_ dev: UInt64, _ len: UInt64, _ text: String) {
        guard len > 0, len <= Self.maxStringBytes, let p = try? arenas.pointer(dev, len) else { return }
        let bytes = Array(text.utf8.prefix(Int(len) - 1))
        bytes.withUnsafeBytes { if let b = $0.baseAddress { p.copyMemory(from: b, byteCount: bytes.count) } }
        p.storeBytes(of: 0, toByteOffset: bytes.count, as: UInt8.self)
    }

    private func register(_ ps: MTLComputePipelineState) -> UInt64 {
        let h = nextHandle
        nextHandle += 1
        handles[h] = ps
        return h
    }

    enum OpError: Error { case invalid(String), noStream(UInt32) }

    private func streamOf(_ e: mpt_sqe) throws -> StreamState {
        let id = UInt32((e.flags & UInt16(MPT_F_STREAM_MASK)) >> UInt16(MPT_F_STREAM_SHIFT))
        guard let s = streams[id] else { throw OpError.noStream(id) }
        return s
    }

    /// Returns false when the session should end.
    private func execute(_ e: mpt_sqe) -> Bool {
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
                guard e.a.2 >= 1, e.a.2 <= arenas.maxChunks else { throw OpError.invalid("arena of \(e.a.2) chunks") }
                do {
                    let a = try arenas.map(nonce: nonce, count: Int(e.a.2))
                    if let rs = a.residency {
                        residencySets.append(rs)
                        for s in streams.values { s.queue.addResidencySet(rs) }
                    }
                    arenaList = arenas.all
                    log("arena \(a.id): \(a.length >> 20) MiB @ gpu 0x\(String(a.gpuBase, radix: 16)) uva=\(a.unifiedAddressable)")
                    ok(UInt64(a.id), a.gpuBase, a.unifiedAddressable ? MPT_CAP_UVA : 0)
                } catch { fail(MPT_ENOTFOUND, "\(error)") }
            case Int(MPT_OP_FREE_ARENA):
                drainAll()  // in-flight GPU work must not reference it any more
                if e.a.0 == 0 || e.a.0 > UInt64(UInt32.max) { fail(MPT_EINVAL, "arena \(e.a.0) cannot be freed") } else if let a = arenas.arena(id: UInt32(e.a.0)) {
                    if let rs = a.residency {
                        for s in streams.values { s.queue.removeResidencySet(rs) }
                        residencySets.removeAll { $0 === rs }
                    }
                    _ = arenas.free(id: a.id)
                    arenaList = arenas.all
                    ok()
                } else {
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
                let s = try streamOf(e)
                try dispatch(e, on: s)
                if e.flags & UInt16(MPT_F_COMMIT) != 0 || s.batched >= options.maxBatch { commit(s) }
                ok()
            case Int(MPT_OP_COPY):
                let s = try streamOf(e)
                let (da, doff) = try arenas.resolve(e.a.0, e.a.2)
                let (sa, soff) = try arenas.resolve(e.a.1, e.a.2)
                if e.a.2 > 0 {
                    blit(s).copy(
                        from: sa.buffer, sourceOffset: sa.bufferOffset + soff, to: da.buffer,
                        destinationOffset: da.bufferOffset + doff, size: Int(e.a.2))
                    s.batched += 1
                }
                if e.flags & UInt16(MPT_F_COMMIT) != 0 { commit(s) }
                ok()
            case Int(MPT_OP_FILL):
                let s = try streamOf(e)
                try fill(e, on: s)
                if e.flags & UInt16(MPT_F_COMMIT) != 0 { commit(s) }
                ok()
            case Int(MPT_OP_FLUSH):
                for s in streams.values where s.hasOpenWork { commit(s) }
                ok()
            case Int(MPT_OP_SYNC):
                drainAll()
                let (ns, st, en, err) = statLock.withLock { () -> (UInt64, Double, Double, Error?) in
                    defer {
                        gpuNsSinceSync = 0
                        firstError = nil
                        for s in streams.values { s.gpuNsSinceSync = 0; s.firstError = nil }
                    }
                    return (gpuNsSinceSync, lastGPUStart, lastGPUEnd, firstError)
                }
                if let err { fail(MPT_EGPU, "\(err)") } else { ok(ns, UInt64(st * 1e9), UInt64(en * 1e9)) }
            case Int(MPT_OP_STREAM_CREATE):
                guard let id = (1..<MPT_MAX_STREAMS).first(where: { streams[$0] == nil }) else {
                    throw OpError.invalid("all \(MPT_MAX_STREAMS - 1) streams in use")
                }
                streams[id] = makeStream(id: id, blocking: e.a.0 & UInt64(MPT_STREAM_NON_BLOCKING) == 0)
                ok(UInt64(id))
            case Int(MPT_OP_STREAM_DESTROY):
                let s = try streamOf(e)
                guard s.id != 0 else { throw OpError.invalid("stream 0 cannot be destroyed") }
                if s.hasOpenWork { commit(s) }  // pending work still completes (it retains its queue)
                streams.removeValue(forKey: s.id)
                ok()
            case Int(MPT_OP_STREAM_SYNC):
                let s = try streamOf(e)
                // Synchronizing the legacy stream waits for every blocking stream as well.
                let targets = s.id == 0 ? streams.values.filter { $0.blocking } : [s]
                for t in targets { drain(t) }
                let (ns, err) = statLock.withLock { () -> (UInt64, Error?) in
                    let ns = targets.reduce(UInt64(0)) { $0 &+ $1.gpuNsSinceSync }
                    let err = targets.compactMap { $0.firstError }.first
                    for t in targets { t.gpuNsSinceSync = 0; t.firstError = nil }
                    return (ns, err)
                }
                if let err { fail(MPT_EGPU, "\(err)") } else { ok(ns) }
            case Int(MPT_OP_STREAM_QUERY):
                let s = try streamOf(e)
                let targets = s.id == 0 ? streams.values.filter { $0.blocking } : [s]
                for t in targets where t.hasOpenWork { commit(t) }  // make progress, never wait
                let idle = targets.allSatisfy { t in
                    guard let c = t.lastCommitted else { return true }
                    return c.status == .completed || c.status == .error
                }
                ok(idle ? 1 : 0)
            case Int(MPT_OP_EVENT_RECORD):
                let s = try streamOf(e)
                // The completion is published with an atomic store: misaligned addresses would fault
                // (EXC_ARM_DA_ALIGN) and take the host down, so alignment is part of validation.
                guard e.a.0 % 16 == 0 else { throw OpError.invalid("event address 0x\(String(e.a.0, radix: 16)) not 16-byte aligned") }
                let p = try arenas.pointer(e.a.0, UInt64(MemoryLayout<mpt_event>.size))
                let addr = UInt(bitPattern: p)
                var entry = events[e.a.0] ?? (device.makeSharedEvent()!, 0)
                entry.value += 1
                events[e.a.0] = entry
                prepare(s)
                endEncoders(s)
                openCB(s).encodeSignalEvent(entry.event, value: entry.value)
                commit(s) { end, err in
                    let ev = UnsafeMutableRawPointer(bitPattern: addr)!
                    ev.storeBytes(of: UInt64(end * 1e9), toByteOffset: 0, as: UInt64.self)
                    ev.storeBytes(of: err == nil ? 0 : MPT_EGPU, toByteOffset: 12, as: Int32.self)
                    mpt_store_release_u32(ev.advanced(by: 8), 1)
                }
                ok()
            case Int(MPT_OP_STREAM_WAIT_EVENT):
                let s = try streamOf(e)
                if let entry = events[e.a.0] {  // never recorded: CUDA treats the wait as satisfied
                    prepare(s)
                    encodeWait(s, entry.event, entry.value)
                }
                ok()
            case Int(MPT_OP_GEMM):
                let s = try streamOf(e)
                try gemm(e, on: s)
                ok()
            case Int(MPT_OP_EVENT_DESTROY):
                events.removeValue(forKey: e.a.0)
                ok()
            case Int(MPT_OP_RELEASE_PIPE):
                handles.removeValue(forKey: e.a.0)
                ok()
            case Int(MPT_OP_STATS):
                writeStats()
                ok(UInt64(MemoryLayout<mpt_stats>.size))
            case Int(MPT_OP_CLOSE):
                drainAll()
                ok()
                return false
            default:
                fail(MPT_EINVAL, "unknown opcode \(e.opcode)")
            }
        } catch let err as ArenaError {
            fail(MPT_EFAULT, "\(err)")
        } catch let err as CompileError {
            fail(MPT_ECOMPILE, "\(err)")
        } catch let err as OpError {
            fail(MPT_EINVAL, "\(err)")
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

    /// Hot-path arena lookup over the snapshot (no lock, no allocation): (arena index, byte offset).
    @inline(__always)
    private func locate(_ dev: UInt64, _ len: UInt64) -> (Int, Int)? {
        for i in arenaList.indices where arenaList[i].contains(dev, len) { return (i, Int(dev - arenaList[i].gpuBase)) }
        return nil
    }

    /// Validates and encodes one dispatch. Allocation-free: the descriptor is copied (TOCTOU) into a
    /// stack buffer, bindings are validated in a first pass and encoded in a second.
    private func dispatch(_ e: mpt_sqe, on s: StreamState) throws {
        guard let ps = handles[e.a.0] else { throw OpError.invalid("unknown pipeline \(e.a.0)") }
        let len = e.a.2
        let header = UInt64(MemoryLayout<mpt_dispatch_desc>.size)
        guard len >= header, len <= UInt64(MPT_SCRATCH_PER_SLOT) else { throw ArenaError.outOfRange(e.a.1, len) }
        guard let (di, doff) = locate(e.a.1, len) else { throw ArenaError.outOfRange(e.a.1, len) }
        let raw = arenaList[di].base.advanced(by: doff)
        try withUnsafeTemporaryAllocation(byteCount: Int(len), alignment: 16) { tmp in
            tmp.copyMemory(from: UnsafeRawBufferPointer(start: raw, count: Int(len)))
            let base = tmp.baseAddress!
            let d = base.loadUnaligned(as: mpt_dispatch_desc.self)
            let nb = Int(d.nbindings)
            guard d.nbindings <= Self.maxBindings, header + UInt64(d.nbindings) * 24 <= len else {
                throw OpError.invalid("\(d.nbindings) bindings do not fit a \(len)-byte descriptor")
            }
            guard d.mode == MPT_DISPATCH_THREADS || d.mode == MPT_DISPATCH_THREADGROUPS else {
                throw OpError.invalid("dispatch mode \(d.mode)")
            }
            let t0 = Int(max(d.tg.0, 1)), t1 = Int(max(d.tg.1, 1)), t2 = Int(max(d.tg.2, 1))
            let g0 = UInt64(max(d.grid.0, 1)), g1 = UInt64(max(d.grid.1, 1)), g2 = UInt64(max(d.grid.2, 1))
            // Per-dimension limits first: only then is the threadgroup product small enough to compute.
            let maxTG = device.maxThreadsPerThreadgroup
            guard t0 <= maxTG.width, t1 <= maxTG.height, t2 <= maxTG.depth, t0 * t1 * t2 <= ps.maxTotalThreadsPerThreadgroup else {
                throw OpError.invalid("threadgroup \(t0)x\(t1)x\(t2) exceeds pipeline limit \(ps.maxTotalThreadsPerThreadgroup)")
            }
            // Overflow-safe total thread count (guest-controlled factors can exceed 2^64).
            let (p01, o1) = g0.multipliedReportingOverflow(by: g1)
            let (p012, o2) = p01.multipliedReportingOverflow(by: g2)
            let (total, o3) = p012.multipliedReportingOverflow(by: d.mode == MPT_DISPATCH_THREADS ? 1 : UInt64(t0 * t1 * t2))
            guard !o1, !o2, !o3, total <= Self.maxThreadsPerDispatch else {
                throw OpError.invalid("dispatch geometry \(g0)x\(g1)x\(g2) x \(t0)x\(t1)x\(t2) too large")
            }
            @inline(__always) func binding(_ i: Int) -> mpt_binding {
                base.advanced(by: Int(header) + i * 24).loadUnaligned(as: mpt_binding.self)
            }
            // Pass 1: validate everything before encoding anything.
            var tgMemTotal = ps.staticThreadgroupMemoryLength
            for i in 0..<nb {
                let b = binding(i)
                guard b.index <= Self.maxBindIndex else { throw OpError.invalid("binding index \(b.index)") }
                switch b.kind {
                case MPT_BIND_BUFFER:
                    guard locate(b.addr, b.len) != nil else { throw ArenaError.outOfRange(b.addr, b.len) }
                case MPT_BIND_BYTES:
                    guard b.len > 0, b.addr <= len, b.len <= len - b.addr, b.len <= 4096 else { throw ArenaError.outOfRange(b.addr, b.len) }
                case MPT_BIND_TGMEM:
                    guard b.len <= UInt64(device.maxThreadgroupMemoryLength) else { throw OpError.invalid("threadgroup memory \(b.len)") }
                    tgMemTotal += (Int(b.len) + 15) & ~15
                default:
                    throw OpError.invalid("binding kind \(b.kind)")
                }
            }
            guard tgMemTotal <= device.maxThreadgroupMemoryLength else {
                throw OpError.invalid("threadgroup memory \(tgMemTotal) > \(device.maxThreadgroupMemoryLength)")
            }
            // Pass 2: encode (cannot fail: the descriptor copy and the arena snapshot are unchanged).
            let enc = compute(s)
            enc.setComputePipelineState(ps)
            for i in 0..<nb {
                let b = binding(i)
                switch b.kind {
                case MPT_BIND_BUFFER:
                    let (ai, off) = locate(b.addr, b.len)!
                    enc.setBuffer(arenaList[ai].buffer, offset: arenaList[ai].bufferOffset + off, index: Int(b.index))
                case MPT_BIND_BYTES:
                    enc.setBytes(base.advanced(by: Int(b.addr)), length: Int(b.len), index: Int(b.index))
                default:
                    if b.len > 0 { enc.setThreadgroupMemoryLength((Int(b.len) + 15) & ~15, index: Int(b.index)) }
                }
            }
            if d.use_all_arenas != 0 {
                for a in arenaList { enc.useResource(a.buffer, usage: [.read, .write]) }
            }
            let tgSize = MTLSize(width: t0, height: t1, depth: t2)
            let grid = MTLSize(width: Int(g0), height: Int(g1), depth: Int(g2))
            if d.mode == MPT_DISPATCH_THREADS { enc.dispatchThreads(grid, threadsPerThreadgroup: tgSize) } else {
                enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tgSize)
            }
        }
        s.batched += 1
        statLock.withLock { stats.dispatches &+= 1 }
    }

    private static let convertSource = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void mpt_f16_to_f32(device const half* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) { if (i < n) y[i] = float(x[i]); }
        kernel void mpt_bf16_to_f32(device const ushort* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) { if (i < n) y[i] = as_type<float>(uint(x[i]) << 16); }
        kernel void mpt_f32_to_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) { if (i < n) y[i] = x[i]; }
        kernel void mpt_f32_to_f16(device const float* x [[buffer(0)]], device half* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                                   uint i [[thread_position_in_grid]]) { if (i < n) y[i] = half(x[i]); }  // round to nearest even
        kernel void mpt_f32_to_bf16(device const float* x [[buffer(0)]], device ushort* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
            if (i >= n) return;
            uint u = as_type<uint>(x[i]);
            y[i] = isnan(x[i]) ? ushort((u >> 16) | 0x40u) : ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);  // RNE
        }
        """

    /// BLAS (column-major) GEMM through MPSMatrixMultiplication. A column-major matrix with leading
    /// dimension ld is the row-major transpose with rowBytes = ld * elementSize, so
    /// C^T = op(B)^T * op(A)^T is computed in MPS's row-major world without moving data.
    /// Types MPS takes directly (fp32, fp16 inputs; fp32 output; fp16 output with fp16 accumulation)
    /// run in place on the arena. Otherwise operands are widened exactly to fp32 in per-stream scratch
    /// buffers (bf16 and fp16 are subsets of fp32), multiplied in fp32, and rounded once (RNE) into
    /// the output type, which matches cuBLAS's fp32-compute semantics.
    private func gemm(_ e: mpt_sqe, on s: StreamState) throws {
        let size = UInt64(MemoryLayout<mpt_gemm_desc>.size)
        guard e.a.1 == size else { throw OpError.invalid("gemm descriptor of \(e.a.1) bytes") }
        let g = try arenas.pointer(e.a.0, size).loadUnaligned(as: mpt_gemm_desc.self)
        let abType = g.dtype & 0xff, cType = (g.dtype >> 8) & 0xff, acc16 = g.dtype & MPT_GEMM_ACC16 != 0
        guard g.dtype & ~(0xffff | MPT_GEMM_ACC16) == 0, abType <= 2, cType <= 2, !acc16 || (abType == 1 && cType == 1) else {
            throw OpError.invalid("gemm dtype 0x\(String(g.dtype, radix: 16))")
        }
        guard g.trans_a <= 1, g.trans_b <= 1, g.batch >= 1, g.batch <= 65536,
            g.m >= 1, g.n >= 1, g.k >= 1, g.m <= 1 << 20, g.n <= 1 << 20, g.k <= 1 << 20
        else { throw OpError.invalid("gemm shape m=\(g.m) n=\(g.n) k=\(g.k) batch=\(g.batch)") }
        func esize(_ t: UInt32) -> Int { t == 0 ? 4 : 2 }
        let (m, n, k) = (Int(g.m), Int(g.n), Int(g.k))
        // Row-major views (rows x cols, row stride ld in elements) of the column-major operands.
        typealias View = (rows: Int, cols: Int, ld: Int)
        let left: View = g.trans_b == 0 ? (n, k, Int(g.ldb)) : (k, n, Int(g.ldb))
        let right: View = g.trans_a == 0 ? (k, m, Int(g.lda)) : (m, k, Int(g.lda))
        let result: View = (n, m, Int(g.ldc))
        /// Validates an operand; returns its arena, byte offset and element count (all batch entries).
        func view(_ dev: UInt64, _ v: View, _ stride: UInt64, _ es: Int) throws -> (GuestArena, Int, Int) {
            guard v.ld >= v.cols else { throw OpError.invalid("leading dimension \(v.ld) < \(v.cols)") }
            let span = UInt64((v.rows - 1) * v.ld + v.cols)
            let (last, o1) = stride.multipliedReportingOverflow(by: UInt64(g.batch - 1))
            let (count, o2) = span.addingReportingOverflow(last)
            let (bytes, o3) = count.multipliedReportingOverflow(by: UInt64(es))
            guard !o1, !o2, !o3, dev % UInt64(es) == 0, count <= UInt64(UInt32.max) else { throw OpError.invalid("gemm operand range") }
            let (a, off) = try arenas.resolve(dev, bytes)
            return (a, off, Int(count))
        }
        let (aL, oL, nL) = try view(g.b, left, g.stride_b, esize(abType))
        let (aR, oR, nR) = try view(g.a, right, g.stride_a, esize(abType))
        let (aC, oC, nC) = try view(g.c, result, g.stride_c, esize(cType))

        // MPS asserts (aborting this whole process) unless every matrix buffer holds
        // offset + rows * rowBytes bytes (the last row's padding included) per batch entry; the guest
        // only guarantees the elements it addresses. Operands that do not satisfy that inside their
        // MTLBuffer, and types MPS cannot read (bf16), are staged through fp32 scratch instead.
        func mpsBytes(_ v: View, _ stride: UInt64, _ es: Int) -> Int {
            let one = v.rows * v.ld * es
            return g.batch == 1 ? one : max(Int(g.batch) * Int(stride) * es, (Int(g.batch) - 1) * Int(stride) * es + one)
        }
        func fits(_ a: GuestArena, _ off: Int, _ need: Int) -> Bool { a.bufferOffset + off + need <= a.buffer.length }
        let esAB = esize(abType), esC = esize(cType)
        let stageIn = abType == UInt32(MPT_GEMM_BF16) || !fits(aL, oL, mpsBytes(left, g.stride_b, esAB)) || !fits(aR, oR, mpsBytes(right, g.stride_a, esAB))
        let stageOut = !(cType == 0 || (acc16 && cType == UInt32(MPT_GEMM_F16))) || !fits(aC, oC, mpsBytes(result, g.stride_c, esC)) || (acc16 && stageIn)
        let inMPS: MPSDataType = stageIn || abType == 0 ? .float32 : .float16
        let outMPS: MPSDataType = stageOut || cType == 0 ? .float32 : .float16

        // MPS encodes its own passes, which our per-stream fence cannot order (fences order the encoders
        // that wait on them, and the arena buffers are untracked). A command-buffer boundary does:
        // earlier work completes first (progress event), and later work waits for the GEMM.
        prepare(s)
        if s.hasOpenWork { commit(s) }
        let cb = openCB(s)

        func scratch(_ slot: Int, _ bytes: Int) throws -> MTLBuffer {
            let bytes = max(bytes, 16)
            if let b = s.gemmScratch[slot], b.length >= bytes { return b }
            guard bytes <= device.maxBufferLength, let b = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
                throw OpError.invalid("gemm staging buffer of \(bytes) bytes")
            }
            b.label = "metalpassthrough.gemm-scratch"
            s.gemmScratch[slot] = b
            return b
        }
        /// Encodes an element-wise conversion (scratch buffers are hazard-tracked: Metal orders the
        /// encoders that touch them inside this command buffer).
        func convert(_ fn: String, from src: MTLBuffer, _ srcOff: Int, to dst: MTLBuffer, _ dstOff: Int, count: Int) throws {
            let ps = try pipelines.pipeline(source: Self.convertSource, function: fn, fastMath: false).state
            var n = UInt32(count)
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(ps)
            enc.setBuffer(src, offset: srcOff, index: 0)
            enc.setBuffer(dst, offset: dstOff, index: 1)
            enc.setBytes(&n, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()
        }
        let toF32 = ["mpt_f32_to_f32", "mpt_f16_to_f32", "mpt_bf16_to_f32"]
        let fromF32 = ["mpt_f32_to_f32", "mpt_f32_to_f16", "mpt_f32_to_bf16"]
        var (bufL, offL, bufR, offR) = (aL.buffer, aL.bufferOffset + oL, aR.buffer, aR.bufferOffset + oR)
        if stageIn {  // widen exactly to fp32
            let l = try scratch(0, mpsBytes(left, g.stride_b, 4)), r = try scratch(1, mpsBytes(right, g.stride_a, 4))
            try convert(toF32[Int(abType)], from: aL.buffer, aL.bufferOffset + oL, to: l, 0, count: nL)
            try convert(toF32[Int(abType)], from: aR.buffer, aR.bufferOffset + oR, to: r, 0, count: nR)
            (bufL, offL, bufR, offR) = (l, 0, r, 0)
        }
        var (bufC, offC) = (aC.buffer, aC.bufferOffset + oC)
        if stageOut {
            // Preload C (also the gaps between columns/batches, so the write-back round-trips them exactly).
            let c = try scratch(2, mpsBytes(result, g.stride_c, 4))
            try convert(toF32[Int(cType)], from: aC.buffer, aC.bufferOffset + oC, to: c, 0, count: nC)
            (bufC, offC) = (c, 0)
        }
        let mm = MPSMatrixMultiplication(
            device: device, transposeLeft: g.trans_b == 1, transposeRight: g.trans_a == 1,
            resultRows: n, resultColumns: m, interiorColumns: k, alpha: Double(g.alpha), beta: Double(g.beta))
        // MPS's own batching silently ignores matrixBytes unless it equals rows * rowBytes (measured:
        // any other stride, including cuBLAS's legal stride 0 broadcast, gives wrong results without an
        // error), while arbitrary offsets are honored. So: native batching only for contiguous
        // batches, otherwise one single-matrix encode per batch entry.
        let contiguous = UInt64(left.rows * left.ld) == g.stride_b && UInt64(right.rows * right.ld) == g.stride_a
            && UInt64(result.rows * result.ld) == g.stride_c
        let esIn = inMPS == .float32 ? 4 : 2, esOut = outMPS == .float32 ? 4 : 2
        func desc(_ v: View, _ stride: UInt64, _ t: MPSDataType, batched: Bool) -> MPSMatrixDescriptor {
            let es = t == .float32 ? 4 : 2
            // Single matrices use the plain descriptor: the multi-matrix form (even with matrices = 1)
            // made MPS pick a markedly slower kernel (measured 47-62% of native).
            return !batched
                ? MPSMatrixDescriptor(rows: v.rows, columns: v.cols, rowBytes: v.ld * es, dataType: t)
                : MPSMatrixDescriptor(rows: v.rows, columns: v.cols, matrices: Int(g.batch), rowBytes: v.ld * es,
                                      matrixBytes: Int(stride) * es, dataType: t)
        }
        if g.batch == 1 || contiguous {
            let batched = g.batch > 1
            mm.batchSize = Int(g.batch)
            mm.encode(
                commandBuffer: cb,
                leftMatrix: MPSMatrix(buffer: bufL, offset: offL, descriptor: desc(left, g.stride_b, inMPS, batched: batched)),
                rightMatrix: MPSMatrix(buffer: bufR, offset: offR, descriptor: desc(right, g.stride_a, inMPS, batched: batched)),
                resultMatrix: MPSMatrix(buffer: bufC, offset: offC, descriptor: desc(result, g.stride_c, outMPS, batched: batched)))
        } else {
            let dl = desc(left, 0, inMPS, batched: false), dr = desc(right, 0, inMPS, batched: false), dc = desc(result, 0, outMPS, batched: false)
            for i in 0..<Int(g.batch) {
                mm.encode(
                    commandBuffer: cb,
                    leftMatrix: MPSMatrix(buffer: bufL, offset: offL + i * Int(g.stride_b) * esIn, descriptor: dl),
                    rightMatrix: MPSMatrix(buffer: bufR, offset: offR + i * Int(g.stride_a) * esIn, descriptor: dr),
                    resultMatrix: MPSMatrix(buffer: bufC, offset: offC + i * Int(g.stride_c) * esOut, descriptor: dc))
            }
        }
        if stageOut {  // fp32 -> output type, rounded once (RNE)
            try convert(fromF32[Int(cType)], from: bufC, 0, to: aC.buffer, aC.bufferOffset + oC, count: nC)
        }
        s.batched += 1
        statLock.withLock { stats.dispatches &+= 1 }
        commit(s)
    }

    private func fill(_ e: mpt_sqe, on s: StreamState) throws {
        let width = e.a.3
        guard width == 1 || width == 2 || width == 4, e.a.1 % width == 0 else {
            throw OpError.invalid("fill of \(e.a.1) bytes with \(width)-byte pattern")
        }
        let (a, off) = try arenas.resolve(e.a.0, e.a.1)
        guard e.a.1 > 0 else { return }
        if width == 1 || (e.a.2 & 0xff) &* 0x0101_0101 == e.a.2 & 0xffff_ffff {
            blit(s).fill(buffer: a.buffer, range: (a.bufferOffset + off)..<(a.bufferOffset + off + Int(e.a.1)), value: UInt8(e.a.2 & 0xff))
            s.batched += 1
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
        let count = e.a.1 / width
        guard count <= UInt64(UInt32.max) else { throw OpError.invalid("fill too large") }
        var v = UInt32(truncatingIfNeeded: e.a.2)
        var n = UInt32(count)
        let enc = compute(s)
        enc.setComputePipelineState(ps)
        enc.setBuffer(a.buffer, offset: a.bufferOffset + off, index: 0)
        enc.setBytes(&v, length: 4, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        s.batched += 1
    }

    private func compilePTX(_ e: mpt_sqe) throws {
        let ptx = try string(e.a.0, e.a.1)
        let entry = try string(e.a.2, e.a.3)
        let flags = UInt32(truncatingIfNeeded: e.a.4)
        let infoSize = UInt64(MemoryLayout<mpt_kernel_info>.size)
        let outLen = e.a.6
        guard outLen >= infoSize + 1, outLen <= Self.maxStringBytes, e.a.5 % 8 == 0 else { throw ArenaError.outOfRange(e.a.5, outLen) }
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
        guard kernel.params.count <= Int(MPT_MAX_PARAMS), kernel.paramBytes <= 4096 else {
            logText("kernel has \(kernel.params.count) parameters / \(kernel.paramBytes) bytes (max \(MPT_MAX_PARAMS) / 4096)")
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
                for (i, p) in kernel.params.enumerated() {
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
            if e.flags & UInt16(MPT_F_NO_CQE) == 0 {
                post(tag: e.tag, status: 0, h,
                     mpt_pack_tg(UInt32(c.state.maxTotalThreadsPerThreadgroup), UInt32(c.state.threadExecutionWidth)),
                     UInt64(c.state.staticThreadgroupMemoryLength))
            }
        } catch {
            logText("generated MSL failed to compile: \(error)\n--- MSL ---\n\(kernel.msl)")
            post(tag: e.tag, status: MPT_ECOMPILE)
        }
    }
}
