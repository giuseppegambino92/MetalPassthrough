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
        public var maxBatch: Int = 64
        public var verbose = false
        public init() {}
    }

    /// Upper bounds for guest-controlled sizes (defence in depth; real limits are lower).
    static let maxStringBytes: UInt64 = 64 << 20
    static let maxBindings: UInt32 = 64
    static let maxBindIndex: UInt32 = 30

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
        let keep = arenas.all
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

    private func drain(_ s: StreamState) {
        if s.hasOpenWork { commit(s) }
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

    private func dispatch(_ e: mpt_sqe, on s: StreamState) throws {
        guard let ps = handles[e.a.0] else { throw OpError.invalid("unknown pipeline \(e.a.0)") }
        let len = e.a.2
        let header = UInt64(MemoryLayout<mpt_dispatch_desc>.size)
        guard len >= header, len <= UInt64(MPT_SCRATCH_PER_SLOT) else { throw ArenaError.outOfRange(e.a.1, len) }
        let raw = try arenas.pointer(e.a.1, len)
        // Copy the descriptor before validating it: the guest may keep writing to it (TOCTOU).
        let desc = [UInt8](UnsafeRawBufferPointer(start: raw, count: Int(len)))
        try desc.withUnsafeBytes { db in
            let base = db.baseAddress!
            let d = base.loadUnaligned(as: mpt_dispatch_desc.self)
            guard d.nbindings <= Self.maxBindings, header + UInt64(d.nbindings) * 24 <= len else {
                throw OpError.invalid("\(d.nbindings) bindings do not fit a \(len)-byte descriptor")
            }
            guard d.mode == MPT_DISPATCH_THREADS || d.mode == MPT_DISPATCH_THREADGROUPS else {
                throw OpError.invalid("dispatch mode \(d.mode)")
            }
            let tgs = [Int(max(d.tg.0, 1)), Int(max(d.tg.1, 1)), Int(max(d.tg.2, 1))]
            let grid = [Int(max(d.grid.0, 1)), Int(max(d.grid.1, 1)), Int(max(d.grid.2, 1))]
            let maxTG = device.maxThreadsPerThreadgroup
            guard tgs[0] * tgs[1] * tgs[2] <= ps.maxTotalThreadsPerThreadgroup, tgs[0] <= maxTG.width, tgs[1] <= maxTG.height,
                tgs[2] <= maxTG.depth
            else { throw OpError.invalid("threadgroup \(tgs) exceeds pipeline limit \(ps.maxTotalThreadsPerThreadgroup)") }
            var tgMemTotal = ps.staticThreadgroupMemoryLength
            // Validate every binding before encoding anything.
            var plan: [(kind: UInt32, index: Int, arena: GuestArena?, off: Int, len: Int)] = []
            for i in 0..<Int(d.nbindings) {
                let b = base.advanced(by: Int(header) + i * 24).loadUnaligned(as: mpt_binding.self)
                guard b.index <= Self.maxBindIndex else { throw OpError.invalid("binding index \(b.index)") }
                switch b.kind {
                case MPT_BIND_BUFFER:
                    let (a, off) = try arenas.resolve(b.addr, b.len)
                    plan.append((b.kind, Int(b.index), a, off, Int(b.len)))
                case MPT_BIND_BYTES:
                    guard b.len > 0, b.addr <= len, b.len <= len - b.addr, b.len <= 4096 else {
                        throw ArenaError.outOfRange(b.addr, b.len)
                    }
                    plan.append((b.kind, Int(b.index), nil, Int(b.addr), Int(b.len)))
                case MPT_BIND_TGMEM:
                    guard b.len <= UInt64(device.maxThreadgroupMemoryLength) else { throw OpError.invalid("threadgroup memory \(b.len)") }
                    let l = (Int(b.len) + 15) & ~15
                    tgMemTotal += l
                    plan.append((b.kind, Int(b.index), nil, 0, l))
                default:
                    throw OpError.invalid("binding kind \(b.kind)")
                }
            }
            guard tgMemTotal <= device.maxThreadgroupMemoryLength else {
                throw OpError.invalid("threadgroup memory \(tgMemTotal) > \(device.maxThreadgroupMemoryLength)")
            }
            let enc = compute(s)
            enc.setComputePipelineState(ps)
            for p in plan {
                switch p.kind {
                case MPT_BIND_BUFFER: enc.setBuffer(p.arena!.buffer, offset: p.arena!.bufferOffset + p.off, index: p.index)
                case MPT_BIND_BYTES: enc.setBytes(base.advanced(by: p.off), length: p.len, index: p.index)
                default: if p.len > 0 { enc.setThreadgroupMemoryLength(p.len, index: p.index) }
                }
            }
            if d.use_all_arenas != 0 {
                for a in arenas.all { enc.useResource(a.buffer, usage: [.read, .write]) }
            }
            let g = MTLSize(width: grid[0], height: grid[1], depth: grid[2])
            let t = MTLSize(width: tgs[0], height: tgs[1], depth: tgs[2])
            if d.mode == MPT_DISPATCH_THREADS { enc.dispatchThreads(g, threadsPerThreadgroup: t) } else {
                enc.dispatchThreadgroups(g, threadsPerThreadgroup: t)
            }
        }
        s.batched += 1
        statLock.withLock { stats.dispatches &+= 1 }
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
        guard outLen >= infoSize + 1, outLen <= Self.maxStringBytes else { throw ArenaError.outOfRange(e.a.5, outLen) }
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
