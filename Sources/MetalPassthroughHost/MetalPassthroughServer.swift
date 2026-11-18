// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import MPTProtocol
import Darwin
import Foundation
import Metal

/// Tracks live sessions in this VM and MetalPassthrough-enabled VMs on this host
/// (one marker file per VM under /tmp/metalpassthrough/vms, pruned by pid liveness).
public final class SessionRegistry: @unchecked Sendable {
    public static let root = URL(fileURLWithPath: "/tmp/metalpassthrough")
    private let lock = NSLock()
    private var sessions = 0
    private let marker: URL

    public init(containerID: String) {
        let dir = Self.root.appendingPathComponent("vms", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        marker = dir.appendingPathComponent("\(getpid())-\(containerID)")
        FileManager.default.createFile(atPath: marker.path, contents: Data(containerID.utf8))
    }

    deinit { try? FileManager.default.removeItem(at: marker) }

    public func remove() { try? FileManager.default.removeItem(at: marker) }

    public struct SessionInfo: Sendable {
        public let session: UInt32
        public let pid: UInt32
        public let comm: String
        public var arenaBytes: UInt64 = 0
        public var dispatches: UInt64 = 0
        public var gpuNs: UInt64 = 0
    }
    private var infos: [UInt32: SessionInfo] = [:]
    private var lastKernelNs: UInt64 = 0

    func opened(_ info: SessionInfo) { lock.withLock { sessions += 1; infos[info.session] = info } }
    func closed(_ sid: UInt32) { lock.withLock { sessions -= 1; infos.removeValue(forKey: sid) } }
    func update(_ sid: UInt32, arena: UInt64, dispatches: UInt64, gpuNs: UInt64, lastKernel: UInt64?) {
        lock.withLock {
            infos[sid]?.arenaBytes = arena
            infos[sid]?.dispatches = dispatches
            infos[sid]?.gpuNs = gpuNs
            if let k = lastKernel, k > 0 { lastKernelNs = k }
        }
    }
    var sessionsInThisVM: UInt32 { lock.withLock { UInt32(sessions) } }
    var snapshot: (procs: [SessionInfo], lastKernel: UInt64) { lock.withLock { (infos.values.sorted { $0.session < $1.session }, lastKernelNs) } }

    func activeContainers() -> UInt32 {
        let dir = Self.root.appendingPathComponent("vms")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        var n: UInt32 = 0
        for name in names {
            guard let pid = Int32(name.split(separator: "-").first ?? "") else { continue }
            if kill(pid, 0) == 0 || errno == EPERM { n += 1 } else { try? FileManager.default.removeItem(at: dir.appendingPathComponent(name)) }
        }
        return n
    }
}

/// Per-VM MetalPassthrough host endpoint. Accepts guest connections (vsock port 2049 via the VZ
/// extension) and runs one MetalComputeDispatcher per connection.
@available(macOS 27.0, *)
public final class MetalPassthroughServer: @unchecked Sendable {
    public let device: MTLDevice
    public let memory: GuestMemoryDevice
    public let guestMemoryBytes: UInt64
    public let pipelines: PipelineCache
    public let info: HostGPUInfo
    public let registry: SessionRegistry
    public var options = MetalComputeDispatcher.Options()
    private let lock = NSLock()
    private var nextSession: UInt32 = 1

    public init(memory: GuestMemoryDevice, guestMemoryBytes: UInt64, containerID: String, verbose: Bool = false) throws {
        guard let d = MTLCreateSystemDefaultDevice() else { throw CompileError.pipeline("no Metal device") }
        device = d
        self.memory = memory
        self.guestMemoryBytes = guestMemoryBytes
        pipelines = PipelineCache(device: d)
        info = HostGPUInfo.probe(device: d)
        registry = SessionRegistry(containerID: containerID)
        options.verbose = verbose
    }

    func log(_ s: String) {
        if options.verbose { FileHandle.standardError.write("[metalpassthrough] \(s)\n".data(using: .utf8)!) }
    }

    /// Accepts a connected stream (vsock or unix). Takes ownership of the FileHandle.
    public func accept(_ conn: FileHandle) {
        let t = Thread { [self] in
            let fd = dup(conn.fileDescriptor)
            try? conn.close()
            defer { Darwin.close(fd) }
            do { try serve(FDStream(fd: fd)) } catch { log("session error: \(error)") }
        }
        t.name = "metalpassthrough-session"
        t.qualityOfService = .userInteractive
        t.start()
    }

    func serve(_ s: FDStream) throws {
        var hello = mpt_hello()
        try withUnsafeMutableBytes(of: &hello) { try s.readExact($0.baseAddress!, $0.count) }
        func reply(_ status: Int32, _ base: UInt64 = 0, _ sid: UInt32 = 0) throws {
            var ack = mpt_hello_ack(magic: MPT_MAGIC, status: status, gpu_base: base, session_id: sid, arena_id: 0, reserved: 0)
            try withUnsafeBytes(of: &ack) { try s.writeAll($0.baseAddress!, $0.count) }
        }
        guard hello.magic == MPT_MAGIC, hello.version == MPT_ABI_VERSION else {
            try reply(MPT_EPROTO)
            return
        }
        let nonce = withUnsafeBytes(of: hello.nonce) { Array($0) }
        let comm = withUnsafeBytes(of: hello.comm) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        let arenas = MetalArenaAllocator(device: device, memory: memory, guestMemoryBytes: guestMemoryBytes)
        let t0 = DispatchTime.now().uptimeNanoseconds
        let arena0: GuestArena
        do { arena0 = try arenas.map(nonce: nonce, count: Int(hello.nchunks)) } catch {
            log("HELLO from pid \(hello.pid) (\(comm)): \(error)")
            try reply(MPT_ENOTFOUND)
            return
        }
        guard arena0.length >= Int(MPT_HEAP_OFFSET) else {
            try reply(MPT_EINVAL)
            return
        }
        // The host owns initialization of the control region (it overlaps the chunk-0 stamp).
        memset(arena0.base, 0, Int(MPT_CONTROL_SIZE))
        let ring = arena0.base.assumingMemoryBound(to: mpt_ring_header.self)
        ring.pointee.version = MPT_ABI_VERSION
        ring.pointee.sq_depth = MPT_SQ_DEPTH
        ring.pointee.cq_depth = MPT_CQ_DEPTH
        ring.pointee.arena_bytes = UInt64(arena0.length)
        mpt_fence_seqcst()
        ring.pointee.magic = MPT_MAGIC
        info.write(to: arena0.base.advanced(by: Int(MPT_INFO_OFFSET)))
        let sid = lock.withLock { () -> UInt32 in
            defer { nextSession += 1 }
            return nextSession
        }
        log(String(format: "session %u: pid %u (%@) arena0 %d MiB mapped zero-copy in %.2f ms, gpu base 0x%llx",
                   sid, hello.pid, comm, arena0.length >> 20, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6, arena0.gpuBase))
        let disp = MetalComputeDispatcher(
            sessionID: sid, device: device, arenas: arenas, arena0: arena0, pipelines: pipelines, info: info, stream: s,
            registry: registry, options: options)
        registry.opened(.init(session: sid, pid: hello.pid, comm: comm, arenaBytes: UInt64(arena0.length)))
        defer { registry.closed(sid) }
        try reply(0, arena0.gpuBase, sid)
        disp.run()
    }
}
