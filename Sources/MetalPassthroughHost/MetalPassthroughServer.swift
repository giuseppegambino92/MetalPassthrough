// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import MPTProtocol
import Darwin
import Foundation
import Metal

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

    /// Keeper bookkeeping per session: the keeper connection is released only after the session's
    /// GPU work has drained and its arenas are unmapped (crash-safe teardown).
    private final class KeeperLink {
        let nonce: [UInt8]
        var fd: Int32 = -1
        var ended = false
        init(nonce: [UInt8]) { self.nonce = nonce }
    }
    private var keepers: [UInt32: KeeperLink] = [:]

    private static func release(_ fd: Int32) {
        var b: UInt8 = 1
        _ = Darwin.write(fd, &b, 1)
        Darwin.close(fd)
    }

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

    /// Keeper connection: authenticate, then park its fd until the session has fully drained.
    func serveKeeper(_ s: FDStream) throws {
        var k = mpt_keeper_hello()
        k.magic = MPT_KEEPER_MAGIC
        try withUnsafeMutableBytes(of: &k) { try s.readExact($0.baseAddress!.advanced(by: 4), $0.count - 4) }
        let nonce = withUnsafeBytes(of: k.nonce) { Array($0) }
        let fd = dup(s.fd)  // the accept thread closes its own descriptor
        Self.setReadTimeout(fd, seconds: 0)
        let action: (release: Bool, accepted: Bool) = lock.withLock {
            guard let link = keepers[k.session_id], link.nonce == nonce, link.fd < 0 else { return (false, false) }
            if link.ended { return (true, true) }
            link.fd = fd
            return (false, true)
        }
        if !action.accepted {
            log("keeper for session \(k.session_id) rejected (unknown session or bad nonce)")
            Darwin.close(fd)
        } else if action.release {
            Self.release(fd)
        }
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

    /// Bounds how long a peer may take to send its handshake (a partial hello must not pin a host
    /// thread forever). Cleared once the session is established.
    private static func setReadTimeout(_ fd: Int32, seconds: Int) {
        var tv = timeval(tv_sec: seconds, tv_usec: 0)
        _ = setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
    }

    /// How sessions are isolated from each other on the GPU (host environment
    /// METALPASSTHROUGH_ISOLATION): `process` (default) runs every guest process's session in its own
    /// GPU worker process, hence its own GPU address space, so a shader's forged pointers cannot
    /// reach other sessions; `shared` runs all sessions of the VM in this process (one GPU address
    /// space per VM: faster session start, no intra-VM isolation).
    public enum Isolation: String, Sendable { case process, shared }
    public var isolation: Isolation =
        Isolation(rawValue: ProcessInfo.processInfo.environment["METALPASSTHROUGH_ISOLATION"] ?? "") ?? .process

    func serve(_ s: FDStream) throws {
        Self.setReadTimeout(s.fd, seconds: 5)
        var magic: UInt32 = 0
        try withUnsafeMutableBytes(of: &magic) { try s.readExact($0.baseAddress!, 4) }
        if magic == MPT_KEEPER_MAGIC { try serveKeeper(s); return }
        var hello = mpt_hello()
        hello.magic = magic
        try withUnsafeMutableBytes(of: &hello) { try s.readExact($0.baseAddress!.advanced(by: 4), $0.count - 4) }
        guard hello.magic == MPT_MAGIC, hello.version == MPT_ABI_VERSION else {
            try SessionRunner.reply(s, MPT_EPROTO)
            return
        }
        guard hello.nchunks >= 1, UInt64(hello.nchunks) <= max(1, guestMemoryBytes / UInt64(MPT_CHUNK_SIZE)) else {
            try SessionRunner.reply(s, MPT_EINVAL)
            return
        }
        let nonce = withUnsafeBytes(of: hello.nonce) { Array($0) }
        let comm = withUnsafeBytes(of: hello.comm) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        let sid = lock.withLock { () -> UInt32 in
            defer { nextSession += 1 }
            return nextSession
        }
        // The keeper may connect as soon as the guest sees the ack: register it first.
        let link = KeeperLink(nonce: nonce)
        lock.withLock { keepers[sid] = link }
        registry.opened(.init(session: sid, pid: hello.pid, comm: comm))
        defer {
            registry.closed(sid)
            let fd = lock.withLock { () -> Int32 in
                link.ended = true
                keepers.removeValue(forKey: sid)
                defer { link.fd = -1 }
                return link.fd
            }
            if fd >= 0 {
                Self.release(fd)
                log("session \(sid): GPU drained, keeper released (guest may now reuse the hugepages)")
            }
        }
        switch isolation {
        case .shared:
            try SessionRunner.run(
                stream: s, hello: hello, sid: sid, device: device, memory: memory, guestMemoryBytes: guestMemoryBytes,
                pipelines: pipelines, info: info, registry: registry, options: options, caps: 0)
        case .process:
            // Returns when the worker exited: its GPU work has drained (or died with the process,
            // which tears down its GPU context), so the keeper can release the pages.
            try runWorker(sessionFD: s.fd, sid: sid, hello: hello)
        }
    }

    // MARK: - Per-session GPU worker processes

    /// Mach memory entries shared into every worker: guest RAM and the session table.
    private var sharedPorts: (ram: mach_port_t, table: mach_port_t, ramBase: UInt64, ramLength: Int)?
    private static let spawnLock = NSLock()  // mach_ports_register is per task

    private static func memoryEntry(_ addr: UnsafeMutableRawPointer, _ length: Int) throws -> mach_port_t {
        var size = memory_object_size_t(length)
        var port: mach_port_t = 0
        let MAP_MEM_VM_SHARE: vm_prot_t = 0x400000  // share the existing mapping's pages (no copy)
        let kr = mach_make_memory_entry_64(
            mach_task_self_, &size, memory_object_offset_t(UInt(bitPattern: addr)), VM_PROT_READ | VM_PROT_WRITE | MAP_MEM_VM_SHARE, &port, 0)
        guard kr == KERN_SUCCESS, size >= memory_object_size_t(length) else { throw GuestMemoryError.machError("mach_make_memory_entry_64", kr) }
        return port
    }

    private func ports() throws -> (ram: mach_port_t, table: mach_port_t, ramBase: UInt64, ramLength: Int) {
        try lock.withLock {
            if let p = sharedPorts { return p }
            let ram = try memory.guestRAMView(expectedSize: guestMemoryBytes)
            let p = (try Self.memoryEntry(ram.bytes, ram.length), try Self.memoryEntry(registry.table, SessionRegistry.tableBytes), ram.base, ram.length)
            sharedPorts = p
            return p
        }
    }

    /// mpt-gpu-worker lookup: $METALPASSTHROUGH_WORKER, next to this executable (mpt-run, the runtime
    /// plugin's bin/), or <exe>/../share/metalpassthrough.
    public static func findWorker() -> String? {
        var candidates: [String] = []
        if let e = ProcessInfo.processInfo.environment["METALPASSTHROUGH_WORKER"] { candidates.append(e) }
        let exe = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath().deletingLastPathComponent()
        candidates += [exe.appendingPathComponent("mpt-gpu-worker").path, exe.appendingPathComponent("../share/metalpassthrough/mpt-gpu-worker").path]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// Spawns `mpt-gpu-worker` with `fd3` as its fd 3 and the shared memory entries registered.
    private func spawnWorker(_ extra: [String], fd3: Int32) throws -> pid_t {
        guard let worker = Self.findWorker() else {
            throw CompileError.pipeline("mpt-gpu-worker not found (set METALPASSTHROUGH_WORKER, or METALPASSTHROUGH_ISOLATION=shared)")
        }
        let p = try ports()
        var args = [worker, "--ram-base", "\(p.ramBase)", "--ram-length", "\(p.ramLength)", "--guest-memory", "\(guestMemoryBytes)"] + extra
        if options.verbose { args.append("--verbose") }
        // Probed (including the bandwidth measurement) once by this process, not per session.
        if let json = try? JSONEncoder().encode(info) { args += ["--device-info", json.base64EncodedString()] }

        var fa: posix_spawn_file_actions_t?
        posix_spawn_file_actions_init(&fa)
        defer { posix_spawn_file_actions_destroy(&fa) }
        posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0)
        posix_spawn_file_actions_addinherit_np(&fa, 1)
        posix_spawn_file_actions_addinherit_np(&fa, 2)
        if fd3 == 3 { posix_spawn_file_actions_addinherit_np(&fa, 3) } else { posix_spawn_file_actions_adddup2(&fa, fd3, 3) }
        var attr: posix_spawnattr_t?
        posix_spawnattr_init(&attr)
        defer { posix_spawnattr_destroy(&attr) }
        // Only fds 0-3 reach the worker: never another session's socket.
        posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_CLOEXEC_DEFAULT))
        let cargs = args.map { strdup($0) } + [nil]
        defer { cargs.forEach { free($0) } }
        var pid: pid_t = 0
        let rc: Int32 = Self.spawnLock.withLock {
            // Registered ports are copied into a task when it is created: hand the worker the guest
            // RAM and session-table entries, and clear them again so nothing else inherits them.
            var regs = [p.ram, p.table]
            _ = regs.withUnsafeMutableBufferPointer { mach_ports_register(mach_task_self_, $0.baseAddress, 2) }
            defer { _ = mach_ports_register(mach_task_self_, nil, 0) }
            return posix_spawn(&pid, worker, &fa, &attr, cargs, environ)
        }
        guard rc == 0 else { throw CompileError.pipeline("spawning \(worker) failed: \(String(cString: strerror(rc)))") }
        return pid
    }

    /// One pre-started worker (Metal device and caches initialized, guest RAM mapped) so a session
    /// does not wait for process start-up and MTLDevice creation (~50 ms).
    private var spare: (pid: pid_t, control: Int32)?
    private var spawningSpare = false

    private func refillSpare() {
        let go: Bool = lock.withLock {
            guard spare == nil, !spawningSpare else { return false }
            spawningSpare = true
            return true
        }
        guard go else { return }
        DispatchQueue.global(qos: .utility).async { [self] in
            var sv: [Int32] = [-1, -1]
            var result: (pid_t, Int32)?
            if socketpair(AF_UNIX, SOCK_STREAM, 0, &sv) == 0 {
                if let pid = try? spawnWorker(["--spare"], fd3: sv[1]) { result = (pid, sv[0]) } else { Darwin.close(sv[0]) }
                Darwin.close(sv[1])
            }
            lock.withLock {
                spawningSpare = false
                if let result { spare = (result.0, result.1) }
            }
        }
    }

    private func runWorker(sessionFD: Int32, sid: UInt32, hello: mpt_hello) throws {
        var pid: pid_t = -1
        if let sp = lock.withLock({ () -> (pid: pid_t, control: Int32)? in defer { spare = nil }; return spare }) {
            var msg = WorkerAssignment(sid: sid, reserved: 0, hello: hello)
            let sent = withUnsafeBytes(of: &msg) { mpt_send_fd(sp.control, sessionFD, $0.baseAddress, $0.count) } == 0
            Darwin.close(sp.control)  // the spare exits if it never got a session
            if sent { pid = sp.pid } else { kill(sp.pid, SIGKILL); var st: Int32 = 0; waitpid(sp.pid, &st, 0) }
        }
        if pid < 0 {
            var h = hello
            let helloHex = withUnsafeBytes(of: &h) { $0.map { String(format: "%02x", $0) }.joined() }
            do { pid = try spawnWorker(["--sid", "\(sid)", "--hello", helloHex], fd3: sessionFD) } catch {
                log("session \(sid): \(error)")
                try SessionRunner.reply(FDStream(fd: sessionFD), MPT_EPROTO)
                return
            }
        }
        refillSpare()
        log("session \(sid): GPU worker pid \(pid)")
        var status: Int32 = 0
        while waitpid(pid, &status, 0) < 0 && errno == EINTR {}
        let signaled = status & 0x7f
        if signaled != 0 && signaled != 0x7f {
            log("session \(sid): GPU worker \(pid) killed by signal \(signaled)")
        } else if (status >> 8) & 0xff != 0 {
            log("session \(sid): GPU worker \(pid) exited with status \((status >> 8) & 0xff)")
        }
    }
}
