// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import Darwin
import Foundation
import Metal
import MPTProtocol

/// One guest session after its HELLO was read: maps arena 0, initializes the control region, sends
/// the ack and runs the dispatcher until the session ends. Runs either inside the VM-owning process
/// (isolation: shared) or inside a per-session GPU worker process (isolation: process).
@available(macOS 27.0, *)
public enum SessionRunner {
    static func reply(_ s: FDStream, _ status: Int32, _ base: UInt64 = 0, _ sid: UInt32 = 0, _ caps: UInt64 = 0) throws {
        var ack = mpt_hello_ack(magic: MPT_MAGIC, status: status, gpu_base: base, session_id: sid, arena_id: 0, caps: caps)
        try withUnsafeBytes(of: &ack) { try s.writeAll($0.baseAddress!, $0.count) }
    }

    static func run(
        stream s: FDStream, hello: mpt_hello, sid: UInt32, device: MTLDevice, memory: any GuestRAMProvider, guestMemoryBytes: UInt64,
        pipelines: PipelineCache, info: HostGPUInfo, registry: SessionRegistry, options: MetalComputeDispatcher.Options, caps: UInt64
    ) throws {
        func log(_ m: String) {
            if options.verbose { FileHandle.standardError.write("[metalpassthrough] \(m)\n".data(using: .utf8)!) }
        }
        let nonce = withUnsafeBytes(of: hello.nonce) { Array($0) }
        let comm = withUnsafeBytes(of: hello.comm) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        let arenas = MetalArenaAllocator(device: device, memory: memory, guestMemoryBytes: guestMemoryBytes)
        guard hello.nchunks >= 1, UInt64(hello.nchunks) <= arenas.maxChunks else {
            try reply(s, MPT_EINVAL)
            return
        }
        let t0 = DispatchTime.now().uptimeNanoseconds
        let arena0: GuestArena
        do { arena0 = try arenas.map(nonce: nonce, count: Int(hello.nchunks)) } catch {
            log("HELLO from pid \(hello.pid) (\(comm)): \(error)")
            try reply(s, MPT_ENOTFOUND)
            return
        }
        guard arena0.length >= Int(MPT_HEAP_OFFSET) else {
            try reply(s, MPT_EINVAL)
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
        log(String(format: "session %u: pid %u (%@) arena0 %d MiB mapped zero-copy in %.2f ms, gpu base 0x%llx, %@",
                   sid, hello.pid, comm, arena0.length >> 20, Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6, arena0.gpuBase,
                   caps & MPT_CAP_PROCESS_ISOLATION != 0 ? "own GPU process \(getpid())" : "shared GPU process"))
        let disp = MetalComputeDispatcher(
            sessionID: sid, device: device, arenas: arenas, arena0: arena0, pipelines: pipelines, info: info, stream: s,
            registry: registry, options: options)
        try reply(s, 0, arena0.gpuBase, sid,
                  caps | MPT_CAP_STREAMS | MPT_CAP_KEEPER | (arena0.unifiedAddressable ? MPT_CAP_UVA : 0))
        var tv = timeval(tv_sec: 0, tv_usec: 0)  // established: the doorbell may legitimately stay idle
        _ = setsockopt(s.fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        disp.run()  // returns only after all GPU work drained and the arenas were unmapped
    }
}

/// Guest RAM inside a GPU worker: a mapping of the Mach memory entry the VM-owning process shared.
final class SharedGuestRAM: GuestRAMProvider, @unchecked Sendable {
    let view: GuestRAMView
    init(view: GuestRAMView) { self.view = view }
    func guestRAMView(expectedSize: UInt64) throws -> GuestRAMView { view }
}

/// Message the VM-owning process sends a pre-started (spare) worker along with the session socket.
struct WorkerAssignment {
    var sid: UInt32 = 0
    var reserved: UInt32 = 0
    var hello = mpt_hello()
}

/// Entry point of `mpt-gpu-worker`: one guest session in its own process (own GPU address space).
/// Spawned by MetalPassthroughServer with two registered Mach ports (guest RAM, session table) and
/// either the session socket as fd 3 (`--sid`/`--hello`), or, as a pre-started spare (`--spare`), a
/// control socket as fd 3 on which it later receives the session socket and its HELLO.
@available(macOS 27.0, *)
public enum MetalPassthroughWorker {
    public static func main() -> Int32 {
        let tStart = DispatchTime.now().uptimeNanoseconds
        var marks: [(String, UInt64)] = []
        func mark(_ n: String) { marks.append((n, DispatchTime.now().uptimeNanoseconds)) }
        var opts: [String: String] = [:]
        var verbose = false, spare = false
        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let a = it.next() {
            if a == "--verbose" { verbose = true } else if a == "--spare" { spare = true } else if a.hasPrefix("--"), let v = it.next() {
                opts[String(a.dropFirst(2))] = v
            }
        }
        func fail(_ m: String) -> Int32 {
            FileHandle.standardError.write("[mpt-gpu-worker] \(m)\n".data(using: .utf8)!)
            return 2
        }
        guard let ramBase = opts["ram-base"].flatMap({ UInt64($0) }), let ramLength = opts["ram-length"].flatMap({ Int($0) }),
            let guestMemory = opts["guest-memory"].flatMap({ UInt64($0) })
        else { return fail("usage: mpt-gpu-worker --ram-base B --ram-length L --guest-memory M (--sid N --hello HEX | --spare); spawned by the host") }

        // The two memory entries the parent registered before spawning us.
        var ports: mach_port_array_t?
        var n: mach_msg_type_number_t = 0
        guard mach_ports_lookup(mach_task_self_, &ports, &n) == KERN_SUCCESS, let ports, n >= 2, ports[0] != 0, ports[1] != 0 else {
            return fail("no shared memory entries (must be spawned by the MetalPassthrough host)")
        }
        let ramPort = ports[0], tablePort = ports[1]
        _ = mach_ports_register(mach_task_self_, nil, 0)  // nothing we might spawn inherits them
        func map(_ port: mach_port_t, _ length: Int) -> UnsafeMutableRawPointer? {
            var addr: mach_vm_address_t = 0
            let kr = mach_vm_map(mach_task_self_, &addr, mach_vm_size_t(length), 0, VM_FLAGS_ANYWHERE, port, 0, 0,
                                 VM_PROT_READ | VM_PROT_WRITE, VM_PROT_READ | VM_PROT_WRITE, VM_INHERIT_NONE)
            return kr == KERN_SUCCESS ? UnsafeMutableRawPointer(bitPattern: UInt(addr)) : nil
        }
        guard let ram = map(ramPort, ramLength), let table = map(tablePort, SessionRegistry.tableBytes) else {
            return fail("mapping the shared guest RAM failed")
        }
        mark("ports+map")
        guard let device = MTLCreateSystemDefaultDevice() else { return fail("no Metal device") }
        mark("MTLDevice")
        let pipelines = PipelineCache(device: device)
        mark("pipeline cache")
        let info = opts["device-info"].flatMap { Data(base64Encoded: $0) }.flatMap { try? JSONDecoder().decode(HostGPUInfo.self, from: $0) }
            ?? HostGPUInfo.probe(device: device)

        // Session assignment: given on the command line (cold start) or received later (spare).
        var sid: UInt32 = 0
        var hello = mpt_hello()
        var sessionFD: Int32 = 3
        if spare {
            var msg = WorkerAssignment()
            var fd: Int32 = -1
            // Blocks until the host hands us a session; EOF means the host is gone or shutting down.
            guard withUnsafeMutableBytes(of: &msg, { mpt_recv_fd(3, &fd, $0.baseAddress, $0.count) }) == 0 else { return 0 }
            Darwin.close(3)
            sid = msg.sid
            hello = msg.hello
            sessionFD = fd
            mark("waited for a session")
        } else {
            guard let s = opts["sid"].flatMap({ UInt32($0) }), let hex = opts["hello"], hex.count == MemoryLayout<mpt_hello>.size * 2 else {
                return fail("missing --sid/--hello")
            }
            let bytes = stride(from: 0, to: hex.count, by: 2).compactMap { UInt8(hex.dropFirst($0).prefix(2), radix: 16) }
            guard bytes.count == MemoryLayout<mpt_hello>.size else { return fail("bad --hello") }
            withUnsafeMutableBytes(of: &hello) { $0.copyBytes(from: bytes) }
            sid = s
        }
        if verbose || ProcessInfo.processInfo.environment["MPT_WORKER_TIMING"] != nil {
            var prev = tStart
            let parts = marks.map { m -> String in defer { prev = m.1 }; return String(format: "%@ %.1f ms", m.0, Double(m.1 - prev) / 1e6) }
            FileHandle.standardError.write("[mpt-gpu-worker] session \(sid)\(spare ? " (pre-started)" : ""): \(parts.joined(separator: ", "))\n".data(using: .utf8)!)
        }
        var options = MetalComputeDispatcher.Options()
        options.verbose = verbose
        do {
            try SessionRunner.run(
                stream: FDStream(fd: sessionFD), hello: hello, sid: sid, device: device,
                memory: SharedGuestRAM(view: GuestRAMView(base: ramBase, bytes: ram, length: ramLength)), guestMemoryBytes: guestMemory,
                pipelines: pipelines, info: info, registry: SessionRegistry(attaching: table), options: options, caps: MPT_CAP_PROCESS_ISOLATION)
        } catch {
            return fail("session \(sid): \(error)")
        }
        return 0
    }
}
