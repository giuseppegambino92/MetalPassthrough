// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import Darwin
import Foundation
import MPTProtocol

/// Live sessions of one VM (for apple-smi's process table) plus MetalPassthrough-enabled VMs on this
/// host (one marker file per VM under /tmp/metalpassthrough/vms, pruned by pid liveness).
///
/// The table lives in a small shared-memory region so that per-session GPU worker processes can
/// publish their statistics and read the VM-wide view: the VM-owning process allocates it and owns
/// slot assignment; every writer updates a slot under that slot's seqlock.
///
/// Layout: 128-byte header (vm last-kernel time) + MPT_MAX_PROCS slots of 128 bytes:
///   0 seq u32 | 4 inUse u32 | 8 session u32 | 12 pid u32 | 16 arena u64 | 24 dispatches u64 |
///   32 gpuNs u64 | 40..56 comm[16]
public final class SessionRegistry: @unchecked Sendable {
    public static let root = URL(fileURLWithPath: "/tmp/metalpassthrough")
    public static let tableBytes = 16384
    static let slotCount = Int(MPT_MAX_PROCS)
    static let slotSize = 128
    static let headerSize = 128

    public let table: UnsafeMutableRawPointer
    private let ownsTable: Bool
    private let marker: URL?
    private let lock = NSLock()  // serializes slot assignment within the owning process

    /// VM-owning process: allocates the shared table and registers this VM.
    public init(containerID: String) {
        var addr: mach_vm_address_t = 0
        let kr = mach_vm_allocate(mach_task_self_, &addr, mach_vm_size_t(Self.tableBytes), VM_FLAGS_ANYWHERE)
        precondition(kr == KERN_SUCCESS, "session table allocation failed: \(kr)")
        table = UnsafeMutableRawPointer(bitPattern: UInt(addr))!  // zero-filled
        ownsTable = true
        let dir = Self.root.appendingPathComponent("vms", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let m = dir.appendingPathComponent("\(getpid())-\(containerID)")
        FileManager.default.createFile(atPath: m.path, contents: Data(containerID.utf8))
        marker = m
    }

    /// GPU worker process: attaches to a table shared by the VM-owning process.
    public init(attaching table: UnsafeMutableRawPointer) {
        self.table = table
        ownsTable = false
        marker = nil
    }

    deinit {
        if let marker { try? FileManager.default.removeItem(at: marker) }
        if ownsTable { mach_vm_deallocate(mach_task_self_, mach_vm_address_t(UInt(bitPattern: table)), mach_vm_size_t(Self.tableBytes)) }
    }

    public func remove() { if let marker { try? FileManager.default.removeItem(at: marker) } }

    public struct SessionInfo: Sendable {
        public let session: UInt32
        public let pid: UInt32
        public let comm: String
        public var arenaBytes: UInt64 = 0
        public var dispatches: UInt64 = 0
        public var gpuNs: UInt64 = 0
    }

    private func slot(_ i: Int) -> UnsafeMutableRawPointer { table.advanced(by: Self.headerSize + i * Self.slotSize) }

    /// Seqlock write: odd sequence while fields change.
    private func write(_ i: Int, _ body: (UnsafeMutableRawPointer) -> Void) {
        let p = slot(i)
        let seq = mpt_load_relaxed_u32(p)
        mpt_store_relaxed_u32(p, seq &+ 1)
        mpt_fence_release()
        body(p)
        mpt_store_release_u32(p, seq &+ 2)
    }

    /// Consistent copy of a slot (nil when unused).
    private func read(_ i: Int) -> SessionInfo? {
        let p = slot(i)
        for _ in 0..<1000 {
            let s0 = mpt_load_acquire_u32(p)
            if s0 & 1 != 0 { continue }
            let used = mpt_load_relaxed_u32(p + 4)
            let sid = mpt_load_relaxed_u32(p + 8), pid = mpt_load_relaxed_u32(p + 12)
            let arena = mpt_load_relaxed_u64(p + 16), disp = mpt_load_relaxed_u64(p + 24), gpu = mpt_load_relaxed_u64(p + 32)
            var name = [UInt8](repeating: 0, count: 16)
            for k in 0..<16 { name[k] = p.load(fromByteOffset: 40 + k, as: UInt8.self) }
            mpt_fence_acquire()
            guard mpt_load_relaxed_u32(p) == s0 else { continue }
            guard used != 0 else { return nil }
            let comm = String(decoding: name.prefix { $0 != 0 }, as: UTF8.self)
            return SessionInfo(session: sid, pid: pid, comm: comm, arenaBytes: arena, dispatches: disp, gpuNs: gpu)
        }
        return nil
    }

    private func find(_ sid: UInt32) -> Int? {
        (0..<Self.slotCount).first { mpt_load_relaxed_u32(slot($0) + 4) != 0 && mpt_load_relaxed_u32(slot($0) + 8) == sid }
    }

    /// Assigns a slot (VM-owning process). Sessions beyond MPT_MAX_PROCS still run, untracked.
    func opened(_ info: SessionInfo) {
        lock.withLock {
            guard let i = (0..<Self.slotCount).first(where: { mpt_load_relaxed_u32(slot($0) + 4) == 0 }) else { return }
            write(i) { p in
                mpt_store_relaxed_u32(p + 8, info.session)
                mpt_store_relaxed_u32(p + 12, info.pid)
                mpt_store_relaxed_u64(p + 16, info.arenaBytes)
                mpt_store_relaxed_u64(p + 24, 0)
                mpt_store_relaxed_u64(p + 32, 0)
                let name = Array(info.comm.utf8.prefix(15)) + [UInt8](repeating: 0, count: 16)
                for k in 0..<16 { p.storeBytes(of: name[k], toByteOffset: 40 + k, as: UInt8.self) }
                mpt_store_relaxed_u32(p + 4, 1)
            }
        }
    }

    func closed(_ sid: UInt32) {
        lock.withLock {
            guard let i = find(sid) else { return }
            write(i) { p in mpt_store_relaxed_u32(p + 4, 0) }
        }
    }

    func update(_ sid: UInt32, arena: UInt64, dispatches: UInt64, gpuNs: UInt64, lastKernel: UInt64?) {
        if let k = lastKernel, k > 0 { mpt_store_relaxed_u64(table, k) }
        guard let i = find(sid) else { return }
        write(i) { p in
            mpt_store_relaxed_u64(p + 16, arena)
            mpt_store_relaxed_u64(p + 24, dispatches)
            mpt_store_relaxed_u64(p + 32, gpuNs)
        }
    }

    var sessionsInThisVM: UInt32 { UInt32((0..<Self.slotCount).filter { mpt_load_relaxed_u32(slot($0) + 4) != 0 }.count) }

    var snapshot: (procs: [SessionInfo], lastKernel: UInt64) {
        ((0..<Self.slotCount).compactMap { read($0) }.sorted { $0.session < $1.session }, mpt_load_relaxed_u64(table))
    }

    func activeContainers() -> UInt32 {
        let dir = Self.root.appendingPathComponent("vms")
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        var n: UInt32 = 0
        for name in names {
            guard let pid = Int32(name.split(separator: "-").first ?? "") else { continue }
            if kill(pid, 0) == 0 || errno == EPERM { n += 1 } else if ownsTable { try? FileManager.default.removeItem(at: dir.appendingPathComponent(name)) }
        }
        return n
    }
}
