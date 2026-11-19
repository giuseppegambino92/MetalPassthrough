// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// Arenas are guest-owned hugepage regions that the host locates in guest RAM, remaps contiguously
// and wraps as a single no-copy MTLBuffer. Thousands of tensors live inside one arena and are
// addressed by device address (gpuAddress + offset), so Metal buffer creation happens once per
// arena, never per tensor.

import MPTProtocol
import Foundation
import Metal

public final class GuestArena: @unchecked Sendable {
    public let id: UInt32
    public let region: RemappedGuestRegion
    public let buffer: MTLBuffer
    /// Byte offset of guest chunk 0 inside `buffer` (host-owned alignment padding before it).
    public let bufferOffset: Int
    /// Device address of guest chunk 0. 2 MiB aligned when unified addressing could be arranged.
    /// (Stored: it is read on every dispatch and must not cost an Objective-C message.)
    public let gpuBase: UInt64
    public let length: Int
    public let base: UnsafeMutableRawPointer
    public var unifiedAddressable: Bool { gpuBase % UInt64(MPT_CHUNK_SIZE) == 0 }

    /// Keeps the arena GPU-resident (page tables populated up front instead of on first dispatch).
    public let residency: MTLResidencySet?

    init(id: UInt32, region: RemappedGuestRegion, buffer: MTLBuffer, residency: MTLResidencySet?) {
        self.id = id
        self.region = region
        self.buffer = buffer
        self.residency = residency
        bufferOffset = region.prefix
        gpuBase = buffer.gpuAddress + UInt64(region.prefix)
        length = region.length - region.prefix
        base = region.base.advanced(by: region.prefix)
    }

    @inline(__always) func contains(_ dev: UInt64, _ len: UInt64) -> Bool {
        dev >= gpuBase && len <= UInt64(length) && dev - gpuBase <= UInt64(length) - len
    }
}

public enum ArenaError: Error, CustomStringConvertible {
    case chunksNotFound(found: Int, expected: Int)
    case bufferRefused(Int)
    case outOfRange(UInt64, UInt64)

    public var description: String {
        switch self {
        case .chunksNotFound(let f, let e): return "located \(f)/\(e) arena chunks in guest RAM (are hugepages reserved?)"
        case .bufferRefused(let l): return "MTLDevice.makeBuffer(bytesNoCopy:) refused \(l) bytes"
        case .outOfRange(let a, let l): return String(format: "device range 0x%llx+%llu is outside every arena", a, l)
        }
    }
}

@available(macOS 27.0, *)
public final class MetalArenaAllocator: @unchecked Sendable {
    static let creationLock = NSLock()
    public let device: MTLDevice
    public let memory: any GuestRAMProvider
    public let guestMemoryBytes: UInt64
    private var arenas: [UInt32: GuestArena] = [:]
    private var nextID: UInt32 = 0
    private let lock = NSLock()

    public init(device: MTLDevice, memory: any GuestRAMProvider, guestMemoryBytes: UInt64) {
        self.device = device
        self.memory = memory
        self.guestMemoryBytes = guestMemoryBytes
    }

    /// Locates `count` stamped chunks in guest RAM and wraps them as one MTLBuffer.
    /// Largest arena a guest may request: all of its RAM in 2 MiB chunks (guards host allocations
    /// sized by guest-controlled counts).
    public var maxChunks: UInt64 { max(1, guestMemoryBytes / UInt64(MPT_CHUNK_SIZE)) }

    /// Host-side switch to exercise translated (non-UVA) addressing end to end.
    static let uvaDisabled = ProcessInfo.processInfo.environment["METALPASSTHROUGH_NO_UVA"] != nil

    public func map(nonce: [UInt8], count: Int) throws -> GuestArena {
        guard nonce.count == 16, count >= 1, UInt64(count) <= maxChunks else { throw ArenaError.chunksNotFound(found: 0, expected: count) }
        guard nonce.contains(where: { $0 != 0 }) else { throw ArenaError.chunksNotFound(found: 0, expected: count) }
        let ram = try memory.guestRAMView(expectedSize: guestMemoryBytes)
        let chunk = Int(MPT_CHUNK_SIZE)
        var found = [UnsafeMutableRawPointer?](repeating: nil, count: count)
        var nfound = 0
        let base = ram.bytes
        nonce.withUnsafeBytes { n in
            var off = 0
            while off + chunk <= ram.length {
                let p = base.advanced(by: off)
                if memcmp(p, n.baseAddress!, 16) == 0 {
                    let stamp = p.assumingMemoryBound(to: mpt_chunk_stamp.self).pointee
                    if stamp.count == UInt64(count), stamp.index < UInt64(count), found[Int(stamp.index)] == nil {
                        found[Int(stamp.index)] = p
                        nfound += 1
                        if nfound == count { break }
                    }
                }
                off += chunk
            }
        }
        guard nfound == count else { throw ArenaError.chunksNotFound(found: nfound, expected: count) }
        // Unified addressing: place guest chunk 0 at a 2 MiB-aligned GPU virtual address so the guest
        // can mremap its hugepages to the *same* numeric address (guest pointer == device pointer).
        // Metal hands out GPU VAs from a bump allocator; predict the next one with a probe buffer,
        // pad the region accordingly and verify. Falls back to translated addressing.
        let sources = found.map { $0! }
        var result: (RemappedGuestRegion, MTLBuffer)?
        try Self.creationLock.withLock {
            for _ in 0..<(Self.uvaDisabled ? 1 : 8) {
                var predicted: UInt64 = 0
                do {
                    guard let probe = device.makeBuffer(length: 16384, options: .storageModeShared) else { break }
                    predicted = probe.gpuAddress
                }
                var pad = Int((UInt64(chunk) - predicted % UInt64(chunk)) % UInt64(chunk))
                if Self.uvaDisabled { pad = 16384 }  // deliberately misaligned
                let region = try RemappedGuestRegion(sources: sources, chunkSize: chunk, prefix: pad)
                guard
                    let buf = device.makeBuffer(
                        bytesNoCopy: region.base, length: region.length, options: [.storageModeShared, .hazardTrackingModeUntracked],
                        deallocator: nil)
                else { throw ArenaError.bufferRefused(region.length) }
                result = (region, buf)
                if (buf.gpuAddress + UInt64(pad)) % UInt64(chunk) == 0 { break }
            }
        }
        guard let (region, buf) = result else { throw ArenaError.bufferRefused(count * chunk) }
        // Make the whole arena resident now so the first kernel does not pay for GPU page-table setup.
        let rsd = MTLResidencySetDescriptor()
        rsd.initialCapacity = 1
        let rs = ProcessInfo.processInfo.environment["METALPASSTHROUGH_NO_RESIDENCY"] == nil ? try? device.makeResidencySet(descriptor: rsd) : nil
        if let rs {
            rs.addAllocation(buf)
            rs.commit()
            rs.requestResidency()
        }
        return lock.withLock {
            let a = GuestArena(id: nextID, region: region, buffer: buf, residency: rs)
            buf.label = "metalpassthrough.arena.\(nextID)"
            arenas[nextID] = a
            nextID += 1
            return a
        }
    }

    public func free(id: UInt32) -> Bool { lock.withLock { arenas.removeValue(forKey: id) != nil } }

    public func arena(id: UInt32) -> GuestArena? { lock.withLock { arenas[id] } }

    public var all: [GuestArena] { lock.withLock { Array(arenas.values) } }

    public var totalBytes: UInt64 { lock.withLock { arenas.values.reduce(0) { $0 + UInt64($1.length) } } }

    /// Resolves a device address range to (arena, byte offset), validating bounds.
    public func resolve(_ dev: UInt64, _ len: UInt64) throws -> (GuestArena, Int) {
        try lock.withLock {
            for a in arenas.values where a.contains(dev, len) { return (a, Int(dev - a.gpuBase)) }
            throw ArenaError.outOfRange(dev, len)
        }
    }

    /// Host pointer for a validated device address range.
    public func pointer(_ dev: UInt64, _ len: UInt64) throws -> UnsafeMutableRawPointer {
        let (a, off) = try resolve(dev, len)
        return a.base.advanced(by: off)
    }
}
