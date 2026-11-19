// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// GuestMemoryDevice — a VZCustomVirtioDevice (macOS 27+) whose only purpose is to give the
// host process direct, cacheable access to the guest's DRAM through
// `guestMemoryMapping(atPhysicalAddress:length:)`.
//
// This is what makes MetalPassthrough genuinely zero-copy: Apple's virtio-fs implementation has no
// DAX window, so a guest `mmap(MAP_SHARED)` of a virtio-fs file goes through the guest page cache
// and is NOT backed by the same physical pages as the host's mapping. Guest RAM, by contrast, is
// ordinary unified memory that the host CPU — and therefore the Apple GPU — can address directly.

import Darwin
import Foundation
import Virtualization

public enum GuestMemoryError: Error, CustomStringConvertible {
    case deviceNotCreated
    case ramNotFound
    case invalidRange(UInt64, UInt64)
    case machError(String, kern_return_t)

    public var description: String {
        switch self {
        case .deviceNotCreated: return "MetalPassthrough virtio device was not created by Virtualization.framework"
        case .ramNotFound: return "could not locate guest RAM in the guest physical address space"
        case .invalidRange(let a, let l): return String(format: "guest physical range 0x%llx+0x%llx is not RAM", a, l)
        case .machError(let op, let kr): return "\(op) failed: \(String(cString: mach_error_string(kr))) (\(kr))"
        }
    }
}

/// A host view of guest DRAM: guest-physical `base`, mapped at `bytes` for `length` bytes.
public struct GuestRAMView: @unchecked Sendable {
    public let base: UInt64
    public let bytes: UnsafeMutableRawPointer
    public let length: Int
}

/// Where the arena allocator finds guest RAM: the VZ device in the VM-owning process, or a Mach
/// memory entry shared into a per-session GPU worker process.
public protocol GuestRAMProvider: AnyObject, Sendable {
    func guestRAMView(expectedSize: UInt64) throws -> GuestRAMView
}

/// Virtio device identity. Device ID 0x7E is unassigned in the virtio 1.3 registry; PCI class
/// 0x12/0x00 is "Processing accelerator". No guest driver binds to it — the guest never needs to.
public enum MetalPassthroughVirtioIdentity {
    public static let deviceID: UInt16 = 0x7E
    public static let pciClass: UInt8 = 0x12
    public static let pciSubclass: UInt8 = 0x00
}

@available(macOS 27.0, *)
public final class GuestMemoryDevice: NSObject, VZCustomVirtioDeviceConfigurationDelegate, VZCustomVirtioDeviceDelegate,
    GuestRAMProvider, @unchecked Sendable
{
    public let queue = DispatchQueue(label: "dev.metalpassthrough.virtio")
    private let lock = NSLock()
    private var device: VZCustomVirtioDevice?
    private var ram: (base: UInt64, mapping: VZGuestMemoryMapping)?

    public override init() { super.init() }

    public func makeConfiguration() -> VZCustomVirtioDeviceConfiguration {
        let c = VZCustomVirtioDeviceConfiguration()
        c.deviceID = MetalPassthroughVirtioIdentity.deviceID
        c.pciClassID = MetalPassthroughVirtioIdentity.pciClass
        c.pciSubclassID = MetalPassthroughVirtioIdentity.pciSubclass
        c.virtioQueueCount = 1
        c.provider = VZCustomVirtioDeviceDelegateProvider(deviceQueue: queue, delegate: self)
        return c
    }

    // MARK: VZCustomVirtioDeviceConfigurationDelegate

    public func customVirtioConfiguration(
        _ deviceConfiguration: VZCustomVirtioDeviceConfiguration, didCreateDevice device: VZCustomVirtioDevice
    ) {
        device.delegate = self
        lock.withLock { self.device = device }
    }

    // MARK: Guest RAM access

    /// Locates guest DRAM and returns a single mapping covering all of it (cached after first call).
    public func guestRAM(expectedSize: UInt64) throws -> (base: UInt64, mapping: VZGuestMemoryMapping) {
        if let ram = lock.withLock({ self.ram }) { return ram }
        guard let device = lock.withLock({ self.device }) else { throw GuestMemoryError.deviceNotCreated }
        let found: (UInt64, VZGuestMemoryMapping)? = queue.sync {
            let probeLen: Int = 2 << 20
            func probe(_ pa: UInt64, _ len: Int = probeLen) -> VZGuestMemoryMapping? {
                device.guestMemoryMapping(atPhysicalAddress: pa, length: len)
            }
            // 1. Coarse search in 256 MB steps over the first 1 TB of guest physical space.
            var hit: UInt64?
            var pa: UInt64 = 0
            while pa < (1 << 40) {
                if probe(pa) != nil { hit = pa; break }
                pa += 256 << 20
            }
            guard var start = hit else { return nil }
            // 2. Walk backwards in 2 MB steps to the true start of the RAM region.
            while start >= UInt64(probeLen), probe(start - UInt64(probeLen)) != nil { start -= UInt64(probeLen) }
            // 3. Find the largest length (2 MB granularity) that maps as one region.
            var lo: UInt64 = UInt64(probeLen)
            var hi: UInt64 = max(expectedSize, lo) + (1 << 30)
            while hi - lo > UInt64(probeLen) {
                let mid = ((lo + hi) / 2) & ~(UInt64(probeLen) - 1)
                if probe(start, Int(mid)) != nil { lo = mid } else { hi = mid }
            }
            guard let m = probe(start, Int(lo)) else { return nil }
            return (start, m)
        }
        guard let (base, mapping) = found else { throw GuestMemoryError.ramNotFound }
        lock.withLock { self.ram = (base, mapping) }
        return (base, mapping)
    }

    public func guestRAMView(expectedSize: UInt64) throws -> GuestRAMView {
        let r = try guestRAM(expectedSize: expectedSize)
        return GuestRAMView(base: r.base, bytes: r.mapping.mutableBytes, length: r.mapping.length)
    }

    /// Host virtual address of a guest physical address.
    public func hostPointer(guestPhysical pa: UInt64, length: Int, expectedRAM: UInt64) throws -> UnsafeMutableRawPointer {
        let ram = try guestRAM(expectedSize: expectedRAM)
        guard pa >= ram.base, pa + UInt64(length) <= ram.base + UInt64(ram.mapping.length) else {
            throw GuestMemoryError.invalidRange(pa, UInt64(length))
        }
        return ram.mapping.mutableBytes.advanced(by: Int(pa - ram.base))
    }
}

/// A contiguous host virtual view over a list of (possibly scattered) guest-physical chunks,
/// built with `mach_vm_remap(copy: false)`. The resulting pages alias guest RAM: no copies.
public final class RemappedGuestRegion: @unchecked Sendable {
    public let base: UnsafeMutableRawPointer
    public let length: Int

    /// `prefix` bytes of zero-filled host memory precede the guest chunks (used to place the
    /// first chunk at a 2 MiB-aligned GPU virtual address).
    public let prefix: Int

    public init(sources: [UnsafeMutableRawPointer], chunkSize: Int, prefix: Int = 0) throws {
        precondition(chunkSize % Int(getpagesize()) == 0, "chunk size must be a multiple of the host page size")
        precondition(prefix % Int(getpagesize()) == 0, "prefix must be page aligned")
        self.prefix = prefix
        let total = prefix + sources.count * chunkSize
        var addr: mach_vm_address_t = 0
        var kr = mach_vm_allocate(mach_task_self_, &addr, mach_vm_size_t(total), VM_FLAGS_ANYWHERE)
        guard kr == KERN_SUCCESS else { throw GuestMemoryError.machError("mach_vm_allocate", kr) }
        for (i, src) in sources.enumerated() {
            var target = addr + mach_vm_address_t(prefix + i * chunkSize)
            var cur: vm_prot_t = 0
            var maxp: vm_prot_t = 0
            kr = mach_vm_remap(
                mach_task_self_, &target, mach_vm_size_t(chunkSize), 0,
                VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
                mach_task_self_, mach_vm_address_t(UInt(bitPattern: src)), 0 /* copy = FALSE: share pages */,
                &cur, &maxp, VM_INHERIT_NONE)
            guard kr == KERN_SUCCESS else {
                mach_vm_deallocate(mach_task_self_, addr, mach_vm_size_t(total))
                throw GuestMemoryError.machError("mach_vm_remap", kr)
            }
        }
        self.base = UnsafeMutableRawPointer(bitPattern: UInt(addr))!
        self.length = total
    }

    deinit {
        mach_vm_deallocate(mach_task_self_, mach_vm_address_t(UInt(bitPattern: base)), mach_vm_size_t(length))
    }
}
