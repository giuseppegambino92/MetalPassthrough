// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

import MPTProtocol
import Foundation
import IOKit
import Metal

public let containerGPUVersion = "0.1.0"

/// Host GPU facts published to every guest session (mpt_device_info).
public struct HostGPUInfo: Sendable, Codable {
    public let name: String
    public let gpuCores: UInt32
    public let metalFamily: String
    public let hostOS: String
    public let unifiedMemory: UInt64
    public let recommendedWorkingSet: UInt64
    public let maxBufferLength: UInt64
    public let maxThreadsPerThreadgroup: UInt32
    public let threadgroupMemory: UInt32
    public let simdWidth: UInt32
    public let hasUnifiedMemory: Bool
    public let measuredBandwidthGBs: Double

    public static func probe(device: MTLDevice) -> HostGPUInfo {
        let families: [(MTLGPUFamily, String)] = [
            (.apple10, "Apple10"), (.apple9, "Apple9"), (.apple8, "Apple8"), (.apple7, "Apple7"), (.apple6, "Apple6"),
        ]
        let family = families.first { device.supportsFamily($0.0) }?.1 ?? "Unknown"
        var memsize: UInt64 = 0
        var sz = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &memsize, &sz, nil, 0)
        let os = ProcessInfo.processInfo.operatingSystemVersion
        var build = [CChar](repeating: 0, count: 32)
        var bsz = build.count
        sysctlbyname("kern.osversion", &build, &bsz, nil, 0)
        let simd: UInt32 = {
            // threadExecutionWidth is a pipeline property; compile a trivial kernel once.
            let src = "kernel void k(device uint* o [[buffer(0)]]) { o[0] = 1; }"
            guard let lib = try? device.makeLibrary(source: src, options: nil),
                let fn = lib.makeFunction(name: "k"),
                let ps = try? device.makeComputePipelineState(function: fn)
            else { return 32 }
            return UInt32(ps.threadExecutionWidth)
        }()
        return HostGPUInfo(
            name: device.name,
            gpuCores: gpuCoreCount() ?? 0,
            metalFamily: family,
            hostOS: "macOS \(os.majorVersion).\(os.minorVersion).\(os.patchVersion) (\(String(decoding: build.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)))",
            unifiedMemory: memsize,
            recommendedWorkingSet: device.recommendedMaxWorkingSetSize,
            maxBufferLength: UInt64(device.maxBufferLength),
            maxThreadsPerThreadgroup: UInt32(device.maxThreadsPerThreadgroup.width),
            threadgroupMemory: UInt32(device.maxThreadgroupMemoryLength),
            simdWidth: simd,
            hasUnifiedMemory: device.hasUnifiedMemory,
            measuredBandwidthGBs: measureBandwidth(device: device)
        )
    }

    /// Reads `gpu-core-count` from the AGXAccelerator IORegistry entry.
    static func gpuCoreCount() -> UInt32? {
        let matching = IOServiceMatching("AGXAccelerator")
        var iter: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iter) == KERN_SUCCESS else { return nil }
        defer { IOObjectRelease(iter) }
        var svc = IOIteratorNext(iter)
        while svc != 0 {
            defer { IOObjectRelease(svc); svc = IOIteratorNext(iter) }
            if let v = IORegistryEntryCreateCFProperty(svc, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? NSNumber
            {
                return v.uint32Value
            }
        }
        return nil
    }

    /// Device-memory copy bandwidth (read + write bytes / GPU time) of a 256 MiB blit, best of 5.
    static func measureBandwidth(device: MTLDevice) -> Double {
        let n = 256 << 20
        guard let q = device.makeCommandQueue(),
            let a = device.makeBuffer(length: n, options: .storageModePrivate),
            let b = device.makeBuffer(length: n, options: .storageModePrivate)
        else { return 0 }
        var best = Double.infinity
        for _ in 0..<5 {
            guard let cb = q.makeCommandBuffer(), let bl = cb.makeBlitCommandEncoder() else { return 0 }
            bl.copy(from: a, sourceOffset: 0, to: b, destinationOffset: 0, size: n)
            bl.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            best = min(best, cb.gpuEndTime - cb.gpuStartTime)
        }
        return best > 0 ? Double(2 * n) / best / 1e9 : 0
    }

    func write(to p: UnsafeMutableRawPointer) {
        let info = p.assumingMemoryBound(to: mpt_device_info.self)
        memset(p, 0, MemoryLayout<mpt_device_info>.size)
        info.pointee.magic = MPT_MAGIC
        info.pointee.gpu_cores = gpuCores
        copyCString(name, into: &info.pointee.name)
        copyCString(metalFamily, into: &info.pointee.metal_family)
        copyCString(hostOS, into: &info.pointee.host_os)
        copyCString(containerGPUVersion, into: &info.pointee.daemon_version)
        info.pointee.unified_memory_bytes = unifiedMemory
        info.pointee.recommended_working_set = recommendedWorkingSet
        info.pointee.max_buffer_length = maxBufferLength
        info.pointee.max_threads_per_tg = maxThreadsPerThreadgroup
        info.pointee.threadgroup_memory = threadgroupMemory
        info.pointee.simd_width = simdWidth
        info.pointee.has_unified_memory = hasUnifiedMemory ? 1 : 0
        info.pointee.measured_bandwidth_gbs = measuredBandwidthGBs
        info.pointee.host_page_size = UInt32(getpagesize())
        info.pointee.abi_version = MPT_ABI_VERSION
    }
}

/// Copies a Swift string into a fixed-size imported C char array (NUL terminated, truncated).
func copyCString<T>(_ s: String, into field: inout T) {
    withUnsafeMutableBytes(of: &field) { raw in
        let bytes = Array(s.utf8.prefix(raw.count - 1))
        raw.copyBytes(from: bytes)
        raw[bytes.count] = 0
    }
}
