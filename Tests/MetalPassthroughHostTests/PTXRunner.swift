// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
// Host-side harness: runs translated PTX on the local Apple GPU with ordinary MTLBuffers, using
// exactly the binding model the guest libcuda uses (packed params at buffer(0), gpuAddress pointers).

import MetalPassthroughHost
import Foundation
import Metal

final class PTXRunner {
    let device = MTLCreateSystemDefaultDevice()!
    lazy var queue = device.makeCommandQueue()!
    lazy var cache = PipelineCache(device: device)
    var buffers: [MTLBuffer] = []

    func buffer<T>(_ values: [T]) -> MTLBuffer {
        let b = values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 16), options: .storageModeShared)! }
        buffers.append(b)
        return b
    }
    func zeros(_ bytes: Int) -> MTLBuffer {
        let b = device.makeBuffer(length: max(bytes, 16), options: .storageModeShared)!
        memset(b.contents(), 0, b.length)
        buffers.append(b)
        return b
    }
    func read<T>(_ b: MTLBuffer, _ n: Int, as: T.Type) -> [T] {
        Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: T.self, capacity: n), count: n))
    }

    enum Arg { case ptr(MTLBuffer), i32(Int32), u32(UInt32), f32(Float), f64(Double), u64(UInt64) }

    /// Returns GPU time in ns.
    @discardableResult
    func launch(_ k: TranslatedKernel, grid: (Int, Int, Int), block: (Int, Int, Int), dynamicShared: Int = 0, _ args: [Arg]) throws -> UInt64 {
        precondition(args.count == k.params.count, "arg count \(args.count) != \(k.params.count)")
        let ps = try cache.pipeline(source: k.msl, function: k.mslName, fastMath: false).state
        var block_ = [UInt8](repeating: 0, count: max(Int(k.paramBytes), 16))
        for (a, p) in zip(args, k.params) {
            func put<T>(_ v: T) { withUnsafeBytes(of: v) { block_.replaceSubrange(Int(p.offset)..<Int(p.offset) + $0.count, with: $0) } }
            switch a {
            case .ptr(let b): put(b.gpuAddress)
            case .i32(let v): put(v)
            case .u32(let v): put(v)
            case .f32(let v): put(v)
            case .f64(let v): put(v)
            case .u64(let v): put(v)
            }
        }
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(ps)
        e.setBytes(block_, length: block_.count, index: 0)
        let shared = Int(k.staticSharedBytes) + dynamicShared
        if k.usesSharedMemory { e.setThreadgroupMemoryLength(max(16, (shared + 15) & ~15), index: 0) }
        for b in buffers { e.useResource(b, usage: [.read, .write]) }
        e.dispatchThreadgroups(MTLSize(width: grid.0, height: grid.1, depth: grid.2),
                               threadsPerThreadgroup: MTLSize(width: block.0, height: block.1, depth: block.2))
        e.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw err }
        return UInt64((cb.gpuEndTime - cb.gpuStartTime) * 1e9)
    }
}
