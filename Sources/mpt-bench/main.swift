// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
//
// Native-host baseline for bench/run.sh: the same kernels (bench/kernels.metal), sizes and
// methodology as the in-guest harness (bench/bench_guest.c), on ordinary MTLBuffers.
// Output: CSV rows  env,bench,size,gpu_ms,wall_ms,value,unit  (median of `reps` runs).

import Foundation
import Metal
import MetalPerformanceShaders

let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: mpt-bench <kernels.metal> [reps]"); exit(2) }
let reps = args.count > 2 ? Int(args[2])! : 7
let d = MTLCreateSystemDefaultDevice()!
let q = d.makeCommandQueue()!
let opts = MTLCompileOptions()
opts.mathMode = .safe
let lib = try d.makeLibrary(source: String(contentsOfFile: args[1], encoding: .utf8), options: opts)
func pso(_ n: String) -> MTLComputePipelineState { try! d.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
let saxpy = pso("saxpy"), empty = pso("empty_kernel"), stencil = pso("stencil5"), sgemm = pso("sgemm")

func now() -> Double { Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW)) / 1e9 }
func median(_ v: [Double]) -> Double { let s = v.sorted(); return s[s.count / 2] }
func row(_ bench: String, _ size: String, _ gpu: Double, _ wall: Double, _ value: Double, _ unit: String) {
    print(String(format: "native,%@,%@,%.4f,%.4f,%.3f,%@", bench, size, gpu * 1e3, wall * 1e3, value, unit))
}

/// Runs `encode` in one command buffer `reps` times; returns median (gpu s, wall s).
func measure(_ encode: (MTLCommandBuffer) -> Void) -> (Double, Double) {
    var g: [Double] = [], w: [Double] = []
    for i in 0...reps {
        let t0 = now()
        let cb = q.makeCommandBuffer()!
        encode(cb)
        cb.commit()
        cb.waitUntilCompleted()
        let t1 = now()
        if i == 0 { continue }  // warm-up
        g.append(cb.gpuEndTime - cb.gpuStartTime)
        w.append(t1 - t0)
    }
    return (median(g), median(w))
}

let big = 1 << 26
let X = d.makeBuffer(length: big * 4, options: .storageModeShared)!
let Y = d.makeBuffer(length: big * 4, options: .storageModeShared)!
let Z = d.makeBuffer(length: big * 4, options: .storageModeShared)!
memset(X.contents(), 0, big * 4); memset(Y.contents(), 0, big * 4); memset(Z.contents(), 0, big * 4)

for lg in [18, 20, 22, 24, 26] {
    let n = 1 << lg
    var a: Float = 2
    let (g, w) = measure { cb in
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(saxpy); e.setBuffer(X, offset: 0, index: 0); e.setBuffer(Y, offset: 0, index: 1)
        e.setBytes(&a, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        e.endEncoding()
    }
    let label = n * 4 >= (1 << 20) ? "\((n * 4) >> 20)MiB" : "\((n * 4) >> 10)KiB"  // Swift: >> binds tighter than *
    row("saxpy", label, g, w, 12.0 * Double(n) / g / 1e9, "GB/s")
}
for n in [1024, 2048, 4096, 8192] {
    var nn = UInt32(n)
    let (g, w) = measure { cb in
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(stencil); e.setBuffer(X, offset: 0, index: 0); e.setBuffer(Y, offset: 0, index: 1)
        e.setBytes(&nn, length: 4, index: 2)
        e.dispatchThreads(MTLSize(width: n, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
        e.endEncoding()
    }
    row("stencil5", "\(n)^2", g, w, 8.0 * Double(n * n) / g / 1e9, "GB/s")
}
for n in [512, 1024, 2048, 4096] {
    var nn = UInt32(n)
    let (g, w) = measure { cb in
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(sgemm); e.setBuffer(X, offset: 0, index: 0); e.setBuffer(Y, offset: 0, index: 1)
        e.setBuffer(Z, offset: 0, index: 2); e.setBytes(&nn, length: 4, index: 3)
        e.dispatchThreadgroups(MTLSize(width: n / 64, height: n / 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        e.endEncoding()
    }
    row("sgemm_simd", "\(n)^3", g, w, 2.0 * pow(Double(n), 3) / g / 1e12, "TFLOP/s")
}
for n in [512, 1024, 2048, 4096] {
    let mm = MPSMatrixMultiplication(device: d, transposeLeft: false, transposeRight: false, resultRows: n, resultColumns: n,
                                     interiorColumns: n, alpha: 1, beta: 0)
    let desc = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: n * 4, dataType: .float32)
    let (g, w) = measure { cb in
        mm.encode(commandBuffer: cb, leftMatrix: MPSMatrix(buffer: X, descriptor: desc), rightMatrix: MPSMatrix(buffer: Y, descriptor: desc),
                  resultMatrix: MPSMatrix(buffer: Z, descriptor: desc))
    }
    row("sgemm_mps", "\(n)^3", g, w, 2.0 * pow(Double(n), 3) / g / 1e12, "TFLOP/s")
}
// Latency of one tiny dispatch, submit -> completion observed by the CPU.
do {
    var w: [Double] = []
    for _ in 0..<200 {
        let t0 = now()
        let cb = q.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(empty); e.setBuffer(X, offset: 0, index: 0)
        e.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        w.append(now() - t0)
    }
    row("dispatch_roundtrip", "1x32thr", 0, median(w), median(w) * 1e6, "us")
}
// Throughput of many small dispatches batched into one command buffer.
do {
    let n = 10_000
    var w: [Double] = []
    for _ in 0..<3 {
        let t0 = now()
        let cb = q.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder(dispatchType: .serial)!
        e.setComputePipelineState(empty); e.setBuffer(X, offset: 0, index: 0)
        for _ in 0..<n {
            e.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        }
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        w.append(now() - t0)
    }
    row("dispatch_throughput", "10000x32thr", 0, median(w), Double(n) / median(w), "dispatch/s")
}
