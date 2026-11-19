// Native Metal check (no MetalPassthrough): do command buffers from two queues run concurrently?
//   swiftc -O bench/concurrency.swift -o /tmp/conc && /tmp/conc
// Two low-occupancy kernels (one 32-thread threadgroup each) on two queues, with their GPU
// timelines (gpuStartTime/gpuEndTime). On M3 Max / macOS 27.2 they overlap completely, also
// when both use one shared untracked buffer (MetalPassthrough's arena layout). An earlier version
// of this check wrote both kernels into one hazard-tracked buffer, which made Metal serialize them.
import Metal
import Foundation
let d = MTLCreateSystemDefaultDevice()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void spin(device float* out [[buffer(0)]], constant int& iters [[buffer(1)]], uint t [[thread_position_in_threadgroup]]) {
  float v = t * 1e-7f; for (int i = 0; i < iters; i++) v = fma(v, 0.9999993f, 1e-7f);
  if (t == 0) out[0] = v;
}
"""
let ps = try! d.makeComputePipelineState(function: try! d.makeLibrary(source: src, options: nil).makeFunction(name: "spin")!)
var iters: Int32 = 1 << 20
func buf(_ untracked: Bool) -> MTLBuffer { d.makeBuffer(length: 4096, options: untracked ? [.storageModeShared, .hazardTrackingModeUntracked] : .storageModeShared)! }
func cb(_ q: MTLCommandQueue, _ b: MTLBuffer) -> MTLCommandBuffer {
  let c = q.makeCommandBuffer()!; let e = c.makeComputeCommandEncoder()!
  e.setComputePipelineState(ps); e.setBuffer(b, offset: 0, index: 0); e.setBytes(&iters, length: 4, index: 1)
  e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)); e.endEncoding(); return c
}
let q1 = d.makeCommandQueue()!, q2 = d.makeCommandQueue()!
for untracked in [false, true] {
  let b1 = buf(untracked), b2 = buf(untracked)
  for _ in 0..<2 { let c = cb(q1, b1); c.commit(); c.waitUntilCompleted() }
  let s = cb(q1, b1); s.commit(); s.waitUntilCompleted()
  let single = (s.gpuEndTime - s.gpuStartTime) * 1e3
  let a = cb(q1, b1), b = cb(q2, b2)
  let t0 = Date(); a.commit(); b.commit(); a.waitUntilCompleted(); b.waitUntilCompleted()
  let wall = Date().timeIntervalSince(t0) * 1e3
  let base = min(a.gpuStartTime, b.gpuStartTime)
  print(String(format: "untracked=%d single %.2f ms | 2 queues wall %.2f ms | A [%.2f, %.2f] B [%.2f, %.2f] ms", untracked ? 1 : 0, single, wall,
    (a.gpuStartTime - base) * 1e3, (a.gpuEndTime - base) * 1e3, (b.gpuStartTime - base) * 1e3, (b.gpuEndTime - base) * 1e3))
}
// Same two kernels as two dispatches in ONE concurrent encoder.
let b1 = buf(true), b2 = buf(true)
let c = q1.makeCommandBuffer()!; let e = c.makeComputeCommandEncoder(dispatchType: .concurrent)!
e.setComputePipelineState(ps); e.setBytes(&iters, length: 4, index: 1)
e.setBuffer(b1, offset: 0, index: 0); e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
e.setBuffer(b2, offset: 0, index: 0); e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
e.endEncoding(); c.commit(); c.waitUntilCompleted()
print(String(format: "1 concurrent encoder, 2 dispatches: GPU %.2f ms", (c.gpuEndTime - c.gpuStartTime) * 1e3))
