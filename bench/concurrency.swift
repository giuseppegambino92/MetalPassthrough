// Native Metal check (no MetalPassthrough): do command buffers from two queues overlap?
//   swiftc -O bench/concurrency.swift -o /tmp/conc && /tmp/conc

import Metal
import Foundation
let d = MTLCreateSystemDefaultDevice()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void spin(device float* out [[buffer(0)]], constant int& iters [[buffer(1)]], uint t [[thread_position_in_threadgroup]]) {
  float v = t * 1e-7f; for (int i = 0; i < iters; i++) v = fma(v, 0.9999993f, 1e-7f);
  threadgroup_barrier(mem_flags::mem_threadgroup); if (t == 0) out[0] = v;
}
"""
let ps = try! d.makeComputePipelineState(function: try! d.makeLibrary(source: src, options: nil).makeFunction(name: "spin")!)
let buf = d.makeBuffer(length: 4096, options: .storageModeShared)!
var iters: Int32 = 1 << 20
let q1 = d.makeCommandQueue()!, q2 = d.makeCommandQueue()!
func cb(_ q: MTLCommandQueue, _ off: Int, groups: Int = 1) -> MTLCommandBuffer {
  let c = q.makeCommandBuffer()!; let e = c.makeComputeCommandEncoder()!
  e.setComputePipelineState(ps); e.setBuffer(buf, offset: off, index: 0); e.setBytes(&iters, length: 4, index: 1)
  e.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)); e.endEncoding(); return c
}
for mode in ["serial-1q", "2 queues", "1 CB 2 groups"] {
  _ = { let c = cb(q1, 0); c.commit(); c.waitUntilCompleted() }()
  let t0 = Date()
  switch mode {
  case "serial-1q": let a = cb(q1, 0); a.commit(); a.waitUntilCompleted()
  case "2 queues": let a = cb(q1, 0), b = cb(q2, 1024); a.commit(); b.commit(); a.waitUntilCompleted(); b.waitUntilCompleted()
  default: let a = cb(q1, 0, groups: 2); a.commit(); a.waitUntilCompleted()
  }
  print(mode, String(format: "%.1f ms", Date().timeIntervalSince(t0) * 1e3))
}
