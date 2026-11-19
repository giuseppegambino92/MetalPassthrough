# Test 4: zero-copy NumPy / PyTorch acceleration inside a Linux container (MetalPassthrough).
# Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
import os
import sys
import time

import numpy as np

import metalpassthrough as mpt

failures = 0


def check(ok, what):
    global failures
    if not ok:
        failures += 1
        print(f"  FAIL: {what}")


def best(fn, reps=3):
    t = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


d = mpt.device()
print(f"device: {d!r}  (unified addressing: {mpt.unified_addressing()})")
print(f"python {sys.version.split()[0]}, numpy {np.__version__}, {os.cpu_count()} vCPUs in this Linux VM")

# ---------------------------------------------------------------- 1. zero-copy SAXPY, 256 MiB tensors
@mpt.metal_kernel
def saxpy():
    """
    kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      constant float& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        y[i] = fma(a, x[i], y[i]);
    }
    """


n = 64 << 20  # 64 Mi float32 = 256 MiB per tensor
x = mpt.empty(n, np.float32)
y = mpt.empty(n, np.float32)
x[:] = np.linspace(-4, 4, n, dtype=np.float32)
y[:] = 1.0
expect = (np.float32(2.0) * x.astype(np.float64) + 1.0).astype(np.float32)  # exact: 2*x is exact, +1 rounds once
saxpy(x, y, np.float32(2.0), grid=n)
gpu_s = mpt.synchronize()
exact = bool(np.array_equal(y, expect))
check(exact, "saxpy bit-exact")
y[:] = 1.0
cpu_s = best(lambda: np.add(np.multiply(np.float32(2.0), x), y), 3)
print(f"[1] SAXPY 2x256 MiB (zero-copy numpy arrays): GPU {gpu_s * 1e3:.2f} ms = {3 * n * 4 / gpu_s / 1e9:.0f} GB/s | "
      f"numpy CPU {cpu_s * 1e3:.1f} ms | speedup {cpu_s / gpu_s:.1f}x | bit-exact={exact}")

# ---------------------------------------------------------------- 2. fused transcendental kernel
@mpt.metal_kernel
def gelu_mix():
    """
    kernel void gelu_mix(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
        float v = x[i];
        float g = 0.5f * v * (1.0f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
        y[i] = g + precise::sin(v) * precise::exp(-v * v);
    }
    """


def gelu_mix_np(v):
    return 0.5 * v * (1.0 + np.tanh(0.7978845608 * (v + 0.044715 * v * v * v))) + np.sin(v) * np.exp(-v * v)


gelu_mix(x, y, grid=n)
gpu_s = mpt.synchronize()
sub = slice(0, n, 4099)
err = np.max(np.abs(y[sub].astype(np.float64) - gelu_mix_np(x[sub].astype(np.float64))))
check(err < 1e-5, f"gelu_mix accuracy {err}")
cpu_s = best(lambda: gelu_mix_np(x), 1)
print(f"[2] fused GELU+sin*exp over 64 Mi floats: GPU {gpu_s * 1e3:.2f} ms | numpy CPU {cpu_s * 1e3:.0f} ms | "
      f"speedup {cpu_s / gpu_s:.0f}x | max abs err vs float64 {err:.1e}")
del x, y

# ---------------------------------------------------------------- 3. SGEMM with simdgroup matrices
SGEMM = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
// C = A * B, row-major; M, N multiples of 64, K multiple of 8. 4 SIMD-groups per 64x64 tile.
kernel void sgemm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                  device float* C [[buffer(2)]], constant uint3& dims [[buffer(3)]],
                  uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint N = dims.y, K = dims.z;
    const uint r0 = tg.y * 64 + (sg / 2) * 32, c0 = tg.x * 64 + (sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < K; k += 8) {
        simdgroup_float8x8 a[4], b[4];
        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], A + (r0 + i * 8) * K + k, K);
        for (uint j = 0; j < 4; j++) simdgroup_load(b[j], B + k * N + c0 + j * 8, N);
        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], C + (r0 + i * 8) * N + c0 + j * 8, N);
}
"""
sgemm = mpt.Kernel(SGEMM, "sgemm")
M = N = K = 4096
rng = np.random.default_rng(0)
A = mpt.asarray(rng.standard_normal((M, K), dtype=np.float32))
B = mpt.asarray(rng.standard_normal((K, N), dtype=np.float32))
C = mpt.zeros((M, N), np.float32)
dims = np.array([M, N, K], dtype=np.uint32)
sgemm(A, B, C, dims, groups=(N // 64, M // 64), threads=128)  # warm-up (pipeline cached)
mpt.synchronize()
reps = 5
for _ in range(reps):
    sgemm(A, B, C, dims, groups=(N // 64, M // 64), threads=128)
gpu_s = mpt.synchronize() / reps
cpu_s = best(lambda: np.matmul(A, B), 1)
ref = np.matmul(A, B)
rel = np.max(np.abs(C - ref)) / np.max(np.abs(ref))
check(rel < 1e-4, f"sgemm relative error {rel}")
print(f"[3] SGEMM {M}x{N}x{K}: GPU {gpu_s * 1e3:.1f} ms = {2 * M * N * K / gpu_s / 1e12:.2f} TFLOP/s | numpy/OpenBLAS CPU "
      f"{cpu_s * 1e3:.0f} ms = {2 * M * N * K / cpu_s / 1e12:.2f} TFLOP/s | speedup {cpu_s / gpu_s:.0f}x | max rel err {rel:.1e}")

# ---------------------------------------------------------------- 3b. optimized GEMM (host MPS)
C2 = mpt.zeros((M, N), np.float32)
mpt.matmul(A, B, C2)
mpt.synchronize()
for _ in range(reps):
    mpt.matmul(A, B, C2)
mps_s = mpt.synchronize() / reps
rel2 = np.max(np.abs(C2 - ref)) / np.max(np.abs(ref))
check(rel2 < 1e-4, f"mpt.matmul relative error {rel2}")
print(f"[3b] mpt.matmul {M}x{N}x{K} (Metal Performance Shaders, zero-copy): {mps_s * 1e3:.1f} ms = "
      f"{2 * M * N * K / mps_s / 1e12:.2f} TFLOP/s | {cpu_s / mps_s:.0f}x numpy | max rel err {rel2:.1e}")

# ---------------------------------------------------------------- 4. PyTorch zero-copy interop
try:
    import torch

    tA, tB, tC = mpt.to_torch(A), mpt.to_torch(B), mpt.to_torch(C)
    shared = tC.data_ptr() == C.ctypes.data and tA.data_ptr() == mpt.device_address(A) if mpt.unified_addressing() else tC.data_ptr() == C.ctypes.data
    torch.set_num_threads(os.cpu_count())
    t_cpu = best(lambda: torch.matmul(tA, tB), 1)
    C[...] = 0
    sgemm(A, B, C, dims, groups=(N // 64, M // 64), threads=128)
    mpt.synchronize()
    # The GPU wrote C; the torch tensor sees it with no copy.
    tref = torch.matmul(tA, tB)
    trel = float((tC - tref).abs().max() / tref.abs().max())
    check(shared and trel < 1e-4, f"torch zero-copy (shared={shared}, err={trel})")
    print(f"[4] torch {torch.__version__}: from_dlpack views share arena pages (data_ptr == device address: {shared}); "
          f"torch CPU matmul {t_cpu * 1e3:.0f} ms vs MetalPassthrough {gpu_s * 1e3:.1f} ms ({t_cpu / gpu_s:.0f}x); max rel err {trel:.1e}")
except ImportError:
    print("[4] torch not installed; skipped")

# ---------------------------------------------------------------- 5. NVIDIA PTX from Python
ptx_path = os.path.join(os.path.dirname(__file__), "..", "cuda", "kernels.ptx")
vadd = mpt.PTXKernel(open(ptx_path).read(), "vector_add")
m = 1 << 24
a = mpt.asarray(np.arange(m, dtype=np.float32))
b = mpt.asarray(np.full(m, 0.5, dtype=np.float32))
c = mpt.zeros(m, np.float32)
vadd(a, b, c, np.int32(m), grid=((m + 255) // 256,), block=(256,))
t = mpt.synchronize()
check(np.array_equal(c, a + b), "ptx vector_add")
print(f"[5] nvcc PTX vector_add via mpt.PTXKernel: {t * 1e3:.2f} ms, {3 * m * 4 / t / 1e9:.0f} GB/s, exact={np.array_equal(c, a + b)}")

# ---------------------------------------------------------------- 6. Triton-generated PTX
tdir = os.path.join(os.path.dirname(__file__), "..", "triton")
tadd = mpt.PTXKernel(open(os.path.join(tdir, "add_kernel.ptx")).read(), "add_kernel")
tsm = mpt.PTXKernel(open(os.path.join(tdir, "softmax_kernel.ptx")).read(), "softmax_kernel")
c[...] = 0
tadd(a, b, c, np.int32(m), 0, 0, grid=((m + 1023) // 1024,), block=(128,))  # + Triton's two scratch pointers
t = mpt.synchronize()
check(np.array_equal(c, a + b), "triton add_kernel")
rows, cols = 4096, 1000
logits = mpt.asarray(np.random.default_rng(1).standard_normal((rows, cols), dtype=np.float32))
probs = mpt.zeros_like(logits)
tsm(probs, logits, np.int32(cols), np.int32(cols), 0, 0, grid=(rows,), block=(128,), shared_mem=16)
ts = mpt.synchronize()
ex = np.exp(logits - logits.max(axis=1, keepdims=True))
serr = float(np.max(np.abs(probs - ex / ex.sum(axis=1, keepdims=True))))
check(serr < 1e-6, f"triton softmax error {serr}")
print(f"[6] Triton 3.5 PTX (compiled on a machine without an NVIDIA GPU): add {t * 1e3:.2f} ms exact={np.array_equal(c, a + b)}, "
      f"softmax {rows}x{cols} {ts * 1e3:.2f} ms max err {serr:.1e}")

s = mpt.stats()
print(f"stats: {s.dispatches} dispatches, {s.command_buffers} command buffers, compiles={s.pipeline_compiles}, "
      f"cache hits={s.pipeline_cache_hits}, arena={s.arena_bytes >> 20} MiB in {s.arenas} arena(s); ring ping {mpt.ping():.2f} us")
print("test4: PASS" if failures == 0 else f"test4: FAILED ({failures})")
sys.exit(1 if failures else 0)
