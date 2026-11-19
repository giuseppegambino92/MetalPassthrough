// Benchmark kernels shared by the native-host and the in-guest harnesses (identical source).
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant float& a [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) { y[i] = fma(a, x[i], y[i]); }

kernel void empty_kernel(device uint* p [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i == 0xffffffffu) p[0] = 1; }

kernel void stencil5(device const float* u [[buffer(0)]], device float* v [[buffer(1)]], constant uint& n [[buffer(2)]],
                     uint2 p [[thread_position_in_grid]]) {
    if (p.x == 0 || p.y == 0 || p.x >= n - 1 || p.y >= n - 1) return;
    uint i = p.y * n + p.x;
    v[i] = 0.2f * ((((u[i] + u[i - 1]) + u[i + 1]) + u[i - n]) + u[i + n]);
}

// 64x64 tile per threadgroup, 4 SIMD-groups x (4x4 simdgroup_float8x8). N % 64 == 0.
kernel void sgemm(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],
                  constant uint& N [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint r0 = tg.y * 64 + (sg / 2) * 32, c0 = tg.x * 64 + (sg % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = 0; k < N; k += 8) {
        simdgroup_float8x8 a[4], b[4];
        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], A + (r0 + i * 8) * N + k, N);
        for (uint j = 0; j < 4; j++) simdgroup_load(b[j], B + k * N + c0 + j * 8, N);
        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++) simdgroup_store(acc[i][j], C + (r0 + i * 8) * N + c0 + j * 8, N);
}
