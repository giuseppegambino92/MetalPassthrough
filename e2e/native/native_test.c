/*
 * Test 2: zero-copy native Metal compute from Linux. 256 MiB float32 tensors live in guest
 * hugepages (mmap'd by libmetalpassthrough); the M3 Max GPU computes on them in place. Results are
 * verified bit-exact against CPU references computed in the guest.
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */
#include "metalpassthrough.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, " [%s]\n", mpt_last_error()); exit(1); } } while (0)

static const char *SRC =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant float& a [[buffer(2)]],\n"
    "                  uint i [[thread_position_in_grid]]) { y[i] = fma(a, x[i], y[i]); }\n"
    "kernel void stencil5(device const float* u [[buffer(0)]], device float* v [[buffer(1)]], constant uint2& n [[buffer(2)]],\n"
    "                     uint2 p [[thread_position_in_grid]]) {\n"
    "  if (p.x == 0 || p.y == 0 || p.x >= n.x - 1 || p.y >= n.y - 1) { if (p.x < n.x && p.y < n.y) v[p.y * n.x + p.x] = u[p.y * n.x + p.x]; return; }\n"
    "  uint i = p.y * n.x + p.x;\n"
    "  v[i] = 0.2f * ((((u[i] + u[i - 1]) + u[i + 1]) + u[i - n.x]) + u[i + n.x]);\n"
    "}\n"
    "kernel void matmul(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],\n"
    "                   constant uint& n [[buffer(3)]], threadgroup float* ta [[threadgroup(0)]], threadgroup float* tb [[threadgroup(1)]],\n"
    "                   uint2 g [[thread_position_in_grid]], uint2 l [[thread_position_in_threadgroup]]) {\n"
    "  float acc = 0.0f;\n"
    "  for (uint t = 0; t < n; t += 16) {\n"
    "    ta[l.y * 16 + l.x] = A[g.y * n + t + l.x];\n"
    "    tb[l.y * 16 + l.x] = B[(t + l.y) * n + g.x];\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "    for (uint k = 0; k < 16; k++) acc = fma(ta[l.y * 16 + k], tb[k * 16 + l.x], acc);\n"
    "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  }\n"
    "  C[g.y * n + g.x] = acc;\n"
    "}\n";

int main(void) {
    mpt_ctx *c = mpt_default();
    CHECK(c, "open");
    const mpt_device_info *d = mpt_device(c);
    printf("Test 2 on %s (%u-core GPU) from Linux, UVA=%d\n", d->name, d->gpu_cores, mpt_unified_addressing(c));
    mpt_pipeline ps, pst, pmm;
    char log[8192];
    CHECK(mpt_compile_msl(c, SRC, "saxpy", 0, &ps, log, sizeof log) == 0, "saxpy %s", log);
    CHECK(mpt_compile_msl(c, SRC, "stencil5", 0, &pst, log, sizeof log) == 0, "stencil %s", log);
    CHECK(mpt_compile_msl(c, SRC, "matmul", 0, &pmm, log, sizeof log) == 0, "matmul %s", log);

    /* ---- SAXPY on 2 x 256 MiB ---- */
    const uint32_t n = 64u << 20;
    float *x = mpt_alloc(c, (size_t)n * 4, 0), *y = mpt_alloc(c, (size_t)n * 4, 0);
    CHECK(x && y, "alloc 2x256 MiB");
    for (uint32_t i = 0; i < n; i++) { x[i] = (float)(i & 0xffff) * 1.0e-4f - 3.0f; y[i] = (float)(i % 977) * 0.5f; }
    float a = 1.61803398875f;
    mpt_arg sa[] = { { MPT_BIND_BUFFER, 0, x, 0 }, { MPT_BIND_BUFFER, 1, y, 0 }, { MPT_BIND_BYTES, 2, &a, 4 } };
    uint32_t g1[3] = { n, 1, 1 }, t1[3] = { 256, 1, 1 };
    uint64_t ns;
    double w0 = now();
    CHECK(mpt_dispatch(c, &ps, g1, t1, MPT_DISPATCH_THREADS, sa, 3, 0) == 0, "dispatch");
    CHECK(mpt_sync(c, &ns) == 0, "sync");
    double wall = now() - w0;
    size_t bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        float want = fmaf(a, x[i], (float)(i % 977) * 0.5f);
        bad += memcmp(&want, &y[i], 4) != 0;
    }
    printf("SAXPY      2x256 MiB : GPU %.2f ms (%.0f GB/s effective), submit->done %.2f ms, bit-exact mismatches %zu\n",
           ns / 1e6, 3.0 * n * 4 / ns, wall * 1e3, bad);
    fails += bad != 0;

    /* ---- 2D stencil, 8192x8192 (256 MiB), 20 Jacobi sweeps batched into command buffers ---- */
    uint32_t nn[2] = { 8192, 8192 };
    size_t cells = (size_t)nn[0] * nn[1];
    float *u = x, *v = y;  /* reuse the two 256 MiB tensors */
    float *ref = malloc(cells * 4), *ref2 = malloc(cells * 4);
    for (size_t i = 0; i < cells; i++) u[i] = ref[i] = (float)((i * 2654435761u) >> 20 & 1023) * (1.0f / 1024);
    const int sweeps = 20;
    uint32_t g2[3] = { nn[0], nn[1], 1 }, t2[3] = { 32, 8, 1 };
    w0 = now();
    for (int s = 0; s < sweeps; s++) {
        float *src = (s & 1) ? v : u, *dst = (s & 1) ? u : v;
        mpt_arg st[] = { { MPT_BIND_BUFFER, 0, src, 0 }, { MPT_BIND_BUFFER, 1, dst, 0 }, { MPT_BIND_BYTES, 2, nn, 8 } };
        CHECK(mpt_dispatch(c, &pst, g2, t2, MPT_DISPATCH_THREADS, st, 3, 0) == 0, "stencil dispatch");
    }
    CHECK(mpt_sync(c, &ns) == 0, "sync");
    wall = now() - w0;
    for (int s = 0; s < sweeps; s++) {
        for (uint32_t yy = 0; yy < nn[1]; yy++)
            for (uint32_t xx = 0; xx < nn[0]; xx++) {
                size_t i = (size_t)yy * nn[0] + xx;
                if (!xx || !yy || xx == nn[0] - 1 || yy == nn[1] - 1) { ref2[i] = ref[i]; continue; }
                ref2[i] = 0.2f * ((((ref[i] + ref[i - 1]) + ref[i + 1]) + ref[i - nn[0]]) + ref[i + nn[0]]);
            }
        float *t = ref; ref = ref2; ref2 = t;
    }
    bad = memcmp(ref, u, cells * 4) ? 0 : 0;
    for (size_t i = 0; i < cells; i++) bad += memcmp(&ref[i], &u[i], 4) != 0;
    printf("Stencil5   8192^2 x%d: GPU %.2f ms total (%.0f GB/s), %.2f ms/sweep, wall %.1f ms, bit-exact mismatches %zu\n", sweeps,
           ns / 1e6, (double)sweeps * 2 * cells * 4 / ns, ns / 1e6 / sweeps, wall * 1e3, bad);
    fails += bad != 0;
    free(ref); free(ref2);

    /* ---- matmul 2048^3 with threadgroup tiles ---- */
    const uint32_t m = 2048;
    float *A = x, *B = x + (size_t)m * m, *C = y;
    for (size_t i = 0; i < (size_t)m * m; i++) { A[i] = (float)((i * 7) % 13) * 0.25f - 1.5f; B[i] = (float)((i * 11) % 17) * 0.125f - 1.0f; }
    mpt_arg ma[] = { { MPT_BIND_BUFFER, 0, A, 0 }, { MPT_BIND_BUFFER, 1, B, 0 }, { MPT_BIND_BUFFER, 2, C, 0 },
                      { MPT_BIND_BYTES, 3, &m, 4 }, { MPT_BIND_TGMEM, 0, NULL, 1024 }, { MPT_BIND_TGMEM, 1, NULL, 1024 } };
    uint32_t g3[3] = { m, m, 1 }, t3[3] = { 16, 16, 1 };
    CHECK(mpt_dispatch(c, &pmm, g3, t3, MPT_DISPATCH_THREADS, ma, 6, 0) == 0, "matmul");
    CHECK(mpt_sync(c, &ns) == 0, "sync");
    bad = 0;
    for (uint32_t i = 0; i < m; i += 61)
        for (uint32_t j = 0; j < m; j++) {
            float acc = 0;
            for (uint32_t k = 0; k < m; k++) acc = fmaf(A[(size_t)i * m + k], B[(size_t)k * m + j], acc);
            bad += memcmp(&acc, &C[(size_t)i * m + j], 4) != 0;
        }
    printf("Matmul     2048^3    : GPU %.2f ms (%.0f GFLOP/s), bit-exact mismatches %zu (34 sampled rows x 2048)\n", ns / 1e6,
           2.0 * m * m * m / ns, bad);
    fails += bad != 0;

    printf(fails ? "native_test: FAILED\n" : "native_test: PASS\n");
    return fails != 0;
}
