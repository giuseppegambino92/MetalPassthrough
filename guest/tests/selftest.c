/*
 * mpt-selftest: end-to-end checks of libmetalpassthrough from inside the Linux container.
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */
#include "metalpassthrough.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, " (%s)\n", mpt_last_error()); return 1; } } while (0)

static const char *SAXPY =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],\n"
    "                  constant float& a [[buffer(2)]], constant uint& n [[buffer(3)]],\n"
    "                  uint i [[thread_position_in_grid]]) {\n"
    "    if (i < n) y[i] = fma(a, x[i], y[i]);\n"
    "}\n";

int main(void) {
    mpt_ctx *c = mpt_default();
    CHECK(c, "mpt_default");
    const mpt_device_info *d = mpt_device(c);
    printf("device        : %s (%u-core GPU, %s), host %s\n", d->name, d->gpu_cores, d->metal_family, d->host_os);
    printf("session       : %u, unified addressing (guest VA == device VA): %s\n", mpt_session_id(c),
           mpt_unified_addressing(c) ? "yes" : "no");

    /* Ring round trip. */
    enum { NP = 2000 };
    static uint64_t rt[NP];
    for (int i = 0; i < NP; i++) CHECK(mpt_ping(c, &rt[i]) == 0, "ping");
    qsort(rt, NP, sizeof rt[0], cmp_u64);
    printf("ring ping     : min %.2f us, median %.2f us, p99 %.2f us\n", rt[0] / 1e3, rt[NP / 2] / 1e3, rt[NP * 99 / 100] / 1e3);

    /* Allocator latency. */
    enum { NA = 100000 };
    static void *ptrs[256];
    uint64_t t0 = now_ns();
    for (int i = 0; i < NA; i++) {
        void *p = mpt_alloc(c, 4096 + (i & 7) * 256, 256);
        CHECK(p, "alloc");
        ptrs[i & 255] = p;
        if ((i & 255) == 255) for (int k = 0; k < 256; k++) mpt_free(c, ptrs[k]);
    }
    printf("mpt_alloc    : %.0f ns per alloc+free (guest-local, no host round trip)\n", (double)(now_ns() - t0) / NA);

    /* SAXPY on arena memory. */
    uint32_t n = 1u << 24;
    float *x = mpt_alloc(c, n * sizeof(float), 0), *y = mpt_alloc(c, n * sizeof(float), 0);
    CHECK(x && y, "alloc 2x64MiB");
    if (mpt_unified_addressing(c)) CHECK(mpt_devaddr(c, x) == (uint64_t)(uintptr_t)x, "UVA mismatch");
    float *ref = malloc(n * sizeof(float));
    for (uint32_t i = 0; i < n; i++) { x[i] = (float)(i % 1000) * 0.001f; y[i] = (float)(i % 7); }
    float a = 2.5f;
    for (uint32_t i = 0; i < n; i++) ref[i] = fmaf(a, x[i], y[i]);
    mpt_pipeline p;
    char log[4096];
    CHECK(mpt_compile_msl(c, SAXPY, "saxpy", 0, &p, log, sizeof log) == 0, "compile: %s", log);
    mpt_arg args[] = {
        { MPT_BIND_BUFFER, 0, x, 0 }, { MPT_BIND_BUFFER, 1, y, 0 },
        { MPT_BIND_BYTES, 2, &a, sizeof a }, { MPT_BIND_BYTES, 3, &n, sizeof n },
    };
    uint32_t grid[3] = { n, 1, 1 }, tg[3] = { 256, 1, 1 };
    CHECK(mpt_dispatch(c, &p, grid, tg, MPT_DISPATCH_THREADS, args, 4, 0) == 0, "dispatch");
    uint64_t gpu_ns = 0;
    CHECK(mpt_sync(c, &gpu_ns) == 0, "sync");
    size_t bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += memcmp(&y[i], &ref[i], 4) != 0;
    printf("saxpy 16M     : GPU %.1f us, %.1f GB/s effective, %zu mismatches (bit-exact vs fmaf)\n", gpu_ns / 1e3,
           3.0 * n * 4 / (double)gpu_ns, bad);
    CHECK(bad == 0, "saxpy mismatch");

    mpt_stats st;
    CHECK(mpt_get_stats(c, &st) == 0, "stats");
    printf("stats         : %llu dispatches, %llu command buffers, %llu compiles, %llu cache hits, arena %llu MiB\n",
           (unsigned long long)st.dispatches, (unsigned long long)st.command_buffers, (unsigned long long)st.pipeline_compiles,
           (unsigned long long)st.pipeline_cache_hits, (unsigned long long)(st.arena_bytes >> 20));
    printf("selftest: PASS\n");
    return 0;
}
