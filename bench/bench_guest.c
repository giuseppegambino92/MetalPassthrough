/*
 * In-guest benchmark harness: same kernels (bench/kernels.metal), sizes and methodology as the
 * native-host baseline (Sources/mpt-bench). GPU time = command-buffer GPU timestamps reported by
 * the host; wall time = from the API call in the Linux guest until the result is visible to it.
 * Output: CSV rows  env,bench,size,gpu_ms,wall_ms,value,unit  (median of `reps`).
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */
#define _GNU_SOURCE
#include "metalpassthrough.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double median(double *v, int n) { qsort(v, n, sizeof *v, cmpd); return v[n / 2]; }
static void row(const char *b, const char *size, double gpu_s, double wall_s, double value, const char *unit) {
    printf("guest,%s,%s,%.4f,%.4f,%.3f,%s\n", b, size, gpu_s * 1e3, wall_s * 1e3, value, unit);
}

static mpt_ctx *c;
static int reps = 7;

/* Runs one dispatch + sync `reps` times (after a warm-up); returns medians. */
static void measure(const mpt_pipeline *p, const uint32_t g[3], const uint32_t t[3], uint32_t mode, mpt_arg *a, uint32_t na,
                    double *gpu, double *wall) {
    double gv[64], wv[64];
    for (int i = 0; i <= reps; i++) {
        uint64_t ns = 0;
        double t0 = now();
        mpt_dispatch(c, p, g, t, mode, a, na, MPT_F_COMMIT);
        mpt_sync(c, &ns);
        double t1 = now();
        if (i) { gv[i - 1] = ns * 1e-9; wv[i - 1] = t1 - t0; }
    }
    *gpu = median(gv, reps);
    *wall = median(wv, reps);
}

static void measure_gemm(int n, float *X, float *Y, float *Z, double *gpu, double *wall) {
    double gv[64], wv[64];
    mpt_gemm_desc d = { .dtype = MPT_GEMM_F32, .m = n, .n = n, .k = n, .lda = n, .ldb = n, .ldc = n, .batch = 1, .alpha = 1, .beta = 0,
                        .a = mpt_devaddr(c, X), .b = mpt_devaddr(c, Y), .c = mpt_devaddr(c, Z) };
    for (int i = 0; i <= reps; i++) {
        uint64_t ns = 0;
        double t0 = now();
        mpt_gemm_on(c, 0, &d);
        mpt_sync(c, &ns);
        double t1 = now();
        if (i) { gv[i - 1] = ns * 1e-9; wv[i - 1] = t1 - t0; }
    }
    *gpu = median(gv, reps);
    *wall = median(wv, reps);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_guest <kernels.metal> [reps]\n"); return 2; }
    if (argc > 2) reps = atoi(argv[2]);
    if (reps > 63) reps = 63;
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 2; }
    static char src[1 << 16];
    src[fread(src, 1, sizeof src - 1, f)] = 0;
    fclose(f);

    /* ---- session / memory costs (fresh sessions of growing arena size) ---- */
    for (size_t mib = 64; mib <= 1024; mib *= 4) {
        double wv[5];
        for (int i = 0; i < 5; i++) {
            mpt_ctx *t;
            double t0 = now();
            if (mpt_open(&t, mib << 20)) { fprintf(stderr, "open %zu MiB: %s\n", mib, mpt_last_error()); return 1; }
            wv[i] = now() - t0;
            mpt_close(t);
        }
        char sz[32];
        snprintf(sz, sizeof sz, "%zuMiB", mib);
        row("session_open", sz, 0, median(wv, 5), median(wv, 5) * 1e3, "ms");
    }

    if (mpt_open(&c, 1100u << 20)) { fprintf(stderr, "%s\n", mpt_last_error()); return 1; }
    char log[8192];
    mpt_pipeline saxpy, empty, stencil, sgemm;
    if (mpt_compile_msl(c, src, "saxpy", 0, &saxpy, log, sizeof log) || mpt_compile_msl(c, src, "empty_kernel", 0, &empty, log, sizeof log) ||
        mpt_compile_msl(c, src, "stencil5", 0, &stencil, log, sizeof log) || mpt_compile_msl(c, src, "sgemm", 0, &sgemm, log, sizeof log)) {
        fprintf(stderr, "%s\n", log);
        return 1;
    }
    const size_t big = (size_t)1 << 26;
    float *X = mpt_alloc(c, big * 4, 0), *Y = mpt_alloc(c, big * 4, 0), *Z = mpt_alloc(c, big * 4, 0);
    if (!X || !Y || !Z) { fprintf(stderr, "alloc: %s\n", mpt_last_error()); return 1; }
    memset(X, 0, big * 4); memset(Y, 0, big * 4); memset(Z, 0, big * 4);
    double g, w;

    for (int lg = 18; lg <= 26; lg += 2) {
        uint32_t n = 1u << lg;
        float a = 2;
        mpt_arg args[] = { { MPT_BIND_BUFFER, 0, X, 0 }, { MPT_BIND_BUFFER, 1, Y, 0 }, { MPT_BIND_BYTES, 2, &a, 4 } };
        uint32_t grid[3] = { n, 1, 1 }, tg[3] = { 256, 1, 1 };
        measure(&saxpy, grid, tg, MPT_DISPATCH_THREADS, args, 3, &g, &w);
        char sz[32];
        if ((size_t)n * 4 >= (1u << 20)) snprintf(sz, sizeof sz, "%zuMiB", (size_t)n * 4 >> 20); else snprintf(sz, sizeof sz, "%zuKiB", (size_t)n * 4 >> 10);
        row("saxpy", sz, g, w, 12.0 * n / g / 1e9, "GB/s");
    }
    for (uint32_t n = 1024; n <= 8192; n *= 2) {
        mpt_arg args[] = { { MPT_BIND_BUFFER, 0, X, 0 }, { MPT_BIND_BUFFER, 1, Y, 0 }, { MPT_BIND_BYTES, 2, &n, 4 } };
        uint32_t grid[3] = { n, n, 1 }, tg[3] = { 32, 8, 1 };
        measure(&stencil, grid, tg, MPT_DISPATCH_THREADS, args, 3, &g, &w);
        char sz[32];
        snprintf(sz, sizeof sz, "%u^2", n);
        row("stencil5", sz, g, w, 8.0 * n * n / g / 1e9, "GB/s");
    }
    for (uint32_t n = 512; n <= 4096; n *= 2) {
        mpt_arg args[] = { { MPT_BIND_BUFFER, 0, X, 0 }, { MPT_BIND_BUFFER, 1, Y, 0 }, { MPT_BIND_BUFFER, 2, Z, 0 }, { MPT_BIND_BYTES, 3, &n, 4 } };
        uint32_t grid[3] = { n / 64, n / 64, 1 }, tg[3] = { 128, 1, 1 };
        measure(&sgemm, grid, tg, MPT_DISPATCH_THREADGROUPS, args, 4, &g, &w);
        char sz[32];
        snprintf(sz, sizeof sz, "%u^3", n);
        row("sgemm_simd", sz, g, w, 2.0 * pow(n, 3) / g / 1e12, "TFLOP/s");
    }
    for (int n = 512; n <= 4096; n *= 2) {
        measure_gemm(n, X, Y, Z, &g, &w);
        char sz[32];
        snprintf(sz, sizeof sz, "%d^3", n);
        row("sgemm_mps", sz, g, w, 2.0 * pow(n, 3) / g / 1e12, "TFLOP/s");
    }
    {   /* one tiny dispatch, submit -> completion observed in the guest */
        double wv[200];
        mpt_arg args[] = { { MPT_BIND_BUFFER, 0, X, 0 } };
        uint32_t grid[3] = { 32, 1, 1 }, tg[3] = { 32, 1, 1 };
        for (int i = 0; i < 200; i++) {
            double t0 = now();
            mpt_dispatch(c, &empty, grid, tg, MPT_DISPATCH_THREADS, args, 1, MPT_F_COMMIT);
            mpt_sync(c, NULL);
            wv[i] = now() - t0;
        }
        row("dispatch_roundtrip", "1x32thr", 0, median(wv, 200), median(wv, 200) * 1e6, "us");
    }
    {   /* many small dispatches, batched by the host into command buffers */
        const int n = 10000;
        double wv[3];
        mpt_arg args[] = { { MPT_BIND_BUFFER, 0, X, 0 } };
        uint32_t grid[3] = { 32, 1, 1 }, tg[3] = { 32, 1, 1 };
        for (int r = 0; r < 3; r++) {
            double t0 = now();
            for (int i = 0; i < n; i++) mpt_dispatch(c, &empty, grid, tg, MPT_DISPATCH_THREADS, args, 1, 0);
            mpt_sync(c, NULL);
            wv[r] = now() - t0;
        }
        row("dispatch_throughput", "10000x32thr", 0, median(wv, 3), n / median(wv, 3), "dispatch/s");
    }
    {   /* control-path latency */
        double hot[1000], cold[100];
        uint64_t rt;
        for (int i = 0; i < 1000; i++) { mpt_ping(c, &rt); hot[i] = rt * 1e-9; }
        for (int i = 0; i < 100; i++) { usleep(20000); mpt_ping(c, &rt); cold[i] = rt * 1e-9; }  /* host worker parked */
        row("ring_rtt_hot", "ping", 0, median(hot, 1000), median(hot, 1000) * 1e6, "us");
        row("ring_rtt_cold", "ping", 0, median(cold, 100), median(cold, 100) * 1e6, "us");
        double av[1000];
        for (int i = 0; i < 1000; i++) {
            double t0 = now();
            for (int k = 0; k < 100; k++) mpt_free(c, mpt_alloc(c, 4096, 256));
            av[i] = (now() - t0) / 100;
        }
        row("alloc_free", "4KiB", 0, median(av, 1000), median(av, 1000) * 1e9, "ns");
    }
    mpt_close(c);
    return 0;
}
