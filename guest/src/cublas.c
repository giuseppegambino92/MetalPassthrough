/*
 * libcublas.so.12 — a cuBLAS subset for MetalPassthrough, compiled against NVIDIA's official
 * <cublas_v2.h> for exact ABI compatibility.
 *
 *   Level 3: cublasSgemm, cublasSgemmStridedBatched -> host MPSMatrixMultiplication (zero-copy)
 *   Level 1: cublasSaxpy, cublasSscal, cublasScopy, cublasSdot, cublasSnrm2 -> Metal kernels
 *
 * All work is ordered on the handle's stream (cublasSetStream). Device pointers are MetalPassthrough
 * device addresses (equal to guest pointers under UVA).
 *
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */
#include <cuda.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metalpassthrough.h"

#define EXPORT __attribute__((visibility("default")))

uint32_t mpt_cuda_stream_id(CUstream s);  /* from MetalPassthrough's libcuda.so.1 */

struct cublasContext {
    mpt_ctx *ctx;
    cudaStream_t stream;
    cublasPointerMode_t pmode;
    cublasMath_t math;
    mpt_pipeline axpy, scal, copy, dot_partial, reduce_final;
    float *scratch;  /* 256 partial sums + result, arena memory */
};

static const char *KERNELS =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void axpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant float& a [[buffer(2)]],\n"
    "                 constant int2& inc [[buffer(3)]], constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {\n"
    "  if (i < n) y[i * inc.y] = fma(a, x[i * inc.x], y[i * inc.y]); }\n"
    "kernel void scal(device float* x [[buffer(0)]], constant float& a [[buffer(1)]], constant int& inc [[buffer(2)]],\n"
    "                 constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < n) x[i * inc] *= a; }\n"
    "kernel void copy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant int2& inc [[buffer(2)]],\n"
    "                 constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < n) y[i * inc.y] = x[i * inc.x]; }\n"
    "kernel void dot_partial(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device float* part [[buffer(2)]],\n"
    "                 constant int2& inc [[buffer(3)]], constant uint& n [[buffer(4)]], threadgroup float* sh [[threadgroup(0)]],\n"
    "                 uint t [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]],\n"
    "                 uint tpg [[threads_per_threadgroup]], uint ng [[threadgroups_per_grid]]) {\n"
    "  float acc = 0.0f;\n"
    "  for (uint i = g * tpg + t; i < n; i += tpg * ng) acc = fma(x[i * inc.x], y[i * inc.y], acc);\n"
    "  sh[t] = acc; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  for (uint s = tpg / 2; s > 0; s >>= 1) { if (t < s) sh[t] += sh[t + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
    "  if (t == 0) part[g] = sh[0]; }\n"
    "kernel void reduce_final(device const float* part [[buffer(0)]], device float* out [[buffer(1)]], constant uint& np [[buffer(2)]],\n"
    "                 constant uint& do_sqrt [[buffer(3)]], threadgroup float* sh [[threadgroup(0)]], uint t [[thread_index_in_threadgroup]]) {\n"
    "  sh[t] = t < np ? part[t] : 0.0f; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  for (uint s = 128; s > 0; s >>= 1) { if (t < s) sh[t] += sh[t + s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
    "  if (t == 0) out[0] = do_sqrt ? sqrt(sh[0]) : sh[0]; }\n";

#define NPART 256

static uint32_t sid(cublasHandle_t h) { return mpt_cuda_stream_id((CUstream)h->stream); }

static cublasStatus_t st(int r) {
    switch (r) {
    case 0: return CUBLAS_STATUS_SUCCESS;
    case MPT_ENOMEM: return CUBLAS_STATUS_ALLOC_FAILED;
    case MPT_EINVAL: case MPT_EFAULT: return CUBLAS_STATUS_INVALID_VALUE;
    case MPT_EGPU: return CUBLAS_STATUS_EXECUTION_FAILED;
    default: return CUBLAS_STATUS_INTERNAL_ERROR;
    }
}

/* Device vector of n elements with stride inc -> guest pointer (validated), or NULL. */
static void *dvec(cublasHandle_t h, const void *dev, int n, int inc) {
    void *p = mpt_hostptr(h->ctx, (uint64_t)(uintptr_t)dev);
    if (!p || n < 0 || inc <= 0) return NULL;
    size_t span = n ? ((size_t)(n - 1) * (size_t)inc + 1) * sizeof(float) : sizeof(float);
    return mpt_check_range(h->ctx, p, span) == 0 ? p : NULL;
}

static float scalar(cublasHandle_t h, const float *v) {
    if (h->pmode == CUBLAS_POINTER_MODE_HOST) return *v;
    float *p = mpt_hostptr(h->ctx, (uint64_t)(uintptr_t)v);
    if (!p) return NAN;
    mpt_stream_sync(h->ctx, sid(h), NULL);  /* device-resident scalar: wait for its producer */
    return *p;
}

EXPORT cublasStatus_t cublasCreate(cublasHandle_t *handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    mpt_ctx *c = mpt_default();
    if (!c) return CUBLAS_STATUS_NOT_INITIALIZED;
    struct cublasContext *h = calloc(1, sizeof *h);
    h->ctx = c;
    h->pmode = CUBLAS_POINTER_MODE_HOST;
    h->math = CUBLAS_DEFAULT_MATH;
    char log[4096];
    if (mpt_compile_msl(c, KERNELS, "axpy", 0, &h->axpy, log, sizeof log) || mpt_compile_msl(c, KERNELS, "scal", 0, &h->scal, log, sizeof log) ||
        mpt_compile_msl(c, KERNELS, "copy", 0, &h->copy, log, sizeof log) ||
        mpt_compile_msl(c, KERNELS, "dot_partial", 0, &h->dot_partial, log, sizeof log) ||
        mpt_compile_msl(c, KERNELS, "reduce_final", 0, &h->reduce_final, log, sizeof log)) {
        fprintf(stderr, "[libcublas/metalpassthrough] kernel build failed: %s\n", log);
        free(h);
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    h->scratch = mpt_alloc(c, (NPART + 64) * sizeof(float), 256);
    if (!h->scratch) { free(h); return CUBLAS_STATUS_ALLOC_FAILED; }
    *handle = h;
    return CUBLAS_STATUS_SUCCESS;
}

EXPORT cublasStatus_t cublasDestroy(cublasHandle_t h) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    mpt_sync(h->ctx, NULL);
    mpt_free(h->ctx, h->scratch);
    free(h);
    return CUBLAS_STATUS_SUCCESS;
}

EXPORT cublasStatus_t cublasGetVersion(cublasHandle_t h, int *v) { *v = CUBLAS_VERSION; return CUBLAS_STATUS_SUCCESS; }
EXPORT cublasStatus_t cublasSetStream(cublasHandle_t h, cudaStream_t s) { if (!h) return CUBLAS_STATUS_NOT_INITIALIZED; h->stream = s; return CUBLAS_STATUS_SUCCESS; }
EXPORT cublasStatus_t cublasGetStream(cublasHandle_t h, cudaStream_t *s) { *s = h->stream; return CUBLAS_STATUS_SUCCESS; }
EXPORT cublasStatus_t cublasSetPointerMode(cublasHandle_t h, cublasPointerMode_t m) { h->pmode = m; return CUBLAS_STATUS_SUCCESS; }
EXPORT cublasStatus_t cublasGetPointerMode(cublasHandle_t h, cublasPointerMode_t *m) { *m = h->pmode; return CUBLAS_STATUS_SUCCESS; }
EXPORT cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t m) { h->math = m; return CUBLAS_STATUS_SUCCESS; }
EXPORT cublasStatus_t cublasGetMathMode(cublasHandle_t h, cublasMath_t *m) { *m = h->math; return CUBLAS_STATUS_SUCCESS; }
EXPORT cublasStatus_t cublasSetWorkspace(cublasHandle_t h, void *w, size_t n) { return CUBLAS_STATUS_SUCCESS; }

EXPORT const char *cublasGetStatusName(cublasStatus_t s) {
    switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    default: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
}
EXPORT const char *cublasGetStatusString(cublasStatus_t s) { return cublasGetStatusName(s); }

/* ---- Level 3 ---- */

static cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const float *alpha,
                           const float *A, int lda, long long sa, const float *B, int ldb, long long sb, const float *beta, float *C,
                           int ldc, long long sc, int batch) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (ta == CUBLAS_OP_C) ta = CUBLAS_OP_T;  /* real types: conjugate transpose == transpose */
    if (tb == CUBLAS_OP_C) tb = CUBLAS_OP_T;
    if (m < 0 || n < 0 || k < 0 || batch < 0 || (ta != CUBLAS_OP_N && ta != CUBLAS_OP_T) || (tb != CUBLAS_OP_N && tb != CUBLAS_OP_T))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (!m || !n || !batch) return CUBLAS_STATUS_SUCCESS;
    mpt_gemm_desc g = {
        .dtype = MPT_GEMM_F32, .trans_a = ta == CUBLAS_OP_T, .trans_b = tb == CUBLAS_OP_T,
        .m = (uint32_t)m, .n = (uint32_t)n, .k = (uint32_t)(k ? k : 1), .lda = (uint32_t)lda, .ldb = (uint32_t)ldb, .ldc = (uint32_t)ldc,
        .batch = (uint32_t)batch, .alpha = k ? scalar(h, alpha) : 0.0f, .beta = scalar(h, beta),
        .a = (uint64_t)(uintptr_t)A, .b = (uint64_t)(uintptr_t)B, .c = (uint64_t)(uintptr_t)C,
        .stride_a = (uint64_t)sa, .stride_b = (uint64_t)sb, .stride_c = (uint64_t)sc,
    };
    return st(mpt_gemm_on(h->ctx, sid(h), &g));
}

EXPORT cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const float *alpha,
                                  const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    return gemm(h, ta, tb, m, n, k, alpha, A, lda, 0, B, ldb, 0, beta, C, ldc, 0, 1);
}

EXPORT cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                                const float *alpha, const float *A, int lda, long long sa, const float *B, int ldb,
                                                long long sb, const float *beta, float *C, int ldc, long long sc, int batch) {
    return gemm(h, ta, tb, m, n, k, alpha, A, lda, sa, B, ldb, sb, beta, C, ldc, sc, batch);
}

/* ---- Level 1 ---- */

static int launch(cublasHandle_t h, mpt_pipeline *p, uint32_t n, mpt_arg *args, uint32_t nargs) {
    uint32_t grid[3] = { n ? n : 1, 1, 1 }, tg[3] = { 256, 1, 1 };
    return mpt_dispatch_on(h->ctx, sid(h), p, grid, tg, MPT_DISPATCH_THREADS, args, nargs, 0);
}

EXPORT cublasStatus_t cublasSaxpy(cublasHandle_t h, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;
    void *px = dvec(h, x, n, incx), *py = dvec(h, y, n, incy);
    if (!px || !py) return incx <= 0 || incy <= 0 ? CUBLAS_STATUS_NOT_SUPPORTED : CUBLAS_STATUS_INVALID_VALUE;
    float a = scalar(h, alpha);
    int inc[2] = { incx, incy };
    uint32_t nn = (uint32_t)n;
    mpt_arg args[] = { { MPT_BIND_BUFFER, 0, px, 0 }, { MPT_BIND_BUFFER, 1, py, 0 }, { MPT_BIND_BYTES, 2, &a, 4 },
                       { MPT_BIND_BYTES, 3, inc, 8 }, { MPT_BIND_BYTES, 4, &nn, 4 } };
    return st(launch(h, &h->axpy, nn, args, 5));
}

EXPORT cublasStatus_t cublasSscal(cublasHandle_t h, int n, const float *alpha, float *x, int incx) {
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;
    void *px = dvec(h, x, n, incx);
    if (!px) return CUBLAS_STATUS_INVALID_VALUE;
    float a = scalar(h, alpha);
    uint32_t nn = (uint32_t)n;
    mpt_arg args[] = { { MPT_BIND_BUFFER, 0, px, 0 }, { MPT_BIND_BYTES, 1, &a, 4 }, { MPT_BIND_BYTES, 2, &incx, 4 }, { MPT_BIND_BYTES, 3, &nn, 4 } };
    return st(launch(h, &h->scal, nn, args, 4));
}

EXPORT cublasStatus_t cublasScopy(cublasHandle_t h, int n, const float *x, int incx, float *y, int incy) {
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;
    void *px = dvec(h, x, n, incx), *py = dvec(h, y, n, incy);
    if (!px || !py) return CUBLAS_STATUS_INVALID_VALUE;
    int inc[2] = { incx, incy };
    uint32_t nn = (uint32_t)n;
    mpt_arg args[] = { { MPT_BIND_BUFFER, 0, px, 0 }, { MPT_BIND_BUFFER, 1, py, 0 }, { MPT_BIND_BYTES, 2, inc, 8 }, { MPT_BIND_BYTES, 3, &nn, 4 } };
    return st(launch(h, &h->copy, nn, args, 4));
}

/* Two-stage reduction on the GPU; the result lands in `result` (device pointer mode) or is read back
 * after a stream sync (host pointer mode, the cuBLAS default). */
static cublasStatus_t reduce(cublasHandle_t h, int n, const float *x, int incx, const float *y, int incy, float *result, int do_sqrt) {
    if (!h || !result) return CUBLAS_STATUS_INVALID_VALUE;
    float *out = h->scratch + NPART;
    if (h->pmode == CUBLAS_POINTER_MODE_DEVICE) {
        out = mpt_hostptr(h->ctx, (uint64_t)(uintptr_t)result);
        if (!out || mpt_check_range(h->ctx, out, 4)) return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0) {
        if (h->pmode == CUBLAS_POINTER_MODE_HOST) *result = 0.0f; else mpt_fill_on(h->ctx, sid(h), mpt_devaddr(h->ctx, out), 4, 0, 4);
        return CUBLAS_STATUS_SUCCESS;
    }
    void *px = dvec(h, x, n, incx), *py = dvec(h, y, n, incy);
    if (!px || !py) return CUBLAS_STATUS_INVALID_VALUE;
    int inc[2] = { incx, incy };
    uint32_t nn = (uint32_t)n, np = NPART, sq = (uint32_t)do_sqrt;
    mpt_arg a1[] = { { MPT_BIND_BUFFER, 0, px, 0 }, { MPT_BIND_BUFFER, 1, py, 0 }, { MPT_BIND_BUFFER, 2, h->scratch, 0 },
                     { MPT_BIND_BYTES, 3, inc, 8 }, { MPT_BIND_BYTES, 4, &nn, 4 }, { MPT_BIND_TGMEM, 0, NULL, 256 * 4 } };
    uint32_t g1[3] = { NPART, 1, 1 }, tg[3] = { 256, 1, 1 };
    int r = mpt_dispatch_on(h->ctx, sid(h), &h->dot_partial, g1, tg, MPT_DISPATCH_THREADGROUPS, a1, 6, 0);
    mpt_arg a2[] = { { MPT_BIND_BUFFER, 0, h->scratch, 0 }, { MPT_BIND_BUFFER, 1, out, 0 }, { MPT_BIND_BYTES, 2, &np, 4 },
                     { MPT_BIND_BYTES, 3, &sq, 4 }, { MPT_BIND_TGMEM, 0, NULL, 256 * 4 } };
    uint32_t g2[3] = { 1, 1, 1 };
    if (!r) r = mpt_dispatch_on(h->ctx, sid(h), &h->reduce_final, g2, tg, MPT_DISPATCH_THREADGROUPS, a2, 5, 0);
    if (!r && h->pmode == CUBLAS_POINTER_MODE_HOST) {
        r = mpt_stream_sync(h->ctx, sid(h), NULL);
        *result = *out;
    }
    return st(r);
}

EXPORT cublasStatus_t cublasSdot(cublasHandle_t h, int n, const float *x, int incx, const float *y, int incy, float *result) {
    return reduce(h, n, x, incx, y, incy, result, 0);
}

EXPORT cublasStatus_t cublasSnrm2(cublasHandle_t h, int n, const float *x, int incx, float *result) {
    return reduce(h, n, x, incx, x, incx, result, 1);
}
