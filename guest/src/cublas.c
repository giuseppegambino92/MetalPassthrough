/*
 * libcublas.so.12 — a cuBLAS subset for MetalPassthrough, compiled against NVIDIA's official
 * <cublas_v2.h> for exact ABI compatibility.
 *
 *   Level 3 -> host MPSMatrixMultiplication on the arena (zero-copy):
 *     cublasSgemm, cublasSgemmStridedBatched, cublasSgemmBatched,
 *     cublasHgemm, cublasHgemmStridedBatched, cublasHgemmBatched,
 *     cublasGemmEx, cublasGemmStridedBatchedEx, cublasGemmBatchedEx
 *       (A/B in fp32, fp16 or bf16; C in fp32, fp16 or bf16; fp32 or fp16 compute)
 *   Level 2: cublasSgemv (unit strides) -> the same GEMM path
 *   Level 1: cublasSaxpy, cublasSscal, cublasScopy, cublasSdot, cublasSnrm2 -> Metal kernels
 *   Not provided: cuBLASLt, complex and fp64 BLAS, most level-2/3 routines (TRSM, SYRK, ...).
 *
 * Configuration calls, and what they do here:
 *   cublasSetMathMode: validated (unknown modes -> CUBLAS_STATUS_INVALID_VALUE) and stored. It never
 *     changes a result: this implementation takes no reduced-precision shortcut (fp32 GEMM is never
 *     run as TF32, fp32-compute GEMMs accumulate in fp32), which is exactly what
 *     CUBLAS_PEDANTIC_MATH / CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION require and what the
 *     default and TF32 modes permit. Only an explicit fp16 compute type (Hgemm, CUBLAS_COMPUTE_16F)
 *     accumulates in fp16, as the API defines.
 *   cublasSetWorkspace: validated like cuBLAS (256-byte aligned device memory, or NULL with size 0)
 *     and stored, but never written: GEMM staging uses host-private Metal memory instead.
 *   Level 1: positive increments only (negative/zero -> CUBLAS_STATUS_NOT_SUPPORTED). Snrm2 sums
 *     squares in fp32 without the reference BLAS rescaling, so it overflows for |x| > ~1.8e19.
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
    void *workspace;
    size_t workspace_bytes;
    mpt_pipeline axpy, scal, copy, dot_partial, reduce_final;
    float *scratch;  /* 256 partial sums + result, arena memory */
};

static const char *KERNELS =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void axpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant float& a [[buffer(2)]],\n"
    "                 constant int2& inc [[buffer(3)]], constant uint& n [[buffer(4)]], uint i [[thread_position_in_grid]]) {\n"
    "  if (i < n) y[ulong(i) * ulong(inc.y)] = fma(a, x[ulong(i) * ulong(inc.x)], y[ulong(i) * ulong(inc.y)]); }\n"
    "kernel void scal(device float* x [[buffer(0)]], constant float& a [[buffer(1)]], constant int& inc [[buffer(2)]],\n"
    "                 constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < n) x[ulong(i) * ulong(inc)] *= a; }\n"
    "kernel void copy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant int2& inc [[buffer(2)]],\n"
    "                 constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) { if (i < n) y[ulong(i) * ulong(inc.y)] = x[ulong(i) * ulong(inc.x)]; }\n"
    "kernel void dot_partial(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]], device float* part [[buffer(2)]],\n"
    "                 constant int2& inc [[buffer(3)]], constant uint& n [[buffer(4)]], threadgroup float* sh [[threadgroup(0)]],\n"
    "                 uint t [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]],\n"
    "                 uint tpg [[threads_per_threadgroup]], uint ng [[threadgroups_per_grid]]) {\n"
    "  float acc = 0.0f;\n"
    "  for (uint i = g * tpg + t; i < n; i += tpg * ng) acc = fma(x[ulong(i) * ulong(inc.x)], y[ulong(i) * ulong(inc.y)], acc);\n"
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

/* Device vector of n elements with stride inc -> guest pointer (validated), or NULL. The extent
 * ((n - 1) * inc + 1) * sizeof(float) is computed with explicit overflow checks: a wrapped size
 * would otherwise pass the range check. */
static void *dvec(cublasHandle_t h, const void *dev, int n, int inc) {
    if (!h || n < 0 || inc <= 0) return NULL;
    void *p = mpt_hostptr(h->ctx, (uint64_t)(uintptr_t)dev);
    if (!p) return NULL;
    size_t elems = 1, bytes;
    if (n > 0 && (__builtin_mul_overflow((size_t)(n - 1), (size_t)inc, &elems) || __builtin_add_overflow(elems, (size_t)1, &elems)))
        return NULL;
    if (__builtin_mul_overflow(elems, sizeof(float), &bytes)) return NULL;
    return mpt_check_range(h->ctx, p, bytes) == 0 ? p : NULL;
}

/* cuBLAS returns NOT_SUPPORTED for increments it cannot handle; distinguish that from bad pointers. */
static cublasStatus_t vec_status(int inc1, int inc2) {
    return inc1 <= 0 || inc2 <= 0 ? CUBLAS_STATUS_NOT_SUPPORTED : CUBLAS_STATUS_INVALID_VALUE;
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
/* See the header comment: validated and stored; results never depend on it (no reduced precision). */
EXPORT cublasStatus_t cublasSetMathMode(cublasHandle_t h, cublasMath_t m) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    unsigned base = (unsigned)m & ~(unsigned)CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
    if (base != CUBLAS_DEFAULT_MATH && base != CUBLAS_TENSOR_OP_MATH && base != CUBLAS_PEDANTIC_MATH && base != CUBLAS_TF32_TENSOR_OP_MATH)
        return CUBLAS_STATUS_INVALID_VALUE;
    h->math = m;
    return CUBLAS_STATUS_SUCCESS;
}
EXPORT cublasStatus_t cublasGetMathMode(cublasHandle_t h, cublasMath_t *m) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!m) return CUBLAS_STATUS_INVALID_VALUE;
    *m = h->math;
    return CUBLAS_STATUS_SUCCESS;
}
/* Validated like cuBLAS and remembered, never written (see the header comment). */
EXPORT cublasStatus_t cublasSetWorkspace(cublasHandle_t h, void *w, size_t n) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!w && n) return CUBLAS_STATUS_INVALID_VALUE;
    if (w) {
        void *p = mpt_hostptr(h->ctx, (uint64_t)(uintptr_t)w);
        if ((uintptr_t)w % 256 || !p || mpt_check_range(h->ctx, p, n ? n : 1)) return CUBLAS_STATUS_INVALID_VALUE;
    }
    h->workspace = w;
    h->workspace_bytes = n;
    return CUBLAS_STATUS_SUCCESS;
}

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

/* cublas_api.h declares the __half entry points only for C++; the ABI is a pointer to 2 bytes. */
typedef struct { uint16_t x; } mpt_half;

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (e == 0x1f) u = sign | 0x7f800000u | (m << 13);
    else if (e) u = sign | ((e + 112) << 23) | (m << 13);
    else if (!m) u = sign;
    else {  /* subnormal */
        e = 113;
        while (!(m & 0x400)) { m <<= 1; e--; }
        u = sign | (e << 23) | ((m & 0x3ff) << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

/* Reads alpha/beta: host or device pointer (cublasPointerMode), fp32 or fp16. */
static float scalar_t(cublasHandle_t h, const void *v, int is_half) {
    const void *p = v;
    if (h->pmode != CUBLAS_POINTER_MODE_HOST) {
        p = mpt_hostptr(h->ctx, (uint64_t)(uintptr_t)v);
        if (!p) return NAN;
        mpt_stream_sync(h->ctx, sid(h), NULL);  /* device-resident scalar: wait for its producer */
    }
    if (is_half) { uint16_t x; memcpy(&x, p, 2); return half_to_float(x); }
    float f;
    memcpy(&f, p, 4);
    return f;
}

static int norm_op(cublasOperation_t *op) {
    if (*op == CUBLAS_OP_C) *op = CUBLAS_OP_T;  /* real types: conjugate transpose == transpose */
    return *op == CUBLAS_OP_N || *op == CUBLAS_OP_T;
}

/* Common GEMM: C = alpha op(A) op(B) + beta C, column-major, `batch` strided entries. */
static cublasStatus_t gemm_core(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, float alpha,
                                const void *A, int lda, long long sa, const void *B, int ldb, long long sb, float beta, void *C, int ldc,
                                long long sc, int batch, uint32_t dtype) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || k < 0 || batch < 0 || sa < 0 || sb < 0 || sc < 0 || !norm_op(&ta) || !norm_op(&tb))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (!m || !n || !batch) return CUBLAS_STATUS_SUCCESS;
    mpt_gemm_desc g = {
        .dtype = dtype, .trans_a = ta == CUBLAS_OP_T, .trans_b = tb == CUBLAS_OP_T,
        .m = (uint32_t)m, .n = (uint32_t)n, .k = (uint32_t)(k ? k : 1), .lda = (uint32_t)lda, .ldb = (uint32_t)ldb, .ldc = (uint32_t)ldc,
        .batch = (uint32_t)batch, .alpha = k ? alpha : 0.0f, .beta = beta,
        .a = (uint64_t)(uintptr_t)A, .b = (uint64_t)(uintptr_t)B, .c = (uint64_t)(uintptr_t)C,
        .stride_a = (uint64_t)sa, .stride_b = (uint64_t)sb, .stride_c = (uint64_t)sc,
    };
    return st(mpt_gemm_on(h->ctx, sid(h), &g));
}

/* Pointer-array batches: the arrays live in device memory; read them once the stream has
 * produced them, then issue one GEMM per entry. */
static cublasStatus_t gemm_ptr_batched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, float alpha,
                                       const void *const *A, int lda, const void *const *B, int ldb, float beta, void *const *C, int ldc,
                                       int batch, uint32_t dtype) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (batch < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (!batch) return CUBLAS_STATUS_SUCCESS;
    size_t bytes = (size_t)batch * sizeof(void *);
    uint64_t *pa = mpt_hostptr(h->ctx, (uint64_t)(uintptr_t)A), *pb = mpt_hostptr(h->ctx, (uint64_t)(uintptr_t)B),
             *pc = mpt_hostptr(h->ctx, (uint64_t)(uintptr_t)C);
    if (!pa || !pb || !pc || mpt_check_range(h->ctx, pa, bytes) || mpt_check_range(h->ctx, pb, bytes) || mpt_check_range(h->ctx, pc, bytes))
        return CUBLAS_STATUS_INVALID_VALUE;
    mpt_stream_sync(h->ctx, sid(h), NULL);
    for (int i = 0; i < batch; i++) {
        cublasStatus_t r = gemm_core(h, ta, tb, m, n, k, alpha, (const void *)(uintptr_t)pa[i], lda, 0, (const void *)(uintptr_t)pb[i], ldb, 0,
                                     beta, (void *)(uintptr_t)pc[i], ldc, 0, 1, dtype);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

EXPORT cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const float *alpha,
                                  const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    return gemm_core(h, ta, tb, m, n, k, scalar_t(h, alpha, 0), A, lda, 0, B, ldb, 0, scalar_t(h, beta, 0), C, ldc, 0, 1, MPT_GEMM_F32);
}

EXPORT cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                                const float *alpha, const float *A, int lda, long long sa, const float *B, int ldb,
                                                long long sb, const float *beta, float *C, int ldc, long long sc, int batch) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    return gemm_core(h, ta, tb, m, n, k, scalar_t(h, alpha, 0), A, lda, sa, B, ldb, sb, scalar_t(h, beta, 0), C, ldc, sc, batch, MPT_GEMM_F32);
}

EXPORT cublasStatus_t cublasSgemmBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                         const float *alpha, const float *const A[], int lda, const float *const B[], int ldb,
                                         const float *beta, float *const C[], int ldc, int batch) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    return gemm_ptr_batched(h, ta, tb, m, n, k, scalar_t(h, alpha, 0), (const void *const *)A, lda, (const void *const *)B, ldb,
                            scalar_t(h, beta, 0), (void *const *)C, ldc, batch, MPT_GEMM_F32);
}

#define H16 MPT_GEMM_DTYPE(MPT_GEMM_F16, MPT_GEMM_F16, 1)

EXPORT cublasStatus_t cublasHgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const mpt_half *alpha,
                                  const mpt_half *A, int lda, const mpt_half *B, int ldb, const mpt_half *beta, mpt_half *C, int ldc) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    return gemm_core(h, ta, tb, m, n, k, scalar_t(h, alpha, 1), A, lda, 0, B, ldb, 0, scalar_t(h, beta, 1), C, ldc, 0, 1, H16);
}

EXPORT cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                                const mpt_half *alpha, const mpt_half *A, int lda, long long sa, const mpt_half *B, int ldb,
                                                long long sb, const mpt_half *beta, mpt_half *C, int ldc, long long sc, int batch) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    return gemm_core(h, ta, tb, m, n, k, scalar_t(h, alpha, 1), A, lda, sa, B, ldb, sb, scalar_t(h, beta, 1), C, ldc, sc, batch, H16);
}

EXPORT cublasStatus_t cublasHgemmBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                         const mpt_half *alpha, const mpt_half *const A[], int lda, const mpt_half *const B[], int ldb,
                                         const mpt_half *beta, mpt_half *const C[], int ldc, int batch) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    return gemm_ptr_batched(h, ta, tb, m, n, k, scalar_t(h, alpha, 1), (const void *const *)A, lda, (const void *const *)B, ldb,
                            scalar_t(h, beta, 1), (void *const *)C, ldc, batch, H16);
}

static int mpt_dtype_of(cudaDataType t) {
    switch ((int)t) {
    case CUDA_R_32F: return MPT_GEMM_F32;
    case CUDA_R_16F: return MPT_GEMM_F16;
    case CUDA_R_16BF: return MPT_GEMM_BF16;
    default: return -1;
    }
}

/* GemmEx type rules: A and B share a type; fp16 compute needs fp16 everywhere (alpha/beta are
 * mpt_half then); fp32 compute (including the TF32/16F/16BF "fast" variants, which run at full fp32
 * precision here) takes float alpha/beta. Legacy callers pass cudaDataType as the compute type. */
static cublasStatus_t ex_dtype(cudaDataType at, cudaDataType bt, cudaDataType ct, int compute, uint32_t *dtype, int *half_scalars) {
    int a = mpt_dtype_of(at), b = mpt_dtype_of(bt), c = mpt_dtype_of(ct);
    if (a < 0 || c < 0 || a != b) return CUBLAS_STATUS_NOT_SUPPORTED;
    int acc16;
    switch (compute) {
    case CUBLAS_COMPUTE_16F: case CUBLAS_COMPUTE_16F_PEDANTIC: case CUDA_R_16F: acc16 = 1; break;
    case CUBLAS_COMPUTE_32F: case CUBLAS_COMPUTE_32F_PEDANTIC: case CUBLAS_COMPUTE_32F_FAST_16F:
    case CUBLAS_COMPUTE_32F_FAST_16BF: case CUBLAS_COMPUTE_32F_FAST_TF32: case CUDA_R_32F: acc16 = 0; break;
    default: return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    if (acc16 && (a != MPT_GEMM_F16 || c != MPT_GEMM_F16)) return CUBLAS_STATUS_NOT_SUPPORTED;
    *dtype = MPT_GEMM_DTYPE(a, c, acc16);
    *half_scalars = acc16;
    return CUBLAS_STATUS_SUCCESS;
}

EXPORT cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const void *alpha,
                                   const void *A, cudaDataType at, int lda, const void *B, cudaDataType bt, int ldb, const void *beta,
                                   void *C, cudaDataType ct, int ldc, cublasComputeType_t compute, cublasGemmAlgo_t algo) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    uint32_t dt;
    int hs;
    cublasStatus_t r = ex_dtype(at, bt, ct, (int)compute, &dt, &hs);
    if (r) return r;
    return gemm_core(h, ta, tb, m, n, k, scalar_t(h, alpha, hs), A, lda, 0, B, ldb, 0, scalar_t(h, beta, hs), C, ldc, 0, 1, dt);
}

EXPORT cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                                 const void *alpha, const void *A, cudaDataType at, int lda, long long sa, const void *B,
                                                 cudaDataType bt, int ldb, long long sb, const void *beta, void *C, cudaDataType ct, int ldc,
                                                 long long sc, int batch, cublasComputeType_t compute, cublasGemmAlgo_t algo) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    uint32_t dt;
    int hs;
    cublasStatus_t r = ex_dtype(at, bt, ct, (int)compute, &dt, &hs);
    if (r) return r;
    return gemm_core(h, ta, tb, m, n, k, scalar_t(h, alpha, hs), A, lda, sa, B, ldb, sb, scalar_t(h, beta, hs), C, ldc, sc, batch, dt);
}

EXPORT cublasStatus_t cublasGemmBatchedEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                          const void *alpha, const void *const A[], cudaDataType at, int lda, const void *const B[],
                                          cudaDataType bt, int ldb, const void *beta, void *const C[], cudaDataType ct, int ldc, int batch,
                                          cublasComputeType_t compute, cublasGemmAlgo_t algo) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    uint32_t dt;
    int hs;
    cublasStatus_t r = ex_dtype(at, bt, ct, (int)compute, &dt, &hs);
    if (r) return r;
    return gemm_ptr_batched(h, ta, tb, m, n, k, scalar_t(h, alpha, hs), A, lda, B, ldb, scalar_t(h, beta, hs), C, ldc, batch, dt);
}

/* ---- Level 2 ---- */

/* y = alpha op(A) x + beta y as an (m x 1) GEMM. Unit strides only. */
EXPORT cublasStatus_t cublasSgemv(cublasHandle_t h, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda,
                                  const float *x, int incx, const float *beta, float *y, int incy) {
    if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || lda < (m > 1 ? m : 1) || !incx || !incy || !norm_op(&trans)) return CUBLAS_STATUS_INVALID_VALUE;
    if (incx != 1 || incy != 1) return CUBLAS_STATUS_NOT_SUPPORTED;
    int rows = trans == CUBLAS_OP_N ? m : n, inner = trans == CUBLAS_OP_N ? n : m;
    if (!rows) return CUBLAS_STATUS_SUCCESS;
    return gemm_core(h, trans, CUBLAS_OP_N, rows, 1, inner, scalar_t(h, alpha, 0), A, lda, 0, x, inner ? inner : 1, 0, scalar_t(h, beta, 0), y,
                     rows, 0, 1, MPT_GEMM_F32);
}

/* ---- Level 1 ---- */

static int launch(cublasHandle_t h, mpt_pipeline *p, uint32_t n, mpt_arg *args, uint32_t nargs) {
    uint32_t grid[3] = { n ? n : 1, 1, 1 }, tg[3] = { 256, 1, 1 };
    return mpt_dispatch_on(h->ctx, sid(h), p, grid, tg, MPT_DISPATCH_THREADS, args, nargs, 0);
}

EXPORT cublasStatus_t cublasSaxpy(cublasHandle_t h, int n, const float *alpha, const float *x, int incx, float *y, int incy) {
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;
    void *px = dvec(h, x, n, incx), *py = dvec(h, y, n, incy);
    if (!px || !py) return vec_status(incx, incy);
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
    if (!px) return vec_status(incx, incx);
    float a = scalar(h, alpha);
    uint32_t nn = (uint32_t)n;
    mpt_arg args[] = { { MPT_BIND_BUFFER, 0, px, 0 }, { MPT_BIND_BYTES, 1, &a, 4 }, { MPT_BIND_BYTES, 2, &incx, 4 }, { MPT_BIND_BYTES, 3, &nn, 4 } };
    return st(launch(h, &h->scal, nn, args, 4));
}

EXPORT cublasStatus_t cublasScopy(cublasHandle_t h, int n, const float *x, int incx, float *y, int incy) {
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;
    void *px = dvec(h, x, n, incx), *py = dvec(h, y, n, incy);
    if (!px || !py) return vec_status(incx, incy);
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
    if (!px || !py) return vec_status(incx, incy);
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
