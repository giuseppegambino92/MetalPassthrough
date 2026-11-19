// cuBLAS mixed precision and batched/level-2 paths: GemmEx (fp16/bf16/fp32 inputs and outputs,
// fp32 and fp16 compute), Hgemm, strided and pointer-array batches, Sgemv. Every result is checked
// against a float64 reference computed from the (rounded) inputs, padding between columns must be
// untouched, and unsupported type combinations must be refused.
//   nvcc -O3 -cudart shared -o cublas_mixed cublas_mixed.cu -lcublas
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s -> %s\n", #x, cudaGetErrorString(e)); exit(2); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { printf("%s -> %s\n", #x, cublasGetStatusString(s)); exit(2); } } while (0)

static int fails = 0;
static unsigned seed = 11;
static float frand() { seed = seed * 1664525u + 1013904223u; return (float)(seed >> 9) / (float)(1u << 23) - 0.5f; }

// Storage <-> float for the three element types.
static size_t esize(cudaDataType t) { return t == CUDA_R_32F ? 4 : 2; }
static void store(void *p, size_t i, cudaDataType t, float v) {
    if (t == CUDA_R_32F) ((float *)p)[i] = v;
    else if (t == CUDA_R_16F) ((__half *)p)[i] = __float2half(v);
    else ((__nv_bfloat16 *)p)[i] = __float2bfloat16(v);
}
static double load(const void *p, size_t i, cudaDataType t) {
    if (t == CUDA_R_32F) return ((const float *)p)[i];
    if (t == CUDA_R_16F) return __half2float(((const __half *)p)[i]);
    return __bfloat162float(((const __nv_bfloat16 *)p)[i]);
}
static const char *tn(cudaDataType t) { return t == CUDA_R_32F ? "f32" : t == CUDA_R_16F ? "f16" : "bf16"; }

struct Mat {
    cudaDataType t; int rows, cols, ld, batch; long long stride;
    std::vector<uint8_t> h; void *d = nullptr;
    Mat(cudaDataType t_, int r, int c, int ld_, int b = 1) : t(t_), rows(r), cols(c), ld(ld_), batch(b) {
        stride = (long long)ld * cols + 7;  // odd gap between batch entries
        h.resize((size_t)(stride * batch) * esize(t));
        for (size_t i = 0; i < (size_t)(stride * batch); i++) store(h.data(), i, t, frand());
        CK(cudaMalloc(&d, h.size()));
        CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
    }
    double at(int b, int r, int c) const { return load(h.data(), (size_t)(b * stride + (long long)c * ld + r), t); }
    ~Mat() { cudaFree(d); }
};

// Checks C (device) against alpha op(A) op(B) + beta C0 in float64; padding must be bit-identical.
static void verify(const char *name, cublasHandle_t, Mat &A, bool ta, Mat &B, bool tb, Mat &C, int m, int n, int k, double alpha, double beta,
                   double tol) {
    std::vector<uint8_t> out(C.h.size());
    CK(cudaMemcpy(out.data(), C.d, out.size(), cudaMemcpyDeviceToHost));
    double worst = 0, scale = 0;
    size_t pad_changed = 0;
    for (int b = 0; b < C.batch; b++) {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                double acc = 0;
                for (int p = 0; p < k; p++) acc += (ta ? A.at(b, p, i) : A.at(b, i, p)) * (tb ? B.at(b, j, p) : B.at(b, p, j));
                double ref = alpha * acc + beta * C.at(b, i, j);
                double got = load(out.data(), (size_t)(b * C.stride + (long long)j * C.ld + i), C.t);
                worst = fmax(worst, fabs(got - ref));
                scale = fmax(scale, fabs(ref));
            }
    }
    for (int b = 0; b < C.batch; b++)  // padding rows (ld > m) and the gap after each batch entry
        for (long long e = 0; e < C.stride; e++) {
            long long col = e / C.ld, row = e % C.ld;
            if (col < n && row < m) continue;
            size_t off = (size_t)(b * C.stride + e) * esize(C.t);
            if (b * C.stride + e < (long long)(C.h.size() / esize(C.t)) && memcmp(&out[off], &C.h[off], esize(C.t))) pad_changed++;
        }
    bool ok = worst <= tol * fmax(scale, 1.0) && !pad_changed;
    fails += !ok;
    printf("%-58s max err %.2e (tol %.1e x %.2f), padding changed: %zu  %s\n", name, worst, tol, fmax(scale, 1.0), pad_changed, ok ? "ok" : "FAIL");
}

int main() {
    cublasHandle_t h;
    CB(cublasCreate(&h));
    const int m = 67, n = 45, k = 93;
    struct Case { cudaDataType ab, c; cublasComputeType_t comp; double tol; };
    Case cases[] = {
        { CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 2e-3 },        // fp16 storage, fp32 accumulate, one rounding
        { CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, 1e-5 },
        { CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, 1e-2 },
        { CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, 1e-5 },
        { CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 1e-5 },
        { CUDA_R_32F, CUDA_R_16BF, CUBLAS_COMPUTE_32F, 1e-2 },
        { CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, 2e-2 },        // fp16 accumulate
    };
    for (auto &cs : cases)
        for (int t = 0; t < 4; t++) {
            bool ta = t & 1, tb = t & 2;
            Mat A(cs.ab, ta ? k : m, ta ? m : k, (ta ? k : m) + 3), B(cs.ab, tb ? n : k, tb ? k : n, (tb ? n : k) + 5), C(cs.c, m, n, m + 2);
            float a32 = 1.25f, b32 = -0.5f;
            __half a16 = __float2half(1.25f), b16 = __float2half(-0.5f);
            bool h16 = cs.comp == CUBLAS_COMPUTE_16F;
            CB(cublasGemmEx(h, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, h16 ? (void *)&a16 : (void *)&a32, A.d,
                            cs.ab, A.ld, B.d, cs.ab, B.ld, h16 ? (void *)&b16 : (void *)&b32, C.d, cs.c, C.ld, cs.comp, CUBLAS_GEMM_DEFAULT));
            char name[128];
            snprintf(name, sizeof name, "GemmEx %s x %s -> %s, compute %s, %c%c", tn(cs.ab), tn(cs.ab), tn(cs.c), h16 ? "16F" : "32F", ta ? 'T' : 'N', tb ? 'T' : 'N');
            verify(name, h, A, ta, B, tb, C, m, n, k, 1.25, -0.5, cs.tol);
        }
    {   // Hgemm (fp16 compute) and strided batched Ex in bf16
        Mat A(CUDA_R_16F, m, k, m), B(CUDA_R_16F, k, n, k), C(CUDA_R_16F, m, n, m);
        __half one = __float2half(1.0f), zero = __float2half(0.0f);
        CB(cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, (const __half *)A.d, A.ld, (const __half *)B.d, B.ld, &zero, (__half *)C.d, C.ld));
        verify("Hgemm NN", h, A, false, B, false, C, m, n, k, 1, 0, 2e-2);
        const int bt = 5;
        Mat As(CUDA_R_16BF, 32, 40, 33, bt), Bs(CUDA_R_16BF, 40, 24, 41, bt), Cs(CUDA_R_16BF, 32, 24, 32, bt);
        float a = 1.0f, b = 0.25f;
        CB(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 32, 24, 40, &a, As.d, CUDA_R_16BF, As.ld, As.stride, Bs.d, CUDA_R_16BF, Bs.ld,
                                      Bs.stride, &b, Cs.d, CUDA_R_16BF, Cs.ld, Cs.stride, bt, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        verify("GemmStridedBatchedEx bf16, 5 entries, gapped strides", h, As, false, Bs, false, Cs, 32, 24, 40, 1, 0.25, 1e-2);
        // fp32, gapped strides (MPS batching would be wrong here) and stride 0 = broadcast A.
        Mat Af(CUDA_R_32F, 32, 40, 33, bt), Bf(CUDA_R_32F, 40, 24, 41, bt), Cf(CUDA_R_32F, 32, 24, 32, bt);
        CB(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, 32, 24, 40, &a, (const float *)Af.d, Af.ld, Af.stride, (const float *)Bf.d, Bf.ld,
                                     Bf.stride, &b, (float *)Cf.d, Cf.ld, Cf.stride, bt));
        verify("SgemmStridedBatched fp32, 5 entries, gapped strides", h, Af, false, Bf, false, Cf, 32, 24, 40, 1, 0.25, 1e-5);
        Mat A0(CUDA_R_32F, 32, 40, 32, 1), Bb(CUDA_R_32F, 40, 24, 40, bt), Cb(CUDA_R_32F, 32, 24, 32, bt);
        CB(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, 32, 24, 40, &a, (const float *)A0.d, A0.ld, 0, (const float *)Bb.d, Bb.ld,
                                     Bb.stride, &b, (float *)Cb.d, Cb.ld, Cb.stride, bt));
        A0.stride = 0;  // reference: every entry reads entry 0 of A
        A0.batch = bt;
        verify("SgemmStridedBatched fp32, stride 0 (broadcast A)", h, A0, false, Bb, false, Cb, 32, 24, 40, 1, 0.25, 1e-5);
        A0.batch = 1;
    }
    {   // pointer-array batches: SgemmBatched
        const int bt = 4;
        std::vector<Mat *> As, Bs, Cs;
        std::vector<const float *> pa, pb;
        std::vector<float *> pc;
        for (int i = 0; i < bt; i++) {
            As.push_back(new Mat(CUDA_R_32F, 20, 30, 20)); Bs.push_back(new Mat(CUDA_R_32F, 30, 10, 30)); Cs.push_back(new Mat(CUDA_R_32F, 20, 10, 20));
            pa.push_back((const float *)As[i]->d); pb.push_back((const float *)Bs[i]->d); pc.push_back((float *)Cs[i]->d);
        }
        const float **dpa; const float **dpb; float **dpc;
        CK(cudaMalloc(&dpa, bt * sizeof(void *))); CK(cudaMalloc(&dpb, bt * sizeof(void *))); CK(cudaMalloc(&dpc, bt * sizeof(void *)));
        CK(cudaMemcpy(dpa, pa.data(), bt * sizeof(void *), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpb, pb.data(), bt * sizeof(void *), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpc, pc.data(), bt * sizeof(void *), cudaMemcpyHostToDevice));
        float a = 0.5f, b = 2.0f;
        CB(cublasSgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, 20, 10, 30, &a, dpa, 20, dpb, 30, &b, dpc, 20, bt));
        for (int i = 0; i < bt; i++) {
            char name[64];
            snprintf(name, sizeof name, "SgemmBatched (pointer arrays) entry %d", i);
            verify(name, h, *As[i], false, *Bs[i], false, *Cs[i], 20, 10, 30, 0.5, 2.0, 1e-5);
            delete As[i]; delete Bs[i]; delete Cs[i];
        }
        cudaFree(dpa); cudaFree(dpb); cudaFree(dpc);
    }
    for (int t = 0; t < 2; t++) {  // Sgemv: y is an (rows x 1) matrix
        Mat A(CUDA_R_32F, 70, 50, 73), x(CUDA_R_32F, t ? 70 : 50, 1, t ? 70 : 50), y(CUDA_R_32F, t ? 50 : 70, 1, t ? 50 : 70);
        float a = 1.5f, b = -1.0f;
        CB(cublasSgemv(h, t ? CUBLAS_OP_T : CUBLAS_OP_N, 70, 50, &a, (const float *)A.d, A.ld, (const float *)x.d, 1, &b, (float *)y.d, 1));
        verify(t ? "Sgemv T" : "Sgemv N", h, A, t, x, false, y, t ? 50 : 70, 1, t ? 70 : 50, 1.5, -1.0, 1e-5);
    }
    {   // refused combinations
        float a = 1, b = 0;
        void *d;
        CK(cudaMalloc(&d, 1 << 16));
        cublasStatus_t s1 = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &a, d, CUDA_R_16F, 8, d, CUDA_R_16BF, 8, &b, d, CUDA_R_32F, 8,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cublasStatus_t s2 = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &a, d, CUDA_R_16BF, 8, d, CUDA_R_16BF, 8, &b, d, CUDA_R_16BF, 8,
                                         CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasStatus_t s3 = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, &a, d, CUDA_R_8I, 8, d, CUDA_R_8I, 8, &b, d, CUDA_R_32I, 8,
                                         CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
        bool ok = s1 == CUBLAS_STATUS_NOT_SUPPORTED && s2 == CUBLAS_STATUS_NOT_SUPPORTED && s3 == CUBLAS_STATUS_NOT_SUPPORTED;
        fails += !ok;
        printf("%-58s %s / %s / %s  %s\n", "refused: f16 x bf16, bf16 with fp16 compute, int8", cublasGetStatusName(s1), cublasGetStatusName(s2),
               cublasGetStatusName(s3), ok ? "ok" : "FAIL");
        cudaFree(d);
    }
    {   // Math modes: validated, and results never depend on them (no reduced-precision shortcuts).
        const int M = 96, N = 80, K = 300;
        Mat A(CUDA_R_32F, M, K, M), B(CUDA_R_32F, K, N, K), C(CUDA_R_32F, M, N, M);
        cublasMath_t modes[] = { CUBLAS_DEFAULT_MATH, CUBLAS_PEDANTIC_MATH, CUBLAS_TF32_TENSOR_OP_MATH, CUBLAS_TENSOR_OP_MATH,
                                 (cublasMath_t)(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION) };
        std::vector<float> ref((size_t)M * N), out((size_t)M * N);
        bool same = true, set_ok = true;
        for (int i = 0; i < 5; i++) {
            cublasMath_t got;
            set_ok &= cublasSetMathMode(h, modes[i]) == CUBLAS_STATUS_SUCCESS && cublasGetMathMode(h, &got) == CUBLAS_STATUS_SUCCESS && got == modes[i];
            float one = 1, zero = 0;
            CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, (const float *)A.d, M, (const float *)B.d, K, &zero, (float *)C.d, M));
            CK(cudaMemcpy((i ? out : ref).data(), C.d, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
            if (i) same &= !memcmp(out.data(), ref.data(), out.size() * 4);
        }
        bool bad_rejected = cublasSetMathMode(h, (cublasMath_t)77) == CUBLAS_STATUS_INVALID_VALUE;
        CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
        double worst = 0;  // full fp32 accuracy (TF32 would be ~1e-3 relative here)
        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++) {
                double acc = 0;
                for (int p2 = 0; p2 < K; p2++) acc += A.at(0, i, p2) * B.at(0, p2, j);
                worst = fmax(worst, fabs(acc - ref[(size_t)j * M + i]));
            }
        bool ok = set_ok && same && bad_rejected && worst < 1e-5;
        fails += !ok;
        printf("%-58s set/get %s, bit-identical across 5 modes %s, invalid mode rejected %s, fp32 err %.1e  %s\n", "math modes", set_ok ? "ok" : "BAD",
               same ? "yes" : "NO", bad_rejected ? "yes" : "NO", worst, ok ? "ok" : "FAIL");
    }
    {   // Workspace: validated like cuBLAS, never written.
        uint8_t *ws;
        CK(cudaMalloc(&ws, 1 << 20));
        CK(cudaMemset(ws, 0x5A, 1 << 20));
        char host_buf[512];
        cublasStatus_t s_null = cublasSetWorkspace(h, nullptr, 0), s_null_size = cublasSetWorkspace(h, nullptr, 64),
                       s_misaligned = cublasSetWorkspace(h, ws + 8, 4096), s_host = cublasSetWorkspace(h, host_buf, 256),
                       s_too_big = cublasSetWorkspace(h, ws, (2 << 20)), s_ok = cublasSetWorkspace(h, ws, 1 << 20);
        Mat A(CUDA_R_16BF, 64, 64, 64), B(CUDA_R_16BF, 64, 64, 64), C(CUDA_R_16BF, 64, 64, 64);  // staged path
        float one = 1, zero = 0;
        CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, 64, 64, 64, &one, A.d, CUDA_R_16BF, 64, B.d, CUDA_R_16BF, 64, &zero, C.d, CUDA_R_16BF, 64,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        std::vector<uint8_t> back(1 << 20);
        CK(cudaMemcpy(back.data(), ws, back.size(), cudaMemcpyDeviceToHost));
        size_t touched = 0;
        for (uint8_t b : back) touched += b != 0x5A;
        bool ok = s_null == CUBLAS_STATUS_SUCCESS && s_null_size == CUBLAS_STATUS_INVALID_VALUE && s_misaligned == CUBLAS_STATUS_INVALID_VALUE &&
                  s_host == CUBLAS_STATUS_INVALID_VALUE && s_too_big == CUBLAS_STATUS_INVALID_VALUE && s_ok == CUBLAS_STATUS_SUCCESS && !touched;
        fails += !ok;
        printf("%-58s NULL/0 %s, NULL/64 %s, misaligned %s, host ptr %s, past end %s, valid %s; bytes written: %zu  %s\n", "cublasSetWorkspace",
               cublasGetStatusName(s_null) + 14, cublasGetStatusName(s_null_size) + 14, cublasGetStatusName(s_misaligned) + 14,
               cublasGetStatusName(s_host) + 14, cublasGetStatusName(s_too_big) + 14, cublasGetStatusName(s_ok) + 14, touched, ok ? "ok" : "FAIL");
        CB(cublasSetWorkspace(h, nullptr, 0));
        cudaFree(ws);
    }
    {   // Level 1: strides, size overflow, unsupported increments.
        const int n = 1000, inc = 3;
        float *x, *y;
        CK(cudaMalloc(&x, (size_t)n * inc * 4)); CK(cudaMalloc(&y, (size_t)n * inc * 4));
        std::vector<float> hx((size_t)n * inc), hy((size_t)n * inc);
        for (size_t i = 0; i < hx.size(); i++) { hx[i] = frand(); hy[i] = frand(); }
        CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(y, hy.data(), hy.size() * 4, cudaMemcpyHostToDevice));
        float a = 0.75f, dot = 0;
        CB(cublasSaxpy(h, n, &a, x, inc, y, inc));
        CB(cublasSdot(h, n, x, inc, y, inc, &dot));
        std::vector<float> gy(hy.size());
        CK(cudaMemcpy(gy.data(), y, gy.size() * 4, cudaMemcpyDeviceToHost));
        double ax = 0, rdot = 0;
        size_t untouched_bad = 0;
        for (int i = 0; i < n * inc; i++) {
            if (i % inc == 0) { double r = (double)a * hx[i] + hy[i]; ax = fmax(ax, fabs(gy[i] - r)); rdot += (double)hx[i] * r; }
            else untouched_bad += gy[i] != hy[i];  // elements between strides must not change
        }
        const int big = 2147483647;  // (n-1)*inc*4 exceeds 2^64 only with these combined; each must be rejected cleanly
        cublasStatus_t o1 = cublasSaxpy(h, big, &a, x, big, y, 1), o2 = cublasScopy(h, 5, x, 1 << 30, y, 1),
                       o3 = cublasSdot(h, big, x, big, y, big, &dot), o4 = cublasSscal(h, big, &a, x, big);
        cublasStatus_t neg = cublasSaxpy(h, n, &a, x, -1, y, 1), neg2 = cublasScopy(h, n, x, 1, y, -2), zero_inc = cublasSscal(h, n, &a, x, 0);
        bool ok = ax < 1e-6 && fabs(dot - rdot) < 1e-3 * fmax(1.0, fabs(rdot)) && !untouched_bad && o1 == CUBLAS_STATUS_INVALID_VALUE &&
                  o2 == CUBLAS_STATUS_INVALID_VALUE && o3 == CUBLAS_STATUS_INVALID_VALUE && o4 == CUBLAS_STATUS_INVALID_VALUE &&
                  neg == CUBLAS_STATUS_NOT_SUPPORTED && neg2 == CUBLAS_STATUS_NOT_SUPPORTED && zero_inc == CUBLAS_STATUS_NOT_SUPPORTED;
        fails += !ok;
        printf("%-58s axpy err %.1e, dot err %.1e, gaps untouched %s; huge n*inc rejected %s; negative/zero inc -> %s  %s\n",
               "level 1 with stride 3, overflow and increments", ax, fabs(dot - rdot), untouched_bad ? "NO" : "yes",
               o1 == CUBLAS_STATUS_INVALID_VALUE && o2 == CUBLAS_STATUS_INVALID_VALUE && o3 == CUBLAS_STATUS_INVALID_VALUE && o4 == CUBLAS_STATUS_INVALID_VALUE ? "yes" : "NO",
               cublasGetStatusName(neg) + 14, ok ? "ok" : "FAIL");
        cudaFree(x); cudaFree(y);
    }
    // Throughput (4096^3).
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int N = 4096;
    struct Perf { const char *name; cudaDataType ab, c; cublasComputeType_t comp; } perf[] = {
        { "fp32 (Sgemm path)", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F },
        { "fp16 x fp16 -> fp16, fp16 compute (Hgemm)", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F },
        { "fp16 x fp16 -> fp32, fp32 compute", CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F },
        { "fp16 x fp16 -> fp16, fp32 compute", CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F },
        { "bf16 x bf16 -> bf16, fp32 compute", CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F },
    };
    void *dA, *dB, *dC;
    CK(cudaMalloc(&dA, (size_t)N * N * 4)); CK(cudaMalloc(&dB, (size_t)N * N * 4)); CK(cudaMalloc(&dC, (size_t)N * N * 4));
    CK(cudaMemset(dA, 0, (size_t)N * N * 4)); CK(cudaMemset(dB, 0, (size_t)N * N * 4)); CK(cudaMemset(dC, 0, (size_t)N * N * 4));
    for (auto &p : perf) {
        float a32 = 1, b32 = 0;
        __half a16 = __float2half(1.0f), b16 = __float2half(0.0f);
        bool h16 = p.comp == CUBLAS_COMPUTE_16F;
        auto run = [&] {
            CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, h16 ? (void *)&a16 : (void *)&a32, dA, p.ab, N, dB, p.ab, N,
                            h16 ? (void *)&b16 : (void *)&b32, dC, p.c, N, p.comp, CUBLAS_GEMM_DEFAULT));
        };
        run();
        CK(cudaEventRecord(e0));
        for (int r = 0; r < 5; r++) run();
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("GemmEx %d^3 %-44s %7.2f ms  %5.2f TFLOP/s\n", N, p.name, ms / 5, 2.0 * N * N * (double)N / (ms / 5 * 1e-3) / 1e12);
    }
    CB(cublasDestroy(h));
    printf(fails ? "cublas_mixed: FAILED (%d)\n" : "cublas_mixed: PASS\n", fails);
    return fails != 0;
}
