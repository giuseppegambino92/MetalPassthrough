// cuBLAS SGEMM: correctness (transposes, leading dimensions, strided batches) and throughput.
//   nvcc -O3 -cudart shared -o cublas_gemm cublas_gemm.cu -lcublas
// Unmodified cuBLAS code; on MetalPassthrough cublasSgemm runs Apple's MPSMatrixMultiplication on
// the container's own memory.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s -> %s\n", #x, cudaGetErrorString(e)); exit(2); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { printf("%s -> %s\n", #x, cublasGetStatusString(s)); exit(2); } } while (0)

static float frand(unsigned &s) { s = s * 1664525u + 1013904223u; return (float)(s >> 9) / (float)(1u << 23) - 0.5f; }

// Column-major reference: C = alpha op(A) op(B) + beta C
static void ref_gemm(bool ta, bool tb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                     double *C, int ldc) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            double acc = 0;
            for (int p = 0; p < k; p++) {
                double a = ta ? A[p + (size_t)i * lda] : A[i + (size_t)p * lda];
                double b = tb ? B[j + (size_t)p * ldb] : B[p + (size_t)j * ldb];
                acc += a * b;
            }
            C[i + (size_t)j * ldc] = alpha * acc + beta * C[i + (size_t)j * ldc];
        }
}

int main() {
    cublasHandle_t h;
    CB(cublasCreate(&h));
    int fails = 0;
    unsigned seed = 7;

    // Correctness: odd shapes, every transpose combination, padded leading dimensions.
    const int m = 67, n = 45, k = 93;
    for (int ta = 0; ta < 2; ta++)
        for (int tb = 0; tb < 2; tb++) {
            int lda = (ta ? k : m) + 3, ldb = (tb ? n : k) + 5, ldc = m + 2;
            std::vector<float> A((size_t)lda * (ta ? m : k)), B((size_t)ldb * (tb ? k : n)), C((size_t)ldc * n);
            for (auto &v : A) v = frand(seed);
            for (auto &v : B) v = frand(seed);
            for (auto &v : C) v = frand(seed);
            std::vector<double> R(C.begin(), C.end());
            float alpha = 1.25f, beta = -0.5f;
            ref_gemm(ta, tb, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta, R.data(), ldc);
            float *dA, *dB, *dC;
            CK(cudaMalloc(&dA, A.size() * 4)); CK(cudaMalloc(&dB, B.size() * 4)); CK(cudaMalloc(&dC, C.size() * 4));
            CK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice));
            CB(cublasSgemm(h, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
            CK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
            double worst = 0;
            for (int j = 0; j < n; j++)
                for (int i = 0; i < m; i++) worst = fmax(worst, fabs(C[i + (size_t)j * ldc] - R[i + (size_t)j * ldc]));
            bool ok = worst < 1e-4;
            fails += !ok;
            printf("sgemm %c%c %dx%dx%d (lda %d ldb %d ldc %d): max |err| %.2e %s\n", ta ? 'T' : 'N', tb ? 'T' : 'N', m, n, k, lda, ldb, ldc,
                   worst, ok ? "ok" : "FAIL");
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }

    // Strided batched.
    {
        const int bm = 32, bn = 48, bk = 40, batch = 9;
        std::vector<float> A((size_t)bm * bk * batch), B((size_t)bk * bn * batch), C((size_t)bm * bn * batch, 0.0f);
        for (auto &v : A) v = frand(seed);
        for (auto &v : B) v = frand(seed);
        float *dA, *dB, *dC, one = 1.0f, zero = 0.0f;
        CK(cudaMalloc(&dA, A.size() * 4)); CK(cudaMalloc(&dB, B.size() * 4)); CK(cudaMalloc(&dC, C.size() * 4));
        CK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
        CB(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, bm, bn, bk, &one, dA, bm, (long long)bm * bk, dB, bk, (long long)bk * bn, &zero,
                                     dC, bm, (long long)bm * bn, batch));
        CK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int b = 0; b < batch; b++) {
            std::vector<double> R((size_t)bm * bn, 0.0);
            ref_gemm(false, false, bm, bn, bk, 1.0f, &A[(size_t)b * bm * bk], bm, &B[(size_t)b * bk * bn], bk, 0.0f, R.data(), bm);
            for (size_t i = 0; i < R.size(); i++) worst = fmax(worst, fabs(C[(size_t)b * bm * bn + i] - R[i]));
        }
        bool ok = worst < 1e-4;
        fails += !ok;
        printf("sgemmStridedBatched %d x (%dx%dx%d): max |err| %.2e %s\n", batch, bm, bn, bk, worst, ok ? "ok" : "FAIL");
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    // Throughput.
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int N : {512, 1024, 2048, 4096, 8192}) {
        size_t bytes = (size_t)N * N * 4;
        float *dA, *dB, *dC, one = 1.0f, zero = 0.0f;
        if (cudaMalloc(&dA, bytes) || cudaMalloc(&dB, bytes) || cudaMalloc(&dC, bytes)) { printf("N=%d: out of memory, skipped\n", N); cudaGetLastError(); break; }
        CK(cudaMemset(dA, 0, bytes)); CK(cudaMemset(dB, 0, bytes));
        CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dA, N, dB, N, &zero, dC, N));  // warm-up
        int reps = N >= 4096 ? 3 : 10;
        CK(cudaEventRecord(e0));
        for (int r = 0; r < reps; r++) CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dA, N, dB, N, &zero, dC, N));
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("cublasSgemm %5d^3: %8.3f ms  %6.2f TFLOP/s\n", N, ms / reps, 2.0 * N * N * N / (ms / reps * 1e-3) / 1e12);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    CB(cublasDestroy(h));
    printf(fails ? "cublas_gemm: FAILED\n" : "cublas_gemm: PASS\n");
    return fails != 0;
}
