// Representative HPC application: conjugate-gradient solver for the 2D Poisson equation
// -Δu = f on an N x N grid (Dirichlet 0), matrix-free 5-point stencil SpMV + cuBLAS level-1.
//   nvcc -O3 -cudart shared -o cg_solver cg_solver.cu -lcublas
// The same algorithm runs on the CPU for comparison; the GPU solution is verified independently
// by recomputing the true residual ||f - A u|| on the CPU.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s -> %s\n", #x, cudaGetErrorString(e)); exit(2); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { printf("%s -> %s\n", #x, cublasGetStatusString(s)); exit(2); } } while (0)

__global__ void laplacian(const float *__restrict__ x, float *__restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    int k = j * n + i;
    float c = 4.0f * x[k];
    if (i > 0) c -= x[k - 1];
    if (i < n - 1) c -= x[k + 1];
    if (j > 0) c -= x[k - n];
    if (j < n - 1) c -= x[k + n];
    y[k] = c;
}

static void cpu_laplacian(const std::vector<float> &x, std::vector<float> &y, int n) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            int k = j * n + i;
            float c = 4.0f * x[k];
            if (i > 0) c -= x[k - 1];
            if (i < n - 1) c -= x[k + 1];
            if (j > 0) c -= x[k - n];
            if (j < n - 1) c -= x[k + n];
            y[k] = c;
        }
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 1024, N = n * n, maxit = argc > 2 ? atoi(argv[2]) : 2000;
    const float tol = 1e-5f;
    std::vector<float> f(N);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            float x = (i + 1.0f) / (n + 1), y = (j + 1.0f) / (n + 1);
            f[j * n + i] = (float)(2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y)) / ((n + 1.0f) * (n + 1.0f));
        }

    // ---- GPU CG ----
    cublasHandle_t h;
    CB(cublasCreate(&h));
    float *u, *r, *p, *Ap;
    CK(cudaMalloc(&u, N * 4)); CK(cudaMalloc(&r, N * 4)); CK(cudaMalloc(&p, N * 4)); CK(cudaMalloc(&Ap, N * 4));
    CK(cudaMemset(u, 0, N * 4));
    CK(cudaMemcpy(r, f.data(), N * 4, cudaMemcpyHostToDevice));
    CB(cublasScopy(h, N, r, 1, p, 1));
    float rr, fnorm;
    CB(cublasSdot(h, N, r, 1, r, 1, &rr));
    fnorm = sqrtf(rr);
    dim3 blk(16, 16), grd((n + 15) / 16, (n + 15) / 16);
    double t0 = now();
    int it = 0;
    for (; it < maxit && sqrtf(rr) > tol * fnorm; it++) {
        laplacian<<<grd, blk>>>(p, Ap, n);
        float pAp;
        CB(cublasSdot(h, N, p, 1, Ap, 1, &pAp));
        float alpha = rr / pAp, nalpha = -alpha;
        CB(cublasSaxpy(h, N, &alpha, p, 1, u, 1));
        CB(cublasSaxpy(h, N, &nalpha, Ap, 1, r, 1));
        float rr_new;
        CB(cublasSdot(h, N, r, 1, r, 1, &rr_new));
        float beta = rr_new / rr;
        rr = rr_new;
        CB(cublasSscal(h, N, &beta, p, 1));
        float one = 1.0f;
        CB(cublasSaxpy(h, N, &one, r, 1, p, 1));
    }
    CK(cudaDeviceSynchronize());
    double gpu_s = now() - t0;
    std::vector<float> ug(N);
    CK(cudaMemcpy(ug.data(), u, N * 4, cudaMemcpyDeviceToHost));

    // Independent verification against the analytic solution u = sin(pi x) sin(pi y) (the f above is
    // its exact Laplacian; discretization error is O(h^2) ~ 1e-6). A float32 residual f - A u would be
    // dominated by cancellation (f ~ 1e-5, u ~ 1), so the solution itself is checked instead.
    double err = 0, umax = 0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            double x = (i + 1.0) / (n + 1), y = (j + 1.0) / (n + 1), ex = sin(M_PI * x) * sin(M_PI * y);
            err = fmax(err, fabs(ug[j * n + i] - ex));
            umax = fmax(umax, fabs(ex));
        }
    double relerr = err / umax;

    // ---- CPU CG (same algorithm, same iteration count) ----
    std::vector<float> uc(N, 0.0f), rc(f), pc(f), Apc(N);
    double rrc = 0;
    for (float v : rc) rrc += (double)v * v;
    t0 = now();
    for (int k2 = 0; k2 < it; k2++) {
        cpu_laplacian(pc, Apc, n);
        double pAp = 0;
        for (int k = 0; k < N; k++) pAp += (double)pc[k] * Apc[k];
        float alpha = (float)(rrc / pAp);
        for (int k = 0; k < N; k++) { uc[k] += alpha * pc[k]; rc[k] -= alpha * Apc[k]; }
        double rn = 0;
        for (float v : rc) rn += (double)v * v;
        float beta = (float)(rn / rrc);
        rrc = rn;
        for (int k = 0; k < N; k++) pc[k] = rc[k] + beta * pc[k];
    }
    double cpu_s = now() - t0;
    double diff = 0, mag = 0;
    for (int k = 0; k < N; k++) { diff = fmax(diff, fabs(ug[k] - uc[k])); mag = fmax(mag, fabs(uc[k])); }

    printf("CG Poisson %dx%d (%d unknowns): %d iterations\n", n, n, N, it);
    printf("  GPU: %.3f s (%.2f ms/iteration, 6 kernels + 2 host-synchronized reductions each)\n", gpu_s, gpu_s / it * 1e3);
    printf("  CPU (same algorithm, 1 thread): %.3f s  -> speedup %.1fx\n", cpu_s, cpu_s / gpu_s);
    printf("  max error vs analytic solution sin(pi x) sin(pi y): %.2e (relative)\n", relerr);
    printf("  max |u_gpu - u_cpu| / max|u| = %.2e\n", diff / mag);
    bool ok = it < maxit && relerr < 1e-3 && diff / mag < 1e-3;
    CB(cublasDestroy(h));
    printf(ok ? "cg_solver: PASS\n" : "cg_solver: FAILED\n");
    return !ok;
}
