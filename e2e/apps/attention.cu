// Transformer attention forward pass (FlashAttention-style: K/V tiles in shared memory, online
// softmax, O(S) memory), single precision, verified against a float64 CPU reference.
//   nvcc -O3 -cudart shared -o attention attention.cu
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s -> %s\n", #x, cudaGetErrorString(e)); exit(2); } } while (0)

constexpr int D = 64;    // head dimension
constexpr int BQ = 64;   // queries per block (one thread each)
constexpr int BK = 32;   // keys per shared-memory tile

// Q, K, V, O: [B, H, S, D] contiguous. grid = (S / BQ, H, B), block = BQ threads.
__global__ void flash_attention(const float *__restrict__ Q, const float *__restrict__ K, const float *__restrict__ V,
                                float *__restrict__ O, int S, float scale) {
    __shared__ float Ks[BK][D];
    __shared__ float Vs[BK][D];
    const size_t head = ((size_t)blockIdx.z * gridDim.y + blockIdx.y) * (size_t)S * D;
    const int qi = blockIdx.x * BQ + threadIdx.x;
    float q[D], acc[D];
    for (int d = 0; d < D; d++) { q[d] = Q[head + (size_t)qi * D + d] * scale; acc[d] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int k0 = 0; k0 < S; k0 += BK) {
        for (int t = threadIdx.x; t < BK * D; t += BQ) {
            Ks[t / D][t % D] = K[head + (size_t)(k0 + t / D) * D + t % D];
            Vs[t / D][t % D] = V[head + (size_t)(k0 + t / D) * D + t % D];
        }
        __syncthreads();
        for (int j = 0; j < BK; j++) {
            float s = 0.0f;
            for (int d = 0; d < D; d++) s = fmaf(q[d], Ks[j][d], s);
            float mn = fmaxf(m, s);
            float corr = __expf(m - mn), p = __expf(s - mn);
            l = l * corr + p;
            for (int d = 0; d < D; d++) acc[d] = fmaf(p, Vs[j][d], acc[d] * corr);
            m = mn;
        }
        __syncthreads();
    }
    for (int d = 0; d < D; d++) O[head + (size_t)qi * D + d] = acc[d] / l;
}

int main(int argc, char **argv) {
    const int B = 2, H = 8, S = argc > 1 ? atoi(argv[1]) : 1024;
    const size_t n = (size_t)B * H * S * D;
    std::vector<float> q(n), k(n), v(n), o(n);
    unsigned s = 1;
    auto rnd = [&] { s = s * 1664525u + 1013904223u; return (float)(s >> 9) / (float)(1u << 23) - 0.5f; };
    for (size_t i = 0; i < n; i++) { q[i] = rnd(); k[i] = rnd(); v[i] = rnd(); }
    float *dq, *dk, *dv, *dout;
    CK(cudaMalloc(&dq, n * 4)); CK(cudaMalloc(&dk, n * 4)); CK(cudaMalloc(&dv, n * 4)); CK(cudaMalloc(&dout, n * 4));
    CK(cudaMemcpy(dq, q.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk, k.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dv, v.data(), n * 4, cudaMemcpyHostToDevice));
    const float scale = 1.0f / sqrtf((float)D);
    dim3 grid(S / BQ, H, B);
    flash_attention<<<grid, BQ>>>(dq, dk, dv, dout, S, scale);  // warm-up + JIT
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int reps = 5;
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++) flash_attention<<<grid, BQ>>>(dq, dk, dv, dout, S, scale);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= reps;
    CK(cudaMemcpy(o.data(), dout, n * 4, cudaMemcpyDeviceToHost));

    // float64 reference (exact softmax) on a sample of heads and all queries.
    double worst = 0;
    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> sc(S);
    for (int bh = 0; bh < B * H; bh += 3) {
        const size_t hd = (size_t)bh * S * D;
        for (int i = 0; i < S; i++) {
            double mx = -1e300;
            for (int j = 0; j < S; j++) {
                double d0 = 0;
                for (int d = 0; d < D; d++) d0 += (double)q[hd + (size_t)i * D + d] * k[hd + (size_t)j * D + d];
                sc[j] = d0 * scale;
                mx = fmax(mx, sc[j]);
            }
            double den = 0;
            for (int j = 0; j < S; j++) { sc[j] = exp(sc[j] - mx); den += sc[j]; }
            for (int d = 0; d < D; d++) {
                double acc = 0;
                for (int j = 0; j < S; j++) acc += sc[j] * v[hd + (size_t)j * D + d];
                worst = fmax(worst, fabs(acc / den - o[hd + (size_t)i * D + d]));
            }
        }
    }
    double cpu_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    int heads_checked = (B * H + 2) / 3;
    double flops = 4.0 * B * H * (double)S * S * D;
    printf("attention B=%d H=%d S=%d D=%d: GPU %.3f ms (%.1f GFLOP/s)\n", B, H, S, D, ms, flops / (ms * 1e-3) / 1e9);
    printf("  float64 CPU reference on %d/%d heads: %.2f s (%.1fx slower per head than the GPU on all heads)\n", heads_checked, B * H, cpu_s,
           (cpu_s / heads_checked) / (ms * 1e-3 / (B * H)));
    printf("  max |gpu - reference| = %.2e\n", worst);
    bool ok = worst < 2e-5;
    printf(ok ? "attention: PASS\n" : "attention: FAILED\n");
    return !ok;
}
