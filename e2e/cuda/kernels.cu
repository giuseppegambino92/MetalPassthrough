// MetalPassthrough CUDA test kernels. Compiled to PTX with NVIDIA nvcc inside a Linux container
// (tests/cuda/build-ptx.sh); the PTX is then executed on the Apple GPU through libcuda.so.1.
#include <cstdint>

extern "C" __global__ void vector_add(const float* __restrict__ a, const float* __restrict__ b,
                                      float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void saxpy(int n, float alpha, const float* x, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        y[i] = alpha * x[i] + y[i];
}

#define TILE 16
// Classic shared-memory tiled SGEMM: C[MxN] = A[MxK] * B[KxN], row-major.
extern "C" __global__ void tiled_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int ac = t * TILE + tx, br = t * TILE + ty;
        As[ty][tx] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[ty][tx] = (br < K && col < N) ? B[br * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// Block reduction: shared-memory tree, then warp shuffles; one atomicAdd per block.
extern "C" __global__ void reduce_sum(const float* in, float* out, int n) {
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    float v = 0.0f;
    for (unsigned i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) v += in[i];
    sdata[tid] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        v = sdata[tid] + sdata[tid + 32];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
        if (tid == 0) atomicAdd(out, v);
    }
}

// 5-point Jacobi stencil on an NxN grid (interior points only).
extern "C" __global__ void stencil2d(const float* in, float* out, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && y > 0 && x < nx - 1 && y < ny - 1) {
        int i = y * nx + x;
        out[i] = 0.2f * (in[i] + in[i - 1] + in[i + 1] + in[i - nx] + in[i + nx]);
    }
}

// Integer atomics + bit ops: 256-bin histogram of bytes.
extern "C" __global__ void histogram256(const unsigned char* data, unsigned int* bins, int n) {
    __shared__ unsigned int local[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) local[i] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&local[data[i]], 1u);
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x) atomicAdd(&bins[i], local[i]);
}

// Transcendentals, min/max, select, integer division.
extern "C" __global__ void mathmix(const float* x, float* y, int* q, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    y[i] = fmaxf(sinf(v) * cosf(v), -0.25f) + sqrtf(fabsf(v)) + __expf(-v * v) + rsqrtf(1.0f + v * v) + log2f(2.0f + v * v);
    q[i] = (i * 7 - 3) / 5 + (i % 3) - min(i, 100) + max(i >> 2, 3);
}

// FP64 kernel (exercises the fp64 -> fp32 lowering mode).
extern "C" __global__ void daxpy(int n, double a, const double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
