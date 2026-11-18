// CUDA Runtime API test, compiled with stock nvcc:  nvcc -O3 -cudart shared -o runtime_test runtime_test.cu
// The resulting aarch64 Linux executable embeds a fatbinary (SASS + PTX); inside a MetalPassthrough
// container libcudart.so.12 registers it and the PTX runs on the Apple GPU.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); exit(1); } } while (0)

struct Affine { float scale; float bias; int n; };

template <typename T>
__global__ void affine(const T* __restrict__ in, T* __restrict__ out, Affine p) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.n; i += gridDim.x * blockDim.x)
        out[i] = in[i] * (T)p.scale + (T)p.bias;
}

// 2D Jacobi heat diffusion step with a shared-memory tile + halo.
#define BS 16
__global__ void heat_step(const float* __restrict__ u, float* __restrict__ v, int nx, int ny, float alpha) {
    __shared__ float tile[BS + 2][BS + 2];
    int x = blockIdx.x * BS + threadIdx.x, y = blockIdx.y * BS + threadIdx.y;
    int lx = threadIdx.x + 1, ly = threadIdx.y + 1;
    auto at = [&](int xx, int yy) { xx = min(max(xx, 0), nx - 1); yy = min(max(yy, 0), ny - 1); return u[yy * nx + xx]; };
    tile[ly][lx] = at(x, y);
    if (threadIdx.x == 0) tile[ly][0] = at(x - 1, y);
    if (threadIdx.x == BS - 1) tile[ly][BS + 1] = at(x + 1, y);
    if (threadIdx.y == 0) tile[0][lx] = at(x, y - 1);
    if (threadIdx.y == BS - 1) tile[BS + 1][lx] = at(x, y + 1);
    __syncthreads();
    if (x < nx && y < ny) {
        float c = tile[ly][lx];
        v[y * nx + x] = c + alpha * (tile[ly][lx - 1] + tile[ly][lx + 1] + tile[ly - 1][lx] + tile[ly + 1][lx] - 4.0f * c);
    }
}

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    __shared__ float s[256];
    float acc = 0.f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc += a[i] * b[i];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) { if (threadIdx.x < k) s[threadIdx.x] += s[threadIdx.x + k]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out, s[0]);
}

int main() {
    int count = 0;
    CK(cudaGetDeviceCount(&count));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    printf("runtime %d / driver %d | %d device: %s, sm_%d%d, %d SMs, warp %d, %.1f GiB, managed=%d\n", rt, drv, count, prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, prop.warpSize, prop.totalGlobalMem / 1073741824.0, prop.managedMemory);
    int fails = 0;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // Template kernel + struct parameter, managed memory.
    {
        const int n = 1 << 22;
        float *in, *out;
        CK(cudaMallocManaged(&in, n * sizeof(float))); CK(cudaMallocManaged(&out, n * sizeof(float)));
        for (int i = 0; i < n; i++) in[i] = (float)(i % 1000);
        Affine p{0.5f, 3.0f, n};
        affine<float><<<256, 256>>>(in, out, p);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        int bad = 0; for (int i = 0; i < n; i++) bad += out[i] != fmaf(in[i], 0.5f, 3.0f) && out[i] != in[i] * 0.5f + 3.0f;
        printf("affine<float><<<256,256>>> (struct by value, managed): %d mismatches\n", bad); fails += bad != 0;
        CK(cudaFree(in)); CK(cudaFree(out));
    }
    // Heat diffusion: 2048^2 grid, 200 steps, ping-pong buffers, compared to a CPU run.
    {
        const int nx = 512, ny = 512, steps = 50; const float alpha = 0.2f;
        size_t bytes = (size_t)nx * ny * sizeof(float);
        float *h = (float*)malloc(bytes), *ref = (float*)malloc(bytes), *tmp = (float*)malloc(bytes);
        for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) h[y * nx + x] = ((x - nx / 2) * (x - nx / 2) + (y - ny / 2) * (y - ny / 2) < 400) ? 100.f : 0.f;
        memcpy(ref, h, bytes);
        float *du, *dv; CK(cudaMalloc(&du, bytes)); CK(cudaMalloc(&dv, bytes));
        CK(cudaMemcpy(du, h, bytes, cudaMemcpyHostToDevice));
        dim3 blk(BS, BS), grd((nx + BS - 1) / BS, (ny + BS - 1) / BS);
        CK(cudaEventRecord(e0));
        for (int s = 0; s < steps; s++) { heat_step<<<grd, blk>>>(du, dv, nx, ny, alpha); float* t = du; du = dv; dv = t; }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaMemcpy(h, du, bytes, cudaMemcpyDeviceToHost));
        for (int s = 0; s < steps; s++) {
            for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
                auto at = [&](int xx, int yy) { xx = xx < 0 ? 0 : (xx >= nx ? nx - 1 : xx); yy = yy < 0 ? 0 : (yy >= ny ? ny - 1 : yy); return ref[yy * nx + xx]; };
                float c = at(x, y);
                tmp[y * nx + x] = c + alpha * (at(x - 1, y) + at(x + 1, y) + at(x, y - 1) + at(x, y + 1) - 4.0f * c);
            }
            memcpy(ref, tmp, bytes);
        }
        double maxerr = 0; for (int i = 0; i < nx * ny; i++) maxerr = fmax(maxerr, fabs(h[i] - ref[i]));
        printf("heat_step<<<(%d,%d),(16,16)>>> x%d: %.3f ms (%.2f us/step), max |gpu-cpu| = %.2e\n", grd.x, grd.y, steps, ms, ms * 1e3 / steps, maxerr);
        fails += maxerr > 1e-3;
        cudaFree(du); cudaFree(dv); free(h); free(ref); free(tmp);
    }
    // Dot product with shared-memory reduction + float atomicAdd.
    {
        const int n = 1 << 24;
        float *a, *b, *out;
        CK(cudaMallocManaged(&a, n * 4)); CK(cudaMallocManaged(&b, n * 4)); CK(cudaMallocManaged(&out, 4));
        double ref = 0; for (int i = 0; i < n; i++) { a[i] = sinf(i * 1e-3f); b[i] = cosf(i * 1e-3f); ref += (double)a[i] * b[i]; }
        *out = 0;
        CK(cudaEventRecord(e0));
        dot_kernel<<<240, 256>>>(a, b, out, n);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("dot_kernel n=%d: %.3f ms, %.1f GB/s, result %.3f vs %.3f\n", n, ms, 8.0 * n / ms / 1e6, *out, ref);
        fails += fabs(*out - ref) > 1e-3 * fabs(ref) + 1;
    }
    printf(fails ? "runtime_test: FAILED\n" : "runtime_test: PASS\n");
    return fails;
}
