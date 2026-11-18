// CUDA stream / event / async-copy / bounds semantics, compiled with stock nvcc:
//   nvcc -O3 -cudart shared -o semantics_test semantics_test.cu -lcuda
// Every check would also pass on an NVIDIA GPU; on MetalPassthrough it exercises real Metal
// queue concurrency, MTLSharedEvent dependencies and range validation.
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } else { printf("  ok: " __VA_ARGS__); printf("\n"); } } while (0)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(2); } } while (0)

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One threadgroup that burns a configurable amount of ALU time, then writes `value`.
__global__ void spin_then_write(float* out, int iters, float value) {
    float v = threadIdx.x * 1e-7f;
    for (int i = 0; i < iters; i++) v = fmaf(v, 0.9999993f, 1e-7f);
    __syncthreads();
    if (threadIdx.x == 0) out[0] = value + (v > 1e30f ? 1.0f : 0.0f);
}

__global__ void copy_scaled(const float* in, float* out, float k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * k;
}

__global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s\n", p.name);
    float *a, *b;
    CK(cudaMalloc(&a, 1024 * sizeof(float)));
    CK(cudaMalloc(&b, 1024 * sizeof(float)));

    // Calibrate a ~40 ms single-threadgroup kernel.
    int iters = 1 << 20;
    for (;;) {
        double t0 = now_ms();
        spin_then_write<<<1, 256>>>(a, iters, 1.0f);
        CK(cudaDeviceSynchronize());
        double dt = now_ms() - t0;
        if (dt > 30 || iters > (1 << 28)) { printf("calibrated: %d iterations = %.1f ms\n", iters, dt); break; }
        iters *= 2;
    }

    printf("[1] two non-blocking streams (independent Metal queues)\n");
    {
        cudaStream_t s1, s2;
        CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        double t0 = now_ms();
        spin_then_write<<<1, 256, 0, s1>>>(a, iters, 1.0f);
        CK(cudaStreamSynchronize(s1));
        double one = now_ms() - t0;
        t0 = now_ms();
        spin_then_write<<<1, 256, 0, s1>>>(a, iters, 1.0f);
        spin_then_write<<<1, 256, 0, s2>>>(b, iters, 2.0f);
        CK(cudaStreamSynchronize(s1));
        CK(cudaStreamSynchronize(s2));
        double both = now_ms() - t0;
        printf("  one kernel %.1f ms, two kernels on two streams %.1f ms (serial would be ~%.1f ms)\n", one, both, 2 * one);
        // Informational: each stream is its own Metal command queue, but whether the GPU overlaps
        // command buffers from different queues is up to the Metal driver. On M3 Max / macOS 27.2
        // native Metal serializes this pattern too (see bench/concurrency), so this is measured,
        // not asserted. Correctness of independent ordering is asserted in [3]-[5].
        printf("  info: overlap factor %.2fx of a single kernel (1.0 = full overlap, 2.0 = serialized)\n", both / one);
        CK(cudaStreamDestroy(s1));
        CK(cudaStreamDestroy(s2));
    }

    printf("[2] cudaStreamQuery never blocks\n");
    {
        cudaStream_t s;
        CK(cudaStreamCreate(&s));
        spin_then_write<<<1, 256, 0, s>>>(a, iters, 3.0f);
        double t0 = now_ms();
        cudaError_t q = cudaStreamQuery(s);
        double dt = now_ms() - t0;
        CHECK(q == cudaErrorNotReady, "query on a busy stream returns cudaErrorNotReady (%s)", cudaGetErrorName(q));
        CHECK(dt < 5.0, "query returned in %.3f ms", dt);
        CK(cudaStreamSynchronize(s));
        CHECK(cudaStreamQuery(s) == cudaSuccess, "query after sync returns cudaSuccess");
        CHECK(cudaGetLastError() == cudaSuccess, "NotReady is not a sticky error");
        CK(cudaStreamDestroy(s));
    }

    printf("[3] cudaStreamWaitEvent orders work across streams\n");
    {
        cudaStream_t producer, consumer;
        cudaEvent_t ready;
        CK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
        CK(cudaEventCreate(&ready));
        CK(cudaMemset(a, 0, 1024 * sizeof(float)));
        CK(cudaDeviceSynchronize());
        spin_then_write<<<1, 256, 0, producer>>>(a, iters, 42.0f);  // slow producer writes a[0] = 42
        CK(cudaEventRecord(ready, producer));
        CK(cudaStreamWaitEvent(consumer, ready, 0));
        copy_scaled<<<4, 256, 0, consumer>>>(a, b, 2.0f, 1024);    // must observe 42
        float r = 0;
        CK(cudaStreamSynchronize(consumer));
        CK(cudaMemcpy(&r, b, sizeof r, cudaMemcpyDeviceToHost));
        CHECK(r == 84.0f, "consumer saw the producer's result (b[0] = %g, expected 84)", r);
        CHECK(cudaEventQuery(ready) == cudaSuccess, "event completed");
        CK(cudaEventDestroy(ready));
        CK(cudaStreamDestroy(producer));
        CK(cudaStreamDestroy(consumer));
    }

    printf("[4] legacy default stream synchronizes with blocking streams\n");
    {
        cudaStream_t s;
        CK(cudaStreamCreate(&s));  // blocking
        CK(cudaMemset(a, 0, 1024 * sizeof(float)));
        spin_then_write<<<1, 256, 0, s>>>(a, iters, 7.0f);  // slow write on blocking stream
        copy_scaled<<<4, 256>>>(a, b, 3.0f, 1024);          // NULL stream must wait for it
        float r = 0;
        CK(cudaMemcpy(&r, b, sizeof r, cudaMemcpyDeviceToHost));
        CHECK(r == 21.0f, "NULL-stream kernel saw the blocking stream's write (b[0] = %g, expected 21)", r);
        CK(cudaStreamDestroy(s));
    }

    printf("[5] async copies are stream ordered (pinned and pageable)\n");
    {
        const int n = 1 << 20;
        cudaStream_t s;
        CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        float *d, *pinned;
        CK(cudaMalloc(&d, n * sizeof(float)));
        CK(cudaMallocHost(&pinned, n * sizeof(float)));
        std::vector<float> pageable(n), back(n);
        for (int i = 0; i < n; i++) { pageable[i] = (float)i; pinned[i] = (float)(2 * i); }
        CK(cudaMemcpyAsync(d, pageable.data(), n * sizeof(float), cudaMemcpyHostToDevice, s));
        for (int i = 0; i < n; i++) pageable[i] = -1.0f;  // staged: overwriting the source is safe
        add_one<<<(n + 255) / 256, 256, 0, s>>>(d, n);
        CK(cudaMemcpyAsync(back.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        int bad = 0;
        for (int i = 0; i < n; i++) bad += back[i] != (float)i + 1.0f;
        CHECK(bad == 0, "pageable HtoD -> kernel -> DtoH in order (%d mismatches)", bad);
        CK(cudaMemcpyAsync(d, pinned, n * sizeof(float), cudaMemcpyHostToDevice, s));
        add_one<<<(n + 255) / 256, 256, 0, s>>>(d, n);
        CK(cudaMemcpyAsync(pinned, d, n * sizeof(float), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        bad = 0;
        for (int i = 0; i < n; i++) bad += pinned[i] != (float)(2 * i) + 1.0f;
        CHECK(bad == 0, "pinned HtoD -> kernel -> DtoH entirely on the GPU (%d mismatches)", bad);
        CK(cudaFree(d));
        CK(cudaFreeHost(pinned));
        CK(cudaStreamDestroy(s));
    }

    printf("[6] memory range validation\n");
    {
        const size_t sz = 1 << 20;
        char *d;
        CK(cudaMalloc(&d, sz));
        std::vector<char> h(sz + 4096, 1);
        CHECK(cudaMemcpy(d, h.data(), sz, cudaMemcpyHostToDevice) == cudaSuccess, "full-size copy accepted");
        CHECK(cudaMemcpy(d + sz - 16, h.data(), 16, cudaMemcpyHostToDevice) == cudaSuccess, "copy ending exactly at the end accepted");
        CHECK(cudaMemcpy(d + sz - 16, h.data(), 17, cudaMemcpyHostToDevice) == cudaErrorInvalidValue, "copy 1 byte past the end rejected");
        CHECK(cudaMemcpy(d, h.data(), sz + 4096, cudaMemcpyHostToDevice) == cudaErrorInvalidValue, "oversized copy rejected");
        CHECK(cudaMemcpy(h.data(), d + 8, SIZE_MAX - 4, cudaMemcpyDeviceToHost) == cudaErrorInvalidValue, "wrapping size rejected");
        CHECK(cudaMemset(d + sz - 8, 0, 9) == cudaErrorInvalidValue, "memset past the end rejected");
        CHECK(cudaMemcpy(d, d + sz / 2, sz / 2 + 1, cudaMemcpyDeviceToDevice) == cudaErrorInvalidValue, "DtoD past the end rejected");
        CUdeviceptr base = 0;
        size_t bsz = 0;
        CHECK(cuMemGetAddressRange(&base, &bsz, (CUdeviceptr)(d + 12345)) == CUDA_SUCCESS && base == (CUdeviceptr)d && bsz >= sz,
              "cuMemGetAddressRange resolves an interior pointer to its allocation");
        CHECK(cudaFree(d + 256) != cudaSuccess, "freeing an interior pointer is rejected");
        cudaGetLastError();
        CK(cudaFree(d));
    }

    CK(cudaFree(a));
    CK(cudaFree(b));
    printf(fails ? "semantics_test: FAILED (%d)\n" : "semantics_test: PASS\n", fails);
    return fails != 0;
}
