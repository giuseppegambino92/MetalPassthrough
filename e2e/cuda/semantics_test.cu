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

    printf("[1] independent streams run concurrently (low-occupancy kernels: 1 threadgroup each)\n");
    {
        float *buf;
        CK(cudaMalloc(&buf, 8 * 1024 * sizeof(float)));
        double one = 0;
        {
            cudaStream_t s;
            CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            spin_then_write<<<1, 256, 0, s>>>(buf, iters, 1.0f);  // warm
            CK(cudaStreamSynchronize(s));
            double t0 = now_ms();
            spin_then_write<<<1, 256, 0, s>>>(buf, iters, 1.0f);
            CK(cudaStreamSynchronize(s));
            one = now_ms() - t0;
            CK(cudaStreamDestroy(s));
        }
        printf("  one kernel alone: %.1f ms\n", one);
        struct { int n; unsigned flags; const char *what; } cases[] = {
            { 2, cudaStreamNonBlocking, "2 non-blocking streams" },
            { 4, cudaStreamNonBlocking, "4 non-blocking streams" },
            { 2, cudaStreamDefault, "2 blocking streams (legacy default-stream rules, no stream-0 work)" },
        };
        for (auto &c : cases) {
            cudaStream_t st[4];
            cudaEvent_t t0e, done[4];
            for (int i = 0; i < c.n; i++) { CK(cudaStreamCreateWithFlags(&st[i], c.flags)); CK(cudaEventCreate(&done[i])); }
            CK(cudaEventCreate(&t0e));
            CK(cudaEventRecord(t0e, st[0]));
            double t0 = now_ms();
            for (int i = 0; i < c.n; i++) {
                spin_then_write<<<1, 256, 0, st[i]>>>(buf + 1024 * i, iters, (float)i);
                CK(cudaEventRecord(done[i], st[i]));
            }
            for (int i = 0; i < c.n; i++) CK(cudaStreamSynchronize(st[i]));
            double wall = now_ms() - t0;
            // Timeline: completion time of each stream's kernel relative to the first launch.
            printf("  %s: wall %.1f ms (serial ~%.1f ms); stream completion timeline:", c.what, wall, c.n * one);
            for (int i = 0; i < c.n; i++) {
                float ms;
                CK(cudaEventElapsedTime(&ms, t0e, done[i]));
                printf(" s%d@%.1f", i + 1, ms);
            }
            printf(" ms\n");
            CHECK(wall < 1.5 * one, "%s overlap: %.2fx the time of one kernel (1.0 = fully concurrent, %d.0 = serialized)", c.what, wall / one, c.n);
            for (int i = 0; i < c.n; i++) { CK(cudaStreamDestroy(st[i])); CK(cudaEventDestroy(done[i])); }
            CK(cudaEventDestroy(t0e));
        }
        {   // The original failure mode: launch, launch, synchronize, with no event in between (an event
            // record ends the stream's batch, which would hide a host that only commits on sync).
            cudaStream_t a1, a2;
            CK(cudaStreamCreateWithFlags(&a1, cudaStreamNonBlocking));
            CK(cudaStreamCreateWithFlags(&a2, cudaStreamNonBlocking));
            double t0 = now_ms();
            spin_then_write<<<1, 256, 0, a1>>>(buf, iters, 1.0f);
            spin_then_write<<<1, 256, 0, a2>>>(buf + 1024, iters, 2.0f);
            CK(cudaStreamSynchronize(a1));
            CK(cudaStreamSynchronize(a2));
            double wall = now_ms() - t0;
            CHECK(wall < 1.5 * one, "2 streams, launch-launch-sync without events: %.2fx the time of one kernel", wall / one);
            CK(cudaStreamDestroy(a1));
            CK(cudaStreamDestroy(a2));
        }
        // GPU-filling kernels cannot overlap (no free cores): informational only.
        float *big;
        const int n = 1 << 24;
        CK(cudaMalloc(&big, 2 * (size_t)n * sizeof(float)));
        cudaStream_t s1, s2;
        CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        add_one<<<n / 256, 256, 0, s1>>>(big, n);
        CK(cudaStreamSynchronize(s1));
        double t0 = now_ms();
        for (int r = 0; r < 20; r++) add_one<<<n / 256, 256, 0, s1>>>(big, n);
        CK(cudaStreamSynchronize(s1));
        double a = now_ms() - t0;
        t0 = now_ms();
        for (int r = 0; r < 20; r++) { add_one<<<n / 256, 256, 0, s1>>>(big, n); add_one<<<n / 256, 256, 0, s2>>>(big + n, n); }
        CK(cudaStreamSynchronize(s1));
        CK(cudaStreamSynchronize(s2));
        double b = now_ms() - t0;
        printf("  info: bandwidth-bound kernels (whole GPU): 1 stream %.1f ms, 2 streams x same work %.1f ms (%.2fx; ~2.0 expected when the GPU is saturated)\n", a, b, b / a);
        CK(cudaStreamDestroy(s1));
        CK(cudaStreamDestroy(s2));
        CK(cudaFree(big));
        CK(cudaFree(buf));
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
