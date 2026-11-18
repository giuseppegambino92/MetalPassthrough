/*
 * Test 3: a plain CUDA Driver API program (builds and runs unmodified on NVIDIA hardware).
 * Built against NVIDIA's cuda.h and linked against NVIDIA's libcuda stub; inside a MetalPassthrough
 * container the dynamic loader picks MetalPassthrough's libcuda.so.1 and the nvcc-generated PTX runs
 * on the Apple GPU.
 *   gcc -O2 driver_test.c -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda -lm
 */
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(x) do { CUresult _r = (x); if (_r != CUDA_SUCCESS) { const char *n = 0, *s = 0; cuGetErrorName(_r, &n); cuGetErrorString(_r, &s); \
    fprintf(stderr, "%s:%d: %s -> %s (%s)\n", __FILE__, __LINE__, #x, n ? n : "?", s ? s : "?"); exit(1); } } while (0)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static unsigned long long s_rng = 88172645463325252ull;
static float frand(void) { s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17; return (float)((s_rng >> 40) & 0xffff) / 65536.0f - 0.5f; }

static char *read_file(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = calloc(1, n + 1);
    if (fread(b, 1, n, f) != (size_t)n) exit(1);
    fclose(f);
    return b;
}

static int failures;
#define EXPECT(cond, ...) do { if (!(cond)) { failures++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

int main(int argc, char **argv) {
    const char *ptx_path = argc > 1 ? argv[1] : "kernels.ptx";
    int N = argc > 2 ? atoi(argv[2]) : 1024;
    CK(cuInit(0));
    int ver, count; CUdevice dev; char name[256]; size_t total;
    CK(cuDriverGetVersion(&ver)); CK(cuDeviceGetCount(&count)); CK(cuDeviceGet(&dev, 0));
    CK(cuDeviceGetName(name, sizeof name, dev)); CK(cuDeviceTotalMem(&total, dev));
    int sms, warp, maxt, cc_major, cc_minor, uva, managed;
    CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    CK(cuDeviceGetAttribute(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev));
    CK(cuDeviceGetAttribute(&maxt, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev));
    CK(cuDeviceGetAttribute(&cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CK(cuDeviceGetAttribute(&cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    CK(cuDeviceGetAttribute(&uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, dev));
    CK(cuDeviceGetAttribute(&managed, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, dev));
    printf("CUDA driver %d.%d | %d device(s): \"%s\"\n", ver / 1000, (ver % 1000) / 10, count, name);
    printf("  %d SMs (GPU cores), warp %d, %d threads/block, sm_%d%d PTX level, %.1f GiB allocatable, UVA=%d managed=%d\n",
           sms, warp, maxt, cc_major, cc_minor, total / 1073741824.0, uva, managed);

    CUcontext ctx; CK(cuCtxCreate(&ctx, 0, dev));
    char *ptx = read_file(ptx_path);
    CUmodule mod;
    double t0 = now();
    CK(cuModuleLoadData(&mod, ptx));
    printf("cuModuleLoadData: nvcc PTX (%zu bytes) translated to Metal + compiled in %.1f ms\n", strlen(ptx), (now() - t0) * 1e3);
    CUevent e0, e1; CK(cuEventCreate(&e0, 0)); CK(cuEventCreate(&e1, 0));
    float ms;

    /* ---- vector_add ---- */
    {
        int n = 1 << 24; size_t bytes = (size_t)n * 4;
        float *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
        for (int i = 0; i < n; i++) { a[i] = frand(); b[i] = frand(); }
        CUdeviceptr da, db, dc; CK(cuMemAlloc(&da, bytes)); CK(cuMemAlloc(&db, bytes)); CK(cuMemAlloc(&dc, bytes));
        CK(cuMemcpyHtoD(da, a, bytes)); CK(cuMemcpyHtoD(db, b, bytes));
        CUfunction f; CK(cuModuleGetFunction(&f, mod, "vector_add"));
        void *args[] = { &da, &db, &dc, &n };
        CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, 0)); /* warm-up */
        CK(cuEventRecord(e0, 0));
        CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, 0));
        CK(cuEventRecord(e1, 0)); CK(cuEventSynchronize(e1)); CK(cuEventElapsedTime(&ms, e0, e1));
        CK(cuMemcpyDtoH(c, dc, bytes));
        int bad = 0; for (int i = 0; i < n; i++) bad += c[i] != a[i] + b[i];
        printf("vector_add   n=%d: %.3f ms, %.1f GB/s, mismatches %d\n", n, ms, 3.0 * bytes / ms / 1e6, bad);
        EXPECT(bad == 0, "vector_add");
        cuMemFree(da); cuMemFree(db); cuMemFree(dc); free(a); free(b); free(c);
    }

    /* ---- tiled_sgemm (shared memory + __syncthreads) ---- */
    {
        int M = N, Nn = N, K = N;
        size_t sa = (size_t)M * K * 4, sb = (size_t)K * Nn * 4, sc = (size_t)M * Nn * 4;
        float *A = malloc(sa), *B = malloc(sb), *C = malloc(sc);
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = frand();
        for (size_t i = 0; i < (size_t)K * Nn; i++) B[i] = frand();
        CUdeviceptr dA, dB, dC; CK(cuMemAlloc(&dA, sa)); CK(cuMemAlloc(&dB, sb)); CK(cuMemAlloc(&dC, sc));
        CK(cuMemcpyHtoD(dA, A, sa)); CK(cuMemcpyHtoD(dB, B, sb));
        CUfunction f; CK(cuModuleGetFunction(&f, mod, "tiled_sgemm"));
        void *args[] = { &dA, &dB, &dC, &M, &Nn, &K };
        CK(cuLaunchKernel(f, (Nn + 15) / 16, (M + 15) / 16, 1, 16, 16, 1, 0, 0, args, 0));
        CK(cuEventRecord(e0, 0));
        CK(cuLaunchKernel(f, (Nn + 15) / 16, (M + 15) / 16, 1, 16, 16, 1, 0, 0, args, 0));
        CK(cuEventRecord(e1, 0)); CK(cuEventSynchronize(e1)); CK(cuEventElapsedTime(&ms, e0, e1));
        CK(cuMemcpyDtoH(C, dC, sc));
        long bad = 0;
        for (int i = 0; i < M; i += (M > 512 ? 7 : 1))  /* sampled rows for large N; every column */
            for (int j = 0; j < Nn; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) acc = fmaf(A[(size_t)i * K + k], B[(size_t)k * Nn + j], acc);
                bad += memcmp(&acc, &C[(size_t)i * Nn + j], 4) != 0;
            }
        printf("tiled_sgemm  %dx%dx%d: %.3f ms, %.1f GFLOP/s, bit-exact mismatches %ld\n", M, Nn, K, ms, 2.0 * M * Nn * K / ms / 1e6, bad);
        EXPECT(bad == 0, "tiled_sgemm");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); free(A); free(B); free(C);
    }

    /* ---- reduce_sum (dynamic shared memory, warp shuffles, float atomics) ---- */
    {
        int n = 1 << 24; size_t bytes = (size_t)n * 4;
        float *x = malloc(bytes); double ref = 0;
        for (int i = 0; i < n; i++) { x[i] = frand(); ref += x[i]; }
        CUdeviceptr dx, dout; CK(cuMemAlloc(&dx, bytes)); CK(cuMemAlloc(&dout, 4));
        CK(cuMemcpyHtoD(dx, x, bytes)); CK(cuMemsetD32(dout, 0, 1));
        CUfunction f; CK(cuModuleGetFunction(&f, mod, "reduce_sum"));
        void *args[] = { &dx, &dout, &n };
        CK(cuEventRecord(e0, 0));
        CK(cuLaunchKernel(f, 240, 1, 1, 256, 1, 1, 256 * 4, 0, args, 0));
        CK(cuEventRecord(e1, 0)); CK(cuEventSynchronize(e1)); CK(cuEventElapsedTime(&ms, e0, e1));
        float got; CK(cuMemcpyDtoH(&got, dout, 4));
        printf("reduce_sum   n=%d: %.3f ms, %.1f GB/s, sum %.4f vs %.4f (fp32 vs fp64 ref)\n", n, ms, bytes / ms / 1e6, got, ref);
        EXPECT(fabs(got - ref) < 1e-3 * (fabs(ref) + 100), "reduce_sum");
        cuMemFree(dx); cuMemFree(dout); free(x);
    }

    /* ---- histogram256 (shared + global integer atomics) ---- */
    {
        int n = 1 << 26;
        unsigned char *d = malloc(n); unsigned ref[256] = { 0 }, got[256];
        for (int i = 0; i < n; i++) { d[i] = (unsigned char)(frand() * 200 + 100); ref[d[i]]++; }
        CUdeviceptr dd, dh; CK(cuMemAlloc(&dd, n)); CK(cuMemAlloc(&dh, 1024));
        CK(cuMemcpyHtoD(dd, d, n)); CK(cuMemsetD8(dh, 0, 1024));
        CUfunction f; CK(cuModuleGetFunction(&f, mod, "histogram256"));
        void *args[] = { &dd, &dh, &n };
        CK(cuEventRecord(e0, 0));
        CK(cuLaunchKernel(f, 480, 1, 1, 256, 1, 1, 0, 0, args, 0));
        CK(cuEventRecord(e1, 0)); CK(cuEventSynchronize(e1)); CK(cuEventElapsedTime(&ms, e0, e1));
        CK(cuMemcpyDtoH(got, dh, 1024));
        printf("histogram256 n=%d: %.3f ms, %.1f GB/s, exact=%s\n", n, ms, n / ms / 1e6, memcmp(got, ref, 1024) ? "NO" : "yes");
        EXPECT(!memcmp(got, ref, 1024), "histogram");
        cuMemFree(dd); cuMemFree(dh); free(d);
    }

    /* ---- daxpy: FP64 emulation (df64) ---- */
    {
        int n = 1 << 22; size_t bytes = (size_t)n * 8; double a = 1.0 / 3.0;
        double *x = malloc(bytes), *y = malloc(bytes), *r = malloc(bytes);
        for (int i = 0; i < n; i++) { x[i] = i * 1.0000001 + 0.25; y[i] = (n - i) * 0.5; }
        CUdeviceptr dx, dy; CK(cuMemAlloc(&dx, bytes)); CK(cuMemAlloc(&dy, bytes));
        CK(cuMemcpyHtoD(dx, x, bytes)); CK(cuMemcpyHtoD(dy, y, bytes));
        CUfunction f; CK(cuModuleGetFunction(&f, mod, "daxpy"));
        void *args[] = { &n, &a, &dx, &dy };
        CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, 0));
        CK(cuCtxSynchronize());
        CK(cuMemcpyDtoH(r, dy, bytes));
        double worst = 0;
        for (int i = 0; i < n; i++) { double w = fma(a, x[i], y[i]); double e = fabs(r[i] - w) / fabs(w); if (e > worst) worst = e; }
        printf("daxpy (f64)  n=%d: worst relative error %.2e (df64 emulation; Apple GPUs have no FP64 ALU)\n", n, worst);
        EXPECT(worst < 1e-12, "daxpy precision");
        cuMemFree(dx); cuMemFree(dy); free(x); free(y); free(r);
    }

    /* ---- managed memory: same pointer on CPU and GPU ---- */
    if (managed) {
        int n = 1 << 20; CUdeviceptr m;
        CK(cuMemAllocManaged(&m, (size_t)n * 4 * 3, CU_MEM_ATTACH_GLOBAL));
        float *p = (float *)(uintptr_t)m;  /* CPU writes straight through the device pointer */
        for (int i = 0; i < n; i++) { p[i] = (float)i; p[n + i] = 2.0f * i; }
        CUdeviceptr pa = m, pb = m + (size_t)n * 4, pc = m + (size_t)n * 8;
        CUfunction f; CK(cuModuleGetFunction(&f, mod, "vector_add"));
        void *args[] = { &pa, &pb, &pc, &n };
        CK(cuLaunchKernel(f, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, 0));
        CK(cuCtxSynchronize());
        int bad = 0; for (int i = 0; i < n; i++) bad += p[2 * n + i] != 3.0f * i;
        printf("managed mem  n=%d: CPU<->GPU through one pointer, mismatches %d\n", n, bad);
        EXPECT(bad == 0, "managed");
        cuMemFree(m);
    }

    cuEventDestroy(e0); cuEventDestroy(e1);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    printf(failures ? "driver_test: FAILED (%d)\n" : "driver_test: PASS\n", failures);
    return failures != 0;
}
