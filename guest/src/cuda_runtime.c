/*
 * libcudart.so.12 — CUDA Runtime API for MetalPassthrough, layered on its libcuda.so.1.
 *
 * Supports executables produced by nvcc with `-cudart shared`: the fatbinary registration that
 * nvcc emits into host code (__cudaRegisterFatBinary / __cudaRegisterFunction), triple-chevron
 * launches (__cudaPushCallConfiguration + cudaLaunchKernel), memory, streams, events, properties.
 * PTX embedded in the fatbinary is JIT-translated to Metal on first launch.
 *
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "metalpassthrough.h"
#include "fatbin.h"

#define EXPORT __attribute__((visibility("default")))

/* ------------------------------------------------------------------ state */

typedef struct fatbin_rec {
    const void *image;     /* __fatBinC_Wrapper_t* */
    CUmodule mod;          /* loaded lazily */
    int load_status;
} fatbin_rec;

typedef struct func_rec {
    const void *host_fn;
    char *device_name;
    fatbin_rec *fb;
    CUfunction fn;
} func_rec;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static func_rec *g_funcs;
static size_t g_nfuncs, g_capfuncs;
static int g_inited;
static __thread cudaError_t t_last;
static __thread int t_device;

typedef struct { dim3 grid, block; size_t shmem; void *stream; } launch_cfg;
static __thread launch_cfg t_cfg_stack[16];
static __thread int t_cfg_top;

static cudaError_t ret(cudaError_t e) { if (e != cudaSuccess) t_last = e; return e; }

static cudaError_t from_cu(CUresult r) {
    switch (r) {
    case CUDA_SUCCESS: return cudaSuccess;
    case CUDA_ERROR_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUDA_ERROR_OUT_OF_MEMORY: return cudaErrorMemoryAllocation;
    case CUDA_ERROR_NOT_INITIALIZED: return cudaErrorInitializationError;
    case CUDA_ERROR_NO_DEVICE: return cudaErrorNoDevice;
    case CUDA_ERROR_INVALID_DEVICE: return cudaErrorInvalidDevice;
    case CUDA_ERROR_INVALID_PTX: return cudaErrorInvalidPtx;
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return cudaErrorNoKernelImageForDevice;
    case CUDA_ERROR_INVALID_IMAGE: return cudaErrorInvalidKernelImage;
    case CUDA_ERROR_LAUNCH_FAILED: return cudaErrorLaunchFailure;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return cudaErrorLaunchOutOfResources;
    case CUDA_ERROR_NOT_FOUND: return cudaErrorSymbolNotFound;
    case CUDA_ERROR_NOT_READY: return cudaErrorNotReady;
    case CUDA_ERROR_INVALID_HANDLE: return cudaErrorInvalidResourceHandle;
    case CUDA_ERROR_NOT_SUPPORTED: return cudaErrorNotSupported;
    default: return cudaErrorUnknown;
    }
}

#define CU(x) do { CUresult _r = (x); if (_r != CUDA_SUCCESS) return ret(from_cu(_r)); } while (0)

static cudaError_t lazy_init(void) {
    if (__atomic_load_n(&g_inited, __ATOMIC_ACQUIRE)) return cudaSuccess;
    pthread_mutex_lock(&g_lock);
    CUresult r = CUDA_SUCCESS;
    if (!g_inited) {
        r = cuInit(0);
        if (r == CUDA_SUCCESS) {
            CUcontext c;
            r = cuDevicePrimaryCtxRetain(&c, 0);
            if (r == CUDA_SUCCESS) cuCtxSetCurrent(c);
            __atomic_store_n(&g_inited, 1, __ATOMIC_RELEASE);
        }
    }
    pthread_mutex_unlock(&g_lock);
    return r == CUDA_SUCCESS ? cudaSuccess : ret(from_cu(r));
}
#define INIT() do { cudaError_t _e = lazy_init(); if (_e != cudaSuccess) return _e; } while (0)

/* ------------------------------------------------------------------ nvcc registration ABI */

EXPORT void **__cudaRegisterFatBinary(void *fatCubin) {
    fatbin_rec *r = calloc(1, sizeof *r);
    r->image = fatCubin;
    return (void **)r;
}
EXPORT void __cudaRegisterFatBinaryEnd(void **h) { (void)h; }
EXPORT void __cudaUnregisterFatBinary(void **h) {
    /* Called from static destructors at process exit: the session is already drained and closed,
     * so modules are intentionally left to process teardown (as NVIDIA's runtime does). */
    (void)h;
}
EXPORT char __cudaInitModule(void **h) { return 1; }

EXPORT void __cudaRegisterFunction(void **h, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit,
                                   uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    pthread_mutex_lock(&g_lock);
    if (g_nfuncs == g_capfuncs) {
        g_capfuncs = g_capfuncs ? g_capfuncs * 2 : 64;
        g_funcs = realloc(g_funcs, g_capfuncs * sizeof *g_funcs);
    }
    g_funcs[g_nfuncs++] = (func_rec){ .host_fn = hostFun, .device_name = strdup(deviceName), .fb = (fatbin_rec *)h };
    pthread_mutex_unlock(&g_lock);
}

EXPORT void __cudaRegisterVar(void **h, char *hostVar, char *deviceAddress, const char *deviceName, int ext, size_t size,
                              int constant, int global) {
    if (getenv("METALPASSTHROUGH_DEBUG"))
        fprintf(stderr, "[libcudart/metalpassthrough] note: __device__/__constant__ variable '%s' is not supported yet\n", deviceName);
}
EXPORT void __cudaRegisterManagedVar(void **h, void **hostVarPtrAddress, char *deviceAddress, const char *deviceName, int ext,
                                     size_t size, int constant, int global) {}
EXPORT void __cudaRegisterTexture(void **h, const void *hostVar, const void **deviceAddress, const char *deviceName, int dim,
                                  int norm, int ext) {}
EXPORT void __cudaRegisterSurface(void **h, const void *hostVar, const void **deviceAddress, const char *deviceName, int dim,
                                  int ext) {}

EXPORT unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shmem, void *stream) {
    if (t_cfg_top >= 16) return 1;
    t_cfg_stack[t_cfg_top++] = (launch_cfg){ grid, block, shmem, stream };
    return 0;
}
EXPORT cudaError_t __cudaPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shmem, void *stream) {
    if (t_cfg_top <= 0) return cudaErrorMissingConfiguration;
    launch_cfg c = t_cfg_stack[--t_cfg_top];
    *grid = c.grid;
    *block = c.block;
    *shmem = c.shmem;
    *(void **)stream = c.stream;
    return cudaSuccess;
}

static cudaError_t resolve(const void *host_fn, CUfunction *out) {
    INIT();
    pthread_mutex_lock(&g_lock);
    func_rec *f = NULL;
    for (size_t i = 0; i < g_nfuncs; i++) if (g_funcs[i].host_fn == host_fn) { f = &g_funcs[i]; break; }
    if (!f) { pthread_mutex_unlock(&g_lock); return ret(cudaErrorInvalidDeviceFunction); }
    if (!f->fn) {
        fatbin_rec *fb = f->fb;
        if (!fb->mod && !fb->load_status) {
            CUresult r = cuModuleLoadData(&fb->mod, fb->image);
            fb->load_status = r == CUDA_SUCCESS ? 1 : -(int)r;
        }
        if (fb->load_status < 0) { pthread_mutex_unlock(&g_lock); return ret(from_cu((CUresult)(-fb->load_status))); }
        CUresult r = cuModuleGetFunction(&f->fn, fb->mod, f->device_name);
        if (r != CUDA_SUCCESS) { pthread_mutex_unlock(&g_lock); return ret(from_cu(r)); }
    }
    *out = f->fn;
    pthread_mutex_unlock(&g_lock);
    return cudaSuccess;
}

EXPORT cudaError_t cudaLaunchKernel(const void *func, dim3 g, dim3 b, void **args, size_t shmem, cudaStream_t stream) {
    CUfunction f;
    cudaError_t e = resolve(func, &f);
    if (e != cudaSuccess) return e;
    CU(cuLaunchKernel(f, g.x, g.y, g.z, b.x, b.y, b.z, (unsigned)shmem, (CUstream)stream, args, NULL));
    return cudaSuccess;
}

EXPORT cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *a, const void *func) {
    CUfunction f;
    cudaError_t e = resolve(func, &f);
    if (e != cudaSuccess) return e;
    memset(a, 0, sizeof *a);
    int v;
    cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, f); a->maxThreadsPerBlock = v;
    cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f); a->sharedSizeBytes = (size_t)v;
    cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_NUM_REGS, f); a->numRegs = v;
    a->ptxVersion = 80;
    a->binaryVersion = 80;
    cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, f); a->maxDynamicSharedSizeBytes = v;
    return cudaSuccess;
}
EXPORT cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) { return cudaSuccess; }
EXPORT cudaError_t cudaFuncSetCacheConfig(const void *func, enum cudaFuncCache c) { return cudaSuccess; }

EXPORT cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *n, const void *func, int block, size_t dyn) {
    CUfunction f;
    cudaError_t e = resolve(func, &f);
    if (e != cudaSuccess) return e;
    CU(cuOccupancyMaxActiveBlocksPerMultiprocessor(n, f, block, dyn));
    return cudaSuccess;
}

/* ------------------------------------------------------------------ device management */

EXPORT cudaError_t cudaGetDeviceCount(int *n) {
    cudaError_t e = lazy_init();
    if (e != cudaSuccess) { *n = 0; return e; }
    CU(cuDeviceGetCount(n));
    return cudaSuccess;
}
EXPORT cudaError_t cudaGetDevice(int *d) { INIT(); *d = t_device; return cudaSuccess; }
EXPORT cudaError_t cudaSetDevice(int d) { INIT(); if (d != 0) return ret(cudaErrorInvalidDevice); t_device = d; return cudaSuccess; }
EXPORT cudaError_t cudaDeviceReset(void) { return cudaSuccess; }
EXPORT cudaError_t cudaSetDeviceFlags(unsigned int flags) { return cudaSuccess; }
EXPORT cudaError_t cudaDeviceSynchronize(void) { INIT(); CU(cuCtxSynchronize()); return cudaSuccess; }
EXPORT cudaError_t cudaThreadSynchronize(void) { return cudaDeviceSynchronize(); }
EXPORT cudaError_t cudaDriverGetVersion(int *v) { return from_cu(cuDriverGetVersion(v)); }
EXPORT cudaError_t cudaRuntimeGetVersion(int *v) { *v = CUDART_VERSION; return cudaSuccess; }
EXPORT cudaError_t cudaDeviceGetAttribute(int *v, enum cudaDeviceAttr attr, int dev) {
    INIT();
    /* cudaDeviceAttr values are defined to match CUdevice_attribute. */
    CU(cuDeviceGetAttribute(v, (CUdevice_attribute)attr, dev));
    return cudaSuccess;
}

EXPORT cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *p, int dev) {
    INIT();
    if (dev != 0) return ret(cudaErrorInvalidDevice);
    memset(p, 0, sizeof *p);
    cuDeviceGetName(p->name, sizeof p->name, 0);
    CUuuid u;
    cuDeviceGetUuid(&u, 0);
    memcpy(&p->uuid, &u, sizeof u);
    size_t total;
    cuDeviceTotalMem(&total, 0);
    p->totalGlobalMem = total;
    int v;
#define A(field, attr) do { cuDeviceGetAttribute(&v, attr, 0); p->field = v; } while (0)
    A(sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    A(sharedMemPerBlockOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);
    A(sharedMemPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    A(regsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
    A(regsPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
    A(warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    A(maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    A(maxThreadsDim[0], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    A(maxThreadsDim[1], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
    A(maxThreadsDim[2], CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
    A(maxGridSize[0], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    A(maxGridSize[1], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    A(maxGridSize[2], CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
    A(totalConstMem, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
    A(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    A(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    A(multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    A(maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
    A(maxBlocksPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR);
    A(integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED);
    A(canMapHostMemory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
    A(unifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
    A(managedMemory, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
    A(concurrentManagedAccess, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS);
    A(textureAlignment, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
    A(memPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH);
#undef A
    p->computeMode = cudaComputeModeDefault;
    p->globalL1CacheSupported = 1;
    p->localL1CacheSupported = 1;
    return cudaSuccess;
}
/* <cuda_runtime_api.h> maps cudaGetDeviceProperties -> _v2 (the function above); also export the legacy name. */
#undef cudaGetDeviceProperties
EXPORT cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *p, int dev);
EXPORT cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *p, int dev) { return cudaGetDeviceProperties_v2(p, dev); }

EXPORT cudaError_t cudaMemGetInfo(size_t *f, size_t *t) { INIT(); CU(cuMemGetInfo(f, t)); return cudaSuccess; }

/* ------------------------------------------------------------------ memory */

EXPORT cudaError_t cudaMalloc(void **p, size_t n) {
    INIT();
    CUdeviceptr d;
    CU(cuMemAlloc(&d, n));
    *p = (void *)(uintptr_t)d;
    return cudaSuccess;
}
EXPORT cudaError_t cudaFree(void *p) { INIT(); if (!p) return cudaSuccess; CU(cuMemFree((CUdeviceptr)(uintptr_t)p)); return cudaSuccess; }
EXPORT cudaError_t cudaMallocHost(void **p, size_t n) { INIT(); CU(cuMemAllocHost(p, n)); return cudaSuccess; }
EXPORT cudaError_t cudaHostAlloc(void **p, size_t n, unsigned int flags) { INIT(); CU(cuMemHostAlloc(p, n, flags)); return cudaSuccess; }
EXPORT cudaError_t cudaFreeHost(void *p) { INIT(); CU(cuMemFreeHost(p)); return cudaSuccess; }
EXPORT cudaError_t cudaHostGetDevicePointer(void **d, void *h, unsigned int flags) {
    INIT();
    CUdeviceptr dp;
    CU(cuMemHostGetDevicePointer(&dp, h, flags));
    *d = (void *)(uintptr_t)dp;
    return cudaSuccess;
}
EXPORT cudaError_t cudaMallocManaged(void **p, size_t n, unsigned int flags) {
    INIT();
    CUdeviceptr d;
    CU(cuMemAllocManaged(&d, n, flags ? flags : CU_MEM_ATTACH_GLOBAL));
    *p = (void *)(uintptr_t)d;
    return cudaSuccess;
}
EXPORT cudaError_t cudaMemcpy(void *dst, const void *src, size_t n, enum cudaMemcpyKind kind) {
    INIT();
    CUdeviceptr d = (CUdeviceptr)(uintptr_t)dst, s = (CUdeviceptr)(uintptr_t)src;
    switch (kind) {
    case cudaMemcpyHostToDevice: CU(cuMemcpyHtoD(d, src, n)); break;
    case cudaMemcpyDeviceToHost: CU(cuMemcpyDtoH(dst, s, n)); break;
    case cudaMemcpyDeviceToDevice: CU(cuMemcpyDtoD(d, s, n)); break;
    case cudaMemcpyHostToHost: memcpy(dst, src, n); break;
    default: CU(cuMemcpy(d, s, n)); break;
    }
    return cudaSuccess;
}
EXPORT cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t n, enum cudaMemcpyKind kind, cudaStream_t st) {
    INIT();
    CUdeviceptr d = (CUdeviceptr)(uintptr_t)dst, s = (CUdeviceptr)(uintptr_t)src;
    switch (kind) {
    case cudaMemcpyHostToDevice: CU(cuMemcpyHtoDAsync(d, src, n, (CUstream)st)); break;
    case cudaMemcpyDeviceToHost: CU(cuMemcpyDtoHAsync(dst, s, n, (CUstream)st)); break;
    case cudaMemcpyDeviceToDevice: CU(cuMemcpyDtoDAsync(d, s, n, (CUstream)st)); break;
    case cudaMemcpyHostToHost: CU(cuStreamSynchronize((CUstream)st)); memcpy(dst, src, n); break;
    default: CU(cuMemcpyAsync(d, s, n, (CUstream)st)); break;
    }
    return cudaSuccess;
}
EXPORT cudaError_t cudaMemset(void *p, int v, size_t n) { INIT(); CU(cuMemsetD8((CUdeviceptr)(uintptr_t)p, (unsigned char)v, n)); return cudaSuccess; }
EXPORT cudaError_t cudaMemsetAsync(void *p, int v, size_t n, cudaStream_t s) {
    INIT();
    CU(cuMemsetD8Async((CUdeviceptr)(uintptr_t)p, (unsigned char)v, n, (CUstream)s));
    return cudaSuccess;
}
EXPORT cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *a, const void *ptr) {
    INIT();
    memset(a, 0, sizeof *a);
    unsigned t = 0;
    cuPointerGetAttribute(&t, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)(uintptr_t)ptr);
    a->type = t == CU_MEMORYTYPE_DEVICE ? cudaMemoryTypeManaged : cudaMemoryTypeUnregistered;
    a->device = 0;
    a->devicePointer = t == CU_MEMORYTYPE_DEVICE ? (void *)ptr : NULL;
    a->hostPointer = (void *)ptr;
    return cudaSuccess;
}

/* ------------------------------------------------------------------ streams & events */

EXPORT cudaError_t cudaStreamCreate(cudaStream_t *s) { INIT(); CU(cuStreamCreate((CUstream *)s, 0)); return cudaSuccess; }
EXPORT cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned int f) { INIT(); CU(cuStreamCreate((CUstream *)s, f)); return cudaSuccess; }
EXPORT cudaError_t cudaStreamDestroy(cudaStream_t s) { CU(cuStreamDestroy((CUstream)s)); return cudaSuccess; }
EXPORT cudaError_t cudaStreamSynchronize(cudaStream_t s) { INIT(); CU(cuStreamSynchronize((CUstream)s)); return cudaSuccess; }
EXPORT cudaError_t cudaStreamQuery(cudaStream_t s) {
    INIT();
    CUresult r = cuStreamQuery((CUstream)s);  /* never blocks; NotReady is not a sticky error */
    return r == CUDA_ERROR_NOT_READY ? cudaErrorNotReady : ret(from_cu(r));
}
EXPORT cudaError_t cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned int f) {
    INIT();
    CU(cuStreamWaitEvent((CUstream)s, (CUevent)e, f));
    return cudaSuccess;
}
EXPORT cudaError_t cudaEventCreate(cudaEvent_t *e) { INIT(); CU(cuEventCreate((CUevent *)e, 0)); return cudaSuccess; }
EXPORT cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned int f) { INIT(); CU(cuEventCreate((CUevent *)e, f)); return cudaSuccess; }
EXPORT cudaError_t cudaEventDestroy(cudaEvent_t e) { CU(cuEventDestroy((CUevent)e)); return cudaSuccess; }
EXPORT cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s) { INIT(); CU(cuEventRecord((CUevent)e, (CUstream)s)); return cudaSuccess; }
EXPORT cudaError_t cudaEventSynchronize(cudaEvent_t e) { INIT(); CU(cuEventSynchronize((CUevent)e)); return cudaSuccess; }
EXPORT cudaError_t cudaEventQuery(cudaEvent_t e) { CUresult r = cuEventQuery((CUevent)e); return r == CUDA_ERROR_NOT_READY ? cudaErrorNotReady : from_cu(r); }
EXPORT cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) {
    INIT();
    CUresult r = cuEventElapsedTime(ms, (CUevent)a, (CUevent)b);
    return r == CUDA_ERROR_NOT_READY ? cudaErrorNotReady : ret(from_cu(r));
}

/* ------------------------------------------------------------------ errors */

EXPORT cudaError_t cudaGetLastError(void) { cudaError_t e = t_last; t_last = cudaSuccess; return e; }
EXPORT cudaError_t cudaPeekAtLastError(void) { return t_last; }

EXPORT const char *cudaGetErrorName(cudaError_t e) {
    switch (e) {
#define E(x) case x: return #x;
    E(cudaSuccess) E(cudaErrorInvalidValue) E(cudaErrorMemoryAllocation) E(cudaErrorInitializationError) E(cudaErrorNoDevice)
    E(cudaErrorInvalidDevice) E(cudaErrorInvalidPtx) E(cudaErrorNoKernelImageForDevice) E(cudaErrorInvalidKernelImage)
    E(cudaErrorLaunchFailure) E(cudaErrorLaunchOutOfResources) E(cudaErrorSymbolNotFound) E(cudaErrorNotReady)
    E(cudaErrorInvalidResourceHandle) E(cudaErrorNotSupported) E(cudaErrorInvalidDeviceFunction) E(cudaErrorMissingConfiguration)
    E(cudaErrorUnknown)
#undef E
    default: return "cudaErrorUnknown";
    }
}
EXPORT const char *cudaGetErrorString(cudaError_t e) {
    switch (e) {
    case cudaSuccess: return "no error";
    case cudaErrorInvalidValue: return "invalid argument";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorNoDevice: return "no CUDA-capable device is detected";
    case cudaErrorInvalidPtx: return "a PTX JIT compilation failed";
    case cudaErrorNoKernelImageForDevice: return "no kernel image is available for execution on the device";
    case cudaErrorLaunchFailure: return "unspecified launch failure";
    case cudaErrorInvalidDeviceFunction: return "invalid device function";
    default: return cudaGetErrorName(e);
    }
}
