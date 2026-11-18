/*
 * libcuda.so.1 — CUDA Driver API on Apple GPUs via MetalPassthrough.
 *
 * Compiled against NVIDIA's official <cuda.h> so every exported symbol has the exact ABI that
 * nvcc-built binaries, ctypes/cffi wrappers and frameworks expect. Device memory is MetalPassthrough
 * arena memory (guest hugepages the Apple GPU addresses directly); PTX modules are translated to
 * Metal Shading Language on the host and run natively on the M-series GPU.
 *
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */
#include <cuda.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "metalpassthrough.h"
#include "fatbin.h"

#define EXPORT __attribute__((visibility("default")))

struct CUctx_st { int refs; unsigned flags; };
struct CUfunc_st {
    struct CUfunc_st *next;
    struct CUmod_st *mod;
    char name[256];
    mpt_pipeline pipe;
    mpt_kernel_info info;
    int max_dynamic_shared;
};
struct CUmod_st { char *ptx; struct CUfunc_st *funcs; };
struct CUstream_st { unsigned flags; int priority; };
struct CUevent_st { mpt_event *ev; unsigned flags; int recorded; };

static struct CUctx_st g_primary = { 0, 0 };
static __thread CUcontext t_current;
static mpt_ctx *g;
static int g_init_status = -1;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_debug;

#define DBG(...) do { if (g_debug) { fprintf(stderr, "[libcuda/metalpassthrough] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

static CUresult map_err(int s) {
    switch (s) {
    case MPT_OK: return CUDA_SUCCESS;
    case MPT_ENOMEM: return CUDA_ERROR_OUT_OF_MEMORY;
    case MPT_EINVAL: return CUDA_ERROR_INVALID_VALUE;
    case MPT_EFAULT: return CUDA_ERROR_INVALID_VALUE;
    case MPT_ETRANSLATE: return CUDA_ERROR_INVALID_PTX;
    case MPT_ECOMPILE: return CUDA_ERROR_INVALID_PTX;
    case MPT_EGPU: return CUDA_ERROR_LAUNCH_FAILED;
    case MPT_ENOENT: return CUDA_ERROR_NOT_FOUND;
    default: return CUDA_ERROR_UNKNOWN;
    }
}

#define REQUIRE_INIT() do { if (g_init_status != 0) return g_init_status < 0 ? CUDA_ERROR_NOT_INITIALIZED : CUDA_ERROR_NO_DEVICE; } while (0)

/* ------------------------------------------------------------------ init / version / device */

EXPORT CUresult cuInit(unsigned int flags) {
    pthread_mutex_lock(&g_lock);
    if (g_init_status < 0) {
        g_debug = getenv("METALPASSTHROUGH_DEBUG") != NULL;
        g = mpt_default();
        g_init_status = g ? 0 : 1;
        if (!g) fprintf(stderr, "[libcuda/metalpassthrough] no device: %s\n", mpt_last_error());
        else DBG("cuInit: %s, UVA=%d", mpt_device(g)->name, mpt_unified_addressing(g));
    }
    pthread_mutex_unlock(&g_lock);
    return g_init_status == 0 ? CUDA_SUCCESS : CUDA_ERROR_NO_DEVICE;
}

EXPORT CUresult cuDriverGetVersion(int *v) { if (!v) return CUDA_ERROR_INVALID_VALUE; *v = CUDA_VERSION; return CUDA_SUCCESS; }

EXPORT CUresult cuDeviceGetCount(int *count) {
    REQUIRE_INIT();
    *count = 1;
    return CUDA_SUCCESS;
}

EXPORT CUresult cuDeviceGet(CUdevice *dev, int ordinal) {
    REQUIRE_INIT();
    if (ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
    *dev = 0;
    return CUDA_SUCCESS;
}

EXPORT CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    REQUIRE_INIT();
    if (dev != 0) return CUDA_ERROR_INVALID_DEVICE;
    snprintf(name, (size_t)len, "%s (MetalPassthrough)", mpt_device(g)->name);
    return CUDA_SUCCESS;
}

EXPORT CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    REQUIRE_INIT();
    memset(uuid, 0, sizeof *uuid);
    const char *n = mpt_device(g)->name;
    for (int i = 0; n[i] && i < 64; i++) uuid->bytes[i % 16] ^= (char)(n[i] * (i + 1));
    return CUDA_SUCCESS;
}
EXPORT CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) { return cuDeviceGetUuid(uuid, dev); }

EXPORT CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    REQUIRE_INIT();
    uint64_t fr, tot;
    mpt_memory_info(g, &fr, &tot);
    *bytes = tot;  /* allocatable unified memory = reserved hugepage pool */
    return CUDA_SUCCESS;
}

EXPORT CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    REQUIRE_INIT();
    *major = 8;  /* PTX feature level accepted by the translator (sm_80 ISA subset) */
    *minor = 0;
    return CUDA_SUCCESS;
}

EXPORT CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute a, CUdevice dev) {
    REQUIRE_INIT();
    const mpt_device_info *d = mpt_device(g);
    int v = 0;
    switch (a) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: v = (int)d->max_threads_per_tg; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: v = (int)d->max_threads_per_tg; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: v = 64; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: v = 0x7fffffff; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: v = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: v = (int)d->threadgroup_memory; break;
    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: v = 64 * 1024; break;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE: v = (int)d->simd_width; break;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH: v = 0x7fffffff; break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: v = (int)d->gpu_cores; break;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: v = 2048; break;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: v = 65536; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: v = 8; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_INTEGRATED: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: v = mpt_unified_addressing(g); break;
    case CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY: v = mpt_unified_addressing(g); break;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: v = mpt_unified_addressing(g); break;
    case CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: v = CU_COMPUTEMODE_DEFAULT; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: v = 0; break;  /* not published by Apple */
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: v = 256; break;
    case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_ECC_ENABLED: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_TCC_DRIVER: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: v = 1; break;
    case CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH: v = 0; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: v = 32; break;
    default: v = 0; break;
    }
    *pi = v;
    return CUDA_SUCCESS;
}

EXPORT CUresult cuDeviceGetProperties(CUdevprop *p, CUdevice dev) {
    REQUIRE_INIT();
    const mpt_device_info *d = mpt_device(g);
    memset(p, 0, sizeof *p);
    p->maxThreadsPerBlock = (int)d->max_threads_per_tg;
    p->maxThreadsDim[0] = p->maxThreadsDim[1] = (int)d->max_threads_per_tg;
    p->maxThreadsDim[2] = 64;
    p->maxGridSize[0] = 0x7fffffff;
    p->maxGridSize[1] = p->maxGridSize[2] = 65535;
    p->sharedMemPerBlock = (int)d->threadgroup_memory;
    p->totalConstantMemory = 65536;
    p->SIMDWidth = (int)d->simd_width;
    p->memPitch = 0x7fffffff;
    p->regsPerBlock = 65536;
    p->textureAlign = 256;
    return CUDA_SUCCESS;
}

/* ------------------------------------------------------------------ contexts */

EXPORT CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    REQUIRE_INIT();
    __atomic_add_fetch(&g_primary.refs, 1, __ATOMIC_RELAXED);
    *pctx = &g_primary;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuDevicePrimaryCtxRelease(CUdevice dev) { __atomic_sub_fetch(&g_primary.refs, 1, __ATOMIC_RELAXED); return CUDA_SUCCESS; }
EXPORT CUresult cuDevicePrimaryCtxReset(CUdevice dev) { return CUDA_SUCCESS; }
EXPORT CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) { g_primary.flags = flags; return CUDA_SUCCESS; }
EXPORT CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    *flags = g_primary.flags;
    *active = g_primary.refs > 0;
    return CUDA_SUCCESS;
}

EXPORT CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    REQUIRE_INIT();
    /* All contexts share the process's MetalPassthrough session (one Metal command queue). */
    __atomic_add_fetch(&g_primary.refs, 1, __ATOMIC_RELAXED);
    g_primary.flags = flags;
    *pctx = &g_primary;
    t_current = &g_primary;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuCtxCreate_v3(CUcontext *pctx, CUexecAffinityParam *p, int n, unsigned int flags, CUdevice dev) {
    return cuCtxCreate(pctx, flags, dev);
}
EXPORT CUresult cuCtxDestroy(CUcontext ctx) {
    if (t_current == ctx) t_current = NULL;
    __atomic_sub_fetch(&g_primary.refs, 1, __ATOMIC_RELAXED);
    return CUDA_SUCCESS;
}
EXPORT CUresult cuCtxSetCurrent(CUcontext ctx) { t_current = ctx; return CUDA_SUCCESS; }
EXPORT CUresult cuCtxGetCurrent(CUcontext *ctx) { *ctx = t_current; return CUDA_SUCCESS; }
EXPORT CUresult cuCtxPushCurrent(CUcontext ctx) { t_current = ctx; return CUDA_SUCCESS; }
EXPORT CUresult cuCtxPopCurrent(CUcontext *ctx) { if (ctx) *ctx = t_current; t_current = NULL; return CUDA_SUCCESS; }
EXPORT CUresult cuCtxGetDevice(CUdevice *dev) { REQUIRE_INIT(); *dev = 0; return CUDA_SUCCESS; }
EXPORT CUresult cuCtxGetFlags(unsigned int *flags) { *flags = g_primary.flags; return CUDA_SUCCESS; }
EXPORT CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *v) { *v = 3020; return CUDA_SUCCESS; }
EXPORT CUresult cuCtxSetCacheConfig(CUfunc_cache c) { return CUDA_SUCCESS; }
EXPORT CUresult cuCtxGetCacheConfig(CUfunc_cache *c) { *c = CU_FUNC_CACHE_PREFER_NONE; return CUDA_SUCCESS; }
EXPORT CUresult cuCtxSetLimit(CUlimit l, size_t v) { return CUDA_SUCCESS; }
EXPORT CUresult cuCtxGetLimit(size_t *v, CUlimit l) { *v = 0; return CUDA_SUCCESS; }

EXPORT CUresult cuCtxSynchronize(void) {
    REQUIRE_INIT();
    return map_err(mpt_sync(g, NULL));
}

/* ------------------------------------------------------------------ memory */

static inline void *hp(CUdeviceptr d) { return mpt_hostptr(g, (uint64_t)d); }

/* Device memory is the guest's own unified memory: host<->device copies are CPU memcpy, ordered
 * after outstanding GPU work (synchronous CUDA copy semantics). */
static CUresult settle(void) { return mpt_pending(g) ? map_err(mpt_sync(g, NULL)) : CUDA_SUCCESS; }

EXPORT CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytes) {
    REQUIRE_INIT();
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    void *p = mpt_alloc(g, bytes ? bytes : 1, 256);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    *dptr = (CUdeviceptr)mpt_devaddr(g, p);
    return CUDA_SUCCESS;
}

EXPORT CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pitch, size_t w, size_t h, unsigned int elem) {
    size_t p = (w + 255) & ~(size_t)255;
    *pitch = p;
    return cuMemAlloc(dptr, p * h);
}

EXPORT CUresult cuMemFree(CUdeviceptr dptr) {
    REQUIRE_INIT();
    if (!dptr) return CUDA_SUCCESS;
    void *p = hp(dptr);
    if (!p) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = settle();  /* never recycle memory the GPU may still touch */
    mpt_free(g, p);
    return r;
}

EXPORT CUresult cuMemGetInfo(size_t *free_b, size_t *total_b) {
    REQUIRE_INIT();
    uint64_t f, t;
    mpt_memory_info(g, &f, &t);
    if (free_b) *free_b = f;
    if (total_b) *total_b = t;
    return CUDA_SUCCESS;
}

EXPORT CUresult cuMemGetAddressRange(CUdeviceptr *base, size_t *size, CUdeviceptr dptr) {
    REQUIRE_INIT();
    void *p = hp(dptr);
    if (!p) return CUDA_ERROR_NOT_FOUND;
    size_t s = mpt_alloc_size(g, p);
    if (!s) return CUDA_ERROR_NOT_FOUND;  /* only exact allocation bases are tracked */
    if (base) *base = dptr;
    if (size) *size = s;
    return CUDA_SUCCESS;
}

/* Pinned / mapped / managed host memory: arena memory is simultaneously host- and GPU-visible. */
EXPORT CUresult cuMemAllocHost(void **pp, size_t bytes) {
    REQUIRE_INIT();
    void *p = mpt_alloc(g, bytes ? bytes : 1, 256);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    *pp = p;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuMemHostAlloc(void **pp, size_t bytes, unsigned int flags) { return cuMemAllocHost(pp, bytes); }
EXPORT CUresult cuMemFreeHost(void *p) { REQUIRE_INIT(); CUresult r = settle(); mpt_free(g, p); return r; }
EXPORT CUresult cuMemHostGetDevicePointer(CUdeviceptr *d, void *p, unsigned int flags) {
    REQUIRE_INIT();
    uint64_t a = mpt_devaddr(g, p);
    if (!a) return CUDA_ERROR_INVALID_VALUE;
    *d = (CUdeviceptr)a;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuMemHostGetFlags(unsigned int *flags, void *p) { *flags = CU_MEMHOSTALLOC_DEVICEMAP; return CUDA_SUCCESS; }
EXPORT CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytes, unsigned int flags) {
    REQUIRE_INIT();
    if (!mpt_unified_addressing(g)) return CUDA_ERROR_NOT_SUPPORTED;  /* needs guest VA == device VA */
    return cuMemAlloc(dptr, bytes);
}
EXPORT CUresult cuMemHostRegister(void *p, size_t bytes, unsigned int flags) {
    /* Arbitrary guest pages are not physically contiguous 2 MiB hugepages: cannot be GPU-mapped. */
    return mpt_devaddr(g, p) ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
}
EXPORT CUresult cuMemHostUnregister(void *p) { return CUDA_SUCCESS; }

EXPORT CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t n) {
    REQUIRE_INIT();
    void *d = hp(dst);
    if (!d) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = settle();
    memcpy(d, src, n);
    return r;
}
EXPORT CUresult cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t n) {
    REQUIRE_INIT();
    void *s = hp(src);
    if (!s) return CUDA_ERROR_INVALID_VALUE;
    CUresult r = settle();
    memcpy(dst, s, n);
    return r;
}
EXPORT CUresult cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t n) {
    REQUIRE_INIT();
    return map_err(mpt_copy(g, dst, src, n));  /* GPU blit, stream ordered */
}
EXPORT CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t n) {
    REQUIRE_INIT();
    int dd = hp(dst) != NULL, sd = hp(src) != NULL;
    if (dd && sd) return cuMemcpyDtoD(dst, src, n);
    if (dd) return cuMemcpyHtoD(dst, (const void *)(uintptr_t)src, n);
    if (sd) return cuMemcpyDtoH((void *)(uintptr_t)dst, src, n);
    memcpy((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, n);
    return CUDA_SUCCESS;
}
EXPORT CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t n, CUstream s) { return cuMemcpyHtoD(dst, src, n); }
EXPORT CUresult cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t n, CUstream s) { return cuMemcpyDtoH(dst, src, n); }
EXPORT CUresult cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s) { return cuMemcpyDtoD(dst, src, n); }
EXPORT CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t n, CUstream s) { return cuMemcpy(dst, src, n); }

EXPORT CUresult cuMemsetD8(CUdeviceptr d, unsigned char v, size_t n) { REQUIRE_INIT(); return map_err(mpt_fill(g, d, n, v, 1)); }
EXPORT CUresult cuMemsetD16(CUdeviceptr d, unsigned short v, size_t n) { REQUIRE_INIT(); return map_err(mpt_fill(g, d, n * 2, v, 2)); }
EXPORT CUresult cuMemsetD32(CUdeviceptr d, unsigned int v, size_t n) { REQUIRE_INIT(); return map_err(mpt_fill(g, d, n * 4, v, 4)); }
EXPORT CUresult cuMemsetD8Async(CUdeviceptr d, unsigned char v, size_t n, CUstream s) { return cuMemsetD8(d, v, n); }
EXPORT CUresult cuMemsetD32Async(CUdeviceptr d, unsigned int v, size_t n, CUstream s) { return cuMemsetD32(d, v, n); }

EXPORT CUresult cuPointerGetAttribute(void *data, CUpointer_attribute a, CUdeviceptr ptr) {
    REQUIRE_INIT();
    int dev = hp(ptr) != NULL;
    switch (a) {
    case CU_POINTER_ATTRIBUTE_MEMORY_TYPE: *(unsigned *)data = dev ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_POINTER: *(CUdeviceptr *)data = ptr; return dev ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    case CU_POINTER_ATTRIBUTE_HOST_POINTER: *(void **)data = hp(ptr); return dev ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
    case CU_POINTER_ATTRIBUTE_IS_MANAGED: *(unsigned *)data = dev && mpt_unified_addressing(g); return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL: *(int *)data = 0; return CUDA_SUCCESS;
    case CU_POINTER_ATTRIBUTE_CONTEXT: *(CUcontext *)data = &g_primary; return CUDA_SUCCESS;
    default: return CUDA_ERROR_INVALID_VALUE;
    }
}

/* ------------------------------------------------------------------ modules & kernels */

static uint32_t compile_flags(void) {
    uint32_t f = 0;
    const char *m = getenv("METALPASSTHROUGH_FP64");
    if (m && !strcmp(m, "fp32")) f |= MPT_CF_FP64_LOWER;
    if (getenv("METALPASSTHROUGH_FAST_MATH")) f |= MPT_CF_FAST_MATH;
    if (getenv("METALPASSTHROUGH_DUMP_MSL")) f |= MPT_CF_DUMP_MSL;
    return f;
}

/* Fills the JIT log option buffers of cuModuleLoadDataEx if present. */
static void jit_log(unsigned n, CUjit_option *opt, void **val, const char *info, const char *errlog) {
    for (unsigned i = 0; opt && i < n; i++) {
        if (opt[i] == CU_JIT_INFO_LOG_BUFFER || opt[i] == CU_JIT_ERROR_LOG_BUFFER) {
            size_t sz = 0;
            for (unsigned k = 0; k < n; k++)
                if (opt[k] == (opt[i] == CU_JIT_INFO_LOG_BUFFER ? CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES : CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES))
                    sz = (size_t)(uintptr_t)val[k];
            if (sz && val[i]) snprintf(val[i], sz, "%s", opt[i] == CU_JIT_INFO_LOG_BUFFER ? info : errlog);
        }
    }
}

/* Eagerly translates + compiles every .entry (CUDA reports JIT errors at module load). */
static CUresult load_ptx(CUmodule *mod, const char *ptx, unsigned nopt, CUjit_option *opt, void **val) {
    struct CUmod_st *m = calloc(1, sizeof *m);
    m->ptx = strdup(ptx);
    const char *p = m->ptx;
    static char log[65536];
    while ((p = strstr(p, ".entry"))) {
        p += 6;
        while (*p == ' ' || *p == '\t') p++;
        char name[256];
        size_t k = 0;
        while (*p && *p != '(' && *p != ' ' && *p != '\t' && *p != '\n' && *p != '{' && k + 1 < sizeof name) name[k++] = *p++;
        name[k] = 0;
        if (!k) continue;
        struct CUfunc_st *f = calloc(1, sizeof *f);
        f->mod = m;
        snprintf(f->name, sizeof f->name, "%s", name);
        int r = mpt_compile_ptx(g, m->ptx, name, compile_flags(), &f->pipe, &f->info, log, sizeof log);
        if (r) {
            fprintf(stderr, "[libcuda/metalpassthrough] PTX JIT of '%s' failed: %s\n", name, log);
            jit_log(nopt, opt, val, "", log);
            free(f);
            /* free module */
            for (struct CUfunc_st *x = m->funcs; x;) { struct CUfunc_st *n = x->next; free(x); x = n; }
            free(m->ptx);
            free(m);
            return map_err(r);
        }
        if (log[0]) DBG("%s: %s", name, log);
        f->next = m->funcs;
        m->funcs = f;
    }
    jit_log(nopt, opt, val, "MetalPassthrough: PTX translated to Metal", "");
    *mod = m;
    return CUDA_SUCCESS;
}

EXPORT CUresult cuModuleLoadDataEx(CUmodule *mod, const void *image, unsigned int nopt, CUjit_option *opt, void **val) {
    REQUIRE_INIT();
    if (!mod || !image) return CUDA_ERROR_INVALID_VALUE;
    int sass = 0;
    char err[256];
    char *ptx = mpt_fatbin_extract_ptx(image, &sass, err, sizeof err);
    if (!ptx) {
        fprintf(stderr, "[libcuda/metalpassthrough] %s\n", err);
        return sass ? CUDA_ERROR_NO_BINARY_FOR_GPU : CUDA_ERROR_INVALID_IMAGE;
    }
    CUresult r = load_ptx(mod, ptx, nopt, opt, val);
    free(ptx);
    return r;
}
EXPORT CUresult cuModuleLoadData(CUmodule *mod, const void *image) { return cuModuleLoadDataEx(mod, image, 0, NULL, NULL); }
EXPORT CUresult cuModuleLoadFatBinary(CUmodule *mod, const void *fatbin) { return cuModuleLoadDataEx(mod, fatbin, 0, NULL, NULL); }

EXPORT CUresult cuModuleLoad(CUmodule *mod, const char *path) {
    REQUIRE_INIT();
    FILE *f = fopen(path, "rb");
    if (!f) return CUDA_ERROR_FILE_NOT_FOUND;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = calloc(1, (size_t)n + 1);
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    if (got != (size_t)n) { free(buf); return CUDA_ERROR_FILE_NOT_FOUND; }
    CUresult r = cuModuleLoadData(mod, buf);
    free(buf);
    return r;
}

EXPORT CUresult cuModuleUnload(CUmodule mod) {
    if (!mod) return CUDA_ERROR_INVALID_HANDLE;
    for (struct CUfunc_st *f = mod->funcs; f;) {
        struct CUfunc_st *n = f->next;
        mpt_release_pipeline(g, &f->pipe);
        free(f);
        f = n;
    }
    free(mod->ptx);
    free(mod);
    return CUDA_SUCCESS;
}

EXPORT CUresult cuModuleGetFunction(CUfunction *fn, CUmodule mod, const char *name) {
    REQUIRE_INIT();
    if (!mod || !fn || !name) return CUDA_ERROR_INVALID_VALUE;
    for (struct CUfunc_st *f = mod->funcs; f; f = f->next)
        if (!strcmp(f->name, name)) { *fn = f; return CUDA_SUCCESS; }
    return CUDA_ERROR_NOT_FOUND;
}

EXPORT CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule mod, const char *name) {
    return CUDA_ERROR_NOT_FOUND;  /* module-scope __device__/__constant__ variables are not supported yet */
}

EXPORT CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute a, CUfunction f) {
    if (!f) return CUDA_ERROR_INVALID_HANDLE;
    switch (a) {
    case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = (int)f->pipe.max_threads_per_tg; break;
    case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: *pi = (int)f->info.shared_bytes; break;
    case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES: *pi = 0; break;
    case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: *pi = 0; break;
    case CU_FUNC_ATTRIBUTE_NUM_REGS: *pi = 32; break;
    case CU_FUNC_ATTRIBUTE_PTX_VERSION: case CU_FUNC_ATTRIBUTE_BINARY_VERSION: *pi = 80; break;
    case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: *pi = f->max_dynamic_shared ? f->max_dynamic_shared : (int)(mpt_device(g)->threadgroup_memory - f->info.shared_bytes); break;
    default: *pi = 0;
    }
    return CUDA_SUCCESS;
}
EXPORT CUresult cuFuncSetAttribute(CUfunction f, CUfunction_attribute a, int v) {
    if (a == CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) f->max_dynamic_shared = v;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuFuncSetCacheConfig(CUfunction f, CUfunc_cache c) { return CUDA_SUCCESS; }
EXPORT CUresult cuFuncGetName(const char **name, CUfunction f) { *name = f->name; return CUDA_SUCCESS; }

EXPORT CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *n, CUfunction f, int block, size_t dyn) {
    *n = block > 0 ? (int)(f->pipe.max_threads_per_tg * 2 / (unsigned)block) : 0;
    if (*n < 1) *n = 1;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuOccupancyMaxPotentialBlockSize(int *grid, int *block, CUfunction f, CUoccupancyB2DSize fn, size_t dyn, int limit) {
    int b = (int)f->pipe.max_threads_per_tg;
    if (limit && b > limit) b = limit;
    *block = b;
    *grid = (int)mpt_device(g)->gpu_cores * 2;
    return CUDA_SUCCESS;
}

EXPORT CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                               unsigned shmem, CUstream stream, void **params, void **extra) {
    REQUIRE_INIT();
    if (!f) return CUDA_ERROR_INVALID_HANDLE;
    if ((uint64_t)bx * by * bz > f->pipe.max_threads_per_tg) return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    if (!gx || !gy || !gz || !bx || !by || !bz) return CUDA_ERROR_INVALID_VALUE;
    uint8_t block[4096];
    uint32_t len = f->info.params_bytes;
    if (len > sizeof block) return CUDA_ERROR_INVALID_VALUE;
    memset(block, 0, len);
    if (params) {
        for (uint32_t i = 0; i < f->info.nparams; i++) memcpy(block + f->info.param_offset[i], params[i], f->info.param_size[i]);
    } else if (extra) {
        void *buf = NULL;
        size_t sz = 0;
        for (int i = 0; extra[i] != CU_LAUNCH_PARAM_END; i += 2) {
            if (extra[i] == CU_LAUNCH_PARAM_BUFFER_POINTER) buf = extra[i + 1];
            else if (extra[i] == CU_LAUNCH_PARAM_BUFFER_SIZE) sz = *(size_t *)extra[i + 1];
        }
        if (!buf || sz > sizeof block) return CUDA_ERROR_INVALID_VALUE;
        memcpy(block, buf, sz < len ? sz : len);
    } else if (f->info.nparams) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    uint32_t tgmem = (f->info.flags & MPT_KF_USES_SHARED) ? f->info.shared_bytes + shmem : 0;
    if (tgmem > mpt_device(g)->threadgroup_memory) return CUDA_ERROR_INVALID_VALUE;
    uint32_t grid[3] = { gx, gy, gz }, tg[3] = { bx, by, bz };
    return map_err(mpt_dispatch_raw(g, &f->pipe, grid, tg, block, len, tgmem, 0));
}

/* ------------------------------------------------------------------ streams & events */

EXPORT CUresult cuStreamCreate(CUstream *s, unsigned int flags) {
    REQUIRE_INIT();
    struct CUstream_st *st = calloc(1, sizeof *st);
    st->flags = flags;
    *s = st;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuStreamCreateWithPriority(CUstream *s, unsigned int flags, int prio) { return cuStreamCreate(s, flags); }
/* All streams execute in submission order on the session's single Metal queue (stricter than CUDA). */
EXPORT CUresult cuStreamDestroy(CUstream s) { free(s); return CUDA_SUCCESS; }
EXPORT CUresult cuStreamSynchronize(CUstream s) { REQUIRE_INIT(); return map_err(mpt_sync(g, NULL)); }
EXPORT CUresult cuStreamQuery(CUstream s) { REQUIRE_INIT(); return mpt_pending(g) ? map_err(mpt_sync(g, NULL)) : CUDA_SUCCESS; }
EXPORT CUresult cuStreamWaitEvent(CUstream s, CUevent e, unsigned int flags) { return CUDA_SUCCESS; }

EXPORT CUresult cuEventCreate(CUevent *e, unsigned int flags) {
    REQUIRE_INIT();
    struct CUevent_st *ev = calloc(1, sizeof *ev);
    ev->ev = mpt_alloc(g, sizeof(mpt_event), 256);
    if (!ev->ev) { free(ev); return CUDA_ERROR_OUT_OF_MEMORY; }
    memset(ev->ev, 0, sizeof *ev->ev);
    ev->flags = flags;
    *e = ev;
    return CUDA_SUCCESS;
}
EXPORT CUresult cuEventDestroy(CUevent e) { if (e) { mpt_free(g, e->ev); free(e); } return CUDA_SUCCESS; }
EXPORT CUresult cuEventRecord(CUevent e, CUstream s) {
    REQUIRE_INIT();
    e->recorded = 1;
    return map_err(mpt_event_record(g, e->ev));
}
EXPORT CUresult cuEventRecordWithFlags(CUevent e, CUstream s, unsigned int flags) { return cuEventRecord(e, s); }
EXPORT CUresult cuEventSynchronize(CUevent e) { REQUIRE_INIT(); return e->recorded ? map_err(mpt_event_wait(g, e->ev)) : CUDA_SUCCESS; }
EXPORT CUresult cuEventQuery(CUevent e) { return (!e->recorded || __atomic_load_n(&e->ev->done, __ATOMIC_ACQUIRE)) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY; }
EXPORT CUresult cuEventElapsedTime(float *ms, CUevent a, CUevent b) {
    REQUIRE_INIT();
    if (!a->recorded || !b->recorded) return CUDA_ERROR_INVALID_HANDLE;
    mpt_event_wait(g, a->ev);
    mpt_event_wait(g, b->ev);
    *ms = (float)((double)((int64_t)b->ev->gpu_end_ns - (int64_t)a->ev->gpu_end_ns) / 1e6);
    return CUDA_SUCCESS;
}
EXPORT CUresult cuEventElapsedTime_v2(float *ms, CUevent a, CUevent b) { return cuEventElapsedTime(ms, a, b); }

/* ------------------------------------------------------------------ errors / misc */

EXPORT CUresult cuGetErrorName(CUresult e, const char **s) {
    switch (e) {
#define E(x) case x: *s = #x; return CUDA_SUCCESS;
    E(CUDA_SUCCESS) E(CUDA_ERROR_INVALID_VALUE) E(CUDA_ERROR_OUT_OF_MEMORY) E(CUDA_ERROR_NOT_INITIALIZED)
    E(CUDA_ERROR_NO_DEVICE) E(CUDA_ERROR_INVALID_DEVICE) E(CUDA_ERROR_INVALID_IMAGE) E(CUDA_ERROR_INVALID_CONTEXT)
    E(CUDA_ERROR_NO_BINARY_FOR_GPU) E(CUDA_ERROR_INVALID_PTX) E(CUDA_ERROR_FILE_NOT_FOUND) E(CUDA_ERROR_INVALID_HANDLE)
    E(CUDA_ERROR_NOT_FOUND) E(CUDA_ERROR_NOT_READY) E(CUDA_ERROR_LAUNCH_FAILED) E(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES)
    E(CUDA_ERROR_NOT_SUPPORTED) E(CUDA_ERROR_UNKNOWN)
#undef E
    default: *s = NULL; return CUDA_ERROR_INVALID_VALUE;
    }
}
EXPORT CUresult cuGetErrorString(CUresult e, const char **s) {
    switch (e) {
    case CUDA_SUCCESS: *s = "no error"; break;
    case CUDA_ERROR_INVALID_VALUE: *s = "invalid argument"; break;
    case CUDA_ERROR_OUT_OF_MEMORY: *s = "out of memory (MetalPassthrough arena: reserve more hugepages)"; break;
    case CUDA_ERROR_NOT_INITIALIZED: *s = "initialization error"; break;
    case CUDA_ERROR_NO_DEVICE: *s = "no CUDA-capable device is detected (MetalPassthrough host not reachable)"; break;
    case CUDA_ERROR_INVALID_PTX: *s = "a PTX JIT compilation failed (MetalPassthrough PTX->Metal)"; break;
    case CUDA_ERROR_NO_BINARY_FOR_GPU: *s = "no kernel image is available for execution on the device (PTX required)"; break;
    case CUDA_ERROR_INVALID_IMAGE: *s = "device kernel image is invalid"; break;
    case CUDA_ERROR_LAUNCH_FAILED: *s = "unspecified launch failure"; break;
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: *s = "too many resources requested for launch"; break;
    case CUDA_ERROR_NOT_FOUND: *s = "named symbol not found"; break;
    case CUDA_ERROR_NOT_SUPPORTED: *s = "operation not supported"; break;
    default: *s = "unknown error"; break;
    }
    return CUDA_SUCCESS;
}

/* cuGetProcAddress: resolve "cuFoo" against this library, preferring the highest _vN variant. */
EXPORT CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags,
                                 CUdriverProcAddressQueryResult *status) {
    static void *self;
    if (!self) self = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    char name[256];
    *pfn = NULL;
    for (int v = 4; v >= 1 && !*pfn; v--) {
        if (v > 1) snprintf(name, sizeof name, "%s_v%d", symbol, v); else snprintf(name, sizeof name, "%s", symbol);
        *pfn = dlsym(self ? self : RTLD_DEFAULT, name);
    }
    if (status) *status = *pfn ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return *pfn ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

EXPORT CUresult cuProfilerStart(void) { return CUDA_SUCCESS; }
EXPORT CUresult cuProfilerStop(void) { return CUDA_SUCCESS; }
