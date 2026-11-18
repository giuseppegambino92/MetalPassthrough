/*
 * libmetalpassthrough — Linux guest client for MetalPassthrough (Metal compute from inside an
 * apple/container Linux VM, zero-copy over unified memory).
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 *
 * All memory returned by mpt_alloc() lives in 2 MiB hugepages that the macOS host maps directly
 * into a Metal buffer: the GPU reads and writes these exact physical pages. Use mpt_devaddr() to
 * obtain the 64-bit device address of a pointer (what Metal kernels and CUDA see).
 */
#ifndef METALPASSTHROUGH_H
#define METALPASSTHROUGH_H

#include <stddef.h>
#include <stdint.h>

#include "metalpassthrough_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MPT_API __attribute__((visibility("default")))

typedef struct mpt_ctx mpt_ctx;

typedef struct {
    uint64_t handle;
    uint32_t max_threads_per_tg;
    uint32_t simd_width;
    uint32_t static_tg_memory;
} mpt_pipeline;

/* Kernel argument for mpt_dispatch(). */
typedef struct {
    uint32_t kind;        /* MPT_BIND_BUFFER | MPT_BIND_BYTES | MPT_BIND_TGMEM */
    uint32_t index;       /* [[buffer(index)]] / [[threadgroup(index)]] */
    const void *ptr;      /* BUFFER: any pointer inside the arena; BYTES: host bytes to inline */
    uint64_t len;         /* BUFFER: bytes accessible (0 = to end of arena); BYTES/TGMEM: length */
} mpt_arg;

/* Session management. arena_bytes == 0 uses $METALPASSTHROUGH_ARENA_MB (default 256). */
MPT_API int mpt_open(mpt_ctx **out, size_t arena_bytes);
MPT_API void mpt_close(mpt_ctx *ctx);
/* Process-wide shared session (opened on first use). NULL on failure; see mpt_last_error(). */
MPT_API mpt_ctx *mpt_default(void);
MPT_API const char *mpt_last_error(void);
MPT_API const char *mpt_strerror(int status);

/* Device / runtime information. */
MPT_API const mpt_device_info *mpt_device(mpt_ctx *ctx);
MPT_API int mpt_get_stats(mpt_ctx *ctx, mpt_stats *out);
MPT_API int mpt_ping(mpt_ctx *ctx, uint64_t *roundtrip_ns);
MPT_API uint32_t mpt_session_id(mpt_ctx *ctx);
/* 1 if every arena pointer is numerically equal to its device address (CUDA-style UVA). */
MPT_API int mpt_unified_addressing(mpt_ctx *ctx);
/* sizeof() of ABI structs by name, for bindings that mirror them (ctypes, Rust FFI). */
MPT_API size_t mpt_abi_sizeof(const char *type);

/* Zero-copy unified memory. */
MPT_API void *mpt_alloc(mpt_ctx *ctx, size_t bytes, size_t align);
MPT_API void mpt_free(mpt_ctx *ctx, void *ptr);
MPT_API size_t mpt_alloc_size(mpt_ctx *ctx, const void *ptr);
MPT_API uint64_t mpt_devaddr(mpt_ctx *ctx, const void *ptr);       /* 0 if not arena memory */
MPT_API void *mpt_hostptr(mpt_ctx *ctx, uint64_t devaddr);         /* NULL if unknown */
MPT_API int mpt_memory_info(mpt_ctx *ctx, uint64_t *free_bytes, uint64_t *total_bytes);

/* Pipelines. log may be NULL. */
MPT_API int mpt_compile_msl(mpt_ctx *ctx, const char *source, const char *function, uint32_t flags,
                              mpt_pipeline *out, char *log, size_t log_len);
MPT_API int mpt_compile_ptx(mpt_ctx *ctx, const char *ptx, const char *entry, uint32_t flags,
                              mpt_pipeline *out, mpt_kernel_info *info, char *log, size_t log_len);
MPT_API int mpt_release_pipeline(mpt_ctx *ctx, const mpt_pipeline *p);

/* Work submission (stream ordered, batched into Metal command buffers). */
MPT_API int mpt_dispatch(mpt_ctx *ctx, const mpt_pipeline *p, const uint32_t grid[3], const uint32_t tg[3],
                           uint32_t mode, const mpt_arg *args, uint32_t nargs, uint32_t flags);
/* Like mpt_dispatch but marks every arena resident (kernels that dereference raw device pointers). */
MPT_API int mpt_dispatch_raw(mpt_ctx *ctx, const mpt_pipeline *p, const uint32_t grid[3], const uint32_t tg[3],
                               const void *params, uint32_t params_len, uint32_t tg_memory, uint32_t flags);
MPT_API int mpt_copy(mpt_ctx *ctx, uint64_t dst_dev, uint64_t src_dev, uint64_t len);
MPT_API int mpt_fill(mpt_ctx *ctx, uint64_t dst_dev, uint64_t len, uint32_t pattern, uint32_t width);
MPT_API int mpt_flush(mpt_ctx *ctx);
/* Waits for all submitted work. gpu_ns (optional) = GPU execution time since the previous sync. */
MPT_API int mpt_sync(mpt_ctx *ctx, uint64_t *gpu_ns);
MPT_API int mpt_pending(mpt_ctx *ctx);   /* 1 if work was submitted since the last sync */

/* GPU timestamp events (ev must be arena memory, e.g. from mpt_alloc). */
MPT_API int mpt_event_record(mpt_ctx *ctx, mpt_event *ev);
MPT_API int mpt_event_wait(mpt_ctx *ctx, mpt_event *ev);

#ifdef __cplusplus
}
#endif
#endif
