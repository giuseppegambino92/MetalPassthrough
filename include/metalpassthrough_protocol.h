/*
 * MetalPassthrough wire/shared-memory ABI. Shared verbatim by the Linux guest (C, aarch64) and the
 * macOS host (Swift via a Clang module). Copyright 2026 MetalPassthrough contributors. Apache-2.0.
 *
 * Data plane
 * ----------
 * A *session* is one guest process <-> one host worker. The guest owns all memory: it mmaps
 * MAP_HUGETLB 2 MiB pages ("chunks"), stamps each chunk with {nonce, index}, and asks the host to
 * locate them in guest RAM. The host remaps the chunks into a contiguous host VA range and wraps
 * that with MTLDevice.makeBuffer(bytesNoCopy:). The resulting MTLBuffer *is* the guest memory.
 *
 * Every reference to guest memory on the wire is a *device address*: arena.gpu_base + offset,
 * where gpu_base is MTLBuffer.gpuAddress. Device addresses are therefore directly dereferenceable
 * by Metal kernels (they are what libcuda hands out as CUdeviceptr).
 *
 * Arena 0 begins with the 64 KiB control region below: ring header, submission queue (SQ),
 * completion queue (CQ), device info block. The *host* initializes the control region after it
 * has located arena 0 (the guest's chunk stamp occupies its first bytes until then); the guest
 * must not touch it before receiving the HELLO ack. The AF_VSOCK connection (host CID 2, port 2049)
 * carries the HELLO handshake and then only 1-byte doorbells.
 */
#ifndef METALPASSTHROUGH_PROTOCOL_H
#define METALPASSTHROUGH_PROTOCOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MPT_MAGIC            0x55504743u /* "MPT" little endian */
#define MPT_ABI_VERSION      2u        /* v2: streams, events with dependencies, capability bits */
#define MPT_VSOCK_PORT       2049u
#define MPT_HOST_CID         2u
#define MPT_CACHELINE        128u        /* Apple Silicon L2 line */
#define MPT_CHUNK_SIZE       2097152u    /* 2 MiB guest hugepage; multiple of the 16 KiB host page */
#define MPT_HOST_PAGE        16384u
#define MPT_CONTROL_SIZE     65536u
#define MPT_SQ_DEPTH         256u
#define MPT_CQ_DEPTH         256u
#define MPT_SCRATCH_PER_SLOT 4096u       /* per-SQ-slot descriptor scratch (see MPT_SCRATCH_OFFSET) */

/* Control region layout (offsets from arena 0 base). */
#define MPT_RING_OFFSET      0u
#define MPT_SQ_OFFSET        1024u
#define MPT_CQ_OFFSET        17408u      /* SQ_OFFSET + SQ_DEPTH * 64 */
#define MPT_INFO_OFFSET      32768u
#define MPT_STATS_OFFSET     49152u
#define MPT_SCRATCH_OFFSET   65536u      /* = CONTROL_SIZE; 1 MiB, slot-indexed */
#define MPT_SCRATCH_SIZE     1048576u    /* SQ_DEPTH * SCRATCH_PER_SLOT */
#define MPT_HEAP_OFFSET      1114112u    /* SCRATCH_OFFSET + SCRATCH_SIZE: first user byte */

/* Layouts are explicitly padded; alignment attributes only add guarantees for C (verified by
 * abi_check.c static asserts). Swift's Clang importer drops over-aligned types, so omit them there. */
#if defined(__swift__)
#define MPT_ALIGNED(n)
#else
#define MPT_ALIGNED(n) __attribute__((aligned(n)))
#endif

/* Fixed-underlying-type enums (Clang extension / C23 / C++) give Swift typed constants. */
#if defined(__clang__) || defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#define MPT_ENUM(T) enum : T
#else
#define MPT_ENUM(T) enum
#endif

/* Chunk stamp written by the guest at the start of every chunk before HELLO / ALLOC_ARENA. The
 * host verifies it and then the guest is free to overwrite it. */
typedef struct {
    uint8_t  nonce[16];
    uint64_t index;
    uint64_t count;
} mpt_chunk_stamp;

/* ---- Handshake over vsock -------------------------------------------------------------- */
typedef struct {
    uint32_t magic;       /* MPT_MAGIC */
    uint32_t version;     /* MPT_ABI_VERSION */
    uint8_t  nonce[16];
    uint32_t nchunks;     /* arena 0 size = nchunks * MPT_CHUNK_SIZE */
    uint32_t pid;         /* guest pid, informational */
    char     comm[16];    /* guest process name, informational */
} mpt_hello;            /* 48 bytes */

typedef struct {
    uint32_t magic;
    int32_t  status;      /* 0 or negative MPT_E* */
    uint64_t gpu_base;    /* MTLBuffer.gpuAddress of arena 0 */
    uint32_t session_id;
    uint32_t arena_id;    /* always 0 */
    uint64_t caps;        /* MPT_CAP_* negotiated for this session */
} mpt_hello_ack;        /* 32 bytes */

/* Keeper handshake (second vsock connection, opened by the guest library right after HELLO).
 * The keeper is a tiny guest process that holds the arena memfds; the host closes this connection
 * (after writing one byte) only once the session's GPU work has drained and its arenas are
 * unmapped. Until then the guest kernel cannot recycle the hugepages, even if the client process
 * was SIGKILLed with GPU work still in flight. Authenticated by session id + the arena-0 nonce. */
#define MPT_KEEPER_MAGIC 0x5045454Bu /* "KEEP" little endian */
typedef struct {
    uint32_t magic;       /* MPT_KEEPER_MAGIC */
    uint32_t session_id;
    uint8_t  nonce[16];
    uint64_t reserved;
} mpt_keeper_hello;      /* 32 bytes */

/* Capabilities (HELLO ack caps / ALLOC_ARENA r2). UVA is *negotiated*: the host only sets
 * MPT_CAP_UVA when it managed to place the arena at a 2 MiB-aligned GPU VA; guests must then
 * still verify their own relocation succeeded, and fall back to translated addressing otherwise. */
#define MPT_CAP_UVA      0x1ull   /* arena gpu_base is 2 MiB aligned: guest may mremap to VA == gpu_base */
#define MPT_CAP_STREAMS  0x2ull   /* independent Metal queues per stream, event waits */
#define MPT_CAP_KEEPER   0x4ull   /* host honours keeper connections (crash-safe teardown) */

/* ---- Ring header (arena 0 + 0). Each line is written by exactly one side. ---------------- */
typedef struct MPT_ALIGNED(MPT_CACHELINE) {
    /* line 0: immutable after HELLO (written by guest before HELLO) */
    uint32_t magic, version;
    uint32_t sq_depth, cq_depth;
    uint64_t arena_bytes;
    uint8_t  _pad0[MPT_CACHELINE - 24];
    /* line 1: guest-owned producer state */
    MPT_ALIGNED(MPT_CACHELINE) uint32_t sq_tail;
    uint32_t guest_waiting;      /* guest is (about to be) blocked on the doorbell for a CQE */
    uint8_t  _pad1[MPT_CACHELINE - 8];
    /* line 2: host-owned consumer state */
    MPT_ALIGNED(MPT_CACHELINE) uint32_t sq_head;
    uint32_t host_sleeping;      /* host worker parked on the doorbell */
    uint8_t  _pad2[MPT_CACHELINE - 8];
    /* line 3: host-owned producer state */
    MPT_ALIGNED(MPT_CACHELINE) uint32_t cq_tail;
    uint8_t  _pad3[MPT_CACHELINE - 4];
    /* line 4: guest-owned consumer state */
    MPT_ALIGNED(MPT_CACHELINE) uint32_t cq_head;
    uint8_t  _pad4[MPT_CACHELINE - 4];
} mpt_ring_header;

/* ---- Submission queue entry (64 bytes) -------------------------------------------------- */
typedef struct MPT_ALIGNED(64) {
    uint16_t opcode;
    uint16_t flags;
    uint32_t tag;         /* echoed in the CQE */
    uint64_t a[7];
} mpt_sqe;

/* ---- Completion queue entry (32 bytes) -------------------------------------------------- */
typedef struct MPT_ALIGNED(32) {
    uint32_t tag;
    int32_t  status;      /* 0 or negative MPT_E* */
    uint64_t r[3];
} mpt_cqe;

/* SQE flags */
#define MPT_F_NO_CQE  0x0001u  /* fire-and-forget: no completion entry is posted on success */
#define MPT_F_COMMIT  0x0002u  /* DISPATCH/COPY/FILL: commit the open command buffer afterwards */
/* Stream selector for stream-ordered ops (DISPATCH, COPY, FILL, FLUSH, EVENT_RECORD, STREAM_*):
 * bits 15:8 of flags. Stream 0 is the CUDA legacy default stream (implicitly synchronizes with
 * every blocking stream); streams 1..255 are created with STREAM_CREATE. */
#define MPT_F_STREAM_SHIFT 8u
#define MPT_F_STREAM_MASK  0xFF00u
#define MPT_MAX_STREAMS    256u
static inline uint16_t mpt_stream_flags(uint32_t stream) { return (uint16_t)((stream & 0xFFu) << MPT_F_STREAM_SHIFT); }

/* Opcodes. Argument conventions: "dev" = device address, "len" = byte length. */
MPT_ENUM(uint16_t) {
    MPT_OP_NOP          = 0,
    MPT_OP_PING         = 1,  /* -> r0 = host CLOCK_UPTIME_RAW ns */
    MPT_OP_DEVICE_INFO  = 2,  /* refreshes the info block -> r0 = sizeof(mpt_device_info) */
    MPT_OP_ALLOC_ARENA  = 3,  /* a0,a1 = nonce lo/hi, a2 = nchunks -> r0 = arena id, r1 = gpu_base */
    MPT_OP_FREE_ARENA   = 4,  /* a0 = arena id */
    MPT_OP_COMPILE_MSL  = 5,  /* a0 = src dev, a1 = src len, a2 = fn dev, a3 = fn len, a4 = MPT_CF_*,
                                  a5 = log dev, a6 = log len  -> r0 = pipeline, r1 = mpt_pack_tg(), r2 = static tg mem */
    MPT_OP_COMPILE_PTX  = 6,  /* same args as COMPILE_MSL with PTX source + entry name; the host
                                  writes a mpt_kernel_info (+ error log) at a5 -> r0,r1,r2 as above */
    MPT_OP_DISPATCH     = 7,  /* a0 = pipeline, a1 = mpt_dispatch_desc dev, a2 = len */
    MPT_OP_SYNC         = 8,  /* commit + wait for all prior work -> r0 = GPU ns since last SYNC,
                                  r1 = last gpuStartTime ns, r2 = last gpuEndTime ns (host clock) */
    MPT_OP_COPY         = 9,  /* a0 = dst dev, a1 = src dev, a2 = len   (GPU blit, stream ordered) */
    MPT_OP_FILL         = 10, /* a0 = dst dev, a1 = len, a2 = 32-bit pattern, a3 = pattern bytes (1|2|4) */
    MPT_OP_EVENT_RECORD = 11, /* a0 = dev of mpt_event; commits; host writes it on GPU completion */
    MPT_OP_RELEASE_PIPE = 12, /* a0 = pipeline */
    MPT_OP_STATS        = 13, /* refreshes the stats block -> r0 = sizeof(mpt_stats) */
    MPT_OP_CLOSE        = 14,
    MPT_OP_FLUSH        = 15, /* commit the open command buffer without waiting */
    MPT_OP_STREAM_CREATE = 16, /* a0 = MPT_STREAM_* flags -> r0 = stream id (1..255) */
    MPT_OP_STREAM_DESTROY = 17, /* stream in flags; pending work still completes */
    MPT_OP_STREAM_SYNC  = 18, /* stream in flags: commit + wait that stream (stream 0: + all blocking
                                  streams) -> r0 = GPU ns of that stream since its last sync */
    MPT_OP_STREAM_QUERY = 19, /* stream in flags: commit pending work, never waits -> r0 = 1 idle, 0 busy */
    MPT_OP_STREAM_WAIT_EVENT = 20, /* stream in flags, a0 = mpt_event dev: later work on the stream waits
                                       for the event's most recent record (no-op if never recorded) */
    MPT_OP_EVENT_DESTROY = 21, /* a0 = mpt_event dev: drop host-side event state */
    MPT_OP_GEMM         = 22, /* a0 = mpt_gemm_desc dev, a1 = len; stream in flags. Host-library GEMM
                                  (MPSMatrixMultiplication) on arena memory, stream ordered */
    MPT_OP__COUNT
};

/* STREAM_CREATE flags */
#define MPT_STREAM_NON_BLOCKING 0x1u  /* no implicit synchronization with stream 0 */

/* Compile flags */
#define MPT_CF_FAST_MATH   0x1u   /* default is IEEE-safe math (bit-exact with CPU fmaf references) */
#define MPT_CF_FP64_LOWER  0x2u   /* PTX: lower .f64 to fp32 (Metal has no double ALU) */
#define MPT_CF_DUMP_MSL    0x4u   /* PTX: append the generated MSL to the log buffer */

static inline uint64_t mpt_pack_tg(uint32_t max_threads, uint32_t exec_width) {
    return (uint64_t)max_threads | ((uint64_t)exec_width << 32);
}

/* Error codes */
MPT_ENUM(int32_t) {
    MPT_OK         = 0,
    MPT_EINVAL     = -22,
    MPT_ENOMEM     = -12,
    MPT_ENOENT     = -2,
    MPT_EFAULT     = -14,  /* device address outside every arena of the session */
    MPT_ECOMPILE   = -1000,
    MPT_ETRANSLATE = -1001,
    MPT_EGPU       = -1002, /* command buffer error */
    MPT_ENOTFOUND  = -1003, /* arena chunks not located in guest RAM */
    MPT_EPROTO     = -1004,
};

/* ---- Dispatch descriptor (lives in guest memory; read by the host at encode time) -------- */
MPT_ENUM(uint32_t) { MPT_DISPATCH_THREADGROUPS = 0, MPT_DISPATCH_THREADS = 1 };
MPT_ENUM(uint32_t) {
    MPT_BIND_BUFFER   = 0, /* addr = device address; bound with setBuffer(arena, offset:) */
    MPT_BIND_BYTES    = 1, /* inline bytes at desc + addr (offset within descriptor), len bytes */
    MPT_BIND_TGMEM    = 2, /* threadgroup memory length = len at threadgroup index `index` */
};
typedef struct {
    uint32_t kind;
    uint32_t index;
    uint64_t addr;
    uint64_t len;
} mpt_binding;             /* 24 bytes */

typedef struct {
    uint32_t grid[3];       /* threadgroups (THREADGROUPS mode) or threads (THREADS mode) */
    uint32_t tg[3];         /* threads per threadgroup */
    uint32_t mode;
    uint32_t nbindings;
    uint32_t use_all_arenas;/* make every arena resident (needed when kernels chase raw pointers) */
    uint32_t _pad;
    mpt_binding bindings[]; /* followed by inline bytes */
} mpt_dispatch_desc;

/* ---- GEMM descriptor (BLAS semantics, column-major like cuBLAS) --------------------------
 * C = alpha * op(A) * op(B) + beta * C, op(A) is m x k, op(B) is k x n, C is m x n. */
/* dtype = A/B element type | C element type << 8 | MPT_GEMM_ACC16 (accumulate in fp16).
 * MPT_GEMM_F32 (= 0) is the plain fp32 SGEMM. alpha/beta are always passed as float. */
enum { MPT_GEMM_F32 = 0, MPT_GEMM_F16 = 1, MPT_GEMM_BF16 = 2 };
#define MPT_GEMM_ACC16 (1u << 16)
#define MPT_GEMM_DTYPE(ab, c, acc16) ((uint32_t)(ab) | ((uint32_t)(c) << 8) | ((acc16) ? MPT_GEMM_ACC16 : 0u))
typedef struct {
    uint32_t dtype;       /* MPT_GEMM_DTYPE(...) */
    uint32_t trans_a;     /* 0 = N, 1 = T */
    uint32_t trans_b;
    uint32_t m, n, k;
    uint32_t lda, ldb, ldc;
    uint32_t batch;       /* >= 1; strided batched when > 1 */
    float    alpha, beta;
    uint64_t a, b, c;     /* device addresses */
    uint64_t stride_a, stride_b, stride_c;  /* elements between batch entries */
} mpt_gemm_desc;          /* 96 bytes */

/* ---- PTX kernel signature returned by COMPILE_PTX ---------------------------------------- */
#define MPT_MAX_PARAMS 64
typedef struct {
    uint32_t nparams;
    uint32_t params_bytes;          /* size of the packed parameter block bound at buffer(0) */
    uint32_t shared_bytes;          /* static .shared size (dynamic shared starts here) */
    uint32_t flags;                 /* MPT_KF_* */
    uint32_t param_offset[MPT_MAX_PARAMS];
    uint32_t param_size[MPT_MAX_PARAMS];
    /* followed by a NUL terminated log (translator warnings / compiler errors / MSL dump) */
} mpt_kernel_info;

#define MPT_KF_USES_SHARED  0x1u  /* bind threadgroup(0) = shared_bytes + dynamic shared */
#define MPT_KF_USES_FP64    0x2u
#define MPT_KF_FP64_LOWERED 0x4u  /* fp32 lowering; otherwise df64 emulation */

/* ---- GPU timestamp event (guest memory, written by host on completion) ------------------- */
typedef struct MPT_ALIGNED(16) {
    uint64_t gpu_end_ns;            /* gpuEndTime of the last command buffer before the record */
    uint32_t done;                  /* set to 1 (release) by the host */
    int32_t  status;
} mpt_event;

/* ---- Device info block (arena 0 + MPT_INFO_OFFSET, host-written) ------------------------ */
typedef struct {
    uint32_t magic;
    uint32_t gpu_cores;             /* IORegistry gpu-core-count */
    char     name[64];              /* MTLDevice.name, e.g. "Apple M3 Max" */
    char     metal_family[32];      /* highest MTLGPUFamily.apple* supported, e.g. "Apple9" */
    char     host_os[48];
    uint64_t unified_memory_bytes;  /* hw.memsize */
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
    uint64_t current_allocated;     /* MTLDevice.currentAllocatedSize */
    uint32_t max_threads_per_tg;
    uint32_t threadgroup_memory;    /* maxThreadgroupMemoryLength */
    uint32_t simd_width;
    uint32_t has_unified_memory;
    double   measured_bandwidth_gbs;/* host blit benchmark at daemon start */
    uint32_t host_page_size;
    uint32_t abi_version;
    char     daemon_version[16];
} mpt_device_info;

#define MPT_MAX_PROCS 64

typedef struct {
    uint32_t pid;
    uint32_t session;
    uint64_t arena_bytes;
    uint64_t dispatches;
    uint64_t gpu_ns;
    char     comm[16];
} mpt_proc_info;                   /* 48 bytes */

/* ---- Stats block (arena 0 + MPT_STATS_OFFSET, host-written) ----------------------------- */
typedef struct {
    uint64_t commands;
    uint64_t dispatches;
    uint64_t command_buffers;
    uint64_t pipeline_compiles;     /* actual Metal compiles */
    uint64_t pipeline_cache_hits;   /* served from the in-memory or binary-archive cache */
    uint64_t last_kernel_gpu_ns;
    uint64_t total_gpu_ns;
    uint64_t arena_bytes;           /* all arenas of this session */
    uint32_t arenas;
    uint32_t active_sessions_vm;    /* sessions in this VM */
    uint32_t active_containers;     /* MetalPassthrough-enabled VMs on this host */
    uint32_t doorbells;             /* times the host worker was woken from park */
    uint64_t pipelines_resident;
    /* VM-wide view (all sessions of this container's VM) */
    uint64_t vm_arena_bytes;
    uint64_t vm_dispatches;
    uint64_t vm_total_gpu_ns;
    uint64_t vm_last_kernel_gpu_ns;
    uint32_t nprocs;
    uint32_t _pad;
    mpt_proc_info procs[MPT_MAX_PROCS];
} mpt_stats;

#ifdef __cplusplus
}
#endif
#endif /* METALPASSTHROUGH_PROTOCOL_H */
