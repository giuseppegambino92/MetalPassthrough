/*
 * libmetalpassthrough — guest side of the MetalPassthrough protocol.
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */

#include "metalpassthrough.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>
#include <linux/vm_sockets.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/wait.h>

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
#ifndef MFD_HUGE_2MB
#define MFD_HUGE_2MB (21U << 26)
#endif
#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 0x0001U
#endif

#define MPT_MAX_ARENAS 64
#define MPT_MAX_FREE   8192

typedef struct { uint64_t off, len; } range_t;

typedef struct {
    uint8_t *base;
    uint64_t len;
    uint64_t gpu_base;
    uint32_t id;
    range_t *free;          /* sorted by offset, coalesced */
    uint32_t nfree;
    range_t *live;          /* live allocations, sorted by offset */
    uint32_t nlive, caplive;
} arena_t;

struct mpt_ctx {
    int fd;
    uint32_t session;
    arena_t arenas[MPT_MAX_ARENAS];
    _Atomic uint32_t narenas;
    mpt_ring_header *ring;
    mpt_sqe *sq;
    mpt_cqe *cq;
    uint8_t *scratch;
    uint64_t scratch_dev;
    uint32_t sq_tail, cq_head, tag;
    int sticky;
    int pending;
    uint64_t spin_ns;
    uint64_t grow_bytes;
    int debug;
    int uva;                         /* every arena satisfies guest VA == device address */
    int closed;                      /* session ended (process exit); calls fail with EPROTO */
    uint64_t caps;                   /* MPT_CAP_* negotiated at HELLO */
    int keeper;                      /* fd-passing socket to the keeper process, or -1 */
    pthread_mutex_t lock;            /* serializes ring submission */
    atomic_flag alock;               /* allocator spinlock */
};

static __thread char g_err[256];
MPT_API const char *mpt_last_error(void) { return g_err; }

static int set_err(int status, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
    return status;
}

MPT_API const char *mpt_strerror(int s) {
    switch (s) {
    case MPT_OK: return "success";
    case MPT_EINVAL: return "invalid argument";
    case MPT_ENOMEM: return "out of arena memory (reserve more hugepages: mpt-run --hugepages N)";
    case MPT_ENOENT: return "no such object";
    case MPT_EFAULT: return "device address outside the session's arenas";
    case MPT_ECOMPILE: return "Metal shader compilation failed";
    case MPT_ETRANSLATE: return "PTX to MSL translation failed";
    case MPT_EGPU: return "GPU command buffer error";
    case MPT_ENOTFOUND: return "host could not locate the arena in guest memory";
    case MPT_EPROTO: return "protocol error / host connection lost";
    default: return "unknown error";
    }
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline void cpu_relax(void) { __asm__ __volatile__("yield" ::: "memory"); }

static uint64_t env_u64(const char *name, uint64_t def) {
    const char *v = getenv(name);
    return (v && *v) ? strtoull(v, NULL, 0) : def;
}

#define DBG(ctx, ...) do { if ((ctx)->debug) { fprintf(stderr, "[libmetalpassthrough] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* ------------------------------------------------------------------ transport */

static int connect_host(void) {
    const char *path = getenv("METALPASSTHROUGH_SOCKET");
    int fd;
    if (path && *path) {
        fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (fd < 0) return -errno;
        struct sockaddr_un a = { .sun_family = AF_UNIX };
        strncpy(a.sun_path, path, sizeof a.sun_path - 1);
        if (connect(fd, (struct sockaddr *)&a, sizeof a) < 0) { int e = -errno; close(fd); return e; }
        return fd;
    }
    fd = socket(AF_VSOCK, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -errno;
    struct sockaddr_vm a = { .svm_family = AF_VSOCK, .svm_cid = MPT_HOST_CID,
                             .svm_port = (unsigned)env_u64("METALPASSTHROUGH_PORT", MPT_VSOCK_PORT) };
    if (connect(fd, (struct sockaddr *)&a, sizeof a) < 0) { int e = -errno; close(fd); return e; }
    return fd;
}

static int io_all(int fd, void *buf, size_t n, int wr) {
    uint8_t *p = buf;
    while (n) {
        ssize_t r = wr ? write(fd, p, n) : read(fd, p, n);
        if (r == 0) return MPT_EPROTO;
        if (r < 0) { if (errno == EINTR) continue; return MPT_EPROTO; }
        p += r;
        n -= (size_t)r;
    }
    return 0;
}

/* Empties the doorbell socket. Returns -1 once the host closed it: Linux vsock reports a peer
 * close as POLLIN + a 0-byte read, not necessarily POLLHUP. */
static int drain(int fd) {
    uint8_t b[64];
    ssize_t n;
    while ((n = recv(fd, b, sizeof b, MSG_DONTWAIT)) > 0) {}
    return n == 0 ? -1 : 0;
}

/* Non-blocking: has the host ended this session? */
static int host_gone(int fd) {
    struct pollfd p = { .fd = fd, .events = POLLIN | POLLRDHUP };
    if (poll(&p, 1, 0) <= 0) return 0;
    if (p.revents & (POLLHUP | POLLERR | POLLRDHUP)) return 1;
    uint8_t b;
    return recv(fd, &b, 1, MSG_DONTWAIT | MSG_PEEK) == 0;
}

/* ------------------------------------------------------------------ hugepage arenas */

/* Maps n hugepage chunks. Prefers a hugetlb memfd (so the keeper process can hold the pages past
 * the client's death); falls back to an anonymous MAP_HUGETLB mapping. *memfd = -1 on fallback. */
static void *map_chunks(uint32_t n, uint8_t nonce[16], int *memfd) {
    size_t len = (size_t)n * MPT_CHUNK_SIZE;
    void *p = MAP_FAILED;
    *memfd = -1;
    if (!getenv("METALPASSTHROUGH_NO_KEEPER")) {
        int fd = (int)syscall(SYS_memfd_create, "metalpassthrough-arena", MFD_HUGETLB | MFD_HUGE_2MB | MFD_CLOEXEC);
        if (fd >= 0) {
            if (ftruncate(fd, (off_t)len) == 0)
                p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
            if (p == MAP_FAILED) close(fd); else *memfd = fd;
        }
    }
    if (p == MAP_FAILED)
        p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
    if (p == MAP_FAILED) return NULL;
    /* Children must not inherit (and COW-split) pages the host GPU addresses physically. */
    madvise(p, len, MADV_DONTFORK);
    if (getrandom(nonce, 16, 0) != 16) {
        for (int i = 0; i < 16; i++) nonce[i] = (uint8_t)(rand() ^ (now_ns() >> (i * 3)));
    }
    for (uint32_t i = 0; i < n; i++) {
        mpt_chunk_stamp *st = (mpt_chunk_stamp *)((uint8_t *)p + (size_t)i * MPT_CHUNK_SIZE);
        memcpy(st->nonce, nonce, 16);
        st->index = i;
        st->count = n;
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
    return p;
}

/* ------------------------------------------------------------------ keeper (crash-safe teardown)
 * A grandchild process that holds the arena memfds and a vsock connection to the host. If the client
 * dies with GPU work in flight, the hugepages stay allocated (to the keeper) until the host has
 * drained the GPU, unmapped the arenas and released the keeper. Runs only async-signal-safe code. */

#define KEEP_KFD 200
#define KEEP_PAIR 201
#define KEEP_MEM0 202

static void keeper_main(int kfd, int pair, int memfd) {
    setsid();  /* leave the client's process group: terminal signals must not kill the keeper */
    dup2(kfd, KEEP_KFD);
    dup2(pair, KEEP_PAIR);
    if (memfd >= 0) dup2(memfd, KEEP_MEM0);
    int null = open("/dev/null", O_RDWR);
    if (null >= 0) { dup2(null, 0); dup2(null, 1); dup2(null, 2); }
    if (syscall(SYS_close_range, 3, KEEP_KFD - 1, 0) != 0)
        for (int fd = 3; fd < KEEP_KFD; fd++) close(fd);
    if (syscall(SYS_close_range, KEEP_MEM0 + 1, ~0U, 0) != 0)
        for (int fd = KEEP_MEM0 + 1; fd < 4096; fd++) close(fd);
    struct pollfd pf[2] = { { .fd = KEEP_KFD, .events = POLLIN }, { .fd = KEEP_PAIR, .events = POLLIN } };
    for (;;) {
        int n = poll(pf, 2, -1);
        if (n < 0) { if (errno == EINTR) continue; _exit(1); }
        if (pf[0].revents) _exit(0);  /* host released us (or the host went away): pages may be reused */
        if (pf[1].revents) {
            char data;
            char cbuf[CMSG_SPACE(sizeof(int))];
            struct iovec iov = { .iov_base = &data, .iov_len = 1 };
            struct msghdr m = { .msg_iov = &iov, .msg_iovlen = 1, .msg_control = cbuf, .msg_controllen = sizeof cbuf };
            ssize_t r = recvmsg(KEEP_PAIR, &m, 0);
            if (r <= 0) { close(KEEP_PAIR); pf[1].fd = -1; continue; }  /* client gone: keep holding */
            /* Received memfds simply stay open in this process: holding the fd holds the pages. */
        }
    }
}

/* Starts the keeper for a freshly established session. Returns the client end of the fd-passing
 * socket (or -1 when the keeper is unavailable; the session still works, without crash safety). */
static int start_keeper(uint32_t session, const uint8_t nonce[16], int memfd0) {
    if (getenv("METALPASSTHROUGH_NO_KEEPER") || memfd0 < 0) return -1;
    int kfd = connect_host();
    if (kfd < 0) return -1;
    mpt_keeper_hello kh = { .magic = MPT_KEEPER_MAGIC, .session_id = session };
    memcpy(kh.nonce, nonce, 16);
    if (io_all(kfd, &kh, sizeof kh, 1)) { close(kfd); return -1; }
    int sp[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sp) != 0) { close(kfd); return -1; }
    pid_t child = fork();
    if (child < 0) { close(kfd); close(sp[0]); close(sp[1]); return -1; }
    if (child == 0) {  /* intermediate child: detach the keeper so init reaps it */
        pid_t k = fork();
        if (k == 0) keeper_main(kfd, sp[1], memfd0);
        _exit(k < 0);
    }
    int st;
    while (waitpid(child, &st, 0) < 0 && errno == EINTR) {}
    close(kfd);
    close(sp[1]);
    return sp[0];
}

static void keeper_hold(int pair, int memfd) {
    if (pair < 0 || memfd < 0) return;
    char data = 'M';
    char cbuf[CMSG_SPACE(sizeof(int))];
    memset(cbuf, 0, sizeof cbuf);
    struct iovec iov = { .iov_base = &data, .iov_len = 1 };
    struct msghdr m = { .msg_iov = &iov, .msg_iovlen = 1, .msg_control = cbuf, .msg_controllen = sizeof cbuf };
    struct cmsghdr *cm = CMSG_FIRSTHDR(&m);
    cm->cmsg_level = SOL_SOCKET;
    cm->cmsg_type = SCM_RIGHTS;
    cm->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cm), &memfd, sizeof(int));
    (void)!sendmsg(pair, &m, MSG_NOSIGNAL);
}

/* Unified virtual addressing: move the hugepage mapping so that guest VA == device address. The host
 * places chunk 0 at a 2 MiB-aligned GPU VA; hugetlb mremap needs that alignment. Returns the
 * (possibly unchanged) base. Never clobbers existing mappings (MAP_FIXED_NOREPLACE probe). */
static uint8_t *relocate_uva(uint8_t *base, size_t len, uint64_t gpu_base, uint64_t caps, int *uva) {
    *uva = 0;
    if (!(caps & MPT_CAP_UVA) || getenv("METALPASSTHROUGH_NO_UVA") || gpu_base % MPT_CHUNK_SIZE || (uintptr_t)base == gpu_base) {
        *uva = (uintptr_t)base == gpu_base;
        return base;
    }
    void *want = (void *)(uintptr_t)gpu_base;
    void *r = mmap(want, len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE | MAP_NORESERVE, -1, 0);
    if (r != want) { if (r != MAP_FAILED) munmap(r, len); return base; }
    void *m = mremap(base, len, len, MREMAP_MAYMOVE | MREMAP_FIXED, want);
    if (m != want) { munmap(want, len); return base; }
    *uva = 1;
    return (uint8_t *)m;
}

static uint64_t hugepages_free(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[128];
    unsigned long long v = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "HugePages_Free: %llu", &v) == 1) break;
    }
    fclose(f);
    return v;
}

static void arena_init(arena_t *a, uint8_t *base, uint64_t len, uint64_t gpu_base, uint32_t id, uint64_t first) {
    a->base = base;
    a->len = len;
    a->gpu_base = gpu_base;
    a->id = id;
    a->free = calloc(MPT_MAX_FREE, sizeof(range_t));
    a->nfree = 1;
    a->free[0].off = first;
    a->free[0].len = len - first;
}

/* ------------------------------------------------------------------ ring */

#define RING_TAIL(c)    ((_Atomic uint32_t *)&(c)->ring->sq_tail)
#define RING_HEAD(c)    ((_Atomic uint32_t *)&(c)->ring->sq_head)
#define RING_SLEEP(c)   ((_Atomic uint32_t *)&(c)->ring->host_sleeping)
#define RING_WAIT(c)    ((_Atomic uint32_t *)&(c)->ring->guest_waiting)
#define RING_CQTAIL(c)  ((_Atomic uint32_t *)&(c)->ring->cq_tail)
#define RING_CQHEAD(c)  ((_Atomic uint32_t *)&(c)->ring->cq_head)

static void doorbell(mpt_ctx *c) {
    atomic_thread_fence(memory_order_seq_cst);
    if (atomic_load_explicit(RING_SLEEP(c), memory_order_seq_cst)) {
        uint8_t b = 1;
        (void)!write(c->fd, &b, 1);
    }
}

#define SQ_DEAD UINT32_MAX
/* Reserves the next SQ slot (waits for space). Caller holds c->lock. Returns SQ_DEAD if the host
 * ended the session while the SQ was full. */
static uint32_t sq_reserve(mpt_ctx *c) {
    for (uint64_t spins = 0; c->sq_tail - atomic_load_explicit(RING_HEAD(c), memory_order_acquire) >= MPT_SQ_DEPTH; spins++) {
        doorbell(c);
        cpu_relax();
        if ((spins & 0xffff) == 0xffff && host_gone(c->fd)) {
            set_err(MPT_EPROTO, "host closed the MetalPassthrough session");
            return SQ_DEAD;
        }
    }
    return c->sq_tail % MPT_SQ_DEPTH;
}

static void sq_publish(mpt_ctx *c, const mpt_sqe *e) {
    c->sq[c->sq_tail % MPT_SQ_DEPTH] = *e;
    c->sq_tail++;
    atomic_store_explicit(RING_TAIL(c), c->sq_tail, memory_order_release);
    doorbell(c);
}

static int cq_wait(mpt_ctx *c, mpt_cqe *out) {
    uint64_t deadline = now_ns() + c->spin_ns;
    for (;;) {
        if (atomic_load_explicit(RING_CQTAIL(c), memory_order_acquire) != c->cq_head) break;
        if (now_ns() < deadline) { cpu_relax(); continue; }
        /* Park on the doorbell socket. */
        atomic_store_explicit(RING_WAIT(c), 1, memory_order_seq_cst);
        if (atomic_load_explicit(RING_CQTAIL(c), memory_order_seq_cst) == c->cq_head) {
            struct pollfd p = { .fd = c->fd, .events = POLLIN | POLLRDHUP };
            int r = poll(&p, 1, 1000);
            if (r > 0) {
                if (p.revents & (POLLHUP | POLLERR | POLLRDHUP)) { atomic_store(RING_WAIT(c), 0); return set_err(MPT_EPROTO, "host closed the MetalPassthrough session"); }
                if (drain(c->fd) < 0) { atomic_store(RING_WAIT(c), 0); return set_err(MPT_EPROTO, "host closed the MetalPassthrough session"); }
            }
        }
        atomic_store_explicit(RING_WAIT(c), 0, memory_order_seq_cst);
        deadline = now_ns() + c->spin_ns;
    }
    *out = c->cq[c->cq_head % MPT_CQ_DEPTH];
    c->cq_head++;
    atomic_store_explicit(RING_CQHEAD(c), c->cq_head, memory_order_release);
    return 0;
}

/* Publishes e (already placed scratch data for the reserved slot) and optionally waits for its CQE. */
static int submit_locked(mpt_ctx *c, mpt_sqe *e, mpt_cqe *out) {
    e->tag = ++c->tag;
    int want = !(e->flags & MPT_F_NO_CQE);
    sq_publish(c, e);
    if (!want) return 0;
    for (;;) {
        mpt_cqe q;
        int r = cq_wait(c, &q);
        if (r) return r;
        if (q.tag == e->tag) {
            if (out) *out = q;
            return q.status;
        }
        /* Completion for an earlier fire-and-forget op: only posted on failure. */
        if (q.status && !c->sticky) c->sticky = q.status;
    }
}

static int submit(mpt_ctx *c, mpt_sqe *e, mpt_cqe *out) {
    if (c->closed) return MPT_EPROTO;
    pthread_mutex_lock(&c->lock);
    if (sq_reserve(c) == SQ_DEAD) { pthread_mutex_unlock(&c->lock); return MPT_EPROTO; }
    int r = submit_locked(c, e, out);
    pthread_mutex_unlock(&c->lock);
    return r;
}

/* ------------------------------------------------------------------ allocator */

static inline void alock(mpt_ctx *c) { while (atomic_flag_test_and_set_explicit(&c->alock, memory_order_acquire)) cpu_relax(); }
static inline void aunlock(mpt_ctx *c) { atomic_flag_clear_explicit(&c->alock, memory_order_release); }

/* Live allocations per arena, sorted by offset: exact-base lookup for free(), containing-range
 * lookup for bounds checks and interior pointers (cuMemGetAddressRange). */
static int live_find(const arena_t *a, uint64_t off) {  /* index of the last entry with .off <= off, or -1 */
    int lo = 0, hi = (int)a->nlive - 1, r = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (a->live[mid].off <= off) { r = mid; lo = mid + 1; } else hi = mid - 1;
    }
    return r;
}

static int live_insert(arena_t *a, uint64_t off, uint64_t len) {
    if (a->nlive == a->caplive) {
        uint32_t nc = a->caplive ? a->caplive * 2 : 256;
        range_t *nl = realloc(a->live, nc * sizeof(range_t));
        if (!nl) return -1;
        a->live = nl;
        a->caplive = nc;
    }
    int k = live_find(a, off) + 1;
    memmove(&a->live[k + 1], &a->live[k], (a->nlive - (uint32_t)k) * sizeof(range_t));
    a->live[k].off = off;
    a->live[k].len = len;
    a->nlive++;
    return 0;
}

static void live_remove(arena_t *a, int k) {
    memmove(&a->live[k], &a->live[k + 1], (a->nlive - (uint32_t)k - 1) * sizeof(range_t));
    a->nlive--;
}

static int arena_alloc(arena_t *a, uint64_t size, uint64_t align, uint64_t *off_out) {
    for (uint32_t i = 0; i < a->nfree; i++) {
        uint64_t start = (a->free[i].off + align - 1) & ~(align - 1);
        uint64_t end = a->free[i].off + a->free[i].len;
        if (start + size > end) continue;
        uint64_t pre = start - a->free[i].off, post = end - (start + size);
        if (pre && post) {
            if (a->nfree >= MPT_MAX_FREE) return -1;
            memmove(&a->free[i + 2], &a->free[i + 1], (a->nfree - i - 1) * sizeof(range_t));
            a->free[i + 1].off = start + size;
            a->free[i + 1].len = post;
            a->free[i].len = pre;
            a->nfree++;
        } else if (pre) {
            a->free[i].len = pre;
        } else if (post) {
            a->free[i].off = start + size;
            a->free[i].len = post;
        } else {
            memmove(&a->free[i], &a->free[i + 1], (a->nfree - i - 1) * sizeof(range_t));
            a->nfree--;
        }
        *off_out = start;
        return 0;
    }
    return -1;
}

static void arena_release(arena_t *a, uint64_t off, uint64_t len) {
    uint32_t lo = 0, hi = a->nfree;
    while (lo < hi) { uint32_t mid = (lo + hi) / 2; if (a->free[mid].off < off) lo = mid + 1; else hi = mid; }
    int merge_prev = lo > 0 && a->free[lo - 1].off + a->free[lo - 1].len == off;
    int merge_next = lo < a->nfree && off + len == a->free[lo].off;
    if (merge_prev && merge_next) {
        a->free[lo - 1].len += len + a->free[lo].len;
        memmove(&a->free[lo], &a->free[lo + 1], (a->nfree - lo - 1) * sizeof(range_t));
        a->nfree--;
    } else if (merge_prev) {
        a->free[lo - 1].len += len;
    } else if (merge_next) {
        a->free[lo].off = off;
        a->free[lo].len += len;
    } else if (a->nfree < MPT_MAX_FREE) {
        memmove(&a->free[lo + 1], &a->free[lo], (a->nfree - lo) * sizeof(range_t));
        a->free[lo].off = off;
        a->free[lo].len = len;
        a->nfree++;
    } /* else: leak the range rather than corrupt the list */
}

static int grow(mpt_ctx *c, uint64_t min_bytes);

MPT_API void *mpt_alloc(mpt_ctx *c, size_t bytes, size_t align) {
    if (!c || !bytes || bytes > (1ull << 48)) return NULL;
    if (align < 256) align = 256;  /* Metal setBuffer offset alignment and cache-line friendliness */
    if (align & (align - 1)) return NULL;
    uint64_t size = (bytes + 255) & ~255ull;
    for (int attempt = 0; attempt < 2; attempt++) {
        alock(c);
        uint32_t n = atomic_load(&c->narenas);
        for (uint32_t i = 0; i < n; i++) {
            uint64_t off;
            arena_t *a = &c->arenas[i];
            if (arena_alloc(a, size, align, &off) == 0) {
                if (live_insert(a, off, size) != 0) { arena_release(a, off, size); break; }
                void *p = a->base + off;
                aunlock(c);
                return p;
            }
        }
        aunlock(c);
        if (attempt == 0 && grow(c, size + align) != 0) break;
    }
    set_err(MPT_ENOMEM, "mpt_alloc(%zu): out of arena memory and %llu free hugepages left", bytes,
            (unsigned long long)hugepages_free());
    return NULL;
}

static arena_t *arena_of(mpt_ctx *c, const void *p) {
    uint32_t n = atomic_load_explicit(&c->narenas, memory_order_acquire);
    for (uint32_t i = 0; i < n; i++) {
        arena_t *a = &c->arenas[i];
        if ((const uint8_t *)p >= a->base && (const uint8_t *)p < a->base + a->len) return a;
    }
    return NULL;
}

MPT_API void mpt_free(mpt_ctx *c, void *p) {
    if (!c || !p) return;
    alock(c);
    arena_t *a = arena_of(c, p);
    if (a) {
        uint64_t off = (uint64_t)((uint8_t *)p - a->base);
        int k = live_find(a, off);
        if (k >= 0 && a->live[k].off == off) {  /* only exact allocation bases are freed */
            uint64_t len = a->live[k].len;
            live_remove(a, k);
            arena_release(a, off, len);
        }
    }
    aunlock(c);
}

MPT_API int mpt_alloc_lookup(mpt_ctx *c, const void *p, void **base, size_t *size) {
    int r = MPT_ENOENT;
    alock(c);
    arena_t *a = arena_of(c, p);
    if (a) {
        uint64_t off = (uint64_t)((const uint8_t *)p - a->base);
        int k = live_find(a, off);
        if (k >= 0 && off < a->live[k].off + a->live[k].len) {
            if (base) *base = a->base + a->live[k].off;
            if (size) *size = a->live[k].len;
            r = 0;
        }
    }
    aunlock(c);
    return r;
}

MPT_API size_t mpt_alloc_size(mpt_ctx *c, const void *p) {
    void *b;
    size_t s;
    return mpt_alloc_lookup(c, p, &b, &s) == 0 && b == p ? s : 0;
}

MPT_API int mpt_check_range(mpt_ctx *c, const void *p, size_t n) {
    void *b;
    size_t s;
    if (!p) return MPT_EFAULT;
    if (mpt_alloc_lookup(c, p, &b, &s) != 0) return MPT_EFAULT;
    uint64_t off = (uint64_t)((const uint8_t *)p - (uint8_t *)b);
    return (n <= s && off <= s - n) ? 0 : MPT_EFAULT;  /* overflow-safe: never computes p + n */
}

MPT_API uint64_t mpt_devaddr(mpt_ctx *c, const void *p) {
    arena_t *a = arena_of(c, p);
    return a ? a->gpu_base + (uint64_t)((const uint8_t *)p - a->base) : 0;
}

MPT_API void *mpt_hostptr(mpt_ctx *c, uint64_t dev) {
    uint32_t n = atomic_load_explicit(&c->narenas, memory_order_acquire);
    for (uint32_t i = 0; i < n; i++) {
        arena_t *a = &c->arenas[i];
        if (dev >= a->gpu_base && dev < a->gpu_base + a->len) return a->base + (dev - a->gpu_base);
    }
    return NULL;
}

MPT_API int mpt_memory_info(mpt_ctx *c, uint64_t *free_b, uint64_t *total_b) {
    uint64_t fr = 0, tot = 0;
    alock(c);
    uint32_t n = atomic_load(&c->narenas);
    for (uint32_t i = 0; i < n; i++) {
        tot += c->arenas[i].len;
        for (uint32_t k = 0; k < c->arenas[i].nfree; k++) fr += c->arenas[i].free[k].len;
    }
    aunlock(c);
    uint64_t hp = hugepages_free() * MPT_CHUNK_SIZE;
    if (free_b) *free_b = fr + hp;
    if (total_b) *total_b = tot + hp;
    return 0;
}

/* Maps an additional arena of at least min_bytes (OP_ALLOC_ARENA). */
static int grow(mpt_ctx *c, uint64_t min_bytes) {
    uint64_t bytes = min_bytes > c->grow_bytes ? min_bytes : c->grow_bytes;
    uint32_t nchunks = (uint32_t)((bytes + MPT_CHUNK_SIZE - 1) / MPT_CHUNK_SIZE);
    uint64_t avail = hugepages_free();
    if (avail < nchunks) {
        if (avail * MPT_CHUNK_SIZE < min_bytes || avail == 0) return set_err(MPT_ENOMEM, "need %u hugepages, %llu free", nchunks, (unsigned long long)avail);
        nchunks = (uint32_t)avail;
    }
    if (atomic_load(&c->narenas) >= MPT_MAX_ARENAS) return set_err(MPT_ENOMEM, "arena table full");
    uint8_t nonce[16];
    int memfd;
    uint8_t *base = map_chunks(nchunks, nonce, &memfd);
    if (!base) return set_err(MPT_ENOMEM, "mmap(MAP_HUGETLB, %u x 2 MiB) failed: %s", nchunks, strerror(errno));
    mpt_sqe e = { .opcode = MPT_OP_ALLOC_ARENA };
    memcpy(&e.a[0], nonce, 8);
    memcpy(&e.a[1], nonce + 8, 8);
    e.a[2] = nchunks;
    mpt_cqe q;
    int r = submit(c, &e, &q);
    if (r) {
        munmap(base, (size_t)nchunks * MPT_CHUNK_SIZE);
        if (memfd >= 0) close(memfd);
        return set_err(r, "ALLOC_ARENA failed: %s", mpt_strerror(r));
    }
    keeper_hold(c->keeper, memfd);
    if (memfd >= 0) close(memfd);
    int uva = 0;
    base = relocate_uva(base, (size_t)nchunks * MPT_CHUNK_SIZE, q.r[1], q.r[2], &uva);
    alock(c);
    if (!uva) c->uva = 0;
    uint32_t n = atomic_load(&c->narenas);
    arena_init(&c->arenas[n], base, (uint64_t)nchunks * MPT_CHUNK_SIZE, q.r[1], (uint32_t)q.r[0], 0);
    atomic_store_explicit(&c->narenas, n + 1, memory_order_release);
    aunlock(c);
    DBG(c, "arena %u: %u MiB at gpu 0x%llx", (unsigned)q.r[0], nchunks * 2, (unsigned long long)q.r[1]);
    return 0;
}

/* ------------------------------------------------------------------ session */

MPT_API int mpt_open(mpt_ctx **out, size_t arena_bytes) {
    *out = NULL;
    if (!arena_bytes) arena_bytes = env_u64("METALPASSTHROUGH_ARENA_MB", 256) << 20;
    uint32_t nchunks = (uint32_t)((arena_bytes + MPT_CHUNK_SIZE - 1) / MPT_CHUNK_SIZE);
    uint32_t minchunks = (MPT_HEAP_OFFSET + MPT_CHUNK_SIZE - 1) / MPT_CHUNK_SIZE + 1;
    if (nchunks < minchunks) nchunks = minchunks;

    mpt_ctx *c = calloc(1, sizeof *c);
    if (!c) return set_err(MPT_ENOMEM, "out of host memory");
    c->debug = (int)env_u64("METALPASSTHROUGH_DEBUG", 0);
    c->spin_ns = env_u64("METALPASSTHROUGH_SPIN_US", 100) * 1000;
    c->grow_bytes = env_u64("METALPASSTHROUGH_GROW_MB", 256) << 20;
    pthread_mutex_init(&c->lock, NULL);
    atomic_flag_clear(&c->alock);

    uint8_t nonce[16];
    int memfd0;
    uint8_t *base = map_chunks(nchunks, nonce, &memfd0);
    c->keeper = -1;
    if (!base) {
        int e = errno;
        free(c);
        return set_err(MPT_ENOMEM,
                       "mmap(MAP_HUGETLB, %u x 2 MiB) failed (%s); %llu hugepages free. MetalPassthrough needs 2 MiB "
                       "hugepages reserved at VM boot (mpt-run --hugepages N / container run --gpu)",
                       nchunks, strerror(e), (unsigned long long)hugepages_free());
    }
    int fd = connect_host();
    if (fd < 0) {
        munmap(base, (size_t)nchunks * MPT_CHUNK_SIZE);
        if (memfd0 >= 0) close(memfd0);
        free(c);
        return set_err(MPT_EPROTO, "cannot reach the MetalPassthrough host (vsock cid 2 port %u): %s",
                       (unsigned)env_u64("METALPASSTHROUGH_PORT", MPT_VSOCK_PORT), strerror(-fd));
    }
    mpt_hello h = { .magic = MPT_MAGIC, .version = MPT_ABI_VERSION, .nchunks = nchunks, .pid = (uint32_t)getpid() };
    memcpy(h.nonce, nonce, 16);
    FILE *f = fopen("/proc/self/comm", "r");
    if (f) { if (fgets(h.comm, sizeof h.comm, f)) h.comm[strcspn(h.comm, "\n")] = 0; fclose(f); }
    mpt_hello_ack ack;
    uint64_t t0 = now_ns();
    if (io_all(fd, &h, sizeof h, 1) || io_all(fd, &ack, sizeof ack, 0) || ack.magic != MPT_MAGIC) {
        close(fd);
        munmap(base, (size_t)nchunks * MPT_CHUNK_SIZE);
        if (memfd0 >= 0) close(memfd0);
        free(c);
        return set_err(MPT_EPROTO, "MetalPassthrough handshake failed");
    }
    if (ack.status) {
        close(fd);
        munmap(base, (size_t)nchunks * MPT_CHUNK_SIZE);
        if (memfd0 >= 0) close(memfd0);
        free(c);
        return set_err(ack.status, "host rejected session: %s", mpt_strerror(ack.status));
    }
    c->fd = fd;
    c->session = ack.session_id;
    c->caps = ack.caps;
    base = relocate_uva(base, (size_t)nchunks * MPT_CHUNK_SIZE, ack.gpu_base, ack.caps, &c->uva);
    c->ring = (mpt_ring_header *)(base + MPT_RING_OFFSET);
    if (__atomic_load_n(&c->ring->magic, __ATOMIC_ACQUIRE) != MPT_MAGIC || c->ring->sq_depth != MPT_SQ_DEPTH) {
        close(fd);
        munmap(base, (size_t)nchunks * MPT_CHUNK_SIZE);
        if (memfd0 >= 0) close(memfd0);
        free(c);
        return set_err(MPT_EPROTO, "ring header not initialized by host");
    }
    c->sq = (mpt_sqe *)(base + MPT_SQ_OFFSET);
    c->cq = (mpt_cqe *)(base + MPT_CQ_OFFSET);
    c->scratch = base + MPT_SCRATCH_OFFSET;
    c->scratch_dev = ack.gpu_base + MPT_SCRATCH_OFFSET;
    arena_init(&c->arenas[0], base, (uint64_t)nchunks * MPT_CHUNK_SIZE, ack.gpu_base, 0, MPT_HEAP_OFFSET);
    atomic_store(&c->narenas, 1);
    if (ack.caps & MPT_CAP_KEEPER) c->keeper = start_keeper(c->session, nonce, memfd0);
    if (memfd0 >= 0) close(memfd0);  /* the keeper holds its own reference */
    DBG(c, "session %u: arena0 %u MiB zero-copy, gpu base 0x%llx, uva=%d, keeper=%s, handshake %.1f us", c->session,
        nchunks * 2, (unsigned long long)ack.gpu_base, c->uva, c->keeper >= 0 ? "yes" : "no", (now_ns() - t0) / 1e3);
    *out = c;
    return 0;
}

MPT_API void mpt_close(mpt_ctx *c) {
    if (!c) return;
    mpt_sqe e = { .opcode = MPT_OP_CLOSE };
    submit(c, &e, NULL);  /* host drains the GPU before acknowledging */
    close(c->fd);
    if (c->keeper >= 0) close(c->keeper);
    uint32_t n = atomic_load(&c->narenas);
    for (uint32_t i = 0; i < n; i++) {
        munmap(c->arenas[i].base, c->arenas[i].len);
        free(c->arenas[i].free);
        free(c->arenas[i].live);
    }
    free(c);
}

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static mpt_ctx *g_ctx;
static char g_open_err[256];
/* At exit: drain the GPU and end the session, but keep the context object valid (later static
 * destructors, e.g. nvcc's __cudaUnregisterFatBinary, may still call in). The kernel reclaims the
 * hugepages after the host has acknowledged CLOSE. */
static void close_default(void) {
    if (!g_ctx || g_ctx->closed) return;
    mpt_sqe e = { .opcode = MPT_OP_CLOSE };
    submit(g_ctx, &e, NULL);
    g_ctx->closed = 1;
    close(g_ctx->fd);
}
static void open_default(void) {
    if (mpt_open(&g_ctx, 0) == 0) atexit(close_default);
    else snprintf(g_open_err, sizeof g_open_err, "%s", g_err);
}
MPT_API mpt_ctx *mpt_default(void) {
    pthread_once(&g_once, open_default);
    if (!g_ctx) snprintf(g_err, sizeof g_err, "%s", g_open_err);
    return g_ctx;
}

MPT_API uint32_t mpt_session_id(mpt_ctx *c) { return c->session; }

/* Struct sizes for language bindings to verify their mirrors of the ABI (0 = unknown name). */
MPT_API size_t mpt_abi_sizeof(const char *type) {
    if (!strcmp(type, "mpt_device_info")) return sizeof(mpt_device_info);
    if (!strcmp(type, "mpt_stats")) return sizeof(mpt_stats);
    if (!strcmp(type, "mpt_kernel_info")) return sizeof(mpt_kernel_info);
    if (!strcmp(type, "mpt_proc_info")) return sizeof(mpt_proc_info);
    if (!strcmp(type, "mpt_event")) return sizeof(mpt_event);
    return 0;
}
MPT_API int mpt_unified_addressing(mpt_ctx *c) { return c->uva; }
MPT_API uint64_t mpt_session_caps(mpt_ctx *c) { return c->caps; }
MPT_API int mpt_crash_safe(mpt_ctx *c) { return c->keeper >= 0; }

/* Test/fuzzing hooks: submit an arbitrary SQE (waits for its CQE unless MPT_F_NO_CQE), and expose
 * the ring so robustness tests can violate the protocol on purpose. Not for applications. */
MPT_API int mpt_submit_raw(mpt_ctx *c, const mpt_sqe *e, mpt_cqe *out) {
    mpt_sqe copy = *e;
    return submit(c, &copy, out);
}
MPT_API mpt_ring_header *mpt_debug_ring(mpt_ctx *c) { return c->ring; }
MPT_API uint64_t mpt_debug_arena_range(mpt_ctx *c, uint64_t *len) {
    if (len) *len = c->arenas[0].len;
    return c->arenas[0].gpu_base;
}

MPT_API const mpt_device_info *mpt_device(mpt_ctx *c) {
    return (const mpt_device_info *)(c->arenas[0].base + MPT_INFO_OFFSET);
}

MPT_API int mpt_get_stats(mpt_ctx *c, mpt_stats *out) {
    mpt_sqe e = { .opcode = MPT_OP_STATS };
    int r = submit(c, &e, NULL);
    if (!r) memcpy(out, c->arenas[0].base + MPT_STATS_OFFSET, sizeof *out);
    return r;
}

MPT_API int mpt_ping(mpt_ctx *c, uint64_t *rt) {
    mpt_sqe e = { .opcode = MPT_OP_PING };
    uint64_t t0 = now_ns();
    int r = submit(c, &e, NULL);
    if (rt) *rt = now_ns() - t0;
    return r;
}

/* ------------------------------------------------------------------ compilation */

static int compile(mpt_ctx *c, uint16_t op, const char *src, const char *fn, uint32_t flags, mpt_pipeline *out,
                   mpt_kernel_info *info, char *log, size_t log_len) {
    size_t sl = strlen(src), fl = strlen(fn);
    size_t outlen = (op == MPT_OP_COMPILE_PTX ? sizeof(mpt_kernel_info) : 0) + 65536;
    uint8_t *tmp = mpt_alloc(c, sl + fl + 2 + outlen, 256);
    if (!tmp) return MPT_ENOMEM;
    memcpy(tmp, src, sl + 1);
    memcpy(tmp + sl + 1, fn, fl + 1);
    uint8_t *o = tmp + ((sl + fl + 2 + 255) & ~255ull);
    if ((size_t)(o - tmp) + outlen > mpt_alloc_size(c, tmp)) outlen = mpt_alloc_size(c, tmp) - (size_t)(o - tmp);
    o[0] = 0;
    mpt_sqe e = { .opcode = op };
    e.a[0] = mpt_devaddr(c, tmp);
    e.a[1] = sl;
    e.a[2] = mpt_devaddr(c, tmp + sl + 1);
    e.a[3] = fl;
    e.a[4] = flags;
    e.a[5] = mpt_devaddr(c, o);
    e.a[6] = outlen;
    mpt_cqe q;
    int r = submit(c, &e, &q);
    const char *text = (const char *)o + (op == MPT_OP_COMPILE_PTX ? sizeof(mpt_kernel_info) : 0);
    if (log && log_len) snprintf(log, log_len, "%s", text);
    if (r == 0) {
        out->handle = q.r[0];
        out->max_threads_per_tg = (uint32_t)q.r[1];
        out->simd_width = (uint32_t)(q.r[1] >> 32);
        out->static_tg_memory = (uint32_t)q.r[2];
        if (info) memcpy(info, o, sizeof *info);
    } else {
        set_err(r, "%s: %.200s", mpt_strerror(r), text);
    }
    mpt_free(c, tmp);
    return r;
}

MPT_API int mpt_compile_msl(mpt_ctx *c, const char *src, const char *fn, uint32_t flags, mpt_pipeline *out, char *log,
                              size_t log_len) {
    return compile(c, MPT_OP_COMPILE_MSL, src, fn, flags, out, NULL, log, log_len);
}

MPT_API int mpt_compile_ptx(mpt_ctx *c, const char *ptx, const char *entry, uint32_t flags, mpt_pipeline *out,
                              mpt_kernel_info *info, char *log, size_t log_len) {
    return compile(c, MPT_OP_COMPILE_PTX, ptx, entry, flags, out, info, log, log_len);
}

MPT_API int mpt_release_pipeline(mpt_ctx *c, const mpt_pipeline *p) {
    mpt_sqe e = { .opcode = MPT_OP_RELEASE_PIPE, .flags = MPT_F_NO_CQE };
    e.a[0] = p->handle;
    return submit(c, &e, NULL);
}

/* ------------------------------------------------------------------ submission */

static int dispatch_common(mpt_ctx *c, uint32_t stream, const mpt_pipeline *p, const uint32_t grid[3], const uint32_t tg[3], uint32_t mode,
                           const mpt_arg *args, uint32_t nargs, uint32_t all_arenas, uint32_t flags) {
    size_t inline_bytes = 0;
    for (uint32_t i = 0; i < nargs; i++)
        if (args[i].kind == MPT_BIND_BYTES) inline_bytes += (args[i].len + 15) & ~15ull;
    if (c->closed) return MPT_EPROTO;
    size_t need = sizeof(mpt_dispatch_desc) + nargs * sizeof(mpt_binding) + inline_bytes;
    if (need > MPT_SCRATCH_PER_SLOT) return set_err(MPT_EINVAL, "dispatch descriptor too large (%zu bytes)", need);
    pthread_mutex_lock(&c->lock);
    uint32_t slot = sq_reserve(c);
    if (slot == SQ_DEAD) { pthread_mutex_unlock(&c->lock); return MPT_EPROTO; }
    uint8_t *desc = c->scratch + (size_t)slot * MPT_SCRATCH_PER_SLOT;
    mpt_dispatch_desc *d = (mpt_dispatch_desc *)desc;
    for (int k = 0; k < 3; k++) { d->grid[k] = grid[k] ? grid[k] : 1; d->tg[k] = tg[k] ? tg[k] : 1; }
    d->mode = mode;
    d->nbindings = nargs;
    d->use_all_arenas = all_arenas;
    size_t boff = sizeof(mpt_dispatch_desc) + nargs * sizeof(mpt_binding);
    for (uint32_t i = 0; i < nargs; i++) {
        mpt_binding *b = &d->bindings[i];
        b->kind = args[i].kind;
        b->index = args[i].index;
        if (args[i].kind == MPT_BIND_BUFFER) {
            arena_t *a = arena_of(c, args[i].ptr);
            if (!a) { pthread_mutex_unlock(&c->lock); return set_err(MPT_EFAULT, "argument %u is not MetalPassthrough arena memory", i); }
            b->addr = mpt_devaddr(c, args[i].ptr);
            uint64_t remain = a->len - (uint64_t)((const uint8_t *)args[i].ptr - a->base);
            if (args[i].len > remain) { pthread_mutex_unlock(&c->lock); return set_err(MPT_EFAULT, "argument %u extends past its arena", i); }
            b->len = args[i].len ? args[i].len : remain;
        } else if (args[i].kind == MPT_BIND_BYTES) {
            memcpy(desc + boff, args[i].ptr, args[i].len);
            b->addr = boff;
            b->len = args[i].len;
            boff += (args[i].len + 15) & ~15ull;
        } else {
            b->addr = 0;
            b->len = args[i].len;
        }
    }
    mpt_sqe e = { .opcode = MPT_OP_DISPATCH, .flags = (uint16_t)(MPT_F_NO_CQE | (flags & MPT_F_COMMIT) | mpt_stream_flags(stream)) };
    e.a[0] = p->handle;
    e.a[1] = c->scratch_dev + (uint64_t)slot * MPT_SCRATCH_PER_SLOT;
    e.a[2] = need;
    int r = submit_locked(c, &e, NULL);
    c->pending = 1;
    pthread_mutex_unlock(&c->lock);
    return r;
}

MPT_API int mpt_dispatch(mpt_ctx *c, const mpt_pipeline *p, const uint32_t grid[3], const uint32_t tg[3], uint32_t mode,
                         const mpt_arg *args, uint32_t nargs, uint32_t flags) {
    return dispatch_common(c, 0, p, grid, tg, mode, args, nargs, 0, flags);
}

MPT_API int mpt_dispatch_on(mpt_ctx *c, uint32_t stream, const mpt_pipeline *p, const uint32_t grid[3], const uint32_t tg[3],
                            uint32_t mode, const mpt_arg *args, uint32_t nargs, uint32_t flags) {
    return dispatch_common(c, stream, p, grid, tg, mode, args, nargs, 0, flags);
}

MPT_API int mpt_dispatch_raw_on(mpt_ctx *c, uint32_t stream, const mpt_pipeline *p, const uint32_t grid[3], const uint32_t tg[3],
                                const void *params, uint32_t params_len, uint32_t tg_memory, uint32_t flags) {
    static const uint8_t zero[16];
    mpt_arg args[2] = {
        { .kind = MPT_BIND_BYTES, .index = 0, .ptr = params_len ? params : zero, .len = params_len ? params_len : 16 },
        { .kind = MPT_BIND_TGMEM, .index = 0, .len = tg_memory },
    };
    return dispatch_common(c, stream, p, grid, tg, MPT_DISPATCH_THREADGROUPS, args, tg_memory ? 2 : 1, 1, flags);
}

MPT_API int mpt_dispatch_raw(mpt_ctx *c, const mpt_pipeline *p, const uint32_t grid[3], const uint32_t tg[3],
                             const void *params, uint32_t params_len, uint32_t tg_memory, uint32_t flags) {
    return mpt_dispatch_raw_on(c, 0, p, grid, tg, params, params_len, tg_memory, flags);
}

static int simple(mpt_ctx *c, uint16_t op, uint16_t flags, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, mpt_cqe *q) {
    mpt_sqe e = { .opcode = op, .flags = flags };
    e.a[0] = a0; e.a[1] = a1; e.a[2] = a2; e.a[3] = a3;
    return submit(c, &e, q);
}

MPT_API int mpt_copy_on(mpt_ctx *c, uint32_t stream, uint64_t dst, uint64_t src, uint64_t len) {
    c->pending = 1;
    return simple(c, MPT_OP_COPY, MPT_F_NO_CQE | mpt_stream_flags(stream), dst, src, len, 0, NULL);
}
MPT_API int mpt_copy(mpt_ctx *c, uint64_t dst, uint64_t src, uint64_t len) { return mpt_copy_on(c, 0, dst, src, len); }

MPT_API int mpt_fill_on(mpt_ctx *c, uint32_t stream, uint64_t dst, uint64_t len, uint32_t pattern, uint32_t width) {
    c->pending = 1;
    return simple(c, MPT_OP_FILL, MPT_F_NO_CQE | mpt_stream_flags(stream), dst, len, pattern, width ? width : 1, NULL);
}
MPT_API int mpt_fill(mpt_ctx *c, uint64_t dst, uint64_t len, uint32_t pattern, uint32_t width) {
    return mpt_fill_on(c, 0, dst, len, pattern, width);
}

MPT_API int mpt_flush(mpt_ctx *c) { return simple(c, MPT_OP_FLUSH, MPT_F_NO_CQE, 0, 0, 0, 0, NULL); }

/* Host-library SGEMM (MPSMatrixMultiplication) on arena memory, stream ordered. The descriptor
 * travels in the reserved SQ slot's scratch area, which stays valid until the host consumed it. */
MPT_API int mpt_gemm_on(mpt_ctx *c, uint32_t stream, const mpt_gemm_desc *g) {
    if (c->closed) return MPT_EPROTO;
    pthread_mutex_lock(&c->lock);
    uint32_t slot = sq_reserve(c);
    if (slot == SQ_DEAD) { pthread_mutex_unlock(&c->lock); return MPT_EPROTO; }
    uint8_t *desc = c->scratch + (size_t)slot * MPT_SCRATCH_PER_SLOT;
    memcpy(desc, g, sizeof *g);
    mpt_sqe e = { .opcode = MPT_OP_GEMM, .flags = (uint16_t)(MPT_F_NO_CQE | mpt_stream_flags(stream)) };
    e.a[0] = c->scratch_dev + (uint64_t)slot * MPT_SCRATCH_PER_SLOT;
    e.a[1] = sizeof *g;
    int r = submit_locked(c, &e, NULL);
    c->pending = 1;
    pthread_mutex_unlock(&c->lock);
    return r;
}

/* Takes the sticky asynchronous error, if any (CUDA reports async failures at the next sync). */
static int take_sticky(mpt_ctx *c, int r) {
    if (!r && c->sticky) r = c->sticky;
    c->sticky = 0;
    return r;
}

MPT_API int mpt_sync(mpt_ctx *c, uint64_t *gpu_ns) {
    mpt_cqe q = { 0 };
    if (c->closed) return MPT_EPROTO;
    pthread_mutex_lock(&c->lock);
    if (sq_reserve(c) == SQ_DEAD) { pthread_mutex_unlock(&c->lock); return MPT_EPROTO; }
    mpt_sqe e = { .opcode = MPT_OP_SYNC };
    int r = take_sticky(c, submit_locked(c, &e, &q));
    c->pending = 0;
    pthread_mutex_unlock(&c->lock);
    if (gpu_ns) *gpu_ns = q.r[0];
    if (r) set_err(r, "GPU work failed: %s", mpt_strerror(r));
    return r;
}

MPT_API int mpt_pending(mpt_ctx *c) { return c->pending; }

/* ---- streams ---- */

MPT_API int mpt_stream_create(mpt_ctx *c, uint32_t flags, uint32_t *stream) {
    if (!(c->caps & MPT_CAP_STREAMS)) return set_err(MPT_EPROTO, "host does not support streams");
    mpt_cqe q;
    int r = simple(c, MPT_OP_STREAM_CREATE, 0, flags, 0, 0, 0, &q);
    if (!r) *stream = (uint32_t)q.r[0];
    return r;
}

MPT_API int mpt_stream_destroy(mpt_ctx *c, uint32_t stream) {
    if (!stream) return MPT_EINVAL;
    return simple(c, MPT_OP_STREAM_DESTROY, mpt_stream_flags(stream), 0, 0, 0, 0, NULL);
}

MPT_API int mpt_stream_sync(mpt_ctx *c, uint32_t stream, uint64_t *gpu_ns) {
    mpt_cqe q = { 0 };
    if (c->closed) return MPT_EPROTO;
    pthread_mutex_lock(&c->lock);
    if (sq_reserve(c) == SQ_DEAD) { pthread_mutex_unlock(&c->lock); return MPT_EPROTO; }
    mpt_sqe e = { .opcode = MPT_OP_STREAM_SYNC, .flags = mpt_stream_flags(stream) };
    int r = take_sticky(c, submit_locked(c, &e, &q));
    pthread_mutex_unlock(&c->lock);
    if (gpu_ns) *gpu_ns = q.r[0];
    return r;
}

MPT_API int mpt_stream_query(mpt_ctx *c, uint32_t stream) {
    mpt_cqe q = { 0 };
    int r = simple(c, MPT_OP_STREAM_QUERY, mpt_stream_flags(stream), 0, 0, 0, 0, &q);
    return r ? r : (int)q.r[0];
}

MPT_API int mpt_stream_wait_event(mpt_ctx *c, uint32_t stream, const mpt_event *ev) {
    uint64_t dev = mpt_devaddr(c, ev);
    if (!dev) return set_err(MPT_EFAULT, "event must live in arena memory");
    return simple(c, MPT_OP_STREAM_WAIT_EVENT, MPT_F_NO_CQE | mpt_stream_flags(stream), dev, 0, 0, 0, NULL);
}

MPT_API int mpt_event_record_on(mpt_ctx *c, uint32_t stream, mpt_event *ev) {
    uint64_t dev = mpt_devaddr(c, ev);
    if (!dev) return set_err(MPT_EFAULT, "event must live in arena memory");
    __atomic_store_n(&ev->done, 0, __ATOMIC_RELEASE);
    return simple(c, MPT_OP_EVENT_RECORD, MPT_F_NO_CQE | mpt_stream_flags(stream), dev, 0, 0, 0, NULL);
}

MPT_API int mpt_event_record(mpt_ctx *c, mpt_event *ev) { return mpt_event_record_on(c, 0, ev); }

MPT_API int mpt_event_destroy(mpt_ctx *c, mpt_event *ev) {
    uint64_t dev = mpt_devaddr(c, ev);
    return dev ? simple(c, MPT_OP_EVENT_DESTROY, MPT_F_NO_CQE, dev, 0, 0, 0, NULL) : MPT_EFAULT;
}

MPT_API int mpt_event_query(mpt_ctx *c, const mpt_event *ev) {
    (void)c;
    return __atomic_load_n(&ev->done, __ATOMIC_ACQUIRE) ? 1 : 0;
}

MPT_API int mpt_event_wait(mpt_ctx *c, mpt_event *ev) {
    uint64_t t0 = now_ns();
    while (!__atomic_load_n(&ev->done, __ATOMIC_ACQUIRE)) {
        if (now_ns() - t0 < c->spin_ns) { cpu_relax(); continue; }
        struct timespec ts = { 0, 20000 };
        nanosleep(&ts, NULL);
    }
    return ev->status;
}
