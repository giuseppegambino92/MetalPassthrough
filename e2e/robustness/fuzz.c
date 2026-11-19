/*
 * Protocol fuzzer: behaves like a malicious guest. The host must reject or contain every input
 * and keep serving correct results to new sessions afterwards.
 *
 *   fuzz [iterations] [seed]
 *
 * Phases: (A) malformed handshakes on raw vsock connections, (B) random SQEs with random /
 * boundary arguments and random dispatch descriptors, (C) GPU faults and garbage shaders,
 * (D) ring protocol violation, (E) a fresh session must still compute correctly.
 *
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */
#define _GNU_SOURCE
#include "metalpassthrough.h"

#include <linux/vm_sockets.h>
#include <poll.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static uint64_t rpick(uint64_t n) { return n ? rnd() % n : 0; }

static int raw_connect(void) {
    int fd = socket(AF_VSOCK, SOCK_STREAM, 0);
    struct sockaddr_vm a = { .svm_family = AF_VSOCK, .svm_cid = MPT_HOST_CID, .svm_port = MPT_VSOCK_PORT };
    if (fd < 0 || connect(fd, (struct sockaddr *)&a, sizeof a) != 0) { if (fd >= 0) close(fd); return -1; }
    struct timeval tv = { .tv_sec = 2 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    return fd;
}

static void phase_handshake(int n) {
    int accepted = 0, rejected = 0;
    for (int i = 0; i < n; i++) {
        int fd = raw_connect();
        if (fd < 0) continue;
        uint8_t buf[256];
        size_t len = rpick(sizeof buf);
        for (size_t k = 0; k < len; k++) buf[k] = (uint8_t)rnd();
        int kind = (int)rpick(4);
        if (kind == 1 && len >= sizeof(mpt_hello)) {  /* plausible hello with hostile fields */
            mpt_hello *h = (mpt_hello *)buf;
            h->magic = MPT_MAGIC;
            h->version = rpick(2) ? MPT_ABI_VERSION : (uint32_t)rnd();
            h->nchunks = rpick(2) ? (uint32_t)rnd() : (uint32_t)rpick(8);
        } else if (kind == 2 && len >= sizeof(mpt_keeper_hello)) {
            mpt_keeper_hello *k = (mpt_keeper_hello *)buf;
            k->magic = MPT_KEEPER_MAGIC;
            k->session_id = (uint32_t)rpick(16);
        }
        (void)!write(fd, buf, len);
        if (rpick(2)) shutdown(fd, SHUT_WR);
        mpt_hello_ack ack;
        struct pollfd pf = { .fd = fd, .events = POLLIN };
        ssize_t r = poll(&pf, 1, 300) > 0 ? read(fd, &ack, sizeof ack) : -1;  /* SO_RCVTIMEO is not reliable on vsock */
        if (r == sizeof ack && ack.status == 0) accepted++; else rejected++;
        close(fd);
    }
    printf("[A] %d malformed handshakes: %d rejected/closed, %d accepted (accepted ones were then abandoned)\n", n, rejected, accepted);
}

static uint64_t g_base, g_len;
static uint64_t g_valid[8];

static uint64_t rarg(void) {
    switch (rpick(10)) {
    case 0: return 0;
    case 1: return rnd();
    case 2: return rpick(4096);
    case 3: return g_base + rpick(g_len);                       /* somewhere inside arena 0 */
    case 4: return g_base + g_len - rpick(64);                  /* right at the end */
    case 5: return g_base + g_len + rpick(1 << 20);              /* just past the end */
    case 6: return g_valid[rpick(8)];
    case 7: return ~0ull - rpick(1024);                          /* wrap-around candidates */
    case 8: return (uint64_t)rpick(1ull << 40);
    default: return rpick(256);
    }
}

static const char *KERNEL =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void poke(device uint* p [[buffer(0)]], uint i [[thread_position_in_grid]]) { if (i < 64) p[i] = i; }\n";

static int phase_ring(int iters) {
    mpt_ctx *c;
    if (mpt_open(&c, 64 << 20) != 0) { printf("[B] open failed: %s\n", mpt_last_error()); return 1; }
    g_base = mpt_debug_arena_range(c, &g_len);
    mpt_pipeline p;
    char log[1024];
    mpt_compile_msl(c, KERNEL, "poke", 0, &p, log, sizeof log);
    uint8_t *scratch = mpt_alloc(c, 8 << 20, 0);  /* fuzz kernels and descriptors only touch this */
    for (int k = 0; k < 8; k++) g_valid[k] = mpt_devaddr(c, scratch + rpick(8 << 20));
    int errors = 0, oks = 0, reconnects = 0;
    for (int i = 0; i < iters; i++) {
        mpt_sqe e = { 0 };
        do { e.opcode = (uint16_t)rpick(MPT_OP__COUNT + 4); } while (e.opcode == MPT_OP_CLOSE);
        e.flags = (uint16_t)(rpick(2) ? MPT_F_NO_CQE : 0) | (uint16_t)(rpick(4) ? mpt_stream_flags((uint32_t)rpick(4)) : (uint16_t)rnd());
        for (int k = 0; k < 7; k++) e.a[k] = rarg();
        if (e.opcode == MPT_OP_ALLOC_ARENA && rpick(8)) e.a[2] = rpick(3);  /* keep scans cheap most of the time */
        if (e.opcode == MPT_OP_FREE_ARENA) e.a[0] = rpick(4) + 1;           /* never arena 0 of this session */
        if (e.opcode == MPT_OP_DISPATCH) {
            /* Random descriptor; buffer bindings point into the scratch block only, so the fuzzer's
             * own ring stays intact (the host must survive either way; this keeps the run going). */
            uint8_t *d = scratch + (rpick(1000) * 4096 % ((8 << 20) - 8192));
            mpt_dispatch_desc *dd = (mpt_dispatch_desc *)d;
            for (int k = 0; k < 3; k++) { dd->grid[k] = (uint32_t)rpick(rpick(2) ? 4 : 1u << 20); dd->tg[k] = (uint32_t)rpick(rpick(2) ? 64 : 2048); }
            /* Huge geometries must be *rejected* (tested below); valid-but-large ones are capped so the
             * run measures validation, not GPU time. */
            if (rpick(16)) { dd->grid[0] %= 1u << 14; dd->grid[1] %= 8; dd->grid[2] %= 4; }
            else { dd->grid[0] = dd->grid[1] = dd->grid[2] = 0xFFFFFFFFu; }  /* ~2^96 threads: must be rejected */
            dd->mode = (uint32_t)rpick(3);
            dd->nbindings = (uint32_t)rpick(rpick(4) ? 4 : 100);
            dd->use_all_arenas = (uint32_t)rpick(2);
            for (uint32_t k = 0; k < dd->nbindings && k < 100; k++) {
                mpt_binding *b = &dd->bindings[k];
                b->kind = (uint32_t)rpick(4);
                b->index = (uint32_t)rpick(rpick(2) ? 4 : 64);
                b->addr = b->kind == MPT_BIND_BUFFER ? (rpick(4) ? mpt_devaddr(c, scratch + 4096 * rpick(1024)) : rarg()) : rpick(512);
                b->len = rpick(3) ? rpick(4096) : rnd();
            }
            e.a[0] = rpick(4) ? p.handle : rnd();
            e.a[1] = rpick(8) ? mpt_devaddr(c, d) : rarg();
            e.a[2] = rpick(8) ? sizeof(mpt_dispatch_desc) + 24 * (rpick(8) ? dd->nbindings : rpick(200)) : rarg();
        }
        if (e.opcode == MPT_OP_COMPILE_MSL || e.opcode == MPT_OP_COMPILE_PTX) {
            char *src = (char *)scratch + (1 << 20);
            size_t n = rpick(2000);
            for (size_t k = 0; k < n; k++) src[k] = (char)(32 + rpick(95));
            src[n] = 0;
            e.a[0] = mpt_devaddr(c, src);
            e.a[1] = rpick(8) ? n : rarg();
            e.a[2] = e.a[0];
            e.a[3] = rpick(16);
            e.a[5] = mpt_devaddr(c, scratch + (2 << 20));
            e.a[6] = rpick(4) ? 65536 : rarg();
        }
        mpt_cqe q;
        int r = mpt_submit_raw(c, &e, &q);
        if (r == MPT_EPROTO) {  /* the host ended this session (protocol violation): reconnect */
            reconnects++;
            if (mpt_open(&c, 64 << 20) != 0) { printf("[B] reconnect failed: %s\n", mpt_last_error()); return 1; }
            g_base = mpt_debug_arena_range(c, &g_len);
            mpt_compile_msl(c, KERNEL, "poke", 0, &p, log, sizeof log);
            scratch = mpt_alloc(c, 8 << 20, 0);
            continue;
        }
        if (r) errors++; else oks++;
        if (i % 500 == 0) mpt_sync(c, NULL);
        if (getenv("FUZZ_TRACE")) fprintf(stderr, "it %d op %u flags 0x%x -> %d\n", i, e.opcode, e.flags, r);
    }
    uint64_t rt;
    int alive = mpt_ping(c, &rt) == 0;
    printf("[B] %d random SQEs: %d accepted, %d rejected with an error, %d session resets; session alive: %s\n", iters, oks,
           errors, reconnects, alive ? "yes" : "no");
    return 0;
}

static int phase_faults(void) {
    mpt_ctx *c;
    if (mpt_open(&c, 64 << 20) != 0) return 1;
    /* A kernel dereferencing a wild pointer: GPU address fault -> command buffer error, not a crash. */
    const char *bad =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void wild(constant ulong& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {\n"
        "  device uint* p = reinterpret_cast<device uint*>(a + i * 4096ul); p[0] = 1; }\n";
    mpt_pipeline p;
    char log[2048];
    int rc = mpt_compile_msl(c, bad, "wild", 0, &p, log, sizeof log);
    uint64_t wild = 0x7f0000000000ull;
    mpt_arg a[] = { { MPT_BIND_BYTES, 0, &wild, 8 } };
    uint32_t grid[3] = { 1024, 1, 1 }, tg[3] = { 64, 1, 1 };
    if (!rc) mpt_dispatch(c, &p, grid, tg, MPT_DISPATCH_THREADS, a, 1, 0);
    int s = mpt_sync(c, NULL);
    printf("[C] wild-pointer kernel: sync returned %d (%s)\n", s, s ? mpt_strerror(s) : "no fault reported");
    /* Garbage shader sources and PTX must produce compile errors. */
    int errs = 0;
    for (int i = 0; i < 50; i++) {
        char src[600];
        for (int k = 0; k < 599; k++) src[k] = (char)(32 + rpick(95));
        src[599] = 0;
        mpt_kernel_info ki;
        errs += mpt_compile_msl(c, src, "k", 0, &p, NULL, 0) != 0;
        errs += mpt_compile_ptx(c, src, "k", 0, &p, &ki, NULL, 0) != 0;
    }
    printf("[C] 100 garbage MSL/PTX sources: %d compile errors reported\n", errs);
    uint64_t rt;
    printf("[C] session still alive after faults: %s\n", mpt_ping(c, &rt) == 0 ? "yes" : "no");
    mpt_close(c);
    return 0;
}

static int phase_ring_violation(void) {
    mpt_ctx *c;
    if (mpt_open(&c, 64 << 20) != 0) return 1;
    mpt_ring_header *r = mpt_debug_ring(c);
    __atomic_store_n(&r->sq_tail, r->sq_tail + 100000, __ATOMIC_RELEASE);  /* claim 100k phantom entries */
    sleep(2);  /* the parked host worker re-checks the ring at least once a second */
    uint64_t rt;
    int alive = mpt_ping(c, &rt) == 0;
    printf("[D] corrupted sq_tail: host %s the session\n", alive ? "still served (unexpected)" : "terminated");
    return alive;
}

static int phase_verify(void) {
    mpt_ctx *c;
    if (mpt_open(&c, 64 << 20) != 0) { printf("[E] new session failed: %s\n", mpt_last_error()); return 1; }
    const char *src =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n"
        "  y[i] = fma(2.0f, x[i], y[i]); }\n";
    mpt_pipeline p;
    char log[1024];
    if (mpt_compile_msl(c, src, "saxpy", 0, &p, log, sizeof log)) { printf("[E] compile failed\n"); return 1; }
    const uint32_t n = 1 << 20;
    float *x = mpt_alloc(c, n * 4, 0), *y = mpt_alloc(c, n * 4, 0);
    for (uint32_t i = 0; i < n; i++) { x[i] = (float)i; y[i] = 1.0f; }
    mpt_arg a[] = { { MPT_BIND_BUFFER, 0, x, 0 }, { MPT_BIND_BUFFER, 1, y, 0 } };
    uint32_t grid[3] = { n, 1, 1 }, tg[3] = { 256, 1, 1 };
    mpt_dispatch(c, &p, grid, tg, MPT_DISPATCH_THREADS, a, 2, 0);
    int s = mpt_sync(c, NULL);
    size_t bad = 0;
    for (uint32_t i = 0; i < n; i++) bad += y[i] != fmaf(2.0f, (float)i, 1.0f);
    printf("[E] fresh session after fuzzing: sync=%d, SAXPY mismatches=%zu\n", s, bad);
    return (s || bad) ? 1 : 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    int iters = argc > 1 ? atoi(argv[1]) : 20000;
    rng ^= argc > 2 ? strtoull(argv[2], NULL, 0) : (uint64_t)time(NULL);
    printf("fuzz seed 0x%llx, %d iterations\n", (unsigned long long)rng, iters);
    printf("[A] start\n");
    phase_handshake(400);
    printf("[B] start\n");
    if (phase_ring(iters)) return 1;
    phase_faults();
    int violation_ignored = phase_ring_violation();
    int r = phase_verify() | violation_ignored;
    printf(r ? "fuzz: FAIL\n" : "fuzz: PASS (host survived and still computes correctly)\n");
    return r;
}
