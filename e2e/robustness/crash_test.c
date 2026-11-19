/*
 * Crash-safety test: a process is SIGKILLed while its GPU kernel is still writing into its arena.
 * A second process then grabs the freed hugepages and watches them. Without the keeper, the kernel
 * recycles the pages immediately and the still-running GPU kernel corrupts the new owner's memory.
 * With the keeper, the pages only return to the pool after the host has drained the GPU.
 *
 *   crash_test victim <arena MiB>         -> prints READY when the long kernel is in flight
 *   crash_test probe  <MiB> <watch ms>    -> waits for hugepages, then counts foreign GPU writes
 *
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */
#define _GNU_SOURCE
#include "metalpassthrough.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define MARKER 0xDEADBEEFu

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static const char *SRC =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void scribble(device atomic_uint* buf [[buffer(0)]], constant uint& words [[buffer(1)]],\n"
    "                     constant uint& rounds [[buffer(2)]], uint i [[thread_position_in_grid]], uint n [[threads_per_grid]]) {\n"
    "  for (uint r = 0; r < rounds; r++)\n"
    "    for (uint w = i; w < words; w += n) atomic_store_explicit(&buf[w], 0xDEADBEEFu, memory_order_relaxed);\n"
    "}\n";

static int victim(size_t mib) {
    mpt_ctx *c;
    if (mpt_open(&c, mib << 20) != 0) { fprintf(stderr, "open: %s\n", mpt_last_error()); return 1; }
    size_t bytes = (mib - 4) << 20;
    uint32_t *buf = mpt_alloc(c, bytes, 0);
    if (!buf) { fprintf(stderr, "alloc: %s\n", mpt_last_error()); return 1; }
    mpt_pipeline p;
    char log[4096];
    if (mpt_compile_msl(c, SRC, "scribble", 0, &p, log, sizeof log)) { fprintf(stderr, "%s\n", log); return 1; }
    uint32_t words = (uint32_t)(bytes / 4), rounds = 1;
    uint32_t grid[3] = { 1u << 16, 1, 1 }, tg[3] = { 256, 1, 1 };
    mpt_arg a[] = { { MPT_BIND_BUFFER, 0, buf, 0 }, { MPT_BIND_BYTES, 1, &words, 4 }, { MPT_BIND_BYTES, 2, &rounds, 4 } };
    uint64_t ns = 0;
    mpt_dispatch(c, &p, grid, tg, MPT_DISPATCH_THREADS, a, 3, 0);
    mpt_sync(c, &ns);
    rounds = (uint32_t)(2.0e9 / (ns ? ns : 1));  /* ~2 s of continuous GPU writes */
    if (rounds < 1) rounds = 1;
    mpt_dispatch(c, &p, grid, tg, MPT_DISPATCH_THREADS, a, 3, MPT_F_COMMIT);
    mpt_flush(c);
    usleep(100 * 1000);  /* let the GPU start */
    printf("READY rounds=%u crash_safe=%d\n", rounds, mpt_crash_safe(c));
    fflush(stdout);
    pause();  /* wait to be SIGKILLed */
    return 0;
}

static int probe(size_t mib, double watch_ms) {
    size_t len = mib << 20;
    double t0 = now_ms(), waited;
    void *p;
    for (;;) {
        p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
        if (p != MAP_FAILED) break;
        if (now_ms() - t0 > 20000) { printf("probe: no hugepages after 20 s\n"); return 2; }
        usleep(2000);
    }
    waited = now_ms() - t0;
    volatile uint32_t *w = p;
    size_t nw = len / 4, seen = 0;
    double end = now_ms() + watch_ms;
    while (now_ms() < end) {
        size_t s = 0;
        for (size_t i = 0; i < nw; i += 1024) s += w[i] == MARKER;  /* fresh pages are zero; only the GPU writes markers */
        if (s > seen) seen = s;
        usleep(10000);
    }
    printf("probe: hugepages available after %.0f ms; foreign GPU writes observed in %zu sampled words\n", waited, seen);
    return seen ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc >= 3 && !strcmp(argv[1], "victim")) return victim((size_t)atol(argv[2]));
    if (argc >= 4 && !strcmp(argv[1], "probe")) return probe((size_t)atol(argv[2]), atof(argv[3]));
    fprintf(stderr, "usage: crash_test victim <MiB> | probe <MiB> <watch ms>\n");
    return 2;
}
