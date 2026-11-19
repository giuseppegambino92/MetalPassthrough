/*
 * Isolation probe: what can a raw-pointer shader reach? Runs a kernel that loads (and optionally
 * stores) through arbitrary 64-bit device addresses and reports, per target, whether the access
 * returned data, whether a store landed, and whether the host reported a GPU fault.
 *   isolation victim <seconds>        allocate 16 MiB of 0xA5A5A5A5, print its device address, wait
 *   isolation probe [victim-addr]     run the probes (victim-addr: another process / VM)
 * Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.
 */
#include "metalpassthrough.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *SRC =
    "#include <metal_stdlib>\nusing namespace metal;\n"
    "kernel void probe(constant ulong* addr [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& store [[buffer(2)]],\n"
    "                  uint i [[thread_position_in_grid]]) {\n"
    "  device volatile uint* p = reinterpret_cast<device volatile uint*>(addr[i]);\n"
    "  out[i] = p[0];\n"
    "  if (store) p[0] = 0x0BADF00Du;\n"
    "}\n";

static mpt_ctx *c;
static mpt_pipeline pso;
static uint32_t *out;

/* Returns the sync status; *val = loaded value. */
static int probe1(uint64_t addr, int store, uint32_t *val) {
    uint32_t st = (uint32_t)store;
    out[0] = 0xDEADBEEF;
    mpt_arg a[] = { { MPT_BIND_BYTES, 0, &addr, 8 }, { MPT_BIND_BUFFER, 1, out, 0 }, { MPT_BIND_BYTES, 2, &st, 4 } };
    uint32_t grid[3] = { 1, 1, 1 }, tg[3] = { 1, 1, 1 };
    int r = mpt_dispatch(c, &pso, grid, tg, MPT_DISPATCH_THREADS, a, 3, 0);
    int s = mpt_sync(c, NULL);
    *val = out[0];
    return r ? r : s;
}

static void report(const char *what, uint64_t addr, int store) {
    uint32_t v;
    int s = probe1(addr, store, &v);
    printf("  %-48s 0x%012" PRIx64 " %s -> %s, loaded 0x%08x\n", what, addr, store ? "load+store" : "load      ",
           s ? mpt_strerror(s) : "no fault reported", v);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: isolation victim <s> | probe [addr]\n"); return 2; }
    if (mpt_open(&c, 64 << 20)) { fprintf(stderr, "%s\n", mpt_last_error()); return 1; }
    if (!strcmp(argv[1], "victim")) {
        uint32_t *v = mpt_alloc(c, 16 << 20, 0);
        for (size_t i = 0; i < (16u << 20) / 4; i++) v[i] = 0xA5A5A5A5u;
        printf("VICTIM 0x%" PRIx64 "\n", mpt_devaddr(c, v));
        fflush(stdout);
        sleep(argc > 2 ? atoi(argv[2]) : 20);
        size_t changed = 0;
        for (size_t i = 0; i < (16u << 20) / 4; i++) changed += v[i] != 0xA5A5A5A5u;
        printf("VICTIM words modified by other processes/VMs: %zu\n", changed);
        return 0;
    }
    char log[4096];
    if (mpt_compile_msl(c, SRC, "probe", 0, &pso, log, sizeof log)) { fprintf(stderr, "%s\n", log); return 1; }
    out = mpt_alloc(c, 4096, 0);
    uint32_t *mine = mpt_alloc(c, 4096, 0);
    mine[0] = 0x12345678;
    uint64_t len, base = mpt_debug_arena_range(c, &len);
    printf("own arena: 0x%" PRIx64 " + %" PRIu64 " MiB\n", base, len >> 20);
    report("own allocation", mpt_devaddr(c, mine), 1);
    printf("  (own allocation now holds 0x%08x)\n", mine[0]);
    report("unmapped address", 0x7f0000000000ull, 0);
    report("unmapped address", 0x7f0000000000ull, 1);
    report("1 MiB below own arena (host padding / host buffers)", base - (1u << 20), 0);
    report("64 MiB below own arena", base - (64u << 20), 0);
    report("just past own arena", base + len, 0);
    if (argc > 2) {
        uint64_t victim = strtoull(argv[2], NULL, 0);
        report("victim address (other process or VM)", victim, 0);
        report("victim address (other process or VM)", victim, 1);
    }
    /* After faults, the session must still work. */
    uint32_t v;
    int s = probe1(mpt_devaddr(c, mine), 0, &v);
    printf("session after probes: %s (own allocation reads 0x%08x)\n", s ? mpt_strerror(s) : "ok", v);
    return 0;
}
