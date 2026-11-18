/* NVIDIA fatbinary container parsing. Layout per public reverse engineering of nvcc output
 * (cuobjdump / gpgpu-sim / cuda-fatbin-decompression). Copyright 2026 MetalPassthrough contributors. Apache-2.0. */
#include "fatbin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t header_size;
    uint64_t size;           /* bytes of entries following the header */
} fat_header;

typedef struct {
    uint16_t kind;           /* 1 = PTX, 2 = ELF (SASS cubin) */
    uint16_t unknown1;
    uint32_t header_size;
    uint64_t size;           /* payload bytes (padded) */
    uint32_t compressed_size;
    uint32_t unknown2;
    uint16_t minor;
    uint16_t major;
    uint32_t arch;           /* e.g. 80 for sm_80 / compute_80 */
    uint32_t obj_name_offset;
    uint32_t obj_name_len;
    uint64_t flags;
    uint64_t zero;
    uint64_t decompressed_size;
} fat_entry;
#pragma pack(pop)

#define FATBIN_FLAG_COMPRESS     0x0000000000002000ull
#define FATBIN_FLAG_COMPRESS_ZSTD 0x0000000000008000ull

long mpt_lz4_decompress(const uint8_t *src, size_t n, uint8_t *dst, size_t cap) {
    const uint8_t *ip = src, *iend = src + n;
    uint8_t *op = dst, *oend = dst + cap;
    while (ip < iend) {
        unsigned token = *ip++;
        size_t lit = token >> 4;
        if (lit == 15) { unsigned b; do { if (ip >= iend) return -1; b = *ip++; lit += b; } while (b == 255); }
        if ((size_t)(iend - ip) < lit || (size_t)(oend - op) < lit) return -1;
        memcpy(op, ip, lit);
        op += lit;
        ip += lit;
        if (ip >= iend) break;  /* last sequence has literals only */
        if (iend - ip < 2) return -1;
        size_t off = (size_t)ip[0] | ((size_t)ip[1] << 8);
        ip += 2;
        if (off == 0 || off > (size_t)(op - dst)) return -1;
        size_t ml = token & 15;
        if (ml == 15) { unsigned b; do { if (ip >= iend) return -1; b = *ip++; ml += b; } while (b == 255); }
        ml += 4;
        if ((size_t)(oend - op) < ml) return -1;
        const uint8_t *m = op - off;
        while (ml--) *op++ = *m++;  /* overlapping copy */
    }
    return (long)(op - dst);
}

static int looks_like_ptx(const char *s) {
    for (int i = 0; i < 4096 && s[i]; i++)
        if (!strncmp(s + i, ".version", 8) || !strncmp(s + i, ".target", 7) || !strncmp(s + i, ".entry", 6)) return 1;
    return 0;
}

char *mpt_fatbin_extract_ptx(const void *image, int *has_sass, char *err, size_t errlen) {
    *has_sass = 0;
    const uint8_t *p = image;
    uint32_t magic;
    memcpy(&magic, p, 4);
    if (magic == FATBINC_MAGIC) {
        const mpt_fatbinc_wrapper *w = image;
        p = (const uint8_t *)w->data;
        memcpy(&magic, p, 4);
    }
    if (magic == 0x464c457fu) { *has_sass = 1; snprintf(err, errlen, "image is a SASS cubin (ELF); MetalPassthrough executes PTX"); return NULL; }
    if (magic != FATBIN_MAGIC) {
        if (looks_like_ptx((const char *)image)) return strdup((const char *)image);
        snprintf(err, errlen, "unrecognized module image (magic 0x%08x)", magic);
        return NULL;
    }
    const fat_header *h = (const fat_header *)p;
    const uint8_t *e = p + h->header_size, *end = e + h->size;
    const fat_entry *best = NULL;
    while (e + sizeof(fat_entry) <= end) {
        const fat_entry *fe = (const fat_entry *)e;
        if (fe->header_size < 16 || fe->size > (uint64_t)(end - e)) break;
        if (fe->kind == 1) { if (!best || fe->arch > best->arch) best = fe; }
        else if (fe->kind == 2) *has_sass = 1;
        e += fe->header_size + fe->size;
    }
    if (!best) {
        snprintf(err, errlen, "fatbinary contains no PTX (only SASS); rebuild with PTX embedded, e.g. -gencode arch=compute_80,code=compute_80");
        return NULL;
    }
    const uint8_t *payload = (const uint8_t *)best + best->header_size;
    char *out;
    if (best->flags & FATBIN_FLAG_COMPRESS_ZSTD) {
        snprintf(err, errlen, "zstd-compressed fatbin PTX is not supported; build with -Xfatbin -compress-all=false or --compress-mode=none");
        return NULL;
    } else if (best->flags & FATBIN_FLAG_COMPRESS) {
        size_t cap = best->decompressed_size + 1;
        out = calloc(1, cap + 64);
        long n = mpt_lz4_decompress(payload, best->compressed_size, (uint8_t *)out, cap + 63);
        if (n < 0) { free(out); snprintf(err, errlen, "fatbin PTX decompression failed"); return NULL; }
        out[n] = 0;
    } else {
        out = calloc(1, best->size + 1);
        memcpy(out, payload, best->size);
    }
    if (!looks_like_ptx(out)) { free(out); snprintf(err, errlen, "fatbin PTX entry is not valid PTX text"); return NULL; }
    return out;
}
