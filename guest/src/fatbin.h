/* NVIDIA fatbinary container parsing (PTX extraction). Copyright 2026 MetalPassthrough contributors. Apache-2.0. */
#ifndef MPT_FATBIN_H
#define MPT_FATBIN_H
#include <stddef.h>
#include <stdint.h>

#define FATBINC_MAGIC 0x466243b1u   /* __fatBinC_Wrapper_t, passed to __cudaRegisterFatBinary */
#define FATBIN_MAGIC  0xBA55ED50u   /* fatbinary header */

typedef struct {
    int magic;
    int version;
    const unsigned long long *data;
    void *filename_or_fatbins;
} mpt_fatbinc_wrapper;

/* Returns a malloc'd NUL-terminated PTX string for the highest-arch PTX entry in the image, or NULL.
 * `image` may be a __fatBinC_Wrapper_t, a raw fatbinary, or PTX text. *has_sass is set if the image
 * only carried SASS (cubin) code, which MetalPassthrough cannot execute. */
char *mpt_fatbin_extract_ptx(const void *image, int *has_sass, char *err, size_t errlen);

/* LZ4 block decompression (fatbin payload compression). Returns bytes written or -1. */
long mpt_lz4_decompress(const uint8_t *src, size_t srclen, uint8_t *dst, size_t dstcap);
#endif
