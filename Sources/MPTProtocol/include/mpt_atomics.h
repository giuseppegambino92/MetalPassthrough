/* Acquire/release helpers so Swift can participate in the guest's C11 ring protocol.
 * Copyright 2026 MetalPassthrough contributors. Apache-2.0. */
#ifndef MPT_ATOMICS_H
#define MPT_ATOMICS_H
#include <stdint.h>
static inline uint32_t mpt_load_acquire_u32(const void *p) { return __atomic_load_n((const uint32_t *)p, __ATOMIC_ACQUIRE); }
static inline uint32_t mpt_load_relaxed_u32(const void *p) { return __atomic_load_n((const uint32_t *)p, __ATOMIC_RELAXED); }
static inline void mpt_store_release_u32(void *p, uint32_t v) { __atomic_store_n((uint32_t *)p, v, __ATOMIC_RELEASE); }
static inline void mpt_store_seqcst_u32(void *p, uint32_t v) { __atomic_store_n((uint32_t *)p, v, __ATOMIC_SEQ_CST); }
static inline uint32_t mpt_load_seqcst_u32(const void *p) { return __atomic_load_n((const uint32_t *)p, __ATOMIC_SEQ_CST); }
static inline void mpt_fence_seqcst(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
static inline void mpt_cpu_relax(void) { __asm__ __volatile__("yield" ::: "memory"); }
static inline uint64_t mpt_load_relaxed_u64(const void *p) { return __atomic_load_n((const uint64_t *)p, __ATOMIC_RELAXED); }
static inline void mpt_store_relaxed_u64(void *p, uint64_t v) { __atomic_store_n((uint64_t *)p, v, __ATOMIC_RELAXED); }
static inline void mpt_fence_acquire(void) { __atomic_thread_fence(__ATOMIC_ACQUIRE); }
static inline void mpt_fence_release(void) { __atomic_thread_fence(__ATOMIC_RELEASE); }
static inline void mpt_store_relaxed_u32(void *p, uint32_t v) { __atomic_store_n((uint32_t *)p, v, __ATOMIC_RELAXED); }
#endif
