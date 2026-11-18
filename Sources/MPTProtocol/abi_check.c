/* Compile-time ABI layout verification (also compiled into the guest library). */
#include "metalpassthrough_protocol.h"
#include <stddef.h>
_Static_assert(sizeof(mpt_sqe) == 64, "sqe");
_Static_assert(sizeof(mpt_cqe) == 32, "cqe");
_Static_assert(sizeof(mpt_hello) == 48, "hello");
_Static_assert(sizeof(mpt_hello_ack) == 32, "hello_ack");
_Static_assert(sizeof(mpt_ring_header) == 5 * MPT_CACHELINE, "ring header");
_Static_assert(offsetof(mpt_ring_header, sq_tail) == 128, "sq_tail line");
_Static_assert(offsetof(mpt_ring_header, sq_head) == 256, "sq_head line");
_Static_assert(offsetof(mpt_ring_header, cq_tail) == 384, "cq_tail line");
_Static_assert(offsetof(mpt_ring_header, cq_head) == 512, "cq_head line");
_Static_assert(sizeof(mpt_ring_header) <= MPT_SQ_OFFSET, "ring fits");
_Static_assert(MPT_CQ_OFFSET + MPT_CQ_DEPTH * sizeof(mpt_cqe) <= MPT_INFO_OFFSET, "cq fits");
_Static_assert(sizeof(mpt_device_info) <= MPT_STATS_OFFSET - MPT_INFO_OFFSET, "info fits");
_Static_assert(sizeof(mpt_stats) <= MPT_CONTROL_SIZE - MPT_STATS_OFFSET, "stats fits");
_Static_assert(sizeof(mpt_binding) == 24, "binding");
_Static_assert(sizeof(mpt_proc_info) == 48, "proc entry");
_Static_assert(sizeof(mpt_dispatch_desc) == 40, "dispatch");
_Static_assert(sizeof(mpt_event) == 16, "event");
_Static_assert(MPT_CHUNK_SIZE % MPT_HOST_PAGE == 0, "chunk/page");
_Static_assert(MPT_HEAP_OFFSET % MPT_HOST_PAGE == 0, "heap aligned");
_Static_assert(MPT_CHUNK_SIZE == (2u << 20), "chunk");
_Static_assert(MPT_CQ_OFFSET == MPT_SQ_OFFSET + MPT_SQ_DEPTH * sizeof(mpt_sqe), "cq offset");
_Static_assert(MPT_SCRATCH_OFFSET == MPT_CONTROL_SIZE, "scratch offset");
_Static_assert(MPT_SCRATCH_SIZE == MPT_SQ_DEPTH * MPT_SCRATCH_PER_SLOT, "scratch size");
_Static_assert(MPT_HEAP_OFFSET == MPT_SCRATCH_OFFSET + MPT_SCRATCH_SIZE, "heap offset");
_Static_assert(MPT_INFO_OFFSET == 32768u && MPT_STATS_OFFSET == 49152u && MPT_CONTROL_SIZE == 65536u, "control layout");
int mpt_abi_version(void) { return MPT_ABI_VERSION; }
