# MetalPassthrough

**MetalPassthrough lets Linux containers on a Mac run native Metal compute shaders and unmodified
NVIDIA CUDA/PTX programs on the host's Apple GPU, with zero-copy access to the container's own
memory.** It plugs into Apple's [`apple/container`](https://github.com/apple/container) and
[`containerization`](https://github.com/apple/containerization) as `container run --gpu`, or runs
standalone via `mpt-run`. It needs no NVIDIA hardware or software at runtime, and it never copies
tensors through a socket.

```
$ mpt-run metalpassthrough-runtime:latest apple-smi      # inside the Linux VM, while a CUDA program and a Python job run
+-----------------------------------------------------------------------------------------+
| APPLE-SMI 0.1.0   MetalPassthrough 0.1.0 (ABI 1)   Metal: Apple9   CUDA bridge: 12.9    |
| Host: macOS 27.2.1 (26B131)   Guest: Linux 6.18.56 in apple/container VM                |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Family   Cores| Unified Memory         | Bandwidth  Max TG    |
| Arena (VM shared, zero-copy)            | Hugepage pool          | Containers  Sessions |
|=========================================|========================|======================|
|   0  Apple M3 Max         Apple9      30| 36.0 GiB (ws 28.1 GiB) |   236 GB/s   1024    |
|       516 MiB [##..........]  12.6%     | 2048x2MiB, 1790 free   |          1         3 |
+-----------------------------------------+------------------------+----------------------+
| Pipelines: 0 built, 9 cached (100% hit) | Last kernel: 6.47 ms   | UVA: VA == dev addr  |
| Ring RTT min/med/p99: 0.67/0.79/2.00 us | GPU busy: 2611.31 ms   | Dispatches: 338      |
+-----------------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|   Session     PID  Process name          Arena MiB    Dispatches        GPU time        |
|=========================================================================================|
|         1       3  driver_test                 256             4        57.19 ms        |
|         2       2  python3                     256           334      2554.12 ms        |
+-----------------------------------------------------------------------------------------+
```

Everything in this README was built and measured on an Apple M3 Max (30-core GPU, 36 GB) with
macOS 27.2.1, apple/container 1.7.1 and containerization 0.54.0. Every number comes from a run
inside a real Linux VM. Nothing is simulated.

---

## What works

| | |
|---|---|
| **Zero-copy unified memory** | Guest memory *is* the Metal buffer. The GPU reads and writes the Linux process's own physical pages. Tensor data never crosses a socket or gets copied. |
| **Unified virtual addressing** | A guest pointer is numerically equal to its GPU device address, so CUDA managed memory and pointer-chasing kernels work unchanged. |
| **Native Metal from Linux** | `libmetalpassthrough.so`: compile MSL, bind arena buffers / bytes / threadgroup memory, batch dispatches into command buffers, GPU timestamps. |
| **CUDA Driver API** | `libcuda.so.1`, compiled against NVIDIA's own `cuda.h` (91 entry points): `cuInit` → `cuMemAlloc` → `cuModuleLoadData(ptx)` → `cuLaunchKernel` → `cuCtxSynchronize`, events, streams, managed and pinned memory. |
| **CUDA Runtime API** | `libcudart.so.12` (58 entry points). Executables built with stock `nvcc -cudart shared` run unmodified: fatbinary registration, `<<<grid, block>>>`, templates, `__shared__`, atomics. |
| **PTX → Metal JIT** | A translator for nvcc-generated PTX (ISA 9.x): integer/float/bit ops, shared/local/generic memory, atomics, warp shuffles/votes, barriers with SIMT-correct reconvergence, and df64 FP64 emulation. |
| **Python** | `import metalpassthrough`: zero-copy NumPy arrays, DLPack / `torch.from_dlpack` views, `@metal_kernel` and `@ptx_kernel` decorators. |
| **`apple-smi`** | An `nvidia-smi` for the Apple GPU from inside Linux. It shows live GPU, memory, pipeline-cache, latency and per-process data. |
| **apple/container integration** | A 49-line upstream patch adds `container run --gpu`. A standalone `mpt-run` works with stock Containerization today. |

## Benchmarks (M3 Max, measured inside the Linux guest)

Two runs are shown. *Quiet* was the first full run on an idle machine. *Loaded* is the final
`make test-all` run, taken while the host had 2.4 GB in swap and about 17 GB in the memory
compressor. CPU baselines degrade much more than GPU work under that load, so the speedup column is
from the quiet run. `build/results/summary.md` holds the raw logs of the latest run.

| Workload (inside Linux VM) | Path | Quiet | Loaded | Check |
|---|---|---|---|---|
| SAXPY, 2 × 256 MiB float32 | native Metal (C) | **248 GB/s** | 229 GB/s | bit-exact vs `fmaf` |
| 5-point stencil 8192², 20 sweeps in one batch | native Metal (C) | **202 GB/s** | 234 GB/s | bit-exact |
| Tiled matmul 2048³ (threadgroup tiles) | native Metal (C) | **1.70 TFLOP/s** | 0.86 TFLOP/s | bit-exact |
| `vector_add`, 16M floats | CUDA Driver API, nvcc PTX | **245 GB/s** | 254 GB/s | bit-exact |
| `tiled_sgemm` 1024³ (`__shared__` + `__syncthreads`) | CUDA Driver API, nvcc PTX | **519 GFLOP/s** | 779 GFLOP/s | bit-exact vs CPU `fmaf` chain |
| `reduce_sum` 16M (`__shfl_down_sync` + `atomicAdd`) | CUDA Driver API, nvcc PTX | 85 GB/s | 150 GB/s | ≤ fp32 rounding |
| `histogram256` 64M bytes (shared + global atomics) | CUDA Driver API, nvcc PTX | 19.7 GB/s | 23.5 GB/s | exact |
| `daxpy` (FP64) 4M | CUDA Driver API, df64 emulation | — | — | worst rel. err **7.2e-15** |
| heat diffusion `<<<(32,32),(16,16)>>>` × 50 | nvcc-built CUDA Runtime binary | 34.7 µs/step | 34.0 µs/step | identical to CPU |
| SAXPY over NumPy arrays (2 × 256 MiB) | Python | 266 GB/s, **24× NumPy** | 253 GB/s | bit-exact |
| fused GELU + sin·exp, 64M floats | Python `@metal_kernel` | 4.2 ms, **145× NumPy** | 4.5 ms | max err 4.3e-7 |
| SGEMM 4096³ (simdgroup matrices) | Python | 1.44 TFLOP/s, 2× OpenBLAS | 0.83 TFLOP/s | rel. err 2.9e-6 |

Unless marked otherwise, the numbers are GPU execution time from `gpuStartTime`/`gpuEndTime`.
Infrastructure costs:

| | |
|---|---|
| Shared-memory ring round trip (hot, both sides spinning) | **0.25 µs** median, 0.17 µs min |
| Ring round trip, host worker parked on the vsock doorbell | 20–30 µs |
| `mpt_alloc` + `mpt_free` (guest-local, no host round trip) | **25 ns** |
| Container boot + `mpt-selftest` end to end (cached rootfs) | **0.65 s** |
| Locating + mapping a 1.1 GiB arena (scan, remap, wrap, make resident) | 42 ms, once per process |

**Overhead against native Metal.** The same SGEMM kernel runs at 1.47 TFLOP/s natively on the host
and 1.44 TFLOP/s from the Linux guest, about 98% of native. SAXPY reaches 248–266 GB/s, against the
host's own measured 250–262 GB/s blit bandwidth. The passthrough costs almost nothing. What limits
the numbers above is the kernels, not MetalPassthrough: the 4096³ SGEMM is a straightforward
simdgroup-matrix kernel, not a tuned MLX/MPS-class GEMM.

## Why not "VirtioFS + mmap"?

The obvious design shares a host directory over VirtioFS, has the guest `mmap(MAP_SHARED)` an arena
file, and has the host `mmap` the same file and call `makeBuffer(bytesNoCopy:)`. **That is not
zero-copy on Virtualization.framework.** Apple's virtio-fs device has no DAX window: no shared-memory
region appears in `VZVirtioFileSystemDeviceConfiguration`. A guest `mmap` of a virtio-fs file is
therefore served from the *guest page cache*. Guest and host map different physical pages, and data
only moves through FUSE reads and writebacks. Coherence is not real-time, and every byte is copied.

MetalPassthrough uses an API that arrived with macOS 27: **`VZCustomVirtioDevice`**. Its
`guestMemoryMapping(atPhysicalAddress:length:)` gives the VM-owning host process a cacheable view of
guest RAM. That is unified memory, so the Apple GPU can address it directly.

## Architecture

```
 Linux guest (apple/container VM, kernel 6.18, 4 KiB pages)              macOS host (runtime helper / mpt-run)
┌─────────────────────────────────────────────────────────┐            ┌─────────────────────────────────────────────────┐
│  CUDA app ──► libcudart.so.12 ──► libcuda.so.1           │            │  MetalPassthroughVZExtension (VZInstanceExtension)  │
│  Python  ──► metalpassthrough (ctypes, NumPy, DLPack)        │            │    • VZCustomVirtioDevice (id 0x7E, class 0x12)  │
│  C/Rust  ──► libmetalpassthrough.so  ◄── apple-smi           │            │    • vsock listener :2049                       │
│                 │                                        │            │                                                 │
│   mmap(MAP_HUGETLB) 2 MiB pages ── stamp {nonce,index}   │  HELLO     │  MetalArenaAllocator                            │
│                 │                                        │ ─vsock───► │    scan guest RAM @2 MiB for nonce stamps       │
│   mremap(hugepages → VA == GPU device address)  (UVA)    │  2049      │    mach_vm_remap chunks → contiguous host VA    │
│                 ▼                                        │            │    pad so chunk0 lands 2 MiB-aligned in GPU VA  │
│  ┌──────────────── arena 0 (guest RAM) ───────────────┐  │            │    makeBuffer(bytesNoCopy:) + MTLResidencySet   │
│  │ 0     ring header (5 × 128 B lines, SPSC)          │◄─┼────────────┼──► same physical pages                          │
│  │ 1K    SQ: 256 × 64 B   ──► host worker polls       │  │ shared     │                                                 │
│  │ 17K   CQ: 256 × 32 B   ◄── completions             │  │ memory     │  MetalComputeDispatcher (one per process)       │
│  │ 32K   device info  48K stats/process table         │  │ (no copy)  │    spin → park on vsock doorbell                │
│  │ 64K   1 MiB per-slot dispatch descriptors          │  │            │    batch dispatches → MTLCommandBuffer          │
│  │ 1.1M  heap: tensors, CUDA allocations, events…     │  │            │    serial compute encoder (CUDA stream order)   │
│  └────────────────────────────────────────────────────┘  │            │  PipelineCache: in-memory + MTLBinaryArchive    │
│        ▲                                   │             │            │  PTXToMSLTranslator → MSL → MTLComputePipeline  │
│        └── 1-byte doorbells over vsock ────┘             │            └────────────────────┬────────────────────────────┘
└─────────────────────────────────────────────────────────┘                                 ▼
                                                                                  Apple M3 Max GPU (30 cores)
```

**Data plane.** The guest owns all memory. `libmetalpassthrough` maps 2 MiB hugepages (reserved at boot
with `hugepages=N`) and stamps each with a random 128-bit nonce. The container has no
`CAP_SYS_ADMIN`, so `/proc/self/pagemap` cannot reveal physical addresses. Instead the host scans
guest RAM at 2 MiB strides for the stamps, which takes about 24 ms for 4 GiB. It then remaps the
found chunks into one contiguous host range and wraps that as a single no-copy `MTLBuffer`. Every
tensor is an offset into that buffer, so Metal buffer creation happens once per arena, never per
allocation.

**Unified addressing.** Metal assigns GPU virtual addresses with a bump allocator. The host predicts
the next address with a probe buffer and prefixes the arena with zero-filled host pages, so that
guest chunk 0 lands on a 2 MiB-aligned GPU VA, then verifies it. The guest then `mremap`s its
hugepages to exactly that numeric address, using `MAP_FIXED_NOREPLACE` and never clobbering. From
then on `ptr == mpt_devaddr(ptr)`: a guest pointer *is* the device pointer.

**Control plane.** A lock-free single-producer/single-consumer ring sits in arena 0, laid out in
128-byte Apple cache lines, each written by exactly one side, with acquire/release atomics on both
sides. The host worker drains the SQ, encodes work into batched command buffers, and spins for 50 µs
when idle before parking on vsock. The guest writes a 1-byte doorbell only when `host_sleeping` is
set. The guest waits for CQEs the same way. Fire-and-forget dispatches post a completion only on
error, which surfaces at the next sync as a sticky error, the way CUDA reports asynchronous failures.

**PTX → MSL.** `Sources/MetalPassthroughHost/PTX` is a real compiler pass, not a template matcher:
- Typed register model. Every PTX virtual register becomes a typed MSL local, and operands are
  reinterpreted to the instruction type.
- Kernel parameters are packed with CUDA ABI alignment at `buffer(0)`.
- All `.shared` memory, static and `.extern`, lives in one threadgroup allocation. `.local` becomes
  a per-thread array.
- Generic pointers carry a state-space tag in bits 63:62, set by `cvta` and dispatched at runtime.

MSL has no `goto`, so control flow becomes a basic-block state machine. Kernels that contain
barriers or SIMD-group operations need more. They use reverse-post-order block numbering with
**min-PC scheduling** (`simd_min(_pc)`), so divergent lanes reconverge before a
`threadgroup_barrier`. Without this, tiled SGEMM is wrong in partial tiles and histograms lose
counts. Apple GPUs have no FP64 ALU, so `.f64` defaults to **df64** (float-float pairs, about
48-bit mantissa, worst-case 7e-15 on `daxpy`). Memory keeps exact IEEE-754 doubles either way.
`METALPASSTHROUGH_FP64=fp32` lowers FP64 to fp32 instead.

## Quickstart

Requirements: Apple Silicon, **macOS 27+** (for `VZCustomVirtioDevice`), Xcode 27, and
`brew install container` (apple/container 1.7.1). Clone outside `~/Documents` and `~/Desktop`,
where vmnet fails on macOS 26+.

```sh
git clone https://github.com/giuseppegambino92/MetalPassthrough.git ~/Projects/MetalPassthrough
cd ~/Projects/MetalPassthrough
container system start
make images      # Linux builder (gcc, rustc, CUDA 12.9 headers + nvcc) and runtime (NumPy, PyTorch CPU) images
make all         # host mpt-run (ad-hoc signed with com.apple.security.virtualization), guest runtime, e2e programs
make test-all    # unit tests + the 4 end-to-end tests inside real Linux containers

# Standalone runner: any image in apple/container's store, GPU attached
.build/release/mpt-run metalpassthrough-runtime:latest apple-smi
.build/release/mpt-run --hugepages 2048 -v $PWD/e2e:/e2e metalpassthrough-runtime:latest \
    python3 /e2e/python/test4_numpy_torch.py
```

### `container run --gpu` (patch for apple/container)

[`patches/0001-feat-gpu-passthrough-metal-zero-copy.patch`](patches/0001-feat-gpu-passthrough-metal-zero-copy.patch)
is a 49-line change to apple/container 1.7.1. It adds `--gpu` to `container run` and
`container create`, reserves guest hugepages, attaches the MetalPassthrough device to the VM, and
bind-mounts the guest runtime. It resolves MetalPassthrough as a sibling checkout
(`../MetalPassthrough`) or from `$METALPASSTHROUGH_PATH`.

```sh
make install                         # guest runtime -> ~/Library/Application Support/dev.metalpassthrough/guest
cd ~/Projects
git clone -b 1.7.1 https://github.com/apple/container.git && cd container
git am ../MetalPassthrough/patches/0001-feat-gpu-passthrough-metal-zero-copy.patch
make container                       # builds bin/ + libexec/ in the checkout
container system stop                # use the patched build with an isolated app root
bin/container system start --app-root $PWD/.test-data --enable-kernel-install
bin/container run --rm --gpu -m 4G ubuntu:22.04 apple-smi
```

`make patch` does the same against `vendor/container`.

Inside a `--gpu` container, `/usr/local/metalpassthrough/{lib,bin,python}` is mounted, and
`LD_LIBRARY_PATH`, `PATH` and `PYTHONPATH` are set. Any glibc-based image works.

### C

```c
#include <metalpassthrough.h>
mpt_ctx *c = mpt_default();
float *x = mpt_alloc(c, n * 4, 0), *y = mpt_alloc(c, n * 4, 0);    // GPU-visible, zero-copy
mpt_pipeline p; mpt_compile_msl(c, src, "saxpy", 0, &p, log, sizeof log);
mpt_arg a[] = {{MPT_BIND_BUFFER, 0, x}, {MPT_BIND_BUFFER, 1, y}, {MPT_BIND_BYTES, 2, &alpha, 4}};
mpt_dispatch(c, &p, (uint32_t[]){n,1,1}, (uint32_t[]){256,1,1}, MPT_DISPATCH_THREADS, a, 3, 0);
mpt_sync(c, &gpu_ns);                                                // y is already in your memory
```

### CUDA

```sh
nvcc -O3 -cudart shared -o app app.cu      # on any Linux arm64 box / in the builder image
./app                                       # inside a --gpu container: runs on the Apple GPU
```

### Python

```python
import numpy as np, metalpassthrough as mpt, torch

@mpt.metal_kernel
def saxpy():
    """
    kernel void saxpy(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      constant float& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        y[i] = fma(a, x[i], y[i]);
    }
    """

x = mpt.asarray(np.random.rand(1 << 24).astype(np.float32)); y = mpt.zeros_like(x)
saxpy(x, y, np.float32(2.0), grid=x.size); mpt.synchronize()
t = torch.from_dlpack(y)        # same pages, no copy

vadd = mpt.PTXKernel(open("kernels.ptx").read(), "vector_add")   # NVIDIA PTX from Python
vadd(x, y, y, np.int32(x.size), grid=(x.size // 256,), block=(256,))
```

## Compatibility and limitations

MetalPassthrough is a working system, not a full CUDA implementation. Know its edges:

- **PTX only, no SASS.** Binaries must embed PTX (nvcc's default embeds it; `-gencode …,code=compute_XX`
  guarantees it). Cubin-only images return `CUDA_ERROR_NO_BINARY_FOR_GPU`.
- **Not translated yet:**
  - Textures and surfaces, tensor cores (`mma`, `wmma`, `ldmatrix`), `cp.async`, clusters.
  - Non-inlined device-function `call`s, including `printf`.
  - Module-scope `__device__`/`__constant__` variables (`cuModuleGetGlobal`).
  - 64-bit atomics: Metal has none apart from min/max.
  - `%clock`/`%globaltimer`.

  Each one fails at JIT time with an error naming the PTX line. A few constructs are approximated
  rather than rejected: directed rounding modes (`.rz/.rm/.rp`) run as round-to-nearest, and
  `bar.sync` with a thread count is a full barrier. Each approximation appears in the JIT log
  (`METALPASSTHROUGH_DEBUG=1`).
- **FP64** is emulated (df64, or fp32 with `METALPASSTHROUGH_FP64=fp32`). It is not IEEE double
  arithmetic.
- **Streams** all execute in order on one Metal queue per process. That is correct, but there is no
  inter-stream concurrency.
- **Host↔device copies** are CPU `memcpy` within guest memory: device memory *is* host memory. Use
  `cudaMallocManaged`/`cudaHostAlloc`, or `mpt.empty`, to skip even that.
- **Hugepages** are reserved at VM boot. `--gpu` defaults to half of the VM memory
  (`METALPASSTHROUGH_GPU_MEMORY_MB` overrides). `mpt-run --hugepages N` sets it explicitly.
- **Isolation.** All sessions of one VM share a host process and Metal device, and a malicious
  shader can read other arenas *of the same VM*. Different containers run in different VM processes.
- **Crash window.** If a guest process is SIGKILLed while its GPU work is in flight, the kernel can
  recycle its hugepages before the host finishes that work. An orderly exit always drains the GPU
  first (`CLOSE`). A guest-kernel helper that pins pages would close this gap.
- **Hugepage stability (assumption).** The design assumes an arena's guest-physical pages don't
  move while mapped. hugetlb pages are never swapped, but the kernel *can* migrate them (memory
  offlining, compaction for contiguous allocations). Nothing in the default apple/container VM does
  that, but it is not enforced, and a pinning guest driver would make it a guarantee.
- **macOS 27+ only** for `--gpu`. On older systems the flag returns an unsupported error, and
  nothing changes without the flag.

## Repository layout

```
include/metalpassthrough_protocol.h   wire + shared-memory ABI (C, imported by Swift; static-asserted)
Sources/MetalPassthroughHost/         Swift 6 host engine
  GuestMemoryDevice.swift           VZCustomVirtioDevice, guest RAM discovery, mach_vm_remap regions
  MetalArenaAllocator.swift         nonce scan, UVA placement, bytesNoCopy MTLBuffers, residency
  MetalComputeDispatcher.swift      ring worker, batching, copies/fills, events, stats
  MetalPassthroughServer.swift          handshake, sessions, VM/host registry
  MetalPassthroughVZExtension.swift     Containerization VZInstanceExtension + VM manager wrapper
  MetalPassthroughAttachment.swift      one-call integration used by mpt-run and `container --gpu`
  PipelineCache.swift               in-memory + persistent MTLBinaryArchive
  PTX/                              PTX parser, PTX→MSL code generator, MSL runtime prelude (df64, atomics…)
Sources/mpt-run/                 standalone runner (Containerization, cached rootfs clones)
guest/                            Linux side: libmetalpassthrough.so, libcuda.so.1, libcudart.so.12, apple-smi (Rust), Python
Tests/MetalPassthroughHostTests/      translator tests: real nvcc PTX executed on the local GPU vs CPU references
e2e/                              Tests 1-4 programs (CUDA C, CUDA C++, native Metal C, Python)
patches/                          apple/container `--gpu` patch (git format-patch)
scripts/                          build-host.sh (sign), test-all.sh
```

## License

Apache License 2.0. See [LICENSE](LICENSE). NVIDIA CUDA headers are used only at build time inside
the builder image, and no NVIDIA software is redistributed or required at runtime. "CUDA" is a
trademark of NVIDIA Corporation, and this project is not affiliated with NVIDIA or Apple.
