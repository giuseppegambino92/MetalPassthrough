# MetalPassthrough

**MetalPassthrough lets Linux containers on a Mac run native Metal compute shaders and unmodified
NVIDIA CUDA/PTX programs on the host's Apple GPU, with zero-copy access to the container's own
memory.** It plugs into an **unmodified** [`apple/container`](https://github.com/apple/container)
as a runtime plugin (`container run --runtime container-runtime-metalpassthrough`), into a patched
one as `container run --gpu`, or runs standalone on
[`containerization`](https://github.com/apple/containerization) via `mpt-run`. It needs no NVIDIA
hardware or software at runtime, and it never copies tensors through a socket. Measured from inside
the Linux VM, kernels run at **99–100% of native Metal speed** on the same Mac
([reproducible benchmark](bench/results/RESULTS.md)).

```
$ mpt-run metalpassthrough-runtime:latest apple-smi      # inside the Linux VM, while a CUDA program and a Python job run
+-----------------------------------------------------------------------------------------+
| APPLE-SMI 0.1.0   MetalPassthrough 0.1.0 (ABI 2)   Metal: Apple9   CUDA bridge: 12.9    |
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
| **CUDA Driver API** | `libcuda.so.1`, compiled against NVIDIA's own `cuda.h` (93 entry points): `cuInit` → `cuMemAlloc` → `cuModuleLoadData(ptx)` → `cuLaunchKernel` → `cuCtxSynchronize`, managed and pinned memory. |
| **CUDA streams and events** | Real streams (one Metal queue each) with CUDA's legacy default-stream rules and `CU_STREAM_NON_BLOCKING`; `cuStreamWaitEvent` is a GPU-side dependency (`MTLSharedEvent`); `cuStreamQuery`/`cuEventQuery` never block; asynchronous copies and memsets are stream-ordered; every copy and memset is range-checked against its allocation. |
| **CUDA Runtime API** | `libcudart.so.12` (58 entry points). Executables built with stock `nvcc -cudart shared` run unmodified: fatbinary registration, `<<<grid, block>>>`, templates, `__shared__`, atomics. |
| **cuBLAS** | `libcublas.so.12` against NVIDIA's `cublas_v2.h`: `cublasSgemm`/`SgemmStridedBatched` (all transposes, padded leading dimensions) run Apple's `MPSMatrixMultiplication` in place on container memory, up to **8.8 TFLOP/s**; level-1 `Saxpy/Sscal/Scopy/Sdot/Snrm2`. |
| **Triton** | PTX produced by Triton 3.5 (compiled AOT on a Mac, no NVIDIA GPU) loads and runs: `add` and a row softmax are verified in CI. |
| **PTX → Metal JIT** | A translator for nvcc-generated PTX (ISA 9.x): integer/float/bit ops, shared/local/generic memory, atomics, warp shuffles/votes, barriers with SIMT-correct reconvergence, and df64 FP64 emulation. |
| **Python** | `import metalpassthrough`: zero-copy NumPy arrays, DLPack / `torch.from_dlpack` views, `@metal_kernel` and `@ptx_kernel` decorators, `mpt.matmul` (MPS GEMM). |
| **Crash-safe teardown** | A SIGKILLed process cannot have its pages recycled while its GPU work still runs: arenas are hugetlb memfds held by a detached keeper process that the host releases only after the GPU drained (tested: 68 874 foreign writes without it, 0 with it). |
| **Hostile-guest hardening** | Every descriptor is copied before validation, sizes are overflow-checked, ring violations end only the offending session. A protocol fuzzer (80 000 random SQEs, malformed handshakes, garbage shaders, corrupted ring indices) and a PTX mutation fuzzer run against the real host. |
| **`apple-smi`** | An `nvidia-smi` for the Apple GPU from inside Linux. It shows live GPU, memory, pipeline-cache, latency and per-process data. |
| **apple/container integration** | A runtime plugin for the **stock** CLI (`--runtime container-runtime-metalpassthrough`), or a 49-line patch that adds `container run --gpu`, or the standalone `mpt-run`. |

## Benchmarks: the same kernels, native vs. inside the Linux guest

`make bench` (`bench/run.sh`) runs **identical kernels and sizes** (`bench/kernels.metal`) once in a
native macOS process on ordinary `MTLBuffer`s (`Sources/mpt-bench`) and once from a Linux container
through MetalPassthrough (`bench/bench_guest.c`). It alternates native and guest rounds so that
system noise hits both equally, and reports medians. *GPU* time comes from command-buffer
timestamps. *Wall* time runs from the API call until the result is visible to the calling CPU,
which for the guest means inside the VM. Full table and raw CSV:
[`bench/results/RESULTS.md`](bench/results/RESULTS.md). Measured on an M3 Max, macOS 27.2.1,
5 rounds × 7 repetitions.

| Kernel | Size | Native | Linux guest | Guest / native (GPU time) | Wall-clock overhead |
|---|---|---:|---:|---:|---:|
| SAXPY | 256 MiB | 270.4 GB/s | 270.7 GB/s | 100% | +0.10 ms |
| SAXPY | 1 MiB | 177.6 GB/s | 176.0 GB/s | 99% | +0.08 ms |
| 5-point stencil | 8192² | 251.5 GB/s | 251.5 GB/s | 100% | +0.08 ms |
| SGEMM, simdgroup-matrix MSL | 4096³ | 1.46 TFLOP/s | 1.47 TFLOP/s | 100% | ≈ 0 |
| SGEMM, Metal Performance Shaders | 4096³ | 8.88 TFLOP/s | 8.92 TFLOP/s | 100% | +0.16 ms |
| SGEMM, Metal Performance Shaders | 512³ | 6.15 TFLOP/s | 6.13 TFLOP/s | 100% | +0.11 ms |
| 10 000 tiny dispatches, batched | — | 1.12 M/s | 1.04 M/s | — | 94% of native throughput |
| one tiny dispatch, submit → result visible | — | 160 µs | 212 µs | — | +52 µs |

**GPU time is the same.** The guest's memory *is* the Metal buffer, so once work reaches the GPU
nothing distinguishes it from native work. The cost is fixed and per synchronization: about
0.06–0.1 ms of wall time when the guest waits for a result, mostly the host waking from its parked
state. Small, synchronous launches pay it, while batched and asynchronous work hides it.

Infrastructure costs, measured in the guest:

| | |
|---|---|
| Shared-memory ring round trip, hot (both sides spinning) | **0.29 µs** |
| Ring round trip, host worker parked on the vsock doorbell | 52 µs |
| `mpt_alloc` + `mpt_free` (guest-local, no host round trip) | **16 ns** |
| Opening a session: map + scan + remap + make resident, 64 MiB / 256 MiB / 1 GiB | 2.8 / 7.6 / 32 ms |
| Host memory of the GPU service (`mpt-run` peak RSS, 1.1 GiB arena) | 76 MiB (the arena itself is guest memory) |
| Container boot + `mpt-selftest` end to end (cached rootfs) | 0.65 s |

**Applications** (inside the VM, from `make test-all`; each verified against an independent
reference):

| Workload | Path | Result | Check |
|---|---|---|---|
| `cublasSgemm` 2048³ / 4096³, unmodified nvcc binary | cuBLAS → MPS | 7.0 / 8.8 TFLOP/s | all transposes, padded ld, strided batch vs float64 |
| CG solver, 2D Poisson, 1M unknowns, 923 iterations | CUDA Runtime + cuBLAS | 0.95 ms/iteration, 3.3× a CPU core | error vs analytic solution 1.6e-5 |
| FlashAttention-style attention, B=2 H=8 S=1024 D=64 | CUDA Runtime, nvcc PTX | 723 GFLOP/s | 7e-8 vs float64 softmax attention |
| Triton 3.5 `softmax`, 4096 × 1000 | Python, Triton PTX | 0.18–0.43 ms | 1.5e-8 |
| SAXPY over NumPy arrays, 2 × 256 MiB | Python | 263 GB/s, 23× NumPy | bit-exact |
| fused GELU + sin·exp, 64M floats | Python `@metal_kernel` | 3.2 ms, 182× NumPy | 4.3e-7 |
| `mpt.matmul` 4096³ | Python → MPS | 8.9 TFLOP/s, 12× OpenBLAS | 2.9e-6 |
| `tiled_sgemm`, `reduce_sum`, `histogram256`, FP64 `daxpy` | CUDA Driver API, nvcc PTX | — | bit-exact / exact / df64 7.2e-15 |

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
│  │ 1.1M  heap: tensors, CUDA allocations, events…     │  │            │    one MTLCommandQueue per CUDA stream + events │
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
make test-all    # unit, end-to-end, workload and robustness tests inside real Linux containers
make bench       # native vs. in-guest benchmark -> bench/results/RESULTS.md

# Standalone runner: any image in apple/container's store, GPU attached
.build/release/mpt-run metalpassthrough-runtime:latest apple-smi
.build/release/mpt-run --hugepages 2048 -v $PWD/e2e:/e2e metalpassthrough-runtime:latest \
    python3 /e2e/python/test4_numpy_torch.py
```

### Stock `container` + runtime plugin (no patch)

apple/container loads runtime plugins from `<install root>/libexec/container-plugins` (Homebrew
links that to `$(brew --prefix)/libexec/container-plugins`, which survives upgrades), and its CLI
already has `--runtime` to pick one. `make install-plugin` builds apple/container's own Linux
runtime helper **from the same release tag as your installed CLI**, with MetalPassthrough compiled
in, installs it as `container-runtime-metalpassthrough` together with the guest runtime, and
restarts the container services.

```sh
make install-plugin
container run --rm --runtime container-runtime-metalpassthrough metalpassthrough-runtime:latest apple-smi
container run --rm --runtime container-runtime-metalpassthrough -m 8G \
    --label dev.metalpassthrough.gpu-memory-mb=4096 ubuntu:22.04 apple-smi     # GPU pool size (default: half the VM)
./scripts/install-plugin.sh --uninstall
```

Containers started without `--runtime` keep using the stock `container-runtime-linux`, untouched.
The plugin still has to match the CLI's XPC types, so it is rebuilt per apple/container release;
a stable device-extension point upstream ([apple/container#1511](https://github.com/apple/container/issues/1511))
would remove even that.

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

Inside a GPU container (plugin or `--gpu`), `/usr/local/metalpassthrough/{lib,bin,python}` is mounted, and
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
- **Stream concurrency.** Each CUDA stream is its own `MTLCommandQueue` with GPU-side event
  dependencies, so ordering is exact. But on the tested M3 Max / macOS 27.2, Metal itself runs
  compute command buffers from different queues one after another, natively too (two 15.5 ms
  kernels on two queues take 31.2 ms, `bench/concurrency.swift`). Independent streams are
  therefore correct but not faster than one stream.
- **Host↔device copies** are CPU `memcpy` within guest memory: device memory *is* host memory. Use
  `cudaMallocManaged`/`cudaHostAlloc`, or `mpt.empty`, to skip even that.
- **Hugepages** are reserved at VM boot. The plugin and `--gpu` default to half of the VM memory
  (`--label dev.metalpassthrough.gpu-memory-mb=N` or `METALPASSTHROUGH_GPU_MEMORY_MB` override).
  `mpt-run --hugepages N` sets it explicitly.
- **Unified addressing is negotiated.** UVA (`ptr == device address`) depends on predicting Metal's
  GPU VA placement. The host verifies it and advertises it as a capability; when it cannot be
  arranged (or `METALPASSTHROUGH_NO_UVA=1` on the host) the runtime falls back to translated
  addresses, and the CUDA test suite passes in both modes.
- **Isolation.** All sessions of one VM share a host process and Metal device, and a malicious
  shader can read other arenas *of the same VM*. Different containers run in different VM processes,
  and a guest can never name memory outside its own VM: the host validates every address against
  that VM's arenas.
- **GPU time is not partitioned.** A guest can submit a kernel that never terminates and monopolize
  the Apple GPU (the host caps a dispatch at 2^40 threads, but cannot bound loop trip counts). macOS
  has no GPU preemption API for that; the session can be killed, the kernel cannot.
- **Crash safety needs the keeper.** The keeper process (on by default; `METALPASSTHROUGH_NO_KEEPER=1`
  disables it) is what keeps a killed process's hugepages allocated until the host drained. If the
  keeper itself is SIGKILLed together with the client (e.g. the whole cgroup is killed while GPU work
  is in flight), the old window reopens. The apple/container guest kernel has `CONFIG_MODULES` off,
  so a pinning guest driver is not an option.
- **Hugepage stability (assumption).** The design assumes an arena's guest-physical pages don't
  move while mapped. hugetlb pages are never swapped, but the kernel *can* migrate them (memory
  offlining, compaction for contiguous allocations). Nothing in the default apple/container VM does
  that, but it is not enforced, and a pinning guest driver would make it a guarantee.
- **macOS 27+ only.** On older systems the plugin and `--gpu` return an unsupported error, and
  nothing changes for other containers.

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
Sources/mpt-run/                  standalone runner (Containerization, cached rootfs clones)
Sources/mpt-bench/                native-host baseline of the benchmark
guest/                            Linux side: libmetalpassthrough.so, libcuda.so.1, libcudart.so.12, libcublas.so.12, apple-smi (Rust), Python
Tests/MetalPassthroughHostTests/  translator tests (nvcc and Triton PTX executed on the GPU vs CPU references), PTX fuzzer
e2e/                              Tests 1-4 (CUDA C, CUDA C++, native Metal C, Python), CUDA semantics test
e2e/apps/                         cuBLAS GEMM, conjugate-gradient Poisson solver, FlashAttention-style attention
e2e/triton/                       Triton 3.5 kernels compiled to PTX without an NVIDIA GPU
e2e/robustness/                   crash-safety test, protocol fuzzer
bench/                            same kernels native vs. guest; run.sh -> results/RESULTS.md
patches/                          0001: `container run --gpu`; 0002: runtime-plugin support
scripts/                          build-host.sh (sign), install-plugin.sh, test-all.sh
```

## License

Apache License 2.0. See [LICENSE](LICENSE). NVIDIA CUDA headers are used only at build time inside
the builder image, and no NVIDIA software is redistributed or required at runtime. "CUDA" is a
trademark of NVIDIA Corporation, and this project is not affiliated with NVIDIA or Apple.
