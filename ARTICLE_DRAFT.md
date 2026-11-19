# How I Built Zero-Copy Metal & CUDA GPU Passthrough for Apple's Linux Containers

> **Show HN title:** Show HN: Zero-copy Metal and CUDA PTX passthrough for Apple's Linux containers
> **Link:** https://github.com/giuseppegambino92/MetalPassthrough

*All numbers were measured on an M3 Max (30-core GPU, 36 GB) with macOS 27.2.1 and
apple/container 1.7.1.*

**TL;DR**
- **The project:** MetalPassthrough lets a Linux container on a Mac run Metal compute shaders, nvcc-built
  CUDA programs and NumPy/PyTorch workloads on the Apple GPU.
- **How it's zero-copy:** the GPU reads and writes the container's own physical memory pages.
- **The ingredients:** a macOS 27 API (`VZCustomVirtioDevice.guestMemoryMapping`), 2 MiB hugepages,
  a predictable GPU address allocator, and a PTX-to-Metal compiler.
- **The cost:** identical GPU time to a native macOS process (measured side by side), plus
  ~50–100 µs each time the guest waits for a result.
- **How to use it:** with the stock Homebrew `container`, as a runtime plugin:
  `container run --runtime container-runtime-metalpassthrough …`.

---

Apple's [`container`](https://github.com/apple/container) runs each Linux container in its own
lightweight VM, using `Virtualization.framework` and a tiny Swift init called `vminitd`. It boots
in under a second and feels native. It also has one glaring gap: the Linux guest cannot see the
GPU. There is no paravirtualized GPU device and no Metal inside the VM. Today's workarounds proxy
HTTP requests to a model server on the host, which is fine for chatbots and useless for anything
else: a custom kernel, a CFD solver, a PyTorch tensor op, a pre-compiled CUDA program.

So I set out to give Linux containers real access to the Apple GPU. I wanted three things:

1. **Zero-copy.** On unified memory, a GPU that copies tensors through a socket wastes the
   hardware's best feature.
2. **Native Metal compute** from C, Rust and Python inside the container.
3. **Unmodified CUDA.** Load NVIDIA PTX through the standard CUDA Driver API, or run an executable
   built with stock `nvcc`, and have it execute on the M3 Max.

All three work. Here is how, including the parts that didn't.

## Dead end #1: "just mmap a file over VirtioFS"

The obvious design shares a directory over VirtioFS, has the guest `mmap(MAP_SHARED)` an arena
file, has the host `mmap` the same file, and calls `MTLDevice.makeBuffer(bytesNoCopy:)`. Same file,
same pages, zero copies. Right?

No. In plain terms: VirtioFS is a *file-sharing protocol*, not shared memory. When the guest reads
a file, the host sends the bytes over a virtio queue, and the guest kernel stores them in its *own*
RAM as page cache. When the guest writes, the dirty pages are later sent back. Two copies of the
data exist, one in guest RAM and one in host RAM, and keeping them in sync is the protocol's job.

Zero-copy file sharing in virtio-fs needs a **DAX window**: a shared-memory region through which
the guest maps host page-cache pages directly. Apple's `VZVirtioFileSystemDeviceConfiguration` has no
such region. Without DAX, a guest `mmap` is served from the *guest's own page cache*. The guest and
the host hold different physical pages, and bytes move between them via FUSE reads and writebacks.
It's a copy, and not even a coherent one. That rules out the design everyone reaches for first.

## The key: `VZCustomVirtioDevice` (macOS 27)

While reading the macOS 27 SDK headers I found a new family of APIs: `VZCustomVirtioDevice`,
`VZVirtioSharedMemoryRegion`, and the one that matters here, `VZGuestMemoryMapping`.

```objc
- (nullable VZGuestMemoryMapping *)guestMemoryMappingAtPhysicalAddress:(uint64_t)physicalAddress
                                                                 length:(size_t)length;
```

A custom virtio device gives the host process that owns the VM a pointer to guest RAM, which is
ordinary, cacheable unified memory. Hand that pointer to Metal and the GPU computes on the guest's
pages in place.

The guest never has to drive the device. With no kernel modules in the guest (`CONFIG_MODULES` is
off) I couldn't have written a driver anyway. The device exists only to unlock the mapping. It
attaches through Containerization's `VZInstanceExtension` hook: `configureVZ` appends to
`config.customVirtioDevices`, and `didCreate` opens a vsock listener.

One subtlety: the delegate callback is an `@optional` Objective-C method. Get the Swift spelling
slightly wrong and nothing fails; it just never gets called. My first proof of concept failed with
"device was not created" until I matched the selector exactly.

## Finding guest memory without privileges

The host can now read guest RAM, but where are the guest's buffers? Inside a container you don't
have `CAP_SYS_ADMIN`, so `/proc/self/pagemap` won't give you physical addresses.

The solution is almost silly. The guest allocates **2 MiB hugepages** (`MAP_HUGETLB`, reserved at
boot with `hugepages=N`) and writes a random 128-bit nonce plus a chunk index at the start of each.
It sends the nonce over vsock. The host scans guest RAM at 2 MiB strides, which takes about 24 ms
for 4 GiB, and finds every chunk.

Hugepages matter twice. Each chunk is physically contiguous, and 2 MiB is a multiple of the host's
16 KiB page. The host can therefore `mach_vm_remap` the scattered chunks into one contiguous virtual
range (`copy = FALSE`, so the pages are shared, not copied) and wrap the whole thing in a single
no-copy `MTLBuffer`.

The first proof of concept: a Python script inside an Alpine container fills 512 MiB of hugepages,
and the M3 Max transforms them in place in **4.49 ms**, roughly 239 GB/s of combined reads and
writes. The guest verifies every value. That was the moment the project became real.

## Making guest pointers *be* device pointers

CUDA programs assume unified virtual addressing. A pointer returned by `cudaMallocManaged` must work
on the CPU and on the GPU with the same numeric value. Metal does give buffers a 64-bit
`gpuAddress`, and integer-to-pointer casts are legal in MSL, which I checked with a three-line
kernel. But that address has nothing to do with where the guest mapped the memory.

Two observations made UVA possible:

1. Metal hands out GPU virtual addresses from a **bump allocator**. Successive buffers land at
   predictable addresses, 16 KiB-granular with a 32 KiB guard.
2. Linux can `mremap` hugetlb mappings, *if* the target is 2 MiB-aligned.

The host creates a probe buffer to learn the next GPU address, then prefixes the arena with just
enough zero-filled host pages that guest chunk 0 lands on a 2 MiB-aligned GPU address, and verifies
the result. The guest then moves its hugepage mapping, with `MAP_FIXED_NOREPLACE` first so nothing
is clobbered, to that exact numeric address. From then on `ptr == device address`. `cuMemAllocManaged`
returns a pointer you can write from the CPU, pass to a kernel, and read back, with no translation
anywhere.

## Paying the GPU page-table cost once: `MTLResidencySet`

The first version had a strange latency profile. SAXPY over 512 MiB took 3.45 ms of GPU time but
78 ms from submit to completion. The difference was Metal mapping every page of a freshly wrapped
no-copy buffer into the GPU's page tables on first use. A `MTLResidencySet` holding the arena,
committed with `requestResidency()` right after the arena is mapped and attached to the session's
command queue, moves that work to session setup: 42 ms, once, for a 1.1 GiB arena. First-dispatch
latency dropped to 9.8 ms. Everything after that is steady state.

## The ring

The data plane needs no sockets; the control plane barely does. The first 64 KiB of arena 0 hold a
single-producer/single-consumer ring:
- 256 × 64-byte submission entries and 256 × 32-byte completions.
- Head and tail indices on separate 128-byte cache lines, Apple Silicon's L2 line size, each line
  written by exactly one side.
- C11 acquire/release on the guest; Swift calling the same `__atomic` builtins through a tiny C
  module on the host.

The host worker spins for 50 µs when idle, then parks on the vsock socket after setting
`host_sleeping`. The guest rings that 1-byte doorbell only when the flag is set. Hot round trips
take **0.25 µs**, and a wake-up from park takes about 20 µs. Dispatches are fire-and-forget. They
batch into a single `MTLCommandBuffer` until the guest syncs or the host goes idle, and failures
surface at the next sync the way CUDA reports asynchronous errors. Twenty stencil sweeps take
54.5 ms wall-clock for 53.2 ms of GPU time.

## PTX → Metal

The CUDA side starts with real compiler output. NVIDIA's CUDA toolkit for Linux arm64 installs
fine in a Debian container on the Mac, and `nvcc -ptx` runs without a GPU, so every test kernel is
genuine nvcc 13.5 PTX for `sm_80`. That honesty paid off immediately. The `sinf` in my "simple"
math kernel compiles into a Payne-Hanek argument reduction with a `.local` stack array, 64-bit
`mad.wide`, `bfi.b64`, and a detour through **FP64**. No hand-written test would have contained
those.

The translator works like this:
- Every PTX register becomes a typed MSL local. Operands are reinterpreted to the instruction type,
  because a `.b32` register may hold a float.
- Parameters are packed with CUDA ABI alignment into a `constant` block. All shared memory, static
  and dynamic, lives in one threadgroup allocation.
- Generic pointers carry a state-space tag in their top bits.

Apple GPUs have no FP64 units, so `.f64` defaults to **df64**: pairs of floats with error-free
transformations. That gives about 48 bits of mantissa; `daxpy` reaches a worst-case relative error
of 7.2e-15. Memory keeps exact IEEE-754 doubles, so the host program never notices.

### Bug #1: barriers and divergence

MSL has no `goto`, so I lowered PTX's unstructured branches to a state machine,
`for(;;) switch(_pc)`. Vector add, SAXPY and the stencil were bit-exact at once. Tiled SGEMM was
wrong in exactly one place, the last partial tile column. The histogram lost a handful of counts,
but only with more than one SIMD-group per threadgroup.

The cause was divergence. Lanes that took a branch went back through the dispatch loop, while lanes
that fell through reached `threadgroup_barrier` directly. Each subset of a SIMD-group hit the
barrier separately, so the barrier released while half the group was still storing its tile.

The fix is old SIMT wisdom. Number the basic blocks in reverse post-order, and on every iteration
let each SIMD-group execute only its **minimum pending block**, `simd_min(_pc)`. Lanes now
reconverge at join points before any barrier. It applies only to kernels with barriers or SIMD-group
operations, so the rest keep the fast path. Since then, tiled SGEMM matches a CPU `fmaf` chain
**bit for bit**, at every odd shape I tried.

### Bug #2: a 3 KB lie

Late in the project, the Python test passed and then segfaulted at interpreter exit. gdb pointed at
the cyclic GC walking a dict with a dangling pointer, and once at "unknown opcode": Python's own
bytecode had been overwritten.

The cause was mine. I had grown the C `mpt_stats` struct by a 64-entry process table for
`apple-smi`, but not its ctypes mirror in Python, so every `mpt.stats()` call wrote about 3 KB past
the end of a Python object. It took longer than it should have because a stale copy of the guest
runtime shadowed my fixes. Bindings now verify every mirrored struct against the C library's
`mpt_abi_sizeof()` at import, and refuse to load on a mismatch.

## Two CUDA personalities

`libcuda.so.1` is compiled against NVIDIA's official `cuda.h`. That guarantees the exported symbols
(`cuMemAlloc_v2`, `cuMemcpyHtoD_v2`, …) have exactly the ABI real binaries import, and the compiler
rejects any signature drift. The test program is a plain Driver API application linked against
NVIDIA's own stub `libcuda.so`. Inside the container, the dynamic loader simply finds ours instead.

`libcudart.so.12` goes further. It implements the hidden registration ABI that `nvcc` emits into
host code (`__cudaRegisterFatBinary`, `__cudaRegisterFunction`, `__cudaPushCallConfiguration`). It
parses the fatbinary container, picks the PTX, and JIT-translates it on first launch. An executable
built with stock `nvcc -cudart shared` runs unmodified: templates, struct-by-value parameters,
`__shared__` halos, `atomicAdd`, managed memory. Its 2D heat-diffusion kernel matches the CPU
exactly at 34 µs per step.

## From demo to runtime: what a review found

The first public version passed its four end-to-end tests, and a careful review still found five
real problems. `cuStreamWaitEvent` was a no-op, `cuStreamQuery` blocked, copies did not check that
they stayed inside their allocation, unified addressing was a heuristic the guest simply trusted,
and a process killed with GPU work in flight could have its memory recycled under a running kernel.
None of these show up in a benchmark. All of them show up in production.

**Streams that mean something.** Every CUDA stream is now its own `MTLCommandQueue`. An event
record signals an `MTLSharedEvent`; `cuStreamWaitEvent` encodes a GPU-side wait on it, so the
dependency costs no CPU round trip. The legacy default stream waits on, and is waited on by, every
blocking stream, exactly as CUDA specifies. Queries read the event's counter and never block.
Then the surprise: on an M3 Max, two independent 15.5 ms kernels on two queues take 31.2 ms
*natively*. Metal runs compute command buffers from different queues back to back. So streams are
now correct, not faster, and the test reports overlap as information, not as a pass condition.

**Teardown that survives `kill -9`.** The guest kernel has `CONFIG_MODULES` off, so there is no
driver to pin pages. The fix lives in user space: arenas are hugetlb `memfd`s, and on first use the
library double-forks a tiny keeper that holds those file descriptors and a vsock connection
authenticated with the session id and a nonce. If the client dies, its pages stay allocated until
the host has drained the GPU and closes the keeper's connection. The crash test kills a process in
the middle of 300 MiB of in-flight writes and has a second process grab the freed hugepages:
68 874 corrupted words without the keeper, zero with it.

**Unified addressing as a capability.** The host now verifies the GPU address it predicted and
advertises UVA in the handshake only when it holds; otherwise the guest library uses translated
addresses. The whole CUDA suite runs in both modes, with the fallback forced by an environment
variable on the host.

**A host that assumes a hostile guest.** A fuzzer speaks the wire protocol directly: 400 malformed
handshakes, 20 000 random submission entries per seed, garbage MSL and PTX, kernels that write
through wild pointers, and a ring whose tail index is corrupted. It found four ways to crash or hang
the host process (a misaligned atomic on a guest-chosen event address, two arithmetic overflows in
dispatch-size checks, and a half-sent handshake that pinned a thread forever) and one guest-side
hang (a Linux vsock reports a closed peer as readable, not as hang-up). All five are fixed, and the
fuzzer runs in the test suite.

**Real workloads.** `libcublas.so.12`, compiled against NVIDIA's `cublas_v2.h`, sends SGEMM to
Apple's `MPSMatrixMultiplication`, running in place on the container's pages. A conjugate-gradient
Poisson solver with a million unknowns, a FlashAttention-style attention kernel, and kernels
compiled by **Triton** on a Mac with no NVIDIA GPU all run, each checked against an independent
reference. Triton's PTX found three parser bugs that nvcc's never triggered, the worst being
that an instruction following a label and a `.loc` debug directive was silently dropped.

## Results

`make bench` runs the same Metal kernels at the same sizes in a native macOS process and from inside
the Linux container, alternating rounds, and reports medians:

| | Native | Linux guest |
|---|---:|---:|
| SAXPY, 256 MiB | 270.4 GB/s | 270.7 GB/s |
| 5-point stencil, 8192² | 251.5 GB/s | 251.5 GB/s |
| SGEMM (MPS), 4096³ | 8.88 TFLOP/s | 8.92 TFLOP/s |
| SGEMM (my simdgroup kernel), 4096³ | 1.46 TFLOP/s | 1.47 TFLOP/s |
| 10 000 tiny dispatches | 1.12 M/s | 1.04 M/s |
| one tiny dispatch, submit → result visible | 160 µs | 212 µs |

GPU time is identical, because once work reaches the GPU nothing distinguishes guest memory from
any other Metal buffer. What the VM costs is a fixed ~50–100 µs each time the guest *waits* for a
result, mostly the host waking from its parked state. Getting the dispatch rate to 94% of native
took two fixes the benchmark exposed: the host validated each dispatch with an allocation in the
hot path, and it cut command buffers every 64 dispatches, which throttled on Metal's in-flight
command-buffer limit. A third finding: passing MPS a multi-matrix descriptor for a single matrix
quietly selected a GEMM kernel at half speed.

Applications, all inside the VM and each checked against an independent reference: `cublasSgemm`
from an unmodified nvcc binary at 8.8 TFLOP/s; a CG Poisson solve with a million unknowns at
0.95 ms per iteration; FlashAttention-style attention at 723 GFLOP/s; NumPy SAXPY at 23× NumPy.

## Shipping it

- A **runtime plugin** works with the stock, Homebrew-installed `container`: apple/container already
  loads runtime plugins and has `--runtime`, so `make install-plugin` builds its own Linux runtime
  helper from the matching release tag with MetalPassthrough compiled in, and
  `container run --runtime container-runtime-metalpassthrough` does the rest. No CLI change.
- A **49-line patch** to apple/container adds `container run --gpu`. Internally it reserves
  hugepages on the kernel command line, wraps the VM manager with the MetalPassthrough extension, and
  bind-mounts the guest runtime. It builds under apple/container's `-warnings-as-errors` and works
  with stock `ubuntu:22.04`.
- **`mpt-run`** does the same with stock Containerization today.
- **`apple-smi`** shows what's running:

```
|   Session     PID  Process name          Arena MiB    Dispatches        GPU time        |
|=========================================================================================|
|         1       3  driver_test                 256             4        50.88 ms        |
|         2       2  python3                     256           507      2273.64 ms        |
```

## What's missing

There are real limits:
- **No SASS.** Binaries must carry PTX.
- **Not translated:** textures, tensor-core instructions, `cp.async`, 64-bit atomics (Metal has
  none), and non-inlined device-function calls.
- **No stream overlap.** Streams are correct, but Metal on this hardware runs them one after another.
- **Shared GPU address space.** All processes in one VM share the host process's GPU address space.
- **GPU time is not partitioned.** A kernel that never terminates monopolizes the GPU; the session
  can be killed, the kernel cannot be preempted.

Each of these is engineering rather than research. The research part, zero-copy unified memory
between a Linux VM and the Apple GPU, turned out to need one new Apple API, a 128-bit nonce, and a
bump allocator I could predict.

*Code, the apple/container patch, and the reproducible test suite (`make test-all`):
https://github.com/giuseppegambino92/MetalPassthrough*
