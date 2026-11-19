<!-- Comment for https://github.com/apple/container/issues/1511 — paste everything below this line. -->

Following up on @delgadof's question about the macOS 27 `VZCustomVirtioDevice` API: it works.
It makes real, **zero-copy** GPU compute possible for Linux containers today, with no virtio-gpu
driver and no guest kernel module.

I built it, and it runs with the **unmodified** `container` 1.7.1 from Homebrew as a runtime
plugin, selected with the existing `--runtime` flag:
**https://github.com/giuseppegambino92/MetalPassthrough**

```sh
make install-plugin   # builds container-runtime-linux 1.7.1 + MetalPassthrough, installs it as a plugin
container run --rm --runtime container-runtime-metalpassthrough ubuntu:22.04 apple-smi
```

**How it works**
- **Shared memory.** A `VZCustomVirtioDevice` gives the runtime helper
  `guestMemoryMapping(atPhysicalAddress:length:)` access to guest RAM. The guest allocates 2 MiB
  hugepages (`hugepages=N` on the kernel command line). The host locates them, remaps them
  contiguously, and wraps them with `MTLDevice.makeBuffer(bytesNoCopy:)`, so the GPU reads and
  writes the container's own physical pages. VirtioFS can't do this, because without a DAX window
  a guest `mmap` lives in the guest page cache.
- **Pointers.** Guest pointers equal GPU device addresses (UVA).
- **Control.** A lock-free ring in that memory carries commands, with a vsock port for doorbells
  only.
- **Guest stack:**
  - `libcuda.so.1` and `libcudart.so.12` translate nvcc PTX to Metal, so unmodified CUDA
    Driver/Runtime programs run on the Apple GPU.
  - A C API and a Python package (NumPy/DLPack/PyTorch zero-copy) give direct Metal access.
  - `apple-smi` reports GPU and process status.

**What the plugin needed from apple/container** (`patches/0002-runtime-plugin.patch`, 2 small
changes to the runtime helper): its XPC label is derived from its executable name instead of being
hard-coded to `container-runtime-linux`, so the same helper can be installed under a second plugin
name; and the GPU attachment is enabled when `runtimeHandler` is that plugin. Everything else is
the stock plugin mechanism. A stable extension point for VM devices in the runtime helper (or
accepting the label change upstream) would let GPU support ship as a plugin with no rebuild per
release.

**Alternatively, a first-class flag** (`patches/0001-feat-gpu-passthrough-metal-zero-copy.patch`,
49 lines against 1.7.1):
- adds `--gpu` on `run`/`create`, plus a `gpu` field in `ContainerConfiguration` decoded with
  `decodeIfPresent`, so existing bundles are unaffected;
- in `RuntimeService`, reserves hugepages, wraps the `VZVirtualMachineManager` with a
  `VZInstanceExtension`, and bind-mounts the guest runtime;
- builds under the repo's `-warnings-as-errors`;
- returns an unsupported error before macOS 27, and changes nothing without the flag.

The CLI could equally be spelled `--gpus all` to match the Compose semantics requested here.

**Measured inside the Linux VM** (M3 Max 30-core, macOS 27.2.1, all results verified against CPU
references):

| | |
|---|---|
| Same kernels, Linux guest vs. native host process ([reproducible](https://github.com/giuseppegambino92/MetalPassthrough/blob/main/bench/results/RESULTS.md)) | 99–100% of native GPU time (SAXPY, stencil, SGEMM, MPS GEMM) |
| `cublasSgemm` 4096³ from an unmodified nvcc binary (→ MPS) | ~8.8 TFLOP/s |
| 10 000 small dispatches | ~93% of native throughput |
| nvcc PTX `tiled_sgemm` via the CUDA Driver API, Triton 3.5 PTX | bit-exact / verified |
| Shared-memory ring round trip (hot) | 0.3 µs |

**Safety.** Every guest-supplied descriptor is copied before validation and bounds-checked
against that VM's own memory; a protocol fuzzer (random commands, malformed handshakes, corrupted
ring indices, GEMM descriptors aimed at MPS's process-aborting assertions) runs against the real
host. A process killed mid-kernel cannot have its pages recycled before the GPU drains (stress-tested
with every word checked). Each VM's GPU work runs in that VM's own host process, so a shader with a
forged pointer cannot reach another container (tested); processes *within* one container share a
GPU address space. A guest *can* keep the GPU busy with a non-terminating kernel, since Metal
offers no preemption for that.

**Limits.** This is compute only, with no graphics or Vulkan. It needs PTX, not SASS. There are
no tensor-core instructions and no 64-bit atomics, and FP64 is emulated. To @hansmbakker's
question: it does not run MLX inside Linux, but Metal kernels, CUDA PTX and NumPy/PyTorch
interop do work.

If a GPU path is on the roadmap, I'd be glad to adapt the patch to whatever shape you prefer:
flag name, opt-in config, or keeping the device behind an extension point.
