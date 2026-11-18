<!-- Comment for https://github.com/apple/container/issues/1511 — paste everything below this line. -->

Following up on @delgadof's question about the macOS 27 `VZCustomVirtioDevice` API: it works.
It makes real, **zero-copy** GPU compute possible for Linux containers today, with no virtio-gpu
driver and no guest kernel module.

I built it as a proof of concept, with a 49-line patch to `container` that adds `--gpu`:
**https://github.com/giuseppegambino92/MetalPassthrough**

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

**Patch** (`patches/0001-feat-gpu-passthrough-metal-zero-copy.patch`, against 1.7.1):
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
| SAXPY, 2 × 256 MiB, native Metal | 248 GB/s, bit-exact |
| 2D stencil, 8192², 20 batched sweeps | 202 GB/s, bit-exact |
| Same SGEMM kernel, guest vs. native host | 1.44 vs 1.47 TFLOP/s (~98%) |
| nvcc PTX `tiled_sgemm` via the CUDA Driver API | bit-exact vs CPU |
| Shared-memory ring round trip (hot) | 0.25 µs |

**Limits.** This is compute only, with no graphics or Vulkan. It needs PTX, not SASS. There are
no tensor-core instructions and no 64-bit atomics, and FP64 is emulated. To @hansmbakker's
question: it does not run MLX inside Linux, but Metal kernels, CUDA PTX and NumPy/PyTorch
interop do work.

If a GPU path is on the roadmap, I'd be glad to adapt the patch to whatever shape you prefer:
flag name, opt-in config, or keeping the device behind an extension point.
