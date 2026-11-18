# MetalPassthrough test results

2026-11-18 22:21 UTC — Apple M3 Max, macOS 27.2.1 (26B131)

| Step | Result | Log |
|---|---|---|
| host: swift build -c release + codesign | PASS | `build-host.log` |
| guest: libmetalpassthrough / libcuda / libcudart / apple-smi | PASS | `build-guest.log` |
| e2e programs (gcc + nvcc, Linux arm64) | PASS | `build-e2e.log` |
| PTX->MSL translator + Metal execution (swift test) | PASS | `unit.log` |
| selftest: ring, allocator, zero-copy SAXPY | PASS | `selftest.log` |
| Test 1: apple-smi hardware detection | PASS | `test1-apple-smi.log` |
| Test 2: zero-copy native Metal (256 MiB, bit-exact) | PASS | `test2-native.log` |
| Test 3a: CUDA Driver API + nvcc PTX | PASS | `test3-driver.log` |
| Test 3b: nvcc-built CUDA Runtime executable | PASS | `test3-runtime.log` |
| Test 4: Python NumPy/PyTorch zero-copy | PASS | `test4-python.log` |

## test1-apple-smi

```
Nov 18 22:21:08 2026 UTC
+-----------------------------------------------------------------------------------------+
| APPLE-SMI 0.1.0   MetalPassthrough 0.1.0 (ABI 1)   Metal: Apple9   CUDA bridge: 12.9    |
| Host: macOS 27.2.1 (26B131)   Guest: Linux 6.18.56 in apple/container VM                |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Family   Cores| Unified Memory         | Bandwidth  Max TG    |
| Arena (VM shared, zero-copy)            | Hugepage pool          | Containers  Sessions |
|=========================================|========================|======================|
|   0  Apple M3 Max         Apple9      30| 36.0 GiB (ws 28.1 GiB) |   255 GB/s   1024    |
|         4 MiB [............]   0.1%     | 2048x2MiB, 2046 free   |          1         1 |
+-----------------------------------------+------------------------+----------------------+
| Pipelines: 0 built, 0 cached (0% hit)   | Last kernel: -         | UVA: VA == dev addr  |
| Ring RTT min/med/p99: 0.25/0.29/0.38 us | GPU busy: -            | Dispatches: 0        |
+-----------------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|   Session     PID  Process name          Arena MiB    Dispatches        GPU time        |
|=========================================================================================|
|   No running processes found                                                            |
+-----------------------------------------------------------------------------------------+
{'name': 'Apple M3 Max', 'gpu_cores': 30, 'metal_family': 'Apple9', 'host_os': 'macOS 27.2.1 (26B131)', 'unified_memory_bytes': 38654705664, 'recommended_working_set': 30150672384, 'measured_bandwidth_gbs': 255.3, 'uva': True, 'arena_bytes_vm': 4194304, 'hugepages_total': 2048, 'hugepages_free': 2046, 'active_containers': 1, 'sessions_vm': 1, 'pipeline_compiles': 0, 'pipeline_cache_hits': 0, 'last_kernel_gpu_ns': 0, 'ring_latency_ns': {'min': 208, 'median': 291, 'p99': 375}, 'processes': []}
apple-smi: PASS
```

## test2-native

```
Test 2 on Apple M3 Max (30-core GPU) from Linux, UVA=1
SAXPY      2x256 MiB : GPU 3.08 ms (261 GB/s effective), submit->done 7.40 ms, bit-exact mismatches 0
Stencil5   8192^2 x20: GPU 44.51 ms total (241 GB/s), 2.23 ms/sweep, wall 45.4 ms, bit-exact mismatches 0
Matmul     2048^3    : GPU 15.82 ms (1086 GFLOP/s), bit-exact mismatches 0 (34 sampled rows x 2048)
native_test: PASS
```

## test3-driver

```
CUDA driver 12.9 | 1 device(s): "Apple M3 Max (MetalPassthrough)"
  30 SMs (GPU cores), warp 32, 1024 threads/block, sm_80 PTX level, 4.0 GiB allocatable, UVA=1 managed=1
cuModuleLoadData: nvcc PTX (23890 bytes) translated to Metal + compiled in 202.4 ms
vector_add   n=16777216: 0.777 ms, 259.0 GB/s, mismatches 0
tiled_sgemm  1024x1024x1024: 3.743 ms, 573.8 GFLOP/s, bit-exact mismatches 0
reduce_sum   n=16777216: 0.780 ms, 86.0 GB/s, sum 1547.7924 vs 1547.7923 (fp32 vs fp64 ref)
histogram256 n=67108864: 3.280 ms, 20.5 GB/s, exact=yes
daxpy (f64)  n=4194304: worst relative error 7.16e-15 (df64 emulation; Apple GPUs have no FP64 ALU)
managed mem  n=1048576: CPU<->GPU through one pointer, mismatches 0
driver_test: PASS
```

## test3-runtime

```
runtime 12090 / driver 12090 | 1 device: Apple M3 Max (MetalPassthrough), sm_80, 30 SMs, warp 32, 4.0 GiB, managed=1
affine<float><<<256,256>>> (struct by value, managed): 0 mismatches
heat_step<<<(32,32),(16,16)>>> x50: 3.791 ms (75.82 us/step), max |gpu-cpu| = 0.00e+00
dot_kernel n=16777216: 1.006 ms, 133.4 GB/s, result 401.741 vs 401.741
runtime_test: PASS
```

## test4-python

```
device: <Apple M3 Max | 30-core GPU | Apple9 | 36 GiB unified | 259 GB/s>  (unified addressing: True)
python 3.10.12, numpy 2.2.6, 8 vCPUs in this Linux VM
[1] SAXPY 2x256 MiB (zero-copy numpy arrays): GPU 3.05 ms = 264 GB/s | numpy CPU 73.4 ms | speedup 24.1x | bit-exact=True
[2] fused GELU+sin*exp over 64 Mi floats: GPU 2.70 ms | numpy CPU 614 ms | speedup 228x | max abs err vs float64 4.3e-07
[3] SGEMM 4096x4096x4096: GPU 93.3 ms = 1.47 TFLOP/s | numpy/OpenBLAS CPU 230 ms = 0.60 TFLOP/s | speedup 2x | max rel err 2.9e-06
[4] torch 2.14.1+cpu: from_dlpack views share arena pages (data_ptr == device address: True); torch CPU matmul 234 ms vs MetalPassthrough 93.3 ms (3x); max rel err 2.9e-06
[5] nvcc PTX vector_add via mpt.PTXKernel: 0.81 ms, 250 GB/s, exact=True
stats: 10 dispatches, 7 command buffers, compiles=0, cache hits=4, arena=1024 MiB in 1 arena(s); ring ping 18.75 us
test4: PASS
```

## selftest

```
device        : Apple M3 Max (30-core GPU, Apple9), host macOS 27.2.1 (26B131)
session       : 1, unified addressing (guest VA == device VA): yes
ring ping     : min 0.21 us, median 0.29 us, p99 0.38 us
mpt_alloc    : 20 ns per alloc+free (guest-local, no host round trip)
saxpy 16M     : GPU 843.4 us, 238.7 GB/s effective, 0 mismatches (bit-exact vs fmaf)
stats         : 1 dispatches, 1 command buffers, 0 compiles, 1 cache hits, arena 256 MiB
selftest: PASS
```

