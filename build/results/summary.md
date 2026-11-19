# MetalPassthrough test results

2026-11-19 07:56 UTC — Apple M3 Max, macOS 27.2.1 (26B131)

| Step | Result | Log |
|---|---|---|
| host: swift build -c release + codesign | PASS | `build-host.log` |
| guest: libmetalpassthrough, libcuda, libcudart, apple-smi | PASS | `build-guest.log` |
| e2e programs (gcc + nvcc, Linux arm64) | PASS | `build-e2e.log` |
| PTX->MSL translator + Metal execution (swift test) | PASS | `unit.log` |
| selftest: ring, allocator, zero-copy SAXPY | PASS | `selftest.log` |
| Test 1: apple-smi hardware detection | PASS | `test1-apple-smi.log` |
| Test 2: zero-copy native Metal (256 MiB, bit-exact) | PASS | `test2-native.log` |
| Test 3a: CUDA Driver API + nvcc PTX | PASS | `test3-driver.log` |
| Test 3b: nvcc-built CUDA Runtime executable | PASS | `test3-runtime.log` |
| Test 3c: CUDA streams, events, async copies, bounds | PASS | `test3-semantics.log` |
| Test 4: Python NumPy/PyTorch/Triton zero-copy | PASS | `test4-python.log` |
| cuBLAS SGEMM (MPS) correctness + throughput | PASS | `apps-cublas.log` |
| CG Poisson solver, 1M unknowns (cuBLAS + SpMV) | PASS | `apps-cg.log` |
| FlashAttention-style transformer attention | PASS | `apps-attention.log` |
| crash safety: SIGKILL with GPU work in flight | PASS | `robust-crash.log` |
| protocol fuzzing (malicious guest), 5000 SQEs | PASS | `robust-fuzz.log` |
| multi-process: 6 concurrent GPU processes | PASS | `robust-multi.log` |
| container run --runtime container-runtime-metalpassthrough | PASS | `plugin.log` |
| no-UVA: selftest | PASS | `nouva-selftest.log` |
| no-UVA: CUDA Driver API | PASS | `nouva-driver.log` |
| no-UVA: CUDA semantics | PASS | `nouva-semantics.log` |

## test1-apple-smi

```
Nov 19 07:56:06 2026 UTC
+-----------------------------------------------------------------------------------------+
| APPLE-SMI 0.1.0   MetalPassthrough 0.1.0 (ABI 2)   Metal: Apple9   CUDA bridge: 12.9    |
| Host: macOS 27.2.1 (26B131)   Guest: Linux 6.18.56 in apple/container VM                |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Family   Cores| Unified Memory         | Bandwidth  Max TG    |
| Arena (VM shared, zero-copy)            | Hugepage pool          | Containers  Sessions |
|=========================================|========================|======================|
|   0  Apple M3 Max         Apple9      30| 36.0 GiB (ws 28.1 GiB) |   260 GB/s   1024    |
|         4 MiB [............]   0.1%     | 2048x2MiB, 2046 free   |          1         1 |
+-----------------------------------------+------------------------+----------------------+
| Pipelines: 0 built, 0 cached (0% hit)   | Last kernel: -         | UVA: VA == dev addr  |
| Ring RTT min/med/p99: 0.21/0.29/0.38 us | GPU busy: -            | Dispatches: 0        |
+-----------------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|   Session     PID  Process name          Arena MiB    Dispatches        GPU time        |
|=========================================================================================|
|   No running processes found                                                            |
+-----------------------------------------------------------------------------------------+
{'name': 'Apple M3 Max', 'gpu_cores': 30, 'metal_family': 'Apple9', 'host_os': 'macOS 27.2.1 (26B131)', 'unified_memory_bytes': 38654705664, 'recommended_working_set': 30150672384, 'measured_bandwidth_gbs': 260.3, 'uva': True, 'arena_bytes_vm': 4194304, 'hugepages_total': 2048, 'hugepages_free': 2046, 'active_containers': 1, 'sessions_vm': 1, 'pipeline_compiles': 0, 'pipeline_cache_hits': 0, 'last_kernel_gpu_ns': 0, 'ring_latency_ns': {'min': 250, 'median': 292, 'p99': 375}, 'processes': []}
apple-smi: PASS
```

## test2-native

```
Test 2 on Apple M3 Max (30-core GPU) from Linux, UVA=1
SAXPY      2x256 MiB : GPU 2.98 ms (270 GB/s effective), submit->done 7.47 ms, bit-exact mismatches 0
Stencil5   8192^2 x20: GPU 43.41 ms total (247 GB/s), 2.17 ms/sweep, wall 44.4 ms, bit-exact mismatches 0
Matmul     2048^3    : GPU 14.72 ms (1167 GFLOP/s), bit-exact mismatches 0 (34 sampled rows x 2048)
native_test: PASS
```

## test3-driver

```
CUDA driver 12.9 | 1 device(s): "Apple M3 Max (MetalPassthrough)"
  30 SMs (GPU cores), warp 32, 1024 threads/block, sm_80 PTX level, 4.0 GiB allocatable, UVA=1 managed=1
cuModuleLoadData: nvcc PTX (23890 bytes) translated to Metal + compiled in 41.1 ms
vector_add   n=16777216: 0.967 ms, 208.3 GB/s, mismatches 0
tiled_sgemm  1024x1024x1024: 4.261 ms, 504.0 GFLOP/s, bit-exact mismatches 0
reduce_sum   n=16777216: 0.912 ms, 73.6 GB/s, sum 1547.7924 vs 1547.7923 (fp32 vs fp64 ref)
histogram256 n=67108864: 3.419 ms, 19.6 GB/s, exact=yes
daxpy (f64)  n=4194304: worst relative error 7.16e-15 (df64 emulation; Apple GPUs have no FP64 ALU)
managed mem  n=1048576: CPU<->GPU through one pointer, mismatches 0
driver_test: PASS
```

## test3-runtime

```
runtime 12090 / driver 12090 | 1 device: Apple M3 Max (MetalPassthrough), sm_80, 30 SMs, warp 32, 4.0 GiB, managed=1
affine<float><<<256,256>>> (struct by value, managed): 0 mismatches
heat_step<<<(32,32),(16,16)>>> x50: 4.627 ms (92.53 us/step), max |gpu-cpu| = 0.00e+00
dot_kernel n=16777216: 1.427 ms, 94.0 GB/s, result 401.741 vs 401.741
runtime_test: PASS
```

## test3-semantics

```
device: Apple M3 Max (MetalPassthrough)
calibrated: 1048576 iterations = 34.7 ms
[1] two non-blocking streams (independent Metal queues)
  one kernel 22.5 ms, two kernels on two streams 45.0 ms (serial would be ~45.0 ms)
  info: overlap factor 2.00x of a single kernel (1.0 = full overlap, 2.0 = serialized)
[2] cudaStreamQuery never blocks
  ok: query on a busy stream returns cudaErrorNotReady (cudaErrorNotReady)
  ok: query returned in 0.059 ms
  ok: query after sync returns cudaSuccess
  ok: NotReady is not a sticky error
[3] cudaStreamWaitEvent orders work across streams
  ok: consumer saw the producer's result (b[0] = 84, expected 84)
  ok: event completed
[4] legacy default stream synchronizes with blocking streams
  ok: NULL-stream kernel saw the blocking stream's write (b[0] = 21, expected 21)
[5] async copies are stream ordered (pinned and pageable)
  ok: pageable HtoD -> kernel -> DtoH in order (0 mismatches)
  ok: pinned HtoD -> kernel -> DtoH entirely on the GPU (0 mismatches)
[6] memory range validation
  ok: full-size copy accepted
  ok: copy ending exactly at the end accepted
  ok: copy 1 byte past the end rejected
  ok: oversized copy rejected
  ok: wrapping size rejected
  ok: memset past the end rejected
  ok: DtoD past the end rejected
  ok: cuMemGetAddressRange resolves an interior pointer to its allocation
  ok: freeing an interior pointer is rejected
semantics_test: PASS
```

## test4-python

```
device: <Apple M3 Max | 30-core GPU | Apple9 | 36 GiB unified | 261 GB/s>  (unified addressing: True)
python 3.10.12, numpy 2.2.6, 8 vCPUs in this Linux VM
[1] SAXPY 2x256 MiB (zero-copy numpy arrays): GPU 3.00 ms = 269 GB/s | numpy CPU 69.4 ms | speedup 23.2x | bit-exact=True
[2] fused GELU+sin*exp over 64 Mi floats: GPU 3.09 ms | numpy CPU 556 ms | speedup 180x | max abs err vs float64 4.3e-07
[3] SGEMM 4096x4096x4096: GPU 93.3 ms = 1.47 TFLOP/s | numpy/OpenBLAS CPU 187 ms = 0.73 TFLOP/s | speedup 2x | max rel err 2.9e-06
[3b] mpt.matmul 4096x4096x4096 (Metal Performance Shaders, zero-copy): 15.4 ms = 8.91 TFLOP/s | 12x numpy | max rel err 2.9e-06
[4] torch 2.14.1+cpu: from_dlpack views share arena pages (data_ptr == device address: True); torch CPU matmul 183 ms vs MetalPassthrough 93.3 ms (2x); max rel err 2.9e-06
[5] nvcc PTX vector_add via mpt.PTXKernel: 0.79 ms, 255 GB/s, exact=True
[6] Triton 3.5 PTX (compiled on a machine without an NVIDIA GPU): add 0.78 ms exact=True, softmax 4096x1000 0.43 ms max err 1.5e-08
stats: 18 dispatches, 21 command buffers, compiles=0, cache hits=6, arena=1400 MiB in 1 arena(s); ring ping 19.33 us
test4: PASS
```

## apps-cublas

```
sgemm NN 67x45x93 (lda 70 ldb 98 ldc 69): max |err| 1.31e-06 ok
sgemm NT 67x45x93 (lda 70 ldb 50 ldc 69): max |err| 9.59e-07 ok
sgemm TN 67x45x93 (lda 96 ldb 98 ldc 69): max |err| 1.42e-06 ok
sgemm TT 67x45x93 (lda 96 ldb 50 ldc 69): max |err| 1.16e-06 ok
sgemmStridedBatched 9 x (32x48x40): max |err| 4.75e-07 ok
cublasSgemm   512^3:    0.412 ms    0.65 TFLOP/s
cublasSgemm  1024^3:    1.222 ms    1.76 TFLOP/s
cublasSgemm  2048^3:    2.763 ms    6.22 TFLOP/s
cublasSgemm  4096^3:   15.615 ms    8.80 TFLOP/s
cublasSgemm  8192^3:  127.337 ms    8.63 TFLOP/s
cublas_gemm: PASS
```

## apps-cg

```
CG Poisson 1024x1024 (1048576 unknowns): 923 iterations
  GPU: 0.849 s (0.92 ms/iteration, 6 kernels + 2 host-synchronized reductions each)
  CPU (same algorithm, 1 thread): 2.976 s  -> speedup 3.5x
  max error vs analytic solution sin(pi x) sin(pi y): 1.61e-05 (relative)
  max |u_gpu - u_cpu| / max|u| = 3.58e-07
cg_solver: PASS
```

## apps-attention

```
attention B=2 H=8 S=1024 D=64: GPU 5.962 ms (720.4 GFLOP/s)
  float64 CPU reference on 6/16 heads: 0.64 s (287.0x slower per head than the GPU on all heads)
  max |gpu - reference| = 6.99e-08
attention: PASS
```

## robust-crash

```
READY rounds=1374 crash_safe=1
probe: hugepages available after 1503 ms; foreign GPU writes observed in 0 sampled words
```

## robust-fuzz

```
fuzz seed 0x9e3779b97e7f54ea, 5000 iterations
[A] start
[A] 400 malformed handshakes: 400 rejected/closed, 0 accepted (accepted ones were then abandoned)
[B] start
Context leak detected, CoreAnalytics returned false
[B] 5000 random SQEs: 3746 accepted, 1254 rejected with an error, 0 session resets; session alive: yes
[C] wild-pointer kernel: sync returned 0 (no fault reported)
[C] 100 garbage MSL/PTX sources: 100 compile errors reported
[C] session still alive after faults: yes
[D] corrupted sq_tail: host terminated the session
[E] fresh session after fuzzing: sync=0, SAXPY mismatches=0
fuzz: PASS (host survived and still computes correctly)
```

## robust-multi

```
selftest: PASS
selftest: PASS
selftest: PASS
driver_test: PASS
driver_test: PASS
driver_test: PASS
```

## plugin

```
ring ping     : min 0.17 us, median 0.29 us, p99 0.38 us
mpt_alloc    : 42 ns per alloc+free (guest-local, no host round trip)
saxpy 16M     : GPU 762.5 us, 264.0 GB/s effective, 0 mismatches (bit-exact vs fmaf)
stats         : 1 dispatches, 1 command buffers, 0 compiles, 1 cache hits, arena 256 MiB
selftest: PASS
device: Apple M3 Max (MetalPassthrough)
calibrated: 1048576 iterations = 34.1 ms
[1] two non-blocking streams (independent Metal queues)
  one kernel 22.4 ms, two kernels on two streams 45.1 ms (serial would be ~44.7 ms)
  info: overlap factor 2.02x of a single kernel (1.0 = full overlap, 2.0 = serialized)
[2] cudaStreamQuery never blocks
  ok: query on a busy stream returns cudaErrorNotReady (cudaErrorNotReady)
  ok: query returned in 0.082 ms
  ok: query after sync returns cudaSuccess
  ok: NotReady is not a sticky error
[3] cudaStreamWaitEvent orders work across streams
  ok: consumer saw the producer's result (b[0] = 84, expected 84)
  ok: event completed
[4] legacy default stream synchronizes with blocking streams
  ok: NULL-stream kernel saw the blocking stream's write (b[0] = 21, expected 21)
[5] async copies are stream ordered (pinned and pageable)
  ok: pageable HtoD -> kernel -> DtoH in order (0 mismatches)
  ok: pinned HtoD -> kernel -> DtoH entirely on the GPU (0 mismatches)
[6] memory range validation
  ok: full-size copy accepted
  ok: copy ending exactly at the end accepted
  ok: copy 1 byte past the end rejected
  ok: oversized copy rejected
  ok: wrapping size rejected
  ok: memset past the end rejected
  ok: DtoD past the end rejected
  ok: cuMemGetAddressRange resolves an interior pointer to its allocation
  ok: freeing an interior pointer is rejected
semantics_test: PASS
CG Poisson 512x512 (262144 unknowns): 356 iterations
  GPU: 0.180 s (0.51 ms/iteration, 6 kernels + 2 host-synchronized reductions each)
  CPU (same algorithm, 1 thread): 0.288 s  -> speedup 1.6x
  max error vs analytic solution sin(pi x) sin(pi y): 7.83e-06 (relative)
  max |u_gpu - u_cpu| / max|u| = 2.38e-07
cg_solver: PASS
```

## selftest

```
device        : Apple M3 Max (30-core GPU, Apple9), host macOS 27.2.1 (26B131)
session       : 1, unified addressing (guest VA == device VA): yes
ring ping     : min 0.21 us, median 0.29 us, p99 0.38 us
mpt_alloc    : 42 ns per alloc+free (guest-local, no host round trip)
saxpy 16M     : GPU 760.8 us, 264.6 GB/s effective, 0 mismatches (bit-exact vs fmaf)
stats         : 1 dispatches, 1 command buffers, 0 compiles, 1 cache hits, arena 256 MiB
selftest: PASS
```

## nouva-driver

```
CUDA driver 12.9 | 1 device(s): "Apple M3 Max (MetalPassthrough)"
  30 SMs (GPU cores), warp 32, 1024 threads/block, sm_80 PTX level, 4.0 GiB allocatable, UVA=0 managed=0
cuModuleLoadData: nvcc PTX (23890 bytes) translated to Metal + compiled in 40.8 ms
vector_add   n=16777216: 0.972 ms, 207.2 GB/s, mismatches 0
tiled_sgemm  512x512x512: 2.056 ms, 130.6 GFLOP/s, bit-exact mismatches 0
reduce_sum   n=16777216: 0.972 ms, 69.1 GB/s, sum 1617.2891 vs 1617.2898 (fp32 vs fp64 ref)
histogram256 n=67108864: 3.459 ms, 19.4 GB/s, exact=yes
daxpy (f64)  n=4194304: worst relative error 7.16e-15 (df64 emulation; Apple GPUs have no FP64 ALU)
driver_test: PASS
```

