# MetalPassthrough test results

2026-11-19 09:17 UTC — Apple M3 Max, macOS 27.2.1 (26B131)

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
| cuBLAS mixed precision (GemmEx fp16/bf16, Hgemm, batched, gemv) | PASS | `apps-cublas-mixed.log` |
| CG Poisson solver, 1M unknowns (cuBLAS + SpMV) | PASS | `apps-cg.log` |
| FlashAttention-style transformer attention | PASS | `apps-attention.log` |
| crash safety: SIGKILL with GPU work in flight | PASS | `robust-crash.log` |
| crash stress: 15 rounds of random SIGKILLs, every word checked | PASS | `robust-crash-stress.log` |
| protocol fuzzing (malicious guest), 5000 SQEs | PASS | `robust-fuzz.log` |
| isolation: a wild-pointer shader cannot reach another VM | PASS | `robust-isolation.log` |
| isolation: another process in the same VM is unreachable | PASS | `robust-isolation-process.log` |
| isolation control: shared mode does leak (probe works) | PASS | `robust-isolation-shared.log` |
| shared mode (METALPASSTHROUGH_ISOLATION=shared): CUDA semantics | PASS | `shared-semantics.log` |
| multi-process: 6 concurrent GPU processes | PASS | `robust-multi.log` |
| container run --runtime container-runtime-metalpassthrough | PASS | `plugin.log` |
| no-UVA: selftest | PASS | `nouva-selftest.log` |
| no-UVA: CUDA Driver API | PASS | `nouva-driver.log` |
| no-UVA: CUDA semantics | PASS | `nouva-semantics.log` |

## test1-apple-smi

```
Nov 19 09:13:58 2026 UTC
+-----------------------------------------------------------------------------------------+
| APPLE-SMI 0.1.0   MetalPassthrough 0.1.0 (ABI 2)   Metal: Apple9   CUDA bridge: 12.9    |
| Host: macOS 27.2.1 (26B131)   Guest: Linux 6.18.56 in apple/container VM                |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Family   Cores| Unified Memory         | Bandwidth  Max TG    |
| Arena (VM shared, zero-copy)            | Hugepage pool          | Containers  Sessions |
|=========================================|========================|======================|
|   0  Apple M3 Max         Apple9      30| 36.0 GiB (ws 28.1 GiB) |   258 GB/s   1024    |
|         4 MiB [............]   0.1%     | 2048x2MiB, 2046 free   |          1         1 |
+-----------------------------------------+------------------------+----------------------+
| Pipelines: 0 built, 0 cached (0% hit)   | Last kernel: -         | UVA: VA == dev addr  |
| Ring RTT min/med/p99: 0.17/0.21/0.79 us | GPU busy: -            | Dispatches: 0        |
+-----------------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                           GPU isolation: per process (own GPU address space) |
|   Session     PID  Process name          Arena MiB    Dispatches        GPU time        |
|=========================================================================================|
|   No running processes found                                                            |
+-----------------------------------------------------------------------------------------+
{'name': 'Apple M3 Max', 'gpu_cores': 30, 'metal_family': 'Apple9', 'host_os': 'macOS 27.2.1 (26B131)', 'unified_memory_bytes': 38654705664, 'recommended_working_set': 30150672384, 'measured_bandwidth_gbs': 258.3, 'uva': True, 'process_isolation': True, 'arena_bytes_vm': 4194304, 'hugepages_total': 2048, 'hugepages_free': 2046, 'active_containers': 1, 'sessions_vm': 1, 'pipeline_compiles': 0, 'pipeline_cache_hits': 0, 'last_kernel_gpu_ns': 0, 'ring_latency_ns': {'min': 166, 'median': 250, 'p99': 375}, 'processes': []}
apple-smi: PASS
```

## test2-native

```
Test 2 on Apple M3 Max (30-core GPU) from Linux, UVA=1
SAXPY      2x256 MiB : GPU 3.48 ms (231 GB/s effective), submit->done 8.02 ms, bit-exact mismatches 0
Stencil5   8192^2 x20: GPU 45.43 ms total (236 GB/s), 2.27 ms/sweep, wall 46.4 ms, bit-exact mismatches 0
Matmul     2048^3    : GPU 20.17 ms (852 GFLOP/s), bit-exact mismatches 0 (34 sampled rows x 2048)
native_test: PASS
```

## test3-driver

```
CUDA driver 12.9 | 1 device(s): "Apple M3 Max (MetalPassthrough)"
  30 SMs (GPU cores), warp 32, 1024 threads/block, sm_80 PTX level, 4.0 GiB allocatable, UVA=1 managed=1
cuModuleLoadData: nvcc PTX (23890 bytes) translated to Metal + compiled in 193.2 ms
vector_add   n=16777216: 0.984 ms, 204.5 GB/s, mismatches 0
tiled_sgemm  1024x1024x1024: 6.095 ms, 352.3 GFLOP/s, bit-exact mismatches 0
reduce_sum   n=16777216: 0.934 ms, 71.9 GB/s, sum 1547.7922 vs 1547.7923 (fp32 vs fp64 ref)
histogram256 n=67108864: 3.683 ms, 18.2 GB/s, exact=yes
daxpy (f64)  n=4194304: worst relative error 7.16e-15 (df64 emulation; Apple GPUs have no FP64 ALU)
managed mem  n=1048576: CPU<->GPU through one pointer, mismatches 0
driver_test: PASS
```

## test3-runtime

```
runtime 12090 / driver 12090 | 1 device: Apple M3 Max (MetalPassthrough), sm_80, 30 SMs, warp 32, 4.0 GiB, managed=1
affine<float><<<256,256>>> (struct by value, managed): 0 mismatches
heat_step<<<(32,32),(16,16)>>> x50: 1.357 ms (27.14 us/step), max |gpu-cpu| = 0.00e+00
dot_kernel n=16777216: 0.666 ms, 201.5 GB/s, result 401.742 vs 401.741
runtime_test: PASS
```

## test3-semantics

```
device: Apple M3 Max (MetalPassthrough)
calibrated: 1048576 iterations = 33.6 ms
[1] independent streams run concurrently (low-occupancy kernels: 1 threadgroup each)
  one kernel alone: 22.7 ms
  2 non-blocking streams: wall 23.3 ms (serial ~45.4 ms); stream completion timeline: s1@22.2 s2@23.0 ms
  ok: 2 non-blocking streams overlap: 1.03x the time of one kernel (1.0 = fully concurrent, 2.0 = serialized)
  4 non-blocking streams: wall 23.2 ms (serial ~90.7 ms); stream completion timeline: s1@22.9 s2@22.5 s3@22.2 s4@23.0 ms
  ok: 4 non-blocking streams overlap: 1.02x the time of one kernel (1.0 = fully concurrent, 4.0 = serialized)
  2 blocking streams (legacy default-stream rules, no stream-0 work): wall 22.7 ms (serial ~45.4 ms); stream completion timeline: s1@22.3 s2@22.4 ms
  ok: 2 blocking streams (legacy default-stream rules, no stream-0 work) overlap: 1.00x the time of one kernel (1.0 = fully concurrent, 2.0 = serialized)
  ok: 2 streams, launch-launch-sync without events: 1.02x the time of one kernel
  info: bandwidth-bound kernels (whole GPU): 1 stream 10.7 ms, 2 streams x same work 22.2 ms (2.07x; ~2.0 expected when the GPU is saturated)
[2] cudaStreamQuery never blocks
  ok: query on a busy stream returns cudaErrorNotReady (cudaErrorNotReady)
  ok: query returned in 0.055 ms
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
device: <Apple M3 Max | 30-core GPU | Apple9 | 36 GiB unified | 251 GB/s>  (unified addressing: True)
python 3.10.12, numpy 2.2.6, 8 vCPUs in this Linux VM
[1] SAXPY 2x256 MiB (zero-copy numpy arrays): GPU 3.10 ms = 260 GB/s | numpy CPU 174.1 ms | speedup 56.2x | bit-exact=True
[2] fused GELU+sin*exp over 64 Mi floats: GPU 2.08 ms | numpy CPU 1429 ms | speedup 688x | max abs err vs float64 4.3e-07
[3] SGEMM 4096x4096x4096: GPU 209.1 ms = 0.66 TFLOP/s | numpy/OpenBLAS CPU 1830 ms = 0.08 TFLOP/s | speedup 9x | max rel err 2.9e-06
[3b] mpt.matmul 4096x4096x4096 (Metal Performance Shaders, zero-copy): 44.4 ms = 3.10 TFLOP/s | 41x numpy | max rel err 2.9e-06
[4] torch 2.14.1+cpu: from_dlpack views share arena pages (data_ptr == device address: True); torch CPU matmul 4260 ms vs MetalPassthrough 209.1 ms (20x); max rel err 2.9e-06
[5] nvcc PTX vector_add via mpt.PTXKernel: 0.80 ms, 251 GB/s, exact=True
[6] Triton 3.5 PTX (compiled on a machine without an NVIDIA GPU): add 0.79 ms exact=True, softmax 4096x1000 0.16 ms max err 1.5e-08
stats: 18 dispatches, 21 command buffers, compiles=0, cache hits=6, arena=1400 MiB in 1 arena(s); ring ping 1037.54 us
test4: PASS
```

## apps-cublas

```
sgemm NN 67x45x93 (lda 70 ldb 98 ldc 69): max |err| 1.31e-06 ok
sgemm NT 67x45x93 (lda 70 ldb 50 ldc 69): max |err| 9.59e-07 ok
sgemm TN 67x45x93 (lda 96 ldb 98 ldc 69): max |err| 1.42e-06 ok
sgemm TT 67x45x93 (lda 96 ldb 50 ldc 69): max |err| 1.16e-06 ok
sgemmStridedBatched 9 x (32x48x40): max |err| 4.75e-07 ok
cublasSgemm   512^3:    0.385 ms    0.70 TFLOP/s
cublasSgemm  1024^3:    0.792 ms    2.71 TFLOP/s
cublasSgemm  2048^3:    3.822 ms    4.49 TFLOP/s
cublasSgemm  4096^3:   32.271 ms    4.26 TFLOP/s
cublasSgemm  8192^3:  501.730 ms    2.19 TFLOP/s
cublas_gemm: PASS
```

## apps-cg

```
CG Poisson 1024x1024 (1048576 unknowns): 923 iterations
  GPU: 1.221 s (1.32 ms/iteration, 6 kernels + 2 host-synchronized reductions each)
  CPU (same algorithm, 1 thread): 7.082 s  -> speedup 5.8x
  max error vs analytic solution sin(pi x) sin(pi y): 1.61e-05 (relative)
  max |u_gpu - u_cpu| / max|u| = 3.58e-07
cg_solver: PASS
```

## apps-attention

```
attention B=2 H=8 S=1024 D=64: GPU 6.002 ms (715.6 GFLOP/s)
  float64 CPU reference on 6/16 heads: 0.65 s (288.7x slower per head than the GPU on all heads)
  max |gpu - reference| = 6.99e-08
attention: PASS
```

## apps-cublas-mixed

```
GemmEx bf16 x bf16 -> bf16, compute 32F, NN                max err 7.67e-03 (tol 1.0e-02 x 3.82), padding changed: 0  ok
GemmEx bf16 x bf16 -> bf16, compute 32F, TN                max err 7.81e-03 (tol 1.0e-02 x 3.80), padding changed: 0  ok
GemmEx bf16 x bf16 -> bf16, compute 32F, NT                max err 1.19e-02 (tol 1.0e-02 x 4.05), padding changed: 0  ok
GemmEx bf16 x bf16 -> bf16, compute 32F, TT                max err 7.80e-03 (tol 1.0e-02 x 3.65), padding changed: 0  ok
GemmEx bf16 x bf16 -> f32, compute 32F, NN                 max err 5.30e-07 (tol 1.0e-05 x 3.77), padding changed: 0  ok
GemmEx bf16 x bf16 -> f32, compute 32F, TN                 max err 5.65e-07 (tol 1.0e-05 x 4.49), padding changed: 0  ok
GemmEx bf16 x bf16 -> f32, compute 32F, NT                 max err 4.95e-07 (tol 1.0e-05 x 3.89), padding changed: 0  ok
GemmEx bf16 x bf16 -> f32, compute 32F, TT                 max err 5.62e-07 (tol 1.0e-05 x 4.19), padding changed: 0  ok
GemmEx f32 x f32 -> f32, compute 32F, NN                   max err 9.58e-07 (tol 1.0e-05 x 3.61), padding changed: 0  ok
GemmEx f32 x f32 -> f32, compute 32F, TN                   max err 1.39e-06 (tol 1.0e-05 x 3.89), padding changed: 0  ok
GemmEx f32 x f32 -> f32, compute 32F, NT                   max err 1.14e-06 (tol 1.0e-05 x 3.42), padding changed: 0  ok
GemmEx f32 x f32 -> f32, compute 32F, TT                   max err 1.37e-06 (tol 1.0e-05 x 3.75), padding changed: 0  ok
GemmEx f32 x f32 -> bf16, compute 32F, NN                  max err 7.65e-03 (tol 1.0e-02 x 3.75), padding changed: 0  ok
GemmEx f32 x f32 -> bf16, compute 32F, TN                  max err 7.78e-03 (tol 1.0e-02 x 3.65), padding changed: 0  ok
GemmEx f32 x f32 -> bf16, compute 32F, NT                  max err 7.78e-03 (tol 1.0e-02 x 3.82), padding changed: 0  ok
GemmEx f32 x f32 -> bf16, compute 32F, TT                  max err 7.77e-03 (tol 1.0e-02 x 3.70), padding changed: 0  ok
GemmEx f16 x f16 -> f16, compute 16F, NN                   max err 9.77e-04 (tol 2.0e-02 x 3.74), padding changed: 0  ok
GemmEx f16 x f16 -> f16, compute 16F, TN                   max err 9.76e-04 (tol 2.0e-02 x 3.52), padding changed: 0  ok
GemmEx f16 x f16 -> f16, compute 16F, NT                   max err 9.73e-04 (tol 2.0e-02 x 4.30), padding changed: 0  ok
GemmEx f16 x f16 -> f16, compute 16F, TT                   max err 1.58e-03 (tol 2.0e-02 x 4.20), padding changed: 0  ok
Hgemm NN                                                   max err 9.52e-04 (tol 2.0e-02 x 2.92), padding changed: 0  ok
GemmStridedBatchedEx bf16, 5 entries, gapped strides       max err 3.90e-03 (tol 1.0e-02 x 1.81), padding changed: 0  ok
SgemmStridedBatched fp32, 5 entries, gapped strides        max err 3.84e-07 (tol 1.0e-05 x 2.26), padding changed: 0  ok
SgemmStridedBatched fp32, stride 0 (broadcast A)           max err 3.36e-07 (tol 1.0e-05 x 2.08), padding changed: 0  ok
SgemmBatched (pointer arrays) entry 0                      max err 9.14e-08 (tol 1.0e-05 x 1.36), padding changed: 0  ok
SgemmBatched (pointer arrays) entry 1                      max err 8.31e-08 (tol 1.0e-05 x 1.38), padding changed: 0  ok
SgemmBatched (pointer arrays) entry 2                      max err 9.02e-08 (tol 1.0e-05 x 1.42), padding changed: 0  ok
SgemmBatched (pointer arrays) entry 3                      max err 7.73e-08 (tol 1.0e-05 x 1.54), padding changed: 0  ok
Sgemv N                                                    max err 2.81e-07 (tol 1.0e-05 x 2.23), padding changed: 0  ok
Sgemv T                                                    max err 3.89e-07 (tol 1.0e-05 x 2.70), padding changed: 0  ok
refused: f16 x bf16, bf16 with fp16 compute, int8          CUBLAS_STATUS_NOT_SUPPORTED / CUBLAS_STATUS_NOT_SUPPORTED / CUBLAS_STATUS_NOT_SUPPORTED  ok
math modes                                                 set/get ok, bit-identical across 5 modes yes, invalid mode rejected yes, fp32 err 3.2e-06  ok
cublasSetWorkspace                                         NULL/0 SUCCESS, NULL/64 INVALID_VALUE, misaligned INVALID_VALUE, host ptr INVALID_VALUE, past end INVALID_VALUE, valid SUCCESS; bytes written: 0  ok
level 1 with stride 3, overflow and increments             axpy err 3.0e-08, dot err 3.6e-06, gaps untouched yes; huge n*inc rejected yes; negative/zero inc -> NOT_SUPPORTED  ok
GemmEx 4096^3 fp32 (Sgemm path)                              29.91 ms   4.60 TFLOP/s
GemmEx 4096^3 fp16 x fp16 -> fp16, fp16 compute (Hgemm)      85.59 ms   1.61 TFLOP/s
GemmEx 4096^3 fp16 x fp16 -> fp32, fp32 compute              32.87 ms   4.18 TFLOP/s
GemmEx 4096^3 fp16 x fp16 -> fp16, fp32 compute              43.12 ms   3.19 TFLOP/s
GemmEx 4096^3 bf16 x bf16 -> bf16, fp32 compute              63.21 ms   2.17 TFLOP/s
cublas_mixed: PASS
```

## robust-crash

```
READY rounds=1557 crash_safe=1
probe: hugepages available after 2485 ms; foreign GPU writes observed in 0 sampled words
```

## robust-crash-stress

```
pool: 512 MiB of hugepages, 15 rounds
round 1: killed after 6 ms (0/3 kernels in flight) | probe: 504 MiB available after 32 ms; 11 full scans of every word; non-zero (foreign) words: 0
round 2: killed after 1587 ms (3/3 kernels in flight) | probe: 504 MiB available after 1715 ms; 9 full scans of every word; non-zero (foreign) words: 0
round 3: killed after 2308 ms (3/3 kernels in flight) | probe: 504 MiB available after 664 ms; 8 full scans of every word; non-zero (foreign) words: 0
[mpt-gpu-worker] session 8: errno(32)
[mpt-gpu-worker] session 9: errno(32)
round 4: killed after 53 ms (0/3 kernels in flight) | probe: 504 MiB available after 2380 ms; 6 full scans of every word; non-zero (foreign) words: 0
round 5: killed after 536 ms (3/3 kernels in flight) | probe: 504 MiB available after 3139 ms; 5 full scans of every word; non-zero (foreign) words: 0
round 6: killed after 1508 ms (3/3 kernels in flight) | probe: 504 MiB available after 138 ms; 4 full scans of every word; non-zero (foreign) words: 0
[mpt-gpu-worker] session 17: errno(32)
[mpt-gpu-worker] session 18: errno(32)
round 7: killed after 62 ms (0/3 kernels in flight) | probe: 504 MiB available after 2326 ms; 4 full scans of every word; non-zero (foreign) words: 0
round 8: killed after 1473 ms (3/3 kernels in flight) | probe: 504 MiB available after 261 ms; 5 full scans of every word; non-zero (foreign) words: 0
round 9: killed after 1523 ms (3/3 kernels in flight) | probe: 504 MiB available after 1500 ms; 2 full scans of every word; non-zero (foreign) words: 0
[mpt-gpu-worker] session 26: errno(32)
[mpt-gpu-worker] session 27: errno(32)
round 10: killed after 78 ms (1/3 kernels in flight) | probe: 504 MiB available after 230 ms; 2 full scans of every word; non-zero (foreign) words: 0
round 11: killed after 778 ms (3/3 kernels in flight) | probe: 504 MiB available after 631 ms; 3 full scans of every word; non-zero (foreign) words: 0
round 12: killed after 2391 ms (3/3 kernels in flight) | probe: 504 MiB available after 86 ms; 3 full scans of every word; non-zero (foreign) words: 0
[mpt-gpu-worker] session 35: errno(32)
[mpt-gpu-worker] session 36: errno(32)
round 13: killed after 15 ms (0/3 kernels in flight) | probe: 504 MiB available after 43 ms; 3 full scans of every word; non-zero (foreign) words: 0
round 14: killed after 1406 ms (3/3 kernels in flight) | probe: 504 MiB available after 2473 ms; 4 full scans of every word; non-zero (foreign) words: 0
round 15: killed after 750 ms (1/3 kernels in flight) | probe: 504 MiB available after 5182 ms; 3 full scans of every word; non-zero (foreign) words: 0
-- memory exhaustion
open 128 MiB: refused in 0 ms: mmap(MAP_HUGETLB, 64 x 2 MiB) failed (Cannot allocate memory); 32 hugepages free. MetalPassthrough needs 2 MiB hugepages reserved at VM boot (mpt-run --hugepages N / container run --gpu)
open 48 MiB: ok in 85 ms, dispatch correct
after exhaustion + kill: probe: 504 MiB available after 18 ms; 3 full scans of every word; non-zero (foreign) words: 0
-- leaks
hugepages free after all rounds: 256 / 256
open 256 MiB: ok in 29 ms, dispatch correct
crash_stress: PASS
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
[G] start
[G] 1250 GEMM descriptors (MPS path, arena-edge operands): 416 executed, 834 rejected; session alive: yes
[D] corrupted sq_tail: host terminated the session
[E] fresh session after fuzzing: sync=0, SAXPY mismatches=0
fuzz: PASS (host survived and still computes correctly)
```

## robust-isolation

```
own arena: 0x10000200000 + 64 MiB
  own allocation                                   0x010000311000 load+store -> no fault reported, loaded 0x12345678
  (own allocation now holds 0x0badf00d)
  unmapped address                                 0x7f0000000000 load       -> no fault reported, loaded 0x00000000
  unmapped address                                 0x7f0000000000 load+store -> no fault reported, loaded 0x00000000
  1 MiB below own arena (host padding / host buffers) 0x010000100000 load       -> no fault reported, loaded 0x00000000
  64 MiB below own arena                           0x00fffc200000 load       -> no fault reported, loaded 0x00000000
  just past own arena                              0x010004200000 load       -> no fault reported, loaded 0x00000000
  victim address (other process or VM)             0x01000a310000 load       -> no fault reported, loaded 0x00000000
  victim address (other process or VM)             0x01000a310000 load+store -> no fault reported, loaded 0x00000000
session after probes: ok (own allocation reads 0x0badf00d)
VICTIM 0x1000a310000
VICTIM words modified by other processes/VMs: 0
```

## robust-isolation-process

```
own arena: 0x10000200000 + 64 MiB
  own allocation                                   0x010000311000 load+store -> no fault reported, loaded 0x12345678
  (own allocation now holds 0x0badf00d)
  unmapped address                                 0x7f0000000000 load       -> no fault reported, loaded 0x00000000
  unmapped address                                 0x7f0000000000 load+store -> no fault reported, loaded 0x00000000
  1 MiB below own arena (host padding / host buffers) 0x010000100000 load       -> no fault reported, loaded 0x00000000
  64 MiB below own arena                           0x00fffc200000 load       -> no fault reported, loaded 0x00000000
  just past own arena                              0x010004200000 load       -> no fault reported, loaded 0x00000000
  victim address (other process or VM)             0x01000a310000 load       -> no fault reported, loaded 0x00000000
  victim address (other process or VM)             0x01000a310000 load+store -> no fault reported, loaded 0x00000000
session after probes: ok (own allocation reads 0x0badf00d)
VICTIM 0x1000a310000
VICTIM words modified by other processes/VMs: 0
ISOLATED
```

## robust-isolation-shared

```
own arena: 0x10010400000 + 64 MiB
  own allocation                                   0x010010511000 load+store -> no fault reported, loaded 0x12345678
  (own allocation now holds 0x0badf00d)
  unmapped address                                 0x7f0000000000 load       -> no fault reported, loaded 0x00000000
  unmapped address                                 0x7f0000000000 load+store -> no fault reported, loaded 0x00000000
  1 MiB below own arena (host padding / host buffers) 0x010010300000 load       -> no fault reported, loaded 0x00000000
  64 MiB below own arena                           0x01000c400000 load       -> no fault reported, loaded 0x294fff5d
  just past own arena                              0x010014400000 load       -> no fault reported, loaded 0x00000000
  victim address (other process or VM)             0x01000a310000 load       -> no fault reported, loaded 0xa5a5a5a5
  victim address (other process or VM)             0x01000a310000 load+store -> no fault reported, loaded 0xa5a5a5a5
session after probes: ok (own allocation reads 0x0badf00d)
VICTIM 0x1000a310000
VICTIM words modified by other processes/VMs: 1
LEAKED
```

## shared-semantics

```
device: Apple M3 Max (MetalPassthrough)
calibrated: 1048576 iterations = 111.1 ms
[1] independent streams run concurrently (low-occupancy kernels: 1 threadgroup each)
  one kernel alone: 22.6 ms
  2 non-blocking streams: wall 25.1 ms (serial ~45.1 ms); stream completion timeline: s1@22.8 s2@23.2 ms
  ok: 2 non-blocking streams overlap: 1.11x the time of one kernel (1.0 = fully concurrent, 2.0 = serialized)
  4 non-blocking streams: wall 25.0 ms (serial ~90.2 ms); stream completion timeline: s1@22.3 s2@22.3 s3@23.2 s4@24.4 ms
  ok: 4 non-blocking streams overlap: 1.11x the time of one kernel (1.0 = fully concurrent, 4.0 = serialized)
  2 blocking streams (legacy default-stream rules, no stream-0 work): wall 23.0 ms (serial ~45.1 ms); stream completion timeline: s1@22.3 s2@22.3 ms
  ok: 2 blocking streams (legacy default-stream rules, no stream-0 work) overlap: 1.02x the time of one kernel (1.0 = fully concurrent, 2.0 = serialized)
  ok: 2 streams, launch-launch-sync without events: 1.01x the time of one kernel
  info: bandwidth-bound kernels (whole GPU): 1 stream 11.4 ms, 2 streams x same work 22.3 ms (1.96x; ~2.0 expected when the GPU is saturated)
[2] cudaStreamQuery never blocks
  ok: query on a busy stream returns cudaErrorNotReady (cudaErrorNotReady)
  ok: query returned in 0.419 ms
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
  one kernel alone: 23.3 ms
  2 non-blocking streams: wall 23.3 ms (serial ~46.5 ms); stream completion timeline: s1@22.8 s2@22.6 ms
  ok: 2 non-blocking streams overlap: 1.00x the time of one kernel (1.0 = fully concurrent, 2.0 = serialized)
  4 non-blocking streams: wall 23.2 ms (serial ~93.1 ms); stream completion timeline: s1@22.4 s2@22.5 s3@22.4 s4@22.9 ms
  ok: 4 non-blocking streams overlap: 1.00x the time of one kernel (1.0 = fully concurrent, 4.0 = serialized)
  2 blocking streams (legacy default-stream rules, no stream-0 work): wall 23.2 ms (serial ~46.5 ms); stream completion timeline: s1@22.3 s2@22.3 ms
  ok: 2 blocking streams (legacy default-stream rules, no stream-0 work) overlap: 1.00x the time of one kernel (1.0 = fully concurrent, 2.0 = serialized)
  ok: 2 streams, launch-launch-sync without events: 0.98x the time of one kernel
  info: bandwidth-bound kernels (whole GPU): 1 stream 11.6 ms, 2 streams x same work 22.8 ms (1.97x; ~2.0 expected when the GPU is saturated)
[2] cudaStreamQuery never blocks
  ok: query on a busy stream returns cudaErrorNotReady (cudaErrorNotReady)
  ok: query returned in 0.409 ms
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
  GPU: 0.372 s (1.04 ms/iteration, 6 kernels + 2 host-synchronized reductions each)
  CPU (same algorithm, 1 thread): 0.540 s  -> speedup 1.5x
  max error vs analytic solution sin(pi x) sin(pi y): 7.83e-06 (relative)
  max |u_gpu - u_cpu| / max|u| = 2.38e-07
cg_solver: PASS
| Processes:                           GPU isolation: per process (own GPU address space) |
```

## selftest

```
device        : Apple M3 Max (30-core GPU, Apple9), host macOS 27.2.1 (26B131)
session       : 1, unified addressing (guest VA == device VA): yes
ring ping     : min 0.17 us, median 0.21 us, p99 0.29 us
mpt_alloc    : 40 ns per alloc+free (guest-local, no host round trip)
saxpy 16M     : GPU 846.0 us, 238.0 GB/s effective, 0 mismatches (bit-exact vs fmaf)
stats         : 1 dispatches, 1 command buffers, 0 compiles, 1 cache hits, arena 256 MiB
selftest: PASS
```

## nouva-driver

```
CUDA driver 12.9 | 1 device(s): "Apple M3 Max (MetalPassthrough)"
  30 SMs (GPU cores), warp 32, 1024 threads/block, sm_80 PTX level, 4.0 GiB allocatable, UVA=0 managed=0
cuModuleLoadData: nvcc PTX (23890 bytes) translated to Metal + compiled in 53.4 ms
vector_add   n=16777216: 2.588 ms, 77.8 GB/s, mismatches 0
tiled_sgemm  512x512x512: 0.868 ms, 309.2 GFLOP/s, bit-exact mismatches 0
reduce_sum   n=16777216: 0.976 ms, 68.7 GB/s, sum 1617.2886 vs 1617.2898 (fp32 vs fp64 ref)
histogram256 n=67108864: 3.709 ms, 18.1 GB/s, exact=yes
daxpy (f64)  n=4194304: worst relative error 7.16e-15 (df64 emulation; Apple GPUs have no FP64 ALU)
driver_test: PASS
```

