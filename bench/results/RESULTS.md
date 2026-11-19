# MetalPassthrough benchmark results

2026-11-19 07:58 UTC · Apple M3 Max · macOS 27.2.1 · median of 5 alternating rounds (native, guest, native, …) × the per-run repetitions

GPU = command-buffer GPU timestamps. Wall = from the API call until the result is visible to the caller (native: CPU in the same process; guest: CPU inside the Linux VM).

| Benchmark | Size | Native GPU ms | Guest GPU ms | Native wall ms | Guest wall ms | Native | Guest | Guest / native |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| saxpy | 1MiB | 0.018 | 0.018 | 0.198 | 0.274 | 177.64 GB/s | 176.00 GB/s | 99% |
| saxpy | 4MiB | 0.048 | 0.048 | 0.231 | 0.294 | 261.92 GB/s | 263.06 GB/s | 100% |
| saxpy | 16MiB | 0.173 | 0.172 | 0.373 | 0.446 | 291.21 GB/s | 291.85 GB/s | 100% |
| saxpy | 64MiB | 0.755 | 0.754 | 0.960 | 1.029 | 266.57 GB/s | 266.97 GB/s | 100% |
| saxpy | 256MiB | 2.978 | 2.975 | 3.200 | 3.301 | 270.40 GB/s | 270.69 GB/s | 100% |
| stencil5 | 1024^2 | 0.014 | 0.013 | 0.181 | 0.235 | 593.88 GB/s | 625.27 GB/s | 105% |
| stencil5 | 2048^2 | 0.057 | 0.057 | 0.232 | 0.295 | 590.40 GB/s | 587.39 GB/s | 99% |
| stencil5 | 4096^2 | 0.493 | 0.494 | 0.671 | 0.738 | 272.38 GB/s | 271.76 GB/s | 100% |
| stencil5 | 8192^2 | 2.135 | 2.135 | 2.348 | 2.425 | 251.46 GB/s | 251.45 GB/s | 100% |
| sgemm_simd | 512^3 | 0.146 | 0.145 | 0.328 | 0.390 | 1.84 TFLOP/s | 1.85 TFLOP/s | 101% |
| sgemm_simd | 1024^3 | 1.540 | 1.524 | 1.745 | 1.827 | 1.40 TFLOP/s | 1.41 TFLOP/s | 101% |
| sgemm_simd | 2048^3 | 10.447 | 10.902 | 10.702 | 11.293 | 1.65 TFLOP/s | 1.58 TFLOP/s | 96% |
| sgemm_simd | 4096^3 | 94.067 | 93.836 | 94.326 | 94.312 | 1.46 TFLOP/s | 1.47 TFLOP/s | 100% |
| sgemm_mps | 512^3 | 0.044 | 0.044 | 0.252 | 0.358 | 6.15 TFLOP/s | 6.13 TFLOP/s | 100% |
| sgemm_mps | 1024^3 | 0.279 | 0.279 | 0.497 | 0.601 | 7.70 TFLOP/s | 7.71 TFLOP/s | 100% |
| sgemm_mps | 2048^3 | 1.953 | 1.949 | 2.239 | 2.386 | 8.80 TFLOP/s | 8.81 TFLOP/s | 100% |
| sgemm_mps | 4096^3 | 15.484 | 15.415 | 15.745 | 15.905 | 8.88 TFLOP/s | 8.92 TFLOP/s | 100% |
| dispatch_roundtrip | 1x32thr | — | — | 0.160 | 0.212 | 159.92 us | 211.83 us | 75% |
| dispatch_throughput | 10000x32thr | — | — | 8.963 | 9.583 | 1115728.99 dispatch/s | 1043478.30 dispatch/s | 94% |

Guest-only measurements:

| Measurement | Size | Value |
|---|---|---:|
| alloc_free | 4KiB | 15.84 ns |
| host_rss_peak | mpt-run | 76.00 MiB |
| ring_rtt_cold | ping | 52.04 us |
| ring_rtt_hot | ping | 0.29 us |
| session_open | 1024MiB | 32.09 ms |
| session_open | 256MiB | 7.60 ms |
| session_open | 64MiB | 2.82 ms |

*Guest / native* is relative performance (100% = native speed; for latencies it is native time ÷ guest time).
