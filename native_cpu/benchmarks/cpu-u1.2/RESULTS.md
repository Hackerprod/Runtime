# CPU-U1.2 reproducible baseline

Status: **PASS**.

## Scope
- This freezes CPU-U1 as a single current baseline; it does not compare against CPU-R6 and makes no historical performance claim.
- The DLL was rebuilt from source commit `06694632cfe4ee5b97013d6a5b2e6fec9fe98f50` with one participant pinned to CPU 0 and profiling disabled.
- Two forced-token workloads (256 and 1792 prefix tokens) use three repetitions each. CPU-E1 is not integrated.

## Immutable identity
- Source commit: `06694632cfe4ee5b97013d6a5b2e6fec9fe98f50`.
- Measurement tree commit: `d06f89f10c3c3e57e90fe00579dcdc2f56dc1326`.
- Baseline DLL SHA-256: `5956ff064ad8e8fd1ff502d18626ce19240682faa55c584cb3d6eaad1775bb3a`.
- Build-info SHA-256: `8dd097707ddb6bd4922bace1572a3dfb8c2f92141634e00e55746703bf927d69`.
- Model SHA-256: `bcab52e67825061ffc3bd4480dbda1d5067bc3a0e51aba386a236c7131766f33`.

## Commands
```powershell
python native_cpu/tools/build.py --cmake "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build-dir native_cpu/build-u1.2
python native_cpu/benchmarks/cpu-u1.2/benchmark.py --out native_cpu/benchmarks/cpu-u1.2/results --repeats 3
python native_cpu/benchmarks/cpu-u1.2/summarize.py
```

`loading_seconds`, FP16 preparation and compact-storage bytes are reported separately; sustained inference timings exclude preparation.

## Sustained baseline timings
| Case | Metric | Median | Minimum | Maximum | Stdev |
|---|---|---:|---:|---:|---:|
| cpu-u1.2-forced-256 | prefill | 2.296372 s | 2.131440 s | 2.492683 s | 0.180849 s |
| cpu-u1.2-forced-256 | decode | 0.343006 s | 0.330290 s | 0.349880 s | 0.009939 s |
| cpu-u1.2-forced-256 | total | 2.626662 s | 2.474447 s | 2.842564 s | 0.184974 s |
| cpu-u1.2-forced-256 | prefill throughput | 111.480 tok/s | 102.701 tok/s | 120.107 tok/s | 8.703 tok/s |
| cpu-u1.2-forced-256 | decode throughput | 93.293 tok/s | 91.460 tok/s | 96.885 tok/s | 2.760 tok/s |
| cpu-u1.2-forced-1792 | prefill | 26.759412 s | 26.742768 s | 27.330824 s | 0.334813 s |
| cpu-u1.2-forced-1792 | decode | 0.773606 s | 0.742619 s | 0.797560 s | 0.027546 s |
| cpu-u1.2-forced-1792 | total | 27.516373 s | 27.502031 s | 28.128384 s | 0.357557 s |
| cpu-u1.2-forced-1792 | prefill throughput | 66.967 tok/s | 65.567 tok/s | 67.009 tok/s | 0.821 tok/s |
| cpu-u1.2-forced-1792 | decode throughput | 41.365 tok/s | 40.122 tok/s | 43.091 tok/s | 1.491 tok/s |

## Preparation and memory
| Case | Load median | Configuration median | FP16 preparation median | Compact bytes |
|---|---:|---:|---:|---:|
| cpu-u1.2-forced-256 | 1.817770 s | 0.000000 s | 0.259275 s | 89,653,248 bytes |
| cpu-u1.2-forced-1792 | 1.423796 s | 0.000000 s | 0.266801 s | 89,653,248 bytes |

## Exact output consistency
- Each repetition stores FP32 prefill logits and every decode-logit SHA-256, plus the fixed token IDs and final position.
- The evaluator compares those output fields across all three repetitions; any divergence rejects the report.
- The effective route is automatic CPU-U1: selective logits, blocked V, shared K/V GQA and exact FP16 FFN storage; FFN row4 and diagnostics remain disabled.

## Evidence verdict
Result: **PASS**.

A PASS establishes this rebuilt CPU-U1 artifact and these measurements as the reproducible baseline for CPU-E1 integration. It does not claim a gain relative to CPU-R6.
