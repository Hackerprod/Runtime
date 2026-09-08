# CPU-E1 results — four-input FP16 GEMV

Evidence status: **PASS**.
Performance decision: **ACCEPTED FOR NEXT INTEGRATION**.

## Scope
- The candidate kernel processes one FP16 weight row at a time, converts each eight-weight fragment once and reuses that FP32 fragment across four independent FP32 inputs.
- The reference performs four independent `gemv_f16` calls. Each candidate output preserves CPU-R6's FMA order, horizontal reduction and scalar tail; the kernel is not connected to `step()`, chat or scheduling.
- Preparation, allocation, parity checks and warmups are outside the timed intervals. The benchmark is pinned to one participant on CPU 0 and reports four-output wall time, not model tokens/s.

## Identity
- Source commit: `ad6e5ae94b65522b7779ef8a0b1b72dfd109c71b`; benchmark tree: `bbc010d73c3bda7151a71d29e1da051ffde7538a`.
- Executable SHA-256: `cf1b0373aa3e077b874bb06a3a380258cfabf99e0fa66ac5537f6efb8b79bca2`.
- Repetitions per timed sample: `64`; warmups: `2`; CPU: `0`.

## Command
```powershell
python native_cpu/benchmarks/cpu-e1/benchmark.py --out native_cpu/benchmarks/cpu-e1/results --repetitions 64
python native_cpu/benchmarks/cpu-e1/summarize.py
```

## Summary — time for four outputs
| Shape | Reference median | Shared-weight median | Median shared-reference | Median benefit | Favorable pairs |
|---|---:|---:|---:|---:|---:|
| 2432x768 | 0.031660600 s | 0.012955650 s | -0.018697000 s | 59.054% | 6/6 |
| 768x2432 | 0.044451400 s | 0.013257900 s | -0.031184150 s | 70.226% | 6/6 |

## Raw pair details
| Shape | Pair | Order | Reference | Shared | Shared-reference | Benefit | Exact parity |
|---|---:|---|---:|---:|---:|---:|---|
| 2432x768 | 1 | reference-shared | 0.031615500 s | 0.012971200 s | -0.018644300 s | 58.972% | True |
| 2432x768 | 2 | shared-reference | 0.033523000 s | 0.013032300 s | -0.020490700 s | 61.124% | True |
| 2432x768 | 3 | reference-shared | 0.031322600 s | 0.012886500 s | -0.018436100 s | 58.859% | True |
| 2432x768 | 4 | shared-reference | 0.031534600 s | 0.012955300 s | -0.018579300 s | 58.917% | True |
| 2432x768 | 5 | reference-shared | 0.031950900 s | 0.012878700 s | -0.019072200 s | 59.692% | True |
| 2432x768 | 6 | shared-reference | 0.031705700 s | 0.012956000 s | -0.018749700 s | 59.137% | True |
| 768x2432 | 1 | reference-shared | 0.043559200 s | 0.013277800 s | -0.030281400 s | 69.518% | True |
| 768x2432 | 2 | shared-reference | 0.044338000 s | 0.013238000 s | -0.031100000 s | 70.143% | True |
| 768x2432 | 3 | reference-shared | 0.045179100 s | 0.013186600 s | -0.031992500 s | 70.813% | True |
| 768x2432 | 4 | shared-reference | 0.044163000 s | 0.013121800 s | -0.031041200 s | 70.288% | True |
| 768x2432 | 5 | reference-shared | 0.044564800 s | 0.013296500 s | -0.031268300 s | 70.164% | True |
| 768x2432 | 6 | shared-reference | 0.045681900 s | 0.013391500 s | -0.032290400 s | 70.685% | True |

## Verification
- The C++ harness compares every output buffer with `memcmp` before timing and checks the output checksums after timing.
- Shapes tested: 2432×768 and 768×2432, plus small incomplete-column fixtures (3×9, 5×17, 7×31 and 4×8) in the native kernel tests.
- F16C availability is recorded in every receipt; no assumption based only on AVX2 is made.

## Verdict
Result: **PASS**; performance: **ACCEPTED FOR NEXT INTEGRATION**.

A performance decision is reproducible only when all six pairs for both large shapes favor the shared-weight kernel. Mixed results remain documented and are not promoted.
