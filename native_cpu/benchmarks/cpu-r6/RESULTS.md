# CPU-R6 results — exact FP16-storage FFN

Status: **accepted**.

## Scope
- X0 is CPU-R5 W1: CPU-R1/R2/R4/R5 enabled, one participant pinned to CPU 0, CPU-R3 row4 disabled.
- X1 adds only exact FP16 storage for all dense FFN gate/up/down matrices. Every weight is validated by FP32→FP16→FP32 bit equality before activation; the kernel converts eight halves to FP32 registers and keeps the existing FMA/reduction order.
- Attention, KV cache, weights on disk, quantization, workers, affinities, statistics ABI and defaults are unchanged. FP32 vectors remain allocated, so the compact payload is additional runtime memory, not a total-RAM reduction.

## Benchmark command
```powershell
python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r6 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r6/results
```

Speed receipts have profiling disabled; diagnostic receipts are separate. Raw receipts: `native_cpu/benchmarks/cpu-r6/results/*.json`; build identity: `native_cpu/benchmarks/cpu-r6/build-info.json`.

## Summary
| Case | Metric | X0 median | X1 median | Median X1-X0 | Δ stdev | Median benefit | Favorable |
|---|---|---:|---:|---:|---:|---:|---:|
| speed 256 | native_generation_seconds | 0.256759 s | 0.223574 s | -0.033377 | 0.047136 | 13.034% | 6/6 |
| speed 256 | decode_tokens_per_second | 124.632339 tok/s | 143.129815 tok/s | 18.728358 | 18.944341 | 14.987% | 6/6 |
| diagnostics 256 | FFN prefill | 1143.892 ms | 877.761 ms | -256.386 ms | 24.788 | 22.540% | 6/6 |
| diagnostics 256 | FFN decode | 142.880 ms | 110.361 ms | -31.177 ms | 13.669 | 21.944% | 6/6 |
| speed 1792 | native_generation_seconds | 0.552185 s | 0.477134 s | -0.074702 | 0.065621 | 13.558% | 6/6 |
| speed 1792 | decode_tokens_per_second | 57.971434 tok/s | 67.074726 tok/s | 9.138192 | 6.618919 | 15.724% | 6/6 |
| diagnostics 1792 | FFN prefill | 8140.596 ms | 6441.883 ms | -1702.627 ms | 176.142 | 20.860% | 6/6 |
| diagnostics 1792 | FFN decode | 146.292 ms | 118.218 ms | -27.353 ms | 13.654 | 18.777% | 6/6 |

## Native contract tests
- `mm_test_kernels` checks signed zeros, FP16-small values, exact round trips, a non-representable value and a final incomplete column block.
- `mm_test_runtime` checks exact logits with odd FFN tails, activation and disable/rejection paths, quantized fallback and row4 mutual exclusion.

## Weight reconstruction
- Artifact: `reconstruction.json`; 24 FFN matrices and 44,826,624 elements: **PASS** (zero bit mismatches required).

## Historical conversation
- Artifact: `chat-parity.json`; six-turn deterministic conversation parity: **PASS**.
- The check compares prompt IDs, sampled IDs, text and finish reasons for X0 and X1.

## Paired details
### Prefix 256: decode generation seconds
| Pair | Order | X0 FP32 storage | X1 FP16 storage | X1-X0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | X0→X1 | 0.265231 s | 0.223212 s | -0.042019 s | 15.842% | True |
| 2 | X1→X0 | 0.253307 s | 0.244279 s | -0.009029 s | 3.564% | True |
| 3 | X0→X1 | 0.255720 s | 0.241876 s | -0.013844 s | 5.414% | True |
| 4 | X1→X0 | 0.257799 s | 0.223935 s | -0.033864 s | 13.136% | True |
| 5 | X0→X1 | 0.354101 s | 0.216553 s | -0.137547 s | 38.844% | True |
| 6 | X1→X0 | 0.254342 s | 0.221452 s | -0.032890 s | 12.931% | True |

### Prefix 256: FFN prefill
| Pair | Order | X0 FP32 storage | X1 FP16 storage | X1-X0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | X0→X1 | 1176871800.000000 ns | 868457600.000000 ns | -308414200.000000 ns | 26.206% | True |
| 2 | X1→X0 | 1142829400.000000 ns | 895805900.000000 ns | -247023500.000000 ns | 21.615% | True |
| 3 | X0→X1 | 1129263100.000000 ns | 865567800.000000 ns | -263695300.000000 ns | 23.351% | True |
| 4 | X1→X0 | 1144953800.000000 ns | 863712200.000000 ns | -281241600.000000 ns | 24.564% | True |
| 5 | X0→X1 | 1133335100.000000 ns | 887063500.000000 ns | -246271600.000000 ns | 21.730% | True |
| 6 | X1→X0 | 1171361100.000000 ns | 922284200.000000 ns | -249076900.000000 ns | 21.264% | True |

### Prefix 256: FFN decode
| Pair | Order | X0 FP32 storage | X1 FP16 storage | X1-X0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | X0→X1 | 142809700.000000 ns | 126788500.000000 ns | -16021200.000000 ns | 11.219% | True |
| 2 | X1→X0 | 141103500.000000 ns | 110555700.000000 ns | -30547800.000000 ns | 21.649% | True |
| 3 | X0→X1 | 146164200.000000 ns | 107733500.000000 ns | -38430700.000000 ns | 26.293% | True |
| 4 | X1→X0 | 166561200.000000 ns | 108824000.000000 ns | -57737200.000000 ns | 34.664% | True |
| 5 | X0→X1 | 141670600.000000 ns | 110165700.000000 ns | -31504900.000000 ns | 22.238% | True |
| 6 | X1→X0 | 142949500.000000 ns | 112101100.000000 ns | -30848400.000000 ns | 21.580% | True |
### Prefix 1792: decode generation seconds
| Pair | Order | X0 FP32 storage | X1 FP16 storage | X1-X0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | X0→X1 | 0.562392 s | 0.495119 s | -0.067273 s | 11.962% | True |
| 2 | X1→X0 | 0.541977 s | 0.459847 s | -0.082130 s | 15.154% | True |
| 3 | X0→X1 | 0.527399 s | 0.482219 s | -0.045179 s | 8.566% | True |
| 4 | X1→X0 | 0.571008 s | 0.472049 s | -0.098959 s | 17.331% | True |
| 5 | X0→X1 | 0.667942 s | 0.459942 s | -0.208001 s | 31.141% | True |
| 6 | X1→X0 | 0.503924 s | 0.484635 s | -0.019289 s | 3.828% | True |

### Prefix 1792: FFN prefill
| Pair | Order | X0 FP32 storage | X1 FP16 storage | X1-X0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | X0→X1 | 8083282700.000000 ns | 6459072700.000000 ns | -1624210000.000000 ns | 20.093% | True |
| 2 | X1→X0 | 8066576900.000000 ns | 6483452000.000000 ns | -1583124900.000000 ns | 19.626% | True |
| 3 | X0→X1 | 8190942800.000000 ns | 6494806300.000000 ns | -1696136500.000000 ns | 20.707% | True |
| 4 | X1→X0 | 8147382000.000000 ns | 6178296100.000000 ns | -1969085900.000000 ns | 24.168% | True |
| 5 | X0→X1 | 8171463500.000000 ns | 6176801800.000000 ns | -1994661700.000000 ns | 24.410% | True |
| 6 | X1→X0 | 8133810900.000000 ns | 6424693600.000000 ns | -1709117300.000000 ns | 21.013% | True |

### Prefix 1792: FFN decode
| Pair | Order | X0 FP32 storage | X1 FP16 storage | X1-X0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | X0→X1 | 143232400.000000 ns | 117456500.000000 ns | -25775900.000000 ns | 17.996% | True |
| 2 | X1→X0 | 147394900.000000 ns | 135560400.000000 ns | -11834500.000000 ns | 8.029% | True |
| 3 | X0→X1 | 145189100.000000 ns | 121989000.000000 ns | -23200100.000000 ns | 15.979% | True |
| 4 | X1→X0 | 163598900.000000 ns | 110331900.000000 ns | -53267000.000000 ns | 32.560% | True |
| 5 | X0→X1 | 143420900.000000 ns | 112283800.000000 ns | -31137100.000000 ns | 21.710% | True |
| 6 | X1→X0 | 147908000.000000 ns | 118978500.000000 ns | -28929500.000000 ns | 19.559% | True |

## Acceptance check
Result: **PASS**.

Acceptance requires direct raw-entry parity, six alternating pairs, all-pair FFN diagnostic reductions, measurable decode benefit in long context, and no reproducible short-context regression. F16C availability and exact reconstruction are activation prerequisites; no benchmark is run with the diagnostic profiler enabled for final speed figures.
