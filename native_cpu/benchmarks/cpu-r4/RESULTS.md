# CPU-R4 results — shared-K GQA Q·K

Status: **accepted**.

## Scope
- K0 is the CPU-R1 + CPU-R2 reference: selective logits, KV reuse, V-blocked attention, one participant pinned to CPU 0.
- K1 enables only the opt-in shared-K Q·K route for exactly two query heads per KV head; each head keeps independent scores, softmax and CPU-R2 V accumulation.
- Other query/KV relationships use the original route. No FFN, KV layout, weights, quantization, workers, affinities or defaults changed.

## Benchmark command
```powershell
python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r4 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r4/results
```

Raw receipts: `native_cpu/benchmarks/cpu-r4/results/*.json`; build identity: `native_cpu/benchmarks/cpu-r4/build-info.json`.

## Summary
| Case | Metric | K0 median | K1 median | Median K1-K0 | Δ stdev | Median benefit | All improved |
|---|---|---:|---:|---:|---:|---:|---|
| speed 256 | native_generation_seconds | 0.276497 s | 0.257946 s | -0.017069 s | 0.007055 | -6.189% | True |
| speed 256 | decode_tokens_per_second | 115.737708 tok/s | 124.059573 tok/s | 7.658122 tok/s | 3.233770 | 6.599% | True |
| QK 256 | prefill_ms | 145.657900 ms | 82.159150 ms | -65.572300 ms | 5.334158 | 44.804% | True |
| QK 256 | decode_ms | 38.622050 ms | 21.182200 ms | -17.158750 ms | 1.675861 | 44.768% | True |
| speed 1792 | native_generation_seconds | 0.677325 s | 0.555059 s | -0.125352 s | 0.101637 | -19.047% | True |
| speed 1792 | decode_tokens_per_second | 47.253737 tok/s | 57.657109 tok/s | 11.442669 tok/s | 5.234187 | 23.530% | True |
| QK 1792 | prefill_ms | 7686.910700 ms | 4558.265450 ms | -3149.576650 ms | 765.566036 | 41.060% | True |
| QK 1792 | decode_ms | 305.996600 ms | 176.456050 ms | -126.112150 ms | 30.002102 | 41.918% | True |

## Historical conversation
- Artifact: `chat-parity.json`; six-turn deterministic conversation parity: **PASS**.
- K0 and K1 compare canonical prompt IDs, sampled IDs, text and finish reasons.

## Paired details
### Prefix 256: generation seconds
| Pair | Order | K0 reference | K1 shared-K | K1-K0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | K0→K1 | 0.278105 s | 0.261927 s | -0.016178 s | 5.817% | True |
| 2 | K1→K0 | 0.273721 s | 0.255761 s | -0.017960 s | 6.561% | True |
| 3 | K0→K1 | 0.274888 s | 0.253314 s | -0.021574 s | 7.848% | True |
| 4 | K1→K0 | 0.279403 s | 0.256754 s | -0.022649 s | 8.106% | True |
| 5 | K0→K1 | 0.278134 s | 0.274957 s | -0.003177 s | 1.142% | True |
| 6 | K1→K0 | 0.272814 s | 0.259138 s | -0.013676 s | 5.013% | True |

### Prefix 256: Q·K prefill
| Pair | Order | K0 reference | K1 shared-K | K1-K0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | K0→K1 | 144.875000 ms | 87.236600 ms | -57.638400 ms | 39.785% | True |
| 2 | K1→K0 | 145.658900 ms | 77.484200 ms | -68.174700 ms | 46.804% | True |
| 3 | K0→K1 | 152.073000 ms | 80.803800 ms | -71.269200 ms | 46.865% | True |
| 4 | K1→K0 | 144.267800 ms | 84.842400 ms | -59.425400 ms | 41.191% | True |
| 5 | K0→K1 | 147.102000 ms | 83.514500 ms | -63.587500 ms | 43.227% | True |
| 6 | K1→K0 | 145.656900 ms | 78.099800 ms | -67.557100 ms | 46.381% | True |

### Prefix 256: Q·K decode
| Pair | Order | K0 reference | K1 shared-K | K1-K0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | K0→K1 | 39.037400 ms | 24.733400 ms | -14.304000 ms | 36.642% | True |
| 2 | K1→K0 | 38.630200 ms | 21.222000 ms | -17.408200 ms | 45.064% | True |
| 3 | K0→K1 | 38.613900 ms | 21.374000 ms | -17.239900 ms | 44.647% | True |
| 4 | K1→K0 | 38.220000 ms | 21.142400 ms | -17.077600 ms | 44.682% | True |
| 5 | K0→K1 | 40.518800 ms | 20.947600 ms | -19.571200 ms | 48.302% | True |
| 6 | K1→K0 | 37.995100 ms | 20.953000 ms | -17.042100 ms | 44.853% | True |
### Prefix 1792: generation seconds
| Pair | Order | K0 reference | K1 shared-K | K1-K0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | K0→K1 | 0.686712 s | 0.549576 s | -0.137136 s | 19.970% | True |
| 2 | K1→K0 | 0.648001 s | 0.526601 s | -0.121400 s | 18.735% | True |
| 3 | K0→K1 | 0.667939 s | 0.538634 s | -0.129304 s | 19.359% | True |
| 4 | K1→K0 | 0.664142 s | 0.561268 s | -0.102873 s | 15.490% | True |
| 5 | K0→K1 | 0.710722 s | 0.601804 s | -0.108918 s | 15.325% | True |
| 6 | K1→K0 | 0.927498 s | 0.560543 s | -0.366955 s | 39.564% | True |

### Prefix 1792: Q·K prefill
| Pair | Order | K0 reference | K1 shared-K | K1-K0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | K0→K1 | 9568.884000 ms | 4585.719900 ms | -4983.164100 ms | 52.077% | True |
| 2 | K1→K0 | 7681.859600 ms | 4506.815500 ms | -3175.044100 ms | 41.332% | True |
| 3 | K0→K1 | 7857.154600 ms | 4410.363100 ms | -3446.791500 ms | 43.868% | True |
| 4 | K1→K0 | 7659.339800 ms | 4535.230600 ms | -3124.109200 ms | 40.788% | True |
| 5 | K0→K1 | 7661.227300 ms | 4581.300300 ms | -3079.927000 ms | 40.201% | True |
| 6 | K1→K0 | 7691.961800 ms | 4753.306900 ms | -2938.654900 ms | 38.204% | True |

### Prefix 1792: Q·K decode
| Pair | Order | K0 reference | K1 shared-K | K1-K0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | K0→K1 | 271.584600 ms | 177.612100 ms | -93.972500 ms | 34.602% | True |
| 2 | K1→K0 | 318.537400 ms | 175.285200 ms | -143.252200 ms | 44.972% | True |
| 3 | K0→K1 | 344.472900 ms | 188.227100 ms | -156.245800 ms | 45.358% | True |
| 4 | K1→K0 | 280.394900 ms | 171.422800 ms | -108.972100 ms | 38.864% | True |
| 5 | K0→K1 | 325.402900 ms | 175.300000 ms | -150.102900 ms | 46.128% | True |
| 6 | K1→K0 | 293.455800 ms | 205.276600 ms | -88.179200 ms | 30.049% | True |

## Acceptance check
Result: **PASS**.
