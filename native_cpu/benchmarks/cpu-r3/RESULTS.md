# CPU-R3 results — four-row FFN FP32 GEMV

Status: **rejected**.

## Scope
- F0 is the prior FP32 GEMV; F1 enables only the opt-in four-row FFN kernel.
- CPU-R1 selective logits, CPU-R2 V-blocked attention, one participant on CPU 0, model weights and defaults are unchanged.
- No quantization, worker repartitioning, affinity changes, attention changes or additional optimizations were included.

## Benchmark command
```powershell
python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r3 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r3/results
```

Raw receipts: `native_cpu/benchmarks/cpu-r3/results/*.json`; build identity: `native_cpu/benchmarks/cpu-r3/build-info.json`.

## Summary
| Case | Metric | F0 median | F1 median | Median F1-F0 | Δ stdev | Median benefit | All improved |
|---|---|---:|---:|---:|---:|---:|---|
| speed 256 | native_generation_seconds | 0.268055 s | 0.271417 s | 0.004867 s | 0.012443 | -1.830% | False |
| speed 256 | decode_tokens_per_second | 119.378814 tok/s | 117.902608 tok/s | -2.161149 tok/s | 4.941768 | -1.796% | False |
| FFN 256 | prefill_ms | 1184.434400 ms | 1215.040450 ms | 29.535950 ms | 5.311248 | -2.484% | False |
| FFN 256 | decode_ms | 148.087700 ms | 152.296150 ms | 3.867750 ms | 1.010832 | -2.617% | False |
| speed 1792 | native_generation_seconds | 0.558640 s | 0.568657 s | 0.001330 s | 0.149658 | -0.219% | False |
| speed 1792 | decode_tokens_per_second | 57.283360 tok/s | 56.274630 tok/s | -0.115431 tok/s | 9.823486 | -0.219% | False |
| FFN 1792 | prefill_ms | 8248.016600 ms | 8665.045300 ms | 352.310150 ms | 657.367397 | -4.291% | False |
| FFN 1792 | decode_ms | 152.053800 ms | 154.464250 ms | 3.028600 ms | 8.295084 | -2.026% | False |

## Historical conversation
- Artifact: `chat-parity.json`; six-turn deterministic conversation parity: **PASS**.
- Profiles: F0 (CPU-R1 + CPU-R2) versus F1 (the same configuration with `ffn_row4` enabled); all prompts, sampled IDs, text and finish reasons are compared.

## Paired details
### Prefix 256: generation seconds
| Pair | Order | F0 reference | F1 row4 | F1-F0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | F0→F1 | 0.277531 s | 0.291783 s | 0.014252 s | -5.135% | True |
| 2 | F1→F0 | 0.267642 s | 0.299736 s | 0.032094 s | -11.991% | True |
| 3 | F0→F1 | 0.271250 s | 0.268109 s | -0.003141 s | 1.158% | True |
| 4 | F1→F0 | 0.266714 s | 0.270074 s | 0.003360 s | -1.260% | True |
| 5 | F0→F1 | 0.264057 s | 0.269497 s | 0.005441 s | -2.060% | True |
| 6 | F1→F0 | 0.268468 s | 0.272760 s | 0.004293 s | -1.599% | True |

### Prefix 256: FFN prefill
| Pair | Order | F0 reference | F1 row4 | F1-F0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | F0→F1 | 1183.552800 ms | 1209.755700 ms | 26.202900 ms | -2.214% | True |
| 2 | F1→F0 | 1184.082800 ms | 1222.990100 ms | 38.907300 ms | -3.286% | True |
| 3 | F0→F1 | 1186.463000 ms | 1217.811000 ms | 31.348000 ms | -2.642% | True |
| 4 | F1→F0 | 1191.859600 ms | 1219.583500 ms | 27.723900 ms | -2.326% | True |
| 5 | F0→F1 | 1180.421300 ms | 1212.269900 ms | 31.848600 ms | -2.698% | True |
| 6 | F1→F0 | 1184.786000 ms | 1208.713600 ms | 23.927600 ms | -2.020% | True |

### Prefix 256: FFN decode
| Pair | Order | F0 reference | F1 row4 | F1-F0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | F0→F1 | 146.807300 ms | 151.426500 ms | 4.619200 ms | -3.146% | True |
| 2 | F1→F0 | 149.451700 ms | 152.376700 ms | 2.925000 ms | -1.957% | True |
| 3 | F0→F1 | 147.444800 ms | 151.053000 ms | 3.608200 ms | -2.447% | True |
| 4 | F1→F0 | 150.072000 ms | 153.640000 ms | 3.568000 ms | -2.378% | True |
| 5 | F0→F1 | 148.088300 ms | 152.215600 ms | 4.127300 ms | -2.787% | True |
| 6 | F1→F0 | 148.087100 ms | 153.899900 ms | 5.812800 ms | -3.925% | True |
### Prefix 1792: generation seconds
| Pair | Order | F0 reference | F1 row4 | F1-F0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | F0→F1 | 0.561412 s | 0.545176 s | -0.016235 s | 2.892% | True |
| 2 | F1→F0 | 0.627946 s | 0.565517 s | -0.062429 s | 9.942% | True |
| 3 | F0→F1 | 0.609518 s | 0.612041 s | 0.002523 s | -0.414% | True |
| 4 | F1→F0 | 0.555868 s | 0.904928 s | 0.349060 s | -62.795% | True |
| 5 | F0→F1 | 0.549863 s | 0.571798 s | 0.021935 s | -3.989% | True |
| 6 | F1→F0 | 0.550721 s | 0.550858 s | 0.000137 s | -0.025% | True |

### Prefix 1792: FFN prefill
| Pair | Order | F0 reference | F1 row4 | F1-F0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | F0→F1 | 8445.758500 ms | 8679.489100 ms | 233.730600 ms | -2.767% | True |
| 2 | F1→F0 | 8429.580100 ms | 8561.043600 ms | 131.463500 ms | -1.560% | True |
| 3 | F0→F1 | 8387.056800 ms | 8650.601500 ms | 263.544700 ms | -3.142% | True |
| 4 | F1→F0 | 8108.976400 ms | 8550.052000 ms | 441.075600 ms | -5.439% | True |
| 5 | F0→F1 | 8021.625500 ms | 9525.237800 ms | 1503.612300 ms | -18.744% | True |
| 6 | F1→F0 | 8041.648300 ms | 9588.844200 ms | 1547.195900 ms | -19.240% | True |

### Prefix 1792: FFN decode
| Pair | Order | F0 reference | F1 row4 | F1-F0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | F0→F1 | 153.660400 ms | 154.998600 ms | 1.338200 ms | -0.871% | True |
| 2 | F1→F0 | 150.973400 ms | 153.929900 ms | 2.956500 ms | -1.958% | True |
| 3 | F0→F1 | 153.134200 ms | 153.549900 ms | 0.415700 ms | -0.271% | True |
| 4 | F1→F0 | 147.253100 ms | 169.742600 ms | 22.489500 ms | -15.273% | True |
| 5 | F0→F1 | 174.746600 ms | 182.179700 ms | 7.433100 ms | -4.254% | True |
| 6 | F1→F0 | 148.040600 ms | 151.141300 ms | 3.100700 ms | -2.094% | True |

## Acceptance check
Result: **REJECT** — decode speed regression at 256, decode benefit inside dispersion at 256, decode throughput regression at 256, FFN prefill regression at 256, FFN decode regression at 256, decode speed regression at 1792, decode benefit inside dispersion at 1792, decode throughput regression at 1792, FFN prefill regression at 1792, FFN decode regression at 1792
