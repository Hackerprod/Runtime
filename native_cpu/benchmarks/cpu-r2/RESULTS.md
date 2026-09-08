# CPU-R2 results — V accumulation locality

Status: **accepted**. The opt-in V-blocked attention route preserved bitwise logits/tokens/text and improved measured attention plus decode on both requested prefix lengths.

## Scope guard
- Only the post-softmax accumulation over V was changed.
- Q·K, softmax, KV layout, FFN, weights, quantization, workers, affinities and defaults were not changed.
- Route is opt-in with `--v-blocked-attention`; default remains the original accumulation route.

## Benchmark command
```powershell
python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r2 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r2/results
```

Benchmark status: `PASS`; final protocol: `True`; raw pair files: `native_cpu/benchmarks/cpu-r2/results/*.json`.
Chat token/text parity: `native_cpu/benchmarks/cpu-r2/chat-parity.json` = `PASS` for the historical six-turn conversation under CPU-R1 flags plus CPU 0 pinning.

## Summary
| Case | Metric | V0 median | V1 median | Median V1-V0 | Δ stdev | Median benefit | All pairs improved |
|---|---|---:|---:|---:|---:|---:|---|
| speed 256 | native_generation_seconds | 0.292972 s | 0.270325 s | -0.022878 s | 0.003404 | 7.843% | True |
| speed 256 | decode_tokens_per_second | 109.226843 tok/s | 118.377580 tok/s | 9.337060 tok/s | 1.317416 | 8.511% | True |
| diagnostics 256 | attention_prefill_ms | 274.867600 ms | 180.891800 ms | -98.653100 ms | 14.893148 | 35.889% | True |
| diagnostics 256 | attention_decode_ms | 72.867850 ms | 47.788900 ms | -24.949300 ms | 1.876455 | 34.219% | True |
| speed 1792 | native_generation_seconds | 0.821233 s | 0.584163 s | -0.222753 s | 0.029960 | 27.179% | True |
| speed 1792 | decode_tokens_per_second | 38.977289 tok/s | 54.795242 tok/s | 14.483856 tok/s | 2.401252 | 37.339% | True |
| diagnostics 1792 | attention_prefill_ms | 14256.592850 ms | 8978.175850 ms | -5391.590400 ms | 328.305664 | 37.610% | True |
| diagnostics 1792 | attention_decode_ms | 528.955800 ms | 330.831700 ms | -197.152900 ms | 19.533786 | 37.339% | True |

## Paired speed details
### Prefix 256: native decode/generation seconds
| Pair | Order | V0 reference | V1 blocked | V1-V0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | V0→V1 | 0.298141 s | 0.271344 s | -0.026796 s | 8.988% | True |
| 2 | V1→V0 | 0.291920 s | 0.268999 s | -0.022921 s | 7.852% | True |
| 3 | V0→V1 | 0.291452 s | 0.268617 s | -0.022835 s | 7.835% | True |
| 4 | V1→V0 | 0.294024 s | 0.269306 s | -0.024717 s | 8.407% | True |
| 5 | V0→V1 | 0.289231 s | 0.272616 s | -0.016614 s | 5.744% | True |
| 6 | V1→V0 | 0.294602 s | 0.271962 s | -0.022640 s | 7.685% | True |

### Prefix 1792: native decode/generation seconds
| Pair | Order | V0 reference | V1 blocked | V1-V0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | V0→V1 | 0.835332 s | 0.557135 s | -0.278197 s | 33.304% | True |
| 2 | V1→V0 | 0.786711 s | 0.554190 s | -0.232521 s | 29.556% | True |
| 3 | V0→V1 | 0.837582 s | 0.631411 s | -0.206170 s | 24.615% | True |
| 4 | V1→V0 | 0.842114 s | 0.606575 s | -0.235538 s | 27.970% | True |
| 5 | V0→V1 | 0.807134 s | 0.594149 s | -0.212986 s | 26.388% | True |
| 6 | V1→V0 | 0.767489 s | 0.574177 s | -0.193311 s | 25.188% | True |

## Paired diagnostic attention details
### Prefix 256: attention prefill
| Pair | Order | V0 reference | V1 blocked | V1-V0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | V0→V1 | 266.118200 ms | 184.522500 ms | -81.595700 ms | 30.661% | True |
| 2 | V1→V0 | 264.505300 ms | 185.672200 ms | -78.833100 ms | 29.804% | True |
| 3 | V0→V1 | 302.608600 ms | 183.810000 ms | -118.798600 ms | 39.258% | True |
| 4 | V1→V0 | 274.281200 ms | 177.973600 ms | -96.307600 ms | 35.113% | True |
| 5 | V0→V1 | 275.454000 ms | 174.455400 ms | -100.998600 ms | 36.666% | True |
| 6 | V1→V0 | 278.797500 ms | 174.700600 ms | -104.096900 ms | 37.338% | True |

### Prefix 256: attention decode
| Pair | Order | V0 reference | V1 blocked | V1-V0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | V0→V1 | 73.258300 ms | 49.205800 ms | -24.052500 ms | 32.832% | True |
| 2 | V1→V0 | 71.449400 ms | 49.096100 ms | -22.353300 ms | 31.285% | True |
| 3 | V0→V1 | 71.402000 ms | 48.565100 ms | -22.836900 ms | 31.984% | True |
| 4 | V1→V0 | 72.591500 ms | 46.745400 ms | -25.846100 ms | 35.605% | True |
| 5 | V0→V1 | 73.676300 ms | 47.012700 ms | -26.663600 ms | 36.190% | True |
| 6 | V1→V0 | 73.144200 ms | 46.701000 ms | -26.443200 ms | 36.152% | True |

### Prefix 1792: attention prefill
| Pair | Order | V0 reference | V1 blocked | V1-V0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | V0→V1 | 14263.616900 ms | 9051.969000 ms | -5211.647900 ms | 36.538% | True |
| 2 | V1→V0 | 13810.251300 ms | 9205.124600 ms | -4605.126700 ms | 33.346% | True |
| 3 | V0→V1 | 14249.568800 ms | 8857.819900 ms | -5391.748900 ms | 37.838% | True |
| 4 | V1→V0 | 14371.416500 ms | 8938.726300 ms | -5432.690200 ms | 37.802% | True |
| 5 | V0→V1 | 14409.057300 ms | 9017.625400 ms | -5391.431900 ms | 37.417% | True |
| 6 | V1→V0 | 13907.491700 ms | 8438.129400 ms | -5469.362300 ms | 39.327% | True |

### Prefix 1792: attention decode
| Pair | Order | V0 reference | V1 blocked | V1-V0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | V0→V1 | 543.886900 ms | 302.018500 ms | -241.868400 ms | 44.470% | True |
| 2 | V1→V0 | 521.890400 ms | 327.623500 ms | -194.266900 ms | 37.224% | True |
| 3 | V0→V1 | 523.832800 ms | 336.005400 ms | -187.827400 ms | 35.856% | True |
| 4 | V1→V0 | 534.078800 ms | 334.039900 ms | -200.038900 ms | 37.455% | True |
| 5 | V0→V1 | 552.704200 ms | 358.803200 ms | -193.901000 ms | 35.082% | True |
| 6 | V1→V0 | 518.958200 ms | 314.434300 ms | -204.523900 ms | 39.410% | True |

## Acceptance check
Result: **PASS**.
