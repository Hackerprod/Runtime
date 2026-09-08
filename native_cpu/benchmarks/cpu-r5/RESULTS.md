# CPU-R5 results — shared-V GQA accumulation

Status: **accepted**.

## Scope
- W0 is the CPU-R1 + CPU-R2 + CPU-R4 reference: one participant pinned to CPU 0, V blocks of 16 dimensions and shared-K GQA.
- W1 enables only the opt-in shared-V route for exactly two query heads per KV head. Softmax distributions and accumulators remain independent; each V block is read once for both heads.
- Other query/KV ratios and missing prerequisites use the existing route and increment the fallback counter. No Q·K, FFN, KV layout, weights, quantization, workers, affinities or defaults changed.

## Benchmark command
```powershell
python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r5 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r5/results
```

Speed receipts have profiling disabled; diagnostic receipts are separate. Raw receipts: `native_cpu/benchmarks/cpu-r5/results/*.json`; build identity: `native_cpu/benchmarks/cpu-r5/build-info.json`.

## Summary
| Case | Metric | W0 median | W1 median | Median W1-W0 | Δ stdev | Median benefit | Favorable |
|---|---|---:|---:|---:|---:|---:|---:|
| speed 256 | native_generation_seconds | 0.262968 s | 0.262368 s | 0.002024 | 0.007562 | -0.777% | 2/6 |
| speed 256 | decode_tokens_per_second | 121.688060 tok/s | 122.005084 tok/s | -0.948078 | 3.582486 | -0.771% | 2/6 |
| diagnostics 256 | V prefill | 94.232 ms | 78.595 ms | -15.929 ms | 5.347 | 16.902% | 6/6 |
| diagnostics 256 | V decode | 23.903 ms | 20.462 ms | -3.915 ms | 1.499 | 16.070% | 6/6 |
| diagnostics 256 | attention prefill | 179.407 ms | 162.442 ms | -16.965 ms | 8.580 | 9.443% | 6/6 |
| diagnostics 256 | attention decode | 46.719 ms | 44.849 ms | -1.745 ms | 3.093 | 3.746% | 5/6 |
| speed 1792 | native_generation_seconds | 0.548328 s | 0.519290 s | -0.035660 | 0.049045 | 6.726% | 6/6 |
| speed 1792 | decode_tokens_per_second | 58.366860 tok/s | 61.624109 tok/s | 4.356324 | 4.342101 | 7.217% | 6/6 |
| diagnostics 1792 | V prefill | 4711.005 ms | 3335.496 ms | -1348.852 ms | 174.981 | 28.636% | 6/6 |
| diagnostics 1792 | V decode | 165.386 ms | 113.212 ms | -53.350 ms | 5.404 | 32.100% | 6/6 |
| diagnostics 1792 | attention prefill | 9672.431 ms | 8071.298 ms | -1522.462 ms | 447.490 | 15.946% | 6/6 |
| diagnostics 1792 | attention decode | 338.497 ms | 293.209 ms | -50.194 ms | 27.563 | 14.706% | 6/6 |

## Native contract tests
- `mm_test_runtime` compares pre-projection attention and final logits byte-for-byte, including distinct head probabilities, a final incomplete V block, cache truncation/replay and non-2:1 fallback.

## Historical conversation
- Artifact: `chat-parity.json`; six-turn deterministic conversation parity: **PASS**.
- The check compares prompt IDs, sampled IDs, text and finish reasons for W0 and W1.

## Paired details
### Prefix 256: decode generation seconds
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 0.263070 s | 0.267053 s | 0.003982 s | -1.514% | True |
| 2 | W1→W0 | 0.266100 s | 0.268016 s | 0.001916 s | -0.720% | True |
| 3 | W0→W1 | 0.255550 s | 0.257683 s | 0.002133 s | -0.835% | True |
| 4 | W1→W0 | 0.257285 s | 0.248226 s | -0.009059 s | 3.521% | True |
| 5 | W0→W1 | 0.262865 s | 0.272208 s | 0.009343 s | -3.554% | True |
| 6 | W1→W0 | 0.263376 s | 0.253744 s | -0.009632 s | 3.657% | True |

### Prefix 256: V prefill
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 102109100.000000 ns | 79422700.000000 ns | -22686400.000000 ns | 22.218% | True |
| 2 | W1→W0 | 101957700.000000 ns | 78287300.000000 ns | -23670400.000000 ns | 23.216% | True |
| 3 | W0→W1 | 90158300.000000 ns | 78603600.000000 ns | -11554700.000000 ns | 12.816% | True |
| 4 | W1→W0 | 93563400.000000 ns | 78019800.000000 ns | -15543600.000000 ns | 16.613% | True |
| 5 | W0→W1 | 94899900.000000 ns | 78586300.000000 ns | -16313600.000000 ns | 17.190% | True |
| 6 | W1→W0 | 91811700.000000 ns | 80649900.000000 ns | -11161800.000000 ns | 12.157% | True |

### Prefix 256: V decode
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 24365400.000000 ns | 21347400.000000 ns | -3018000.000000 ns | 12.386% | True |
| 2 | W1→W0 | 24901400.000000 ns | 21003300.000000 ns | -3898100.000000 ns | 15.654% | True |
| 3 | W0→W1 | 23582400.000000 ns | 23014300.000000 ns | -568100.000000 ns | 2.409% | True |
| 4 | W1→W0 | 23952200.000000 ns | 19152700.000000 ns | -4799500.000000 ns | 20.038% | True |
| 5 | W0→W1 | 23853600.000000 ns | 19921200.000000 ns | -3932400.000000 ns | 16.486% | True |
| 6 | W1→W0 | 23751600.000000 ns | 19594500.000000 ns | -4157100.000000 ns | 17.502% | True |

### Prefix 256: attention prefill
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 193828300.000000 ns | 161751200.000000 ns | -32077100.000000 ns | 16.549% | True |
| 2 | W1→W0 | 193633300.000000 ns | 166816900.000000 ns | -26816400.000000 ns | 13.849% | True |
| 3 | W0→W1 | 173951600.000000 ns | 160690400.000000 ns | -13261200.000000 ns | 7.623% | True |
| 4 | W1→W0 | 177122200.000000 ns | 162170600.000000 ns | -14951600.000000 ns | 8.441% | True |
| 5 | W0→W1 | 181691400.000000 ns | 162712900.000000 ns | -18978500.000000 ns | 10.445% | True |
| 6 | W1→W0 | 176653200.000000 ns | 166987500.000000 ns | -9665700.000000 ns | 5.472% | True |

### Prefix 256: attention decode
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 46722400.000000 ns | 45217600.000000 ns | -1504800.000000 ns | 3.221% | True |
| 2 | W1→W0 | 47820700.000000 ns | 46982200.000000 ns | -838500.000000 ns | 1.753% | True |
| 3 | W0→W1 | 46655600.000000 ns | 50820400.000000 ns | 4164800.000000 ns | -8.927% | True |
| 4 | W1→W0 | 47347400.000000 ns | 42019000.000000 ns | -5328400.000000 ns | 11.254% | True |
| 5 | W0→W1 | 46465100.000000 ns | 44480600.000000 ns | -1984500.000000 ns | 4.271% | True |
| 6 | W1→W0 | 46716000.000000 ns | 44412300.000000 ns | -2303700.000000 ns | 4.931% | True |
### Prefix 1792: decode generation seconds
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 0.542057 s | 0.529916 s | -0.012141 s | 2.240% | True |
| 2 | W1→W0 | 0.579826 s | 0.516701 s | -0.063125 s | 10.887% | True |
| 3 | W0→W1 | 0.527636 s | 0.488233 s | -0.039403 s | 7.468% | True |
| 4 | W1→W0 | 0.554599 s | 0.532790 s | -0.021810 s | 3.933% | True |
| 5 | W0→W1 | 0.533422 s | 0.501504 s | -0.031917 s | 5.984% | True |
| 6 | W1→W0 | 0.667913 s | 0.521880 s | -0.146033 s | 21.864% | True |

### Prefix 1792: V prefill
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 4762886100.000000 ns | 3259832700.000000 ns | -1503053400.000000 ns | 31.558% | True |
| 2 | W1→W0 | 4531732300.000000 ns | 3434607000.000000 ns | -1097125300.000000 ns | 24.210% | True |
| 3 | W0→W1 | 4655221400.000000 ns | 3358299600.000000 ns | -1296921800.000000 ns | 27.860% | True |
| 4 | W1→W0 | 4749609300.000000 ns | 3411613600.000000 ns | -1337995700.000000 ns | 28.171% | True |
| 5 | W0→W1 | 4769801100.000000 ns | 3166274200.000000 ns | -1603526900.000000 ns | 33.618% | True |
| 6 | W1→W0 | 4672400500.000000 ns | 3312691900.000000 ns | -1359708600.000000 ns | 29.101% | True |

### Prefix 1792: V decode
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 164798600.000000 ns | 111109700.000000 ns | -53688900.000000 ns | 32.578% | True |
| 2 | W1→W0 | 164223000.000000 ns | 120255000.000000 ns | -43968000.000000 ns | 26.773% | True |
| 3 | W0→W1 | 169827300.000000 ns | 111687900.000000 ns | -58139400.000000 ns | 34.234% | True |
| 4 | W1→W0 | 168038800.000000 ns | 114004000.000000 ns | -54034800.000000 ns | 32.156% | True |
| 5 | W0→W1 | 165339500.000000 ns | 119470200.000000 ns | -45869300.000000 ns | 27.742% | True |
| 6 | W1→W0 | 165432100.000000 ns | 112420000.000000 ns | -53012100.000000 ns | 32.045% | True |

### Prefix 1792: attention prefill
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 9801159500.000000 ns | 7951242700.000000 ns | -1849916800.000000 ns | 18.874% | True |
| 2 | W1→W0 | 9245919200.000000 ns | 8211600600.000000 ns | -1034318600.000000 ns | 11.187% | True |
| 3 | W0→W1 | 9677031500.000000 ns | 8480492400.000000 ns | -1196539100.000000 ns | 12.365% | True |
| 4 | W1→W0 | 9382906800.000000 ns | 8117650900.000000 ns | -1265255900.000000 ns | 13.485% | True |
| 5 | W0→W1 | 9667831100.000000 ns | 7888163400.000000 ns | -1779667700.000000 ns | 18.408% | True |
| 6 | W1→W0 | 10196714500.000000 ns | 8024944200.000000 ns | -2171770300.000000 ns | 21.299% | True |

### Prefix 1792: attention decode
| Pair | Order | W0 reference | W1 shared-V | W1-W0 | Benefit | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | W0→W1 | 351163700.000000 ns | 288013300.000000 ns | -63150400.000000 ns | 17.983% | True |
| 2 | W1→W0 | 325379700.000000 ns | 301295400.000000 ns | -24084300.000000 ns | 7.402% | True |
| 3 | W0→W1 | 371775200.000000 ns | 297827000.000000 ns | -73948200.000000 ns | 19.891% | True |
| 4 | W1→W0 | 325829800.000000 ns | 288591600.000000 ns | -37238200.000000 ns | 11.429% | True |
| 5 | W0→W1 | 325713200.000000 ns | 304027500.000000 ns | -21685700.000000 ns | 6.658% | True |
| 6 | W1→W0 | 375009600.000000 ns | 287241600.000000 ns | -87768000.000000 ns | 23.404% | True |

## Acceptance check
Result: **PASS**.

Acceptance uses exact direct-entry parity, six paired orders, all-pair V-time reductions, and a long-context decode benefit. A short-context delta is treated as a regression only when its paired median exceeds the measured one-sigma dispersion; this run stays within that dispersion.
