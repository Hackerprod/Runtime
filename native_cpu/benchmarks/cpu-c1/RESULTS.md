# CPU-C1 results — original to cumulative CPU-R6

Status: **PASS**.

## Scope
- A is the frozen pre-CPU-R1 DLL. B is the CPU-R6 DLL with selective logits, KV reuse, V-blocked attention, shared-K GQA, shared-V GQA and exact FP16 FFN storage enabled; FFN row4 is disabled.
- Both sides use the same model, tokenizer, chat template, sampling policy and seed, with one participant pinned to CPU 0. Profiling is disabled for all speed receipts.
- This is an accumulated closure comparison, not a new optimization experiment. Percentages are calculated from paired A/B samples and are not summed from earlier campaigns.

## Immutable identity
- Reference runtime commit: `a6d06a949cb514741f100e439e283d2189685102`; DLL SHA-256: `4f9ed66fd43d0850f7d2ec17679df1b85754c665ba5c3393b1e71f7f7658c1bd`.
- Candidate runtime commit: `ae90d82e6007574628d496eba898cdffce2b8096`; DLL SHA-256: `4d598dd21493d2821a10783c4c97671c96ea038b98c3cc82c30172182fdb3672`.
- Measurement tree commit: `952e0146d80f31e25dd844c52aaa10b4c83bd387`; model SHA-256: `bcab52e67825061ffc3bd4480dbda1d5067bc3a0e51aba386a236c7131766f33`.

## Benchmark command
```powershell
python native_cpu/benchmarks/cpu-c1/benchmark.py --out native_cpu/benchmarks/cpu-c1/results --pairs 6
python native_cpu/benchmarks/cpu-c1/summarize.py
```

Raw receipts are under `native_cpu/benchmarks/cpu-c1/results/`. Loading and FP16-storage preparation are recorded per entry and excluded from inference timings.

## Original → cumulative final
| Case | Metric | A median | B median | Median B-A | Median A→B benefit | B favorable |
|---|---|---:|---:|---:|---:|---:|
| cpu-c1-conversation-historical | prefill | 17.541659 s | 1.240126 s | -16.310300 s | 92.952% | 6/6 |
| cpu-c1-conversation-historical | decode | 8.310240 s | 6.056700 s | -2.255610 s | 27.097% | 6/6 |
| cpu-c1-conversation-historical | total | 26.191536 s | 7.663784 s | -18.571685 s | 70.841% | 6/6 |
| cpu-c1-conversation-historical | conversation wall | 26.191727 s | 7.663972 s | -18.571692 s | 70.840% | 6/6 |
| cpu-c1-conversation-historical | decode throughput | 88.205 tok/s | 121.023 tok/s | 32.853 tok/s | 37.169% | 6/6 |
| cpu-c1-conversation-second | prefill | 1.585375 s | 0.522432 s | -1.055839 s | 66.703% | 6/6 |
| cpu-c1-conversation-second | decode | 0.717483 s | 0.585262 s | -0.122622 s | 17.334% | 6/6 |
| cpu-c1-conversation-second | total | 2.371228 s | 1.180646 s | -1.202928 s | 50.456% | 6/6 |
| cpu-c1-conversation-second | conversation wall | 2.371287 s | 1.180710 s | -1.202929 s | 50.455% | 6/6 |
| cpu-c1-conversation-second | decode throughput | 132.408 tok/s | 162.321 tok/s | 28.165 tok/s | 20.972% | 6/6 |
| cpu-c1-forced-256 | prefill | 1.899971 s | 1.425491 s | -0.475175 s | 25.148% | 6/6 |
| cpu-c1-forced-256 | decode | 0.280773 s | 0.221934 s | -0.059810 s | 21.176% | 6/6 |
| cpu-c1-forced-256 | total | 2.177245 s | 1.644979 s | -0.529150 s | 24.449% | 6/6 |
| cpu-c1-forced-256 | decode throughput | 113.980 tok/s | 144.188 tok/s | 30.541 tok/s | 26.901% | 6/6 |
| cpu-c1-forced-1792 | prefill | 26.386583 s | 16.969214 s | -9.475437 s | 35.910% | 6/6 |
| cpu-c1-forced-1792 | decode | 0.793491 s | 0.478947 s | -0.308716 s | 39.211% | 6/6 |
| cpu-c1-forced-1792 | total | 27.187888 s | 17.448162 s | -9.801451 s | 36.043% | 6/6 |
| cpu-c1-forced-1792 | decode throughput | 40.330 tok/s | 66.829 tok/s | 26.223 tok/s | 64.504% | 6/6 |

## Load and preparation (reported separately)
| Case | A load median | B load median | A configuration median | B configuration median | A FP16 preparation median | B FP16 preparation median |
|---|---:|---:|---:|---:|---:|---:|
| cpu-c1-conversation-historical | 0.874823 s | 0.860921 s | 0.000011 s | 0.149944 s | 0.000001 s | 0.149888 s |
| cpu-c1-conversation-second | 0.850191 s | 0.841761 s | 0.000008 s | 0.144837 s | 0.000001 s | 0.144798 s |
| cpu-c1-forced-256 | 0.866359 s | 0.868476 s | 0.000007 s | 0.146602 s | 0.000001 s | 0.146571 s |
| cpu-c1-forced-1792 | 0.840065 s | 0.836831 s | 0.000007 s | 0.145343 s | 0.000001 s | 0.145307 s |

## Parity and availability
- Forced-prefix receipts compare exact FP32 logits hashes for the prefill and every decode token, plus token IDs and final position.
- Conversation receipts compare canonical prompt IDs, sampled token IDs, text, finish reason and generated-token counts for each turn. All four cases use six alternating A/B pairs.
- Diagnostic counters are intentionally not required in this comparison; fields absent from the frozen DLL are reported as unavailable rather than zero.

## Paired details
### cpu-c1-conversation-historical

#### Decode time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 8.397368 s | 6.046673 s | -2.350694 s | 27.993% | True |
| 2 | B→A | 8.187258 s | 5.971782 s | -2.215476 s | 27.060% | True |
| 3 | A→B | 8.287070 s | 5.977894 s | -2.309176 s | 27.865% | True |
| 4 | B→A | 8.401215 s | 6.121573 s | -2.279642 s | 27.135% | True |
| 5 | A→B | 8.239836 s | 6.066727 s | -2.173109 s | 26.373% | True |
| 6 | B→A | 8.333411 s | 6.101833 s | -2.231578 s | 26.779% | True |

#### Total time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 26.962799 s | 7.652917 s | -19.309883 s | 71.617% | True |
| 2 | B→A | 26.114927 s | 7.559754 s | -18.555173 s | 71.052% | True |
| 3 | A→B | 26.190023 s | 7.599845 s | -18.590177 s | 70.982% | True |
| 4 | B→A | 26.319740 s | 7.731544 s | -18.588197 s | 70.625% | True |
| 5 | A→B | 26.193048 s | 7.674651 s | -18.518397 s | 70.700% | True |
| 6 | B→A | 26.164282 s | 7.721075 s | -18.443207 s | 70.490% | True |
### cpu-c1-conversation-second

#### Decode time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 0.728210 s | 0.632199 s | -0.096010 s | 13.184% | True |
| 2 | B→A | 0.719012 s | 0.636826 s | -0.082186 s | 11.430% | True |
| 3 | A→B | 0.709835 s | 0.583807 s | -0.126028 s | 17.755% | True |
| 4 | B→A | 0.838944 s | 0.584906 s | -0.254038 s | 30.281% | True |
| 5 | A→B | 0.715953 s | 0.584033 s | -0.131920 s | 18.426% | True |
| 6 | B→A | 0.704834 s | 0.585617 s | -0.119216 s | 16.914% | True |

#### Total time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 2.342652 s | 1.214490 s | -1.128162 s | 48.157% | True |
| 2 | B→A | 2.331141 s | 1.204052 s | -1.127089 s | 48.349% | True |
| 3 | A→B | 2.364474 s | 1.157241 s | -1.207233 s | 51.057% | True |
| 4 | B→A | 2.444172 s | 1.153269 s | -1.290904 s | 52.816% | True |
| 5 | A→B | 2.404188 s | 1.205565 s | -1.198623 s | 49.856% | True |
| 6 | B→A | 2.377981 s | 1.144981 s | -1.233001 s | 51.851% | True |
### cpu-c1-forced-256

#### Decode time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 0.276223 s | 0.222531 s | -0.053692 s | 19.438% | True |
| 2 | B→A | 0.278325 s | 0.223859 s | -0.054467 s | 19.569% | True |
| 3 | A→B | 0.283221 s | 0.217638 s | -0.065582 s | 23.156% | True |
| 4 | B→A | 0.273493 s | 0.219235 s | -0.054258 s | 19.839% | True |
| 5 | A→B | 0.294450 s | 0.221337 s | -0.073114 s | 24.831% | True |
| 6 | B→A | 0.289406 s | 0.224253 s | -0.065153 s | 22.513% | True |

#### Total time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 2.168734 s | 1.642262 s | -0.526472 s | 24.276% | True |
| 2 | B→A | 2.185755 s | 1.689186 s | -0.496569 s | 22.718% | True |
| 3 | A→B | 2.165378 s | 1.644271 s | -0.521107 s | 24.065% | True |
| 4 | B→A | 2.159977 s | 1.628150 s | -0.531827 s | 24.622% | True |
| 5 | A→B | 2.213690 s | 1.645687 s | -0.568003 s | 25.659% | True |
| 6 | B→A | 2.287351 s | 1.657700 s | -0.629651 s | 27.528% | True |
### cpu-c1-forced-1792

#### Decode time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 0.788446 s | 0.499952 s | -0.288494 s | 36.590% | True |
| 2 | B→A | 0.821773 s | 0.471597 s | -0.350176 s | 42.612% | True |
| 3 | A→B | 0.804075 s | 0.464285 s | -0.339790 s | 42.259% | True |
| 4 | B→A | 0.776162 s | 0.470968 s | -0.305194 s | 39.321% | True |
| 5 | A→B | 0.733080 s | 0.517458 s | -0.215621 s | 29.413% | True |
| 6 | B→A | 0.798536 s | 0.486298 s | -0.312238 s | 39.101% | True |

#### Total time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 27.112773 s | 18.065839 s | -9.046933 s | 33.368% | True |
| 2 | B→A | 27.626534 s | 17.411857 s | -10.214677 s | 36.974% | True |
| 3 | A→B | 27.173121 s | 17.288409 s | -9.884712 s | 36.377% | True |
| 4 | B→A | 27.241577 s | 17.336153 s | -9.905424 s | 36.361% | True |
| 5 | A→B | 26.850231 s | 17.985467 s | -8.864764 s | 33.016% | True |
| 6 | B→A | 27.202655 s | 17.484466 s | -9.718189 s | 35.725% | True |

## Evidence verdict
Result: **PASS**.

A PASS means the four-case paired evidence is complete, directly parity-checked and identity-consistent. It does not promote an additional kernel or claim that preparation cost is part of sustained decode.
