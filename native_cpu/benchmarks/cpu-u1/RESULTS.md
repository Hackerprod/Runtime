# CPU-U1 results — frozen CPU-R6 to consolidated production route

Status: **PASS**.

## Scope
- A is the frozen CPU-R6 DLL configured by the compatible validation client. B is the consolidated CPU-U1 DLL; its optimized route is selected automatically with no launcher switches. FFN row4 remains disabled.
- Both sides use the same model, tokenizer, chat template, sampling policy and seed, with one participant pinned to CPU 0. Profiling is disabled for all speed receipts.
- This is a consolidation comparison, not a new optimization experiment. Percentages are calculated from paired A/B samples and are not summed from earlier campaigns.

## Immutable identity
- Reference runtime commit: `ae90d82e6007574628d496eba898cdffce2b8096`; DLL SHA-256: `5261fef560537f48bc215890a14288edef2f5c78bda50e7a87cb2cb88fe0daeb`.
- Candidate runtime commit: `06694632cfe4ee5b97013d6a5b2e6fec9fe98f50`; DLL SHA-256: `494eb6bc48afe1ecd41f532b6a4c3a2310ef2b8e73408fc8809b32bcf6ebca2a`.
- Measurement tree commit: `1d31ddce6b45ec2230670a45fd4a2d6115da5783`; model SHA-256: `bcab52e67825061ffc3bd4480dbda1d5067bc3a0e51aba386a236c7131766f33`.

## Benchmark command
```powershell
python native_cpu/benchmarks/cpu-u1/benchmark.py --out native_cpu/benchmarks/cpu-u1/results --pairs 6
python native_cpu/benchmarks/cpu-u1/summarize.py
```

Raw receipts are under `native_cpu/benchmarks/cpu-u1/results/`. Loading and FP16-storage preparation are recorded per entry and excluded from inference timings.

## Frozen CPU-R6 → consolidated CPU-U1
| Case | Metric | A median | B median | Median B-A | Median A→B benefit | B favorable |
|---|---|---:|---:|---:|---:|---:|
| cpu-u1-conversation-historical | prefill | 1.708860 s | 1.515453 s | -0.055566 s | 3.651% | 4/6 |
| cpu-u1-conversation-historical | decode | 8.040087 s | 7.372731 s | -0.564845 s | 6.784% | 4/6 |
| cpu-u1-conversation-historical | total | 10.156035 s | 9.370386 s | -0.517455 s | 5.014% | 4/6 |
| cpu-u1-conversation-historical | conversation wall | 10.156250 s | 9.370583 s | -0.517447 s | 5.014% | 4/6 |
| cpu-u1-conversation-historical | decode throughput | 91.885 tok/s | 99.481 tok/s | 6.665 tok/s | 7.501% | 4/6 |
| cpu-u1-conversation-second | prefill | 0.611229 s | 0.613081 s | -0.000872 s | 0.167% | 3/6 |
| cpu-u1-conversation-second | decode | 0.672494 s | 0.690492 s | 0.025794 s | -3.884% | 2/6 |
| cpu-u1-conversation-second | total | 1.319900 s | 1.435962 s | 0.119321 s | -9.137% | 3/6 |
| cpu-u1-conversation-second | conversation wall | 1.319971 s | 1.436020 s | 0.119309 s | -9.136% | 3/6 |
| cpu-u1-conversation-second | decode throughput | 141.267 tok/s | 137.606 tok/s | -5.348 tok/s | -3.738% | 2/6 |
| cpu-u1-forced-256 | prefill | 1.685737 s | 1.665871 s | -0.029373 s | 1.787% | 5/6 |
| cpu-u1-forced-256 | decode | 0.263597 s | 0.255094 s | -0.008046 s | 3.055% | 4/6 |
| cpu-u1-forced-256 | total | 1.949333 s | 1.918385 s | -0.031020 s | 1.590% | 4/6 |
| cpu-u1-forced-256 | decode throughput | 121.399 tok/s | 125.445 tok/s | 3.839 tok/s | 3.159% | 4/6 |
| cpu-u1-forced-1792 | prefill | 24.813380 s | 23.454488 s | -0.638596 s | 2.148% | 3/6 |
| cpu-u1-forced-1792 | decode | 0.545277 s | 0.750786 s | 0.168572 s | -31.080% | 1/6 |
| cpu-u1-forced-1792 | total | 25.360696 s | 24.158568 s | -0.480817 s | 1.479% | 3/6 |
| cpu-u1-forced-1792 | decode throughput | 58.686 tok/s | 42.626 tok/s | -13.761 tok/s | -23.332% | 1/6 |

## Load and preparation (reported separately)
| Case | A load median | B load median | A configuration median | B configuration median | A FP16 preparation median | B FP16 preparation median |
|---|---:|---:|---:|---:|---:|---:|
| cpu-u1-conversation-historical | 1.122626 s | 1.306978 s | 0.199589 s | 0.000000 s | 0.199542 s | 0.199326 s |
| cpu-u1-conversation-second | 1.107729 s | 1.303206 s | 0.204621 s | 0.000000 s | 0.204585 s | 0.198878 s |
| cpu-u1-forced-256 | 1.105005 s | 1.337059 s | 0.198558 s | 0.000000 s | 0.198523 s | 0.203906 s |
| cpu-u1-forced-1792 | 1.952951 s | 2.724387 s | 0.194043 s | 0.000000 s | 0.194007 s | 0.211027 s |

## Compact FFN memory
| Case | A compact bytes | B compact bytes |
|---|---:|---:|
| cpu-u1-conversation-historical | unavailable | 89,653,248 bytes |
| cpu-u1-conversation-second | unavailable | 89,653,248 bytes |
| cpu-u1-forced-256 | unavailable | 89,653,248 bytes |
| cpu-u1-forced-1792 | unavailable | 89,653,248 bytes |

The consolidated route prepares exact FP16 FFN storage during model load;
the preparation counter above is reported separately and is not included in
sustained prefill/decode timings. The retained FP32 tensors mean these bytes are
additional representation memory, not a total-RAM reduction.

## Parity and availability
- Forced-prefix receipts compare exact FP32 logits hashes for the prefill and every decode token, plus token IDs and final position.
- Conversation receipts compare canonical prompt IDs, sampled token IDs, text, finish reason and generated-token counts for each turn. All four cases use six alternating A/B pairs.
- Diagnostic counters are intentionally not required in this comparison; fields absent from the frozen DLL are reported as unavailable rather than zero.

## Paired details
### cpu-u1-conversation-historical

#### Decode time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 8.940949 s | 6.868288 s | -2.072661 s | 23.182% | True |
| 2 | B→A | 8.750164 s | 7.784978 s | -0.965186 s | 11.030% | True |
| 3 | A→B | 14.405154 s | 7.191137 s | -7.214017 s | 50.079% | True |
| 4 | B→A | 7.330010 s | 7.865626 s | 0.535616 s | -7.307% | True |
| 5 | A→B | 6.482182 s | 6.317678 s | -0.164504 s | 2.538% | True |
| 6 | B→A | 6.883635 s | 7.554325 s | 0.670691 s | -9.743% | True |

#### Total time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 11.166295 s | 8.759557 s | -2.406739 s | 21.554% | True |
| 2 | B→A | 11.024807 s | 10.204040 s | -0.820768 s | 7.445% | True |
| 3 | A→B | 16.661048 s | 9.240439 s | -7.420608 s | 44.539% | True |
| 4 | B→A | 9.287262 s | 9.858433 s | 0.571170 s | -6.150% | True |
| 5 | A→B | 8.287888 s | 8.073747 s | -0.214141 s | 2.584% | True |
| 6 | B→A | 9.129958 s | 9.500332 s | 0.370374 s | -4.057% | True |
### cpu-u1-conversation-second

#### Decode time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 0.677719 s | 0.647163 s | -0.030556 s | 4.509% | True |
| 2 | B→A | 0.675594 s | 0.806755 s | 0.131161 s | -19.414% | True |
| 3 | A→B | 0.654729 s | 0.681641 s | 0.026913 s | -4.111% | True |
| 4 | B→A | 0.670319 s | 0.653536 s | -0.016784 s | 2.504% | True |
| 5 | A→B | 0.638516 s | 0.881142 s | 0.242625 s | -37.998% | True |
| 6 | B→A | 0.674669 s | 0.699343 s | 0.024675 s | -3.657% | True |

#### Total time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 1.358682 s | 1.264833 s | -0.093849 s | 6.907% | True |
| 2 | B→A | 1.313211 s | 1.767696 s | 0.454485 s | -34.609% | True |
| 3 | A→B | 1.326590 s | 1.317112 s | -0.009478 s | 0.715% | True |
| 4 | B→A | 1.336894 s | 1.282639 s | -0.054255 s | 4.058% | True |
| 5 | A→B | 1.309660 s | 1.596618 s | 0.286958 s | -21.911% | True |
| 6 | B→A | 1.306693 s | 1.554813 s | 0.248120 s | -18.988% | True |
### cpu-u1-forced-256

#### Decode time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 0.241644 s | 0.281846 s | 0.040201 s | -16.637% | True |
| 2 | B→A | 0.262761 s | 0.252478 s | -0.010283 s | 3.914% | True |
| 3 | A→B | 0.264432 s | 0.258624 s | -0.005808 s | 2.197% | True |
| 4 | B→A | 0.248597 s | 0.250695 s | 0.002098 s | -0.844% | True |
| 5 | A→B | 0.273097 s | 0.254332 s | -0.018765 s | 6.871% | True |
| 6 | B→A | 0.442955 s | 0.255856 s | -0.187099 s | 42.239% | True |

#### Total time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 1.849668 s | 1.854829 s | 0.005161 s | -0.279% | True |
| 2 | B→A | 1.962422 s | 1.928432 s | -0.033989 s | 1.732% | True |
| 3 | A→B | 1.936245 s | 1.908194 s | -0.028051 s | 1.449% | True |
| 4 | B→A | 1.896412 s | 1.915729 s | 0.019317 s | -1.019% | True |
| 5 | A→B | 2.180178 s | 1.921041 s | -0.259137 s | 11.886% | True |
| 6 | B→A | 3.714353 s | 1.926011 s | -1.788342 s | 48.147% | True |
### cpu-u1-forced-1792

#### Decode time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 0.544868 s | 1.509804 s | 0.964937 s | -177.096% | True |
| 2 | B→A | 0.489084 s | 1.905875 s | 1.416791 s | -289.682% | True |
| 3 | A→B | 0.554035 s | 0.491953 s | -0.062082 s | 11.205% | True |
| 4 | B→A | 0.545687 s | 0.665008 s | 0.119321 s | -21.866% | True |
| 5 | A→B | 0.540597 s | 0.758420 s | 0.217823 s | -40.293% | True |
| 6 | B→A | 0.645419 s | 0.743152 s | 0.097734 s | -15.143% | True |

#### Total time
| Pair | Order | A | B | B-A | A→B | Parity |
|---:|---|---:|---:|---:|---:|---|
| 1 | A→B | 28.420790 s | 31.435356 s | 3.014566 s | -10.607% | True |
| 2 | B→A | 23.517495 s | 33.233723 s | 9.716228 s | -41.315% | True |
| 3 | A→B | 25.463310 s | 19.918606 s | -5.544704 s | 21.775% | True |
| 4 | B→A | 34.273517 s | 24.285302 s | -9.988216 s | 29.143% | True |
| 5 | A→B | 25.258081 s | 22.635259 s | -2.622822 s | 10.384% | True |
| 6 | B→A | 22.370648 s | 24.031835 s | 1.661187 s | -7.426% | True |

## Evidence verdict
Result: **PASS**.

A PASS means the four-case paired evidence is complete, directly parity-checked and identity-consistent. CPU-U1 becomes the current production baseline; preparation cost is reported separately from sustained inference.
