# CPU-E5 results — exact FP16 attention projection viability

Evidence status: **PASS**.

## Scope
- The original MiniMind-3 checkpoint stores q/k/v/o projection weights as FP16. Every value from all 8 layers is checked by FP16→FP32→FP16 bit round-trip and against the existing MMCPU001 FP32 artifact.
- This delivery is isolated to `gemv_f16`, four independent `gemv_f16` calls, and `gemv_f16_x4`. It does not modify or call `step()`, prefill, chat, speculative decode, workers or runtime configuration.
- The x1 reference is `gemv_f32`; the x1 candidate is `gemv_f16`. The x4 reference is four independent `gemv_f16` calls, matching CPU-E1; the x4 candidate is `gemv_f16_x4`.
- Each receipt benchmarks all eight matrices of one projection family in aggregate, with distinct deterministic inputs, two warmups, three alternating pairs, one participant pinned to CPU 0, and preparation/parity outside timing.

## Identity
- Source commit: `f02d448b21c062d5f1ed301e68c3e7bf264b5f1d`.
- Checkpoint SHA-256: `3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`; MMCPU001 artifact SHA-256: `bcab52e67825061ffc3bd4480dbda1d5067bc3a0e51aba386a236c7131766f33`.
- Benchmark executable SHA-256: `2242e0edacc8705f7f1ec480dfb368757813f707125744ef839f6b7033daa8ab`.
- Matrices: `32`; elements: `14155776`; FP16 round-trip mismatches: `0`; artifact mismatches: `0`.
- Repetitions: `8`; warmups: `2`; CPU: `0`; participants: `1`.

## Command
```powershell
python native_cpu/benchmarks/cpu-e5/benchmark.py --out native_cpu/benchmarks/cpu-e5/results --pairs 3 --repetitions 8
python native_cpu/benchmarks/cpu-e5/summarize.py
```

## Aggregate timing — all eight matrices per projection
| Projection | Shape | x1 FP32 median | x1 FP16 median | x1 median benefit | x1 favorable | x4 independent FP16 median | x4 shared FP16 median | x4 median benefit | x4 favorable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q_proj | 768×768 | 0.002748 s | 0.002514 s | 8.952% | 3/3 | 0.010188 s | 0.004113 s | 59.969% | 3/3 |
| k_proj | 384×768 | 0.001311 s | 0.001247 s | 4.768% | 3/3 | 0.004993 s | 0.002015 s | 59.587% | 3/3 |
| v_proj | 384×768 | 0.001347 s | 0.001274 s | 6.805% | 3/3 | 0.005117 s | 0.002016 s | 59.697% | 3/3 |
| o_proj | 768×768 | 0.002743 s | 0.002467 s | 10.307% | 3/3 | 0.009788 s | 0.004046 s | 58.610% | 3/3 |

## Raw paired totals
| Projection | Pair | x1 FP32 | x1 FP16 | x1 benefit | x4 independent | x4 shared | x4 benefit |
|---|---:|---:|---:|---:|---:|---:|---:|
| q_proj | 1 | 0.002746 s | 0.002514 s | 8.428% | 0.010703 s | 0.004186 s | 60.894% |
| q_proj | 2 | 0.003188 s | 0.002746 s | 13.852% | 0.010188 s | 0.004113 s | 59.632% |
| q_proj | 3 | 0.002748 s | 0.002502 s | 8.952% | 0.010108 s | 0.004046 s | 59.969% |
| k_proj | 1 | 0.001311 s | 0.001247 s | 4.807% | 0.004998 s | 0.002012 s | 59.740% |
| k_proj | 2 | 0.001306 s | 0.001244 s | 4.762% | 0.004993 s | 0.002018 s | 59.587% |
| k_proj | 3 | 0.001315 s | 0.001252 s | 4.768% | 0.004985 s | 0.002015 s | 59.578% |
| v_proj | 1 | 0.001333 s | 0.001242 s | 6.805% | 0.004982 s | 0.002015 s | 59.561% |
| v_proj | 2 | 0.001347 s | 0.001274 s | 5.355% | 0.005117 s | 0.002062 s | 59.697% |
| v_proj | 3 | 0.001380 s | 0.001283 s | 7.007% | 0.005144 s | 0.002016 s | 60.813% |
| o_proj | 1 | 0.002827 s | 0.002467 s | 12.712% | 0.009986 s | 0.004046 s | 59.482% |
| o_proj | 2 | 0.002743 s | 0.002470 s | 9.961% | 0.009788 s | 0.004062 s | 58.501% |
| o_proj | 3 | 0.002717 s | 0.002436 s | 10.307% | 0.009763 s | 0.004041 s | 58.610% |

## Decision
Evidence: **PASS**. This report only establishes exact reconstructibility and isolated kernel measurements; it does not integrate FP16 attention storage into production.

The next integration decision must use these raw receipts: a clear, reproducible advantage in both x1 and x4 is required before connecting the compact attention weights to CPU-E4. Mixed or negative results remain experimental and are not promoted.

Raw receipts: `native_cpu/benchmarks/cpu-e5/results/*.json`; reconstruction: `native_cpu/benchmarks/cpu-e5/results/reconstruction.json`.
