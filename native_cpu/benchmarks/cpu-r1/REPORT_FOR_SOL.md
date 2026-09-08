# CPU-R1 — complete review report for Sol

**Runtime commit to audit and reproduce: [`0a1a853282353c9fe0b18672b39be75fb1b1128f`](https://github.com/Hackerprod/Runtime/commit/0a1a853282353c9fe0b18672b39be75fb1b1128f).**

CPU-R1 preserves native responses and removes redundant work. On the audited
six-turn conversation, combined selective logits and KV reuse reduce median
summed turn time from **26.926 s to 10.533 s**. The median of the six paired
reductions is **61.20%**. A second conversation shows a **40.03%** paired
reduction. This is primarily a prefill improvement, **not faster autoregressive
decode or better model knowledge**. Four participants do not improve this run;
no profile or optimization has been promoted to the default launcher.

## 1. Repository, commits and build identity

Repository: [Hackerprod/Runtime](https://github.com/Hackerprod/Runtime), branch
`main`. The initial tree was clean; its complete identity was frozen before
edits in [baseline-identity.json](baseline-identity.json).

| Work unit | Exact commit | Delivered behavior |
|---|---|---|
| Frozen base | [`a6d06a949cb514741f100e439e283d2189685102`](https://github.com/Hackerprod/Runtime/commit/a6d06a949cb514741f100e439e283d2189685102) | Existing serial/T0 reference, copied DLL and source identity. |
| 1 — Instrumentation | [`7d5ab58792fc64fa3554f16a94179c9ca4b84410`](https://github.com/Hackerprod/Runtime/commit/7d5ab58792fc64fa3554f16a94179c9ca4b84410) | Explicit turn/phase counters; optional engine and participant profiling. |
| 2 — Selective logits | [`9a1c0ebb8d69fcaa7d6b21b32e5e9173a20f2225`](https://github.com/Hackerprod/Runtime/commit/9a1c0ebb8d69fcaa7d6b21b32e5e9173a20f2225) | Opt-in last-head-only evaluation; all-heads reference retained. |
| 3 — Persistent KV | [`0a1a853282353c9fe0b18672b39be75fb1b1128f`](https://github.com/Hackerprod/Runtime/commit/0a1a853282353c9fe0b18672b39be75fb1b1128f) | Exact token-prefix reuse, logical truncation, mutation epochs and transactional recovery. |
| 4 — Review package | The immutable revision containing this report, linked in the delivery message | Benchmark/validation scripts, tests, raw measurements and documentation only; **no runtime rebuild or code change after work unit 3**. |

The tested DLL was built from the exact C++ source bytes in work unit 3, before
that commit was packaged. Source and DLL hashes match
[stage3-build-info.json](stage3-build-info.json). All final benchmark records
identify the same runtime commit and DLL. The runtime tree remained unchanged
during measurement; stage-4 evidence/scripts were untracked until packaging.
The final delivery message supplies the exact fourth commit and verifies the
post-push tree state. A document cannot embed its own future Git commit hash;
its immutable URL is the unambiguous identity of the report revision.

| Artifact | SHA-256 |
|---|---|
| Tested DLL | `21efa3f95294d195dcc0841c3975ba31842a5f9425efb5160c5f6c38093acf99` |
| Frozen DLL | `4f9ed66fd43d0850f7d2ec17679df1b85754c665ba5c3393b1e71f7f7658c1bd` |
| FP32 model | `bcab52e67825061ffc3bd4480dbda1d5067bc3a0e51aba386a236c7131766f33` |
| Tokenizer JSON | `71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668` |
| Chat template | `97c6a9ef3f8d35044f435bb7773f1af160a7328ab6e8d2e9d67702941320a711` |
| Unchanged FP32 kernels | `1afcf2844a5917a39cc52907ce0033fd7793153863df79ee236827f7190bfdd4` |

Build: MSVC **19.50.35727.0**, Release, C++20, `/O2 /Ob2 /fp:precise`;
AVX2 auto kernel for real native route comparisons. Full flags, source hashes,
model/tokenizer provenance and packaging identity are in the build manifest,
baseline identity and [delivery-identity.json](delivery-identity.json).
Weights, binaries and credentials are not included in Git.

## 2. Implementation and invariants

### Instrumentation

`begin_turn/reset_stats` is independent of model reset/truncation. Prefill and
generation are explicitly selected, never inferred from the number of `eval`
calls. Metrics separate canonical prompt length, actual reused/evaluated
tokens, sampled IDs including EOS, emitted tokens, phase-specific vocabulary
heads, cache position/action/reason and frontend/native/sampling time.

The diagnostic profiler is off by default. It measures disjoint QKV, KV
attention, output projection, FFN GEMVs, vocabulary and remaining regions.
Participant compute and controller completion-wait are separate, overlapping
observations. The controller wait starts **after its own shard finishes**;
dispatch/notification overhead remains inside projection wall time. There is
no hot-loop output or theoretical-byte-to-DRAM-bandwidth substitution.

### Selective logits

`step(token, emit_logits)` still executes every layer and commits KV at every
position. With the opt-in switch, a nonempty batch computes only its final
vocabulary head. One-token decode still computes one head. The same DLL keeps
the all-heads reference. Discarded positions validate internal numerical state;
they do not execute the discarded projection merely to check its output.
Requested logits retain finite-output checks and stale logits are invalidated.

### Persistent KV

`cached_token_ids` contains only successfully evaluated tokens. Reuse requires
an exact common prefix of the newly rendered canonical prompt at the same
positions and `runtime.position == len(cached_token_ids)`. A native handle and
mutation epoch detect external changes even at the same final position.

`mm_truncate` rejects advances and invalid positions before mutation, limits
attention logically, invalidates logits and advances the epoch. Discarded
storage need not be zeroed. An empty pending suffix causes one-token rewind
and canonical re-evaluation, not empty evaluation or arbitrary logits reuse.

EOS and the last length-limited output token are not cached unless evaluated.
Re-tokenization and history eviction use canonical IDs, not decoded-text
matching or shifted suffix reuse. Invalid input is rejected before mutation;
post-start failures invalidate bookkeeping and attempt reset. Reset failure
leaves the cache untrusted. History commits only after full response success;
recovery does not add sampler calls. `/clear` preserves existing system/RNG
semantics. Old DLLs fall back to full reset and report zero reuse explicitly.

## 3. Correctness evidence

| Check | Result | Evidence |
|---|---|---|
| Stage 1 build/CTest/Python | CTest 2/2; 85 tests + 103 subtests | [build](stage1-build.log), [tests](stage1-pytest.log), [frozen parity](stage1-smoke.json) |
| Stage 2 build/CTest/Python | CTest 2/2; 87 tests + 103 subtests | [build](stage2-build.log), [tests](stage2-pytest.log), [fidelity](stage2-fidelity.json) |
| Stage 3 build/CTest/Python | CTest 2/2; 116 tests + 103 subtests | [build](stage3-build.log), [tests](stage3-pytest.log) |
| Final full Python suite | **143 tests + 103 subtests passed** | [full log](final-pytest-release.log) |
| Repeated native tests | Both CTest cases passed 10 consecutive runs each | [stress log](final-ctest-stress.log) |
| Native real-weight fidelity | **PASS**, 8,708 captured sampling-decision vectors; exact logits/IDs/text across both workloads, greedy and seeded routes, including frozen DLL | [raw fidelity](stage3-fidelity.json), [execution log](stage3-fidelity-canonical.log), [validator](validate.py) |
| External PyTorch | English/Spanish/Chinese; FP32 scalar and auto PASS at unchanged `atol=rtol=1e-4`, exact greedy streams | [raw result](pytorch-parity.json), [log](pytorch-parity.log) |
| Actual launchers | Default/opt-in parity, KV-enabled `/clear`, guarded no-Torch native inference, old-DLL fallback PASS | [raw result](frontend-smoke-pass.json), [script](check_frontends.py) |
| Final paired measurements | **27 cases × 6 pairs = 162 pairs / 324 entries**, every pair PASS; report independently recomputed parity and summaries | [summary](results/summary.json), [log](final-benchmark.log), [validator/generator](report.py) |

Coverage includes scalar/auto and 1/2/4 native participants, whole and
partitioned evaluation with prior KV, exact/full/short/divergent prefixes,
invalid truncation and replay, max-new-tokens 0/1/normal, immediate EOS,
length termination, re-tokenization, system-preserving history eviction,
external mutation, runtime replacement and injected transactional failures.
There is no native fidelity exception or relaxed tolerance hiding a mismatch.
PyTorch's external tolerance is not confused with native bitwise equality.

Initial harness defects and Windows sandbox permission failures were fixed
and re-run, not hidden or counted as passes. The failed/interrupted evidence
and reasons are documented under [excluded attempts](REPRODUCE.md#attempts-excluded-from-final-speed-evidence).

## 4. R0–R3 attribution: one participant, CPU 0

R0 = all heads/no reuse; R1 = selective/no reuse; R2 = all heads/reuse;
R3 = selective/reuse. Each row has its **own paired R0 reference**. Values are
median summed turn seconds; reduction is the median of six individual paired
reductions, not a ratio of separately rounded medians.

| Workload | Comparison | Reference s | Candidate s | Paired reduction | Reduction stdev, percentage points |
|---|---|---:|---:|---:|---:|
| Audited six turns | R0 → R1 | 28.063 | 26.150 | 5.70% | 2.30 |
| Audited six turns | R0 → R2 | 26.924 | 10.549 | 60.88% | 0.29 |
| Audited six turns | R0 → R3 | 26.926 | 10.533 | **61.20%** | 3.35 |
| Different two turns | R0 → R1 | 2.481 | 2.322 | 4.37% | 4.73 |
| Different two turns | R0 → R2 | 2.435 | 1.514 | 37.92% | 0.70 |
| Different two turns | R0 → R3 | 2.573 | 1.519 | **40.03%** | 6.91 |

For historical R0:R3, prefill median falls **18.142 → 1.694 s** while native
generation remains **8.368 → 8.394 s** (87.595 → 87.328 evaluated decode tok/s).
Sampling is approximately 0.390 s in both. This does **not** demonstrate an
autoregressive decode gain. The isolated R1 decode difference is not attributed
to skipping prefill heads: the generation path is unchanged.

All raw times, sample standard deviations, ranges, frontend overhead and
conversation wall times are in [RESULTS.md](RESULTS.md) and the linked pairs.

### Reconcile the audited 2,071 versus 194 tokens

| Turn | Canonical prompt | Reused R2/R3 | Executed prefill R2/R3 | Decode evaluations | Non-EOS generated | Sampled incl. EOS |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 63 | 0 | 63 | 21 | 21 | 22 |
| 2 | 110 | 84 | 26 | 213 | 213 | 214 |
| 3 | 350 | 323 | 27 | 32 | 32 | 33 |
| 4 | 403 | 382 | 21 | 0 | 0 | 1 |
| 5 | 430 | 403 | 27 | 255 | 256 | 256 |
| 6 | 715 | 685 | 30 | 212 | 212 | 213 |
| **Total** | **2,071** | **1,877** | **194** | **733** | **734** | **739** |

R0/R1 execute all 2,071 prompt tokens; R2/R3 execute 194: **90.63% less prefill
token work**, derived from the trace, not hardcoded. Prefill head counts are
R0 **2,071**, R1 **6**, R2 **194**, R3 **6**; every variant has **733** decode
heads. No empty-suffix re-evaluation was needed in this particular trace.
Turn 4 is immediate EOS; turn 5 demonstrates why generated != evaluated.

### Per-turn latency, matched R0:R3 six-pair medians

| Turn | R0 prefill s | R3 prefill s | R0 native decode s | R3 native decode s | R0 total s | R3 total s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.4402 | 0.4162 | 0.1560 | 0.1551 | 0.6094 | 0.5831 |
| 2 | 0.7784 | 0.1811 | 1.8604 | 1.8456 | 2.7494 | 2.1377 |
| 3 | 2.8666 | 0.2456 | 0.3236 | 0.3178 | 3.2097 | 0.5835 |
| 4 | 3.3770 | 0.1992 | 0.0000 | 0.0000 | 3.3788 | 0.2010 |
| 5 | 3.6689 | 0.2618 | 3.0064 | 3.0190 | 6.7974 | 3.4224 |
| 6 | 7.0002 | 0.3721 | 3.0322 | 3.0379 | 10.1528 | 3.5258 |

Medians of individual turns need not sum to the median conversation total.

## 5. S0/S1/P2/P4 scaling on R3

| Profile | Participants and affinity | Row weights |
|---|---|---|
| S0 | 1, unbound | 1 |
| S1 | 1, logical CPU 0 | 1 |
| P2 | 2, logical CPUs 0 and 6 | Historical 1332:992 |
| P4 | 4, logical CPUs 0,2,4,6 | Prespecified equal rows; not fitted from INT8 |

Live Windows topology reports four physical cores with siblings 0/1, 2/3,
4/5 and 6/7. CPU 0's core has the higher reported efficiency class. P4 uses
distinct physical cores, not four SMT siblings.

| Workload | Comparison | S1 median s | Candidate median s | Paired reduction | Reduction stdev, percentage points |
|---|---|---:|---:|---:|---:|
| Audited six turns | S1 → S0 | 10.995 | 11.826 | -8.60% | 2.01 |
| Audited six turns | S1 → P2 | 10.999 | 10.630 | **3.26%** | 2.68 |
| Audited six turns | S1 → P4 | 11.279 | 12.139 | **-5.14%** | 3.86 |
| Different two turns | S1 → S0 | 1.521 | 1.566 | -2.87% | 4.55 |
| Different two turns | S1 → P2 | 1.542 | 1.494 | 1.49% | 4.80 |
| Different two turns | S1 → P4 | 1.521 | 1.570 | -3.21% | 2.40 |

P2's gains are small and not positive in every pair. P4 is slower on the
median; keeping workers persistent did not make four-way scaling efficient.
The original serial/default launcher is unchanged.

### Forced engine workloads

Every profile evaluates the same fixed 32-token continuation after each empty
64/256/1024/1792-token prefill. This removes changing generation length as a
speed confound. [Full forced tables](RESULTS.md#forced-contexts) include each
paired baseline, prefill/decode seconds, tok/s and dispersion.

For example, at 1,792 prefix tokens S1:P2 decode is **43.210 → 44.850 tok/s**;
the independently paired S1:P4 comparison is **42.897 → 41.668 tok/s**.
These are different paired reference sets, not numbers pooled into one S1
estimate. Throughput degrades with context length, but context-specific
diagnostics beyond 256 were not collected in this delivery.

## 6. Diagnostic findings and uncertainty

Separate diagnostics use a 256-token prefill and fixed 32-token continuation.
All final speed results above use profiling **off**. Category percentages
below are medians of within-entry ratios to summed disjoint engine regions.

| Profile/phase | FFN GEMVs | KV attention | Controller completion-wait, overlapping |
|---|---:|---:|---:|
| S1 prefill | 63.89% | 14.23% | 0.02% |
| S1 decode | 51.64% | 24.97% | 0.01% |
| P2 prefill | 64.04% | 13.70% | 0.34% |
| P2 decode | 51.94% | 24.05% | 0.30% |
| P4 prefill | 61.91% | 14.66% | 9.23% |
| P4 decode | 51.21% | 24.26% | 6.76% |

Wait is already inside projection wall time and **must not be added** to the
category percentages. P4's median prefill completion-wait is **192.165 ms**,
versus **6.112 ms** for P2; decode wait is **21.177 versus 0.841 ms**.
The [full diagnostic table](RESULTS.md#diagnostics-milliseconds) reports QKV,
output, vocabulary, remaining operations and compute time for each participant
separately. It does not sum worker times as elapsed time.

| Profiling on minus off | Paired median ms | Stdev ms | Range ms |
|---|---:|---:|---:|
| S1 | -2.960 | 382.657 | -937.001 to 54.748 |
| P2 | 6.021 | 23.133 | -1.415 to 60.124 |
| P4 | -33.072 | 168.069 | -140.436 to 325.862 |

This quantifies observed overhead **and its poor precision**. Negative values
do not mean profiling accelerates execution; variability masks small overhead
in S1/P4. No unfavorable or noisy final pair was discarded. Completion-wait
does not capture all coordination overhead, and these results do not prove
DRAM saturation.

## 7. Environment, reproducibility and next step

Host: **AMD Ryzen AI 5 330**, Windows 11, Python 3.14.3. The final campaign ran
2026-09-08 **01:40:26–02:40:32 UTC**. All 648 before/after environment samples
reported AC power, battery 100% and the same High Performance power-plan GUID.
System CPU samples had median **8.1%**, maximum **91.7%**; four samples exceeded
50%. This was an active desktop, not an isolated laboratory. All pairs are
retained; no unrelated processes were closed, and no other tests, builds or
benchmarks were launched concurrently.

There are no temperature, energy, cache-miss, DRAM-traffic or bandwidth
measurements. The profiler covers the 256-token diagnostic workload, not every
context length. Model loading is excluded from entry speed metrics. Native
wall time includes Python/ctypes/copies/bookkeeping; buffered prefill latency
is not observed streaming TTFT. These limits constrain causal claims.

**Proposed next change, for Sol's review:** target the dominant FFN GEMV work
with a separately validated row/matrix-locality experiment and reduced
per-matrix coordination, keeping FP32 accumulation order and exact parity.
Before prioritizing an attention rewrite for long contexts, extend diagnostics
to 1,024/1,792 to establish its actual share there. Do not introduce an adaptive
scheduler or assume memory saturation from P4's regression.

What improved: prompt work and conversation latency, with exact native
behavior. What did not: model quality, autoregressive decode from KV/selective
alone, and four-participant scaling. What remains unmeasured: the hardware and
long-context diagnostic counters above. No part of the agreed CPU-R1 benchmark
matrix remains pending.

### Review and reproduce

1. Review the three functional commits linked at the top and the invariants in
   [CPU_R1.md](../../CPU_R1.md).
2. Follow [REPRODUCE.md](REPRODUCE.md) for exact build, tests, fidelity,
   frontend, environment and benchmark commands.
3. Audit [RESULTS.md](RESULTS.md), [162 raw paired records](results),
   [fidelity](stage3-fidelity.json), [build metadata](stage3-build-info.json)
   and [delivery identity](delivery-identity.json).
4. To compare interactively from the repository root:

```bat
compare_cpu\Native_CPP.cmd
compare_cpu\Native_CPP.cmd --selective-logits --reuse-kv
```

These are reference versus opt-in optimization; defaults were not changed.
