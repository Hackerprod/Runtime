# Reproduce CPU-R1

CPU-R1 compares unchanged dense MiniMind FP32 weights in the same native DLL.
The default launcher, FP32 accumulation kernels, tokenizer and chat policy are
unchanged. Selective logits, KV reuse and diagnostics are opt-in.

## Build and tests

Commands below run from the repository root in PowerShell on the tested Windows
machine. Python is `C:\Python314\python.exe`; MSVC and CMake discovery are handled
by the build tool. Model files are intentionally not stored in Git.

```powershell
C:\Python314\python.exe -m native_cpu.tools.build
C:\Python314\python.exe -m pytest native_cpu/tests native_cpu/benchmarks/cpu-r1 -q -p no:cacheprovider
```

The host sandbox can deny access to pytest-created temporary directories. The
final full test run used an approved unsandboxed process and a fresh explicit
`--basetemp native_cpu/validation/cpu-r1/pytest-final-04`. Earlier permission
failures are preserved in logs, not reported as functional test failures.

## Fidelity (not speed)

```powershell
C:\Python314\python.exe native_cpu/benchmarks/cpu-r1/validate.py --workload all --output native_cpu/validation/cpu-r1/reproduced-fidelity.json
C:\Python314\python.exe native_cpu/benchmarks/cpu-r1/check_frontends.py --output native_cpu/validation/cpu-r1/reproduced-frontends.json
C:\Python314\python.exe -m native_cpu.tools.validate_parity --library native_cpu/build/Release/minimind_cpu.dll --fp32-model native_cpu/artifacts/minimind-fp32.bin --fp32-manifest native_cpu/artifacts/minimind-fp32.bin.manifest.json --checkpoint-dir checkpoints/minimind-3-hf --build-info native_cpu/build/build-info.json --bench-repeats 0 --decode-tokens 16 --output native_cpu/validation/cpu-r1/reproduced-pytorch.json
```

The native validator compares every sampling-decision vector bitwise, along
with sampled IDs (including EOS), canonical prompt IDs and text. It covers
greedy and seeded generation, both conversations, R0/R1/R2/R3/P2/P4 and the
frozen reference. It also compares whole/token/partitioned evaluation with a
previous prefix and logical truncation. On failure it emits the first differing
turn, actual position, token and full logit vectors. Successful runs retain
hashes rather than all vectors. These instrumented fidelity times are **not**
used as final performance results.

PyTorch is a separate external comparison: its established `atol=rtol=1e-4`
and exact greedy token-stream requirements are unchanged. Q4 is out of scope.

`validate.py` and the legacy frontend check require the frozen DLL at
`native_cpu/validation/cpu-r1/frozen/minimind_cpu.dll`. It was copied before
changes from base commit `a6d06a949cb514741f100e439e283d2189685102`. To reproduce
on another machine, build that commit in a separate checkout and supply its
DLL through `validate.py --frozen`; the local legacy frontend check expects
the default path. A rebuild may have a different binary hash: record it, do not
pretend it is the preserved binary. The original frozen identity and compiler
settings are in [baseline-identity.json](baseline-identity.json).

## Final benchmark

```powershell
C:\Python314\python.exe native_cpu/benchmarks/cpu-r1/environment.py --output native_cpu/validation/cpu-r1/reproduced-environment.json
C:\Python314\python.exe native_cpu/benchmarks/cpu-r1/benchmark.py --pairs 6 --warmups 1 --suite all --out native_cpu/validation/cpu-r1/reproduced-results
C:\Python314\python.exe native_cpu/benchmarks/cpu-r1/report.py --results native_cpu/validation/cpu-r1/reproduced-results --output native_cpu/validation/cpu-r1/reproduced-report.md
```

Use a fresh output path. `--resume` accepts only matching identities, case,
pair number, order and prior PASS status. The report independently recomputes
raw parity and summaries; it refuses an incomplete or unbalanced final matrix.

The final submitted run uses `native_cpu/benchmarks/cpu-r1/results` and logs to
[final-benchmark.log](final-benchmark.log). No other benchmark, build, test suite
or model inference was run concurrently. Normal desktop applications were
not terminated. AC/battery, active power-plan GUID and sampled system CPU load
are recorded before and after each profile; physical topology is captured
separately. No static weights were fitted during this delivery.

### Fixed protocol

| Boundary | Configuration |
|---|---|
| Attribution | R0: all heads/no reuse; R1: selective/no reuse; R2: all heads/reuse; R3: selective/reuse. All pinned to CPU 0. |
| Scaling | R3: S0 unbound; S1 CPU 0; P2 CPUs 0,6, rows 1332:992; P4 CPUs 0,2,4,6, equal rows. |
| Pairing | R0:R1/R2/R3 and S1:S0/P2/P4 on each conversation; six alternating AB/BA pairs each. |
| Conversations | Exact six audited programming prompts, maximum 256; two different prompts, maximum 64. |
| Sampling | Unchanged Chinese system message, temperature 0.9, top-k 50, top-p 0.85, seed 0, context 2048. |
| Forced engine load | Prefixes 64/256/1024/1792, identical prespecified 32-token continuation across profiles and context lengths; six scaling pairs each. |
| Warmup | One 64-token prefill and eight one-token evaluations, followed by model/stat reset, per entry. |
| Diagnostics | Separate six off/on pairs for S1/P2/P4, prefix 256 + fixed 32 continuation. |

All final speed comparisons have diagnostics off. Native-call timing includes
Python/ctypes, copying and session bookkeeping; it is not pure kernel time.
Forced-loop hashing is outside individual timed native calls and its extra
wall time is reported separately. Conversation timing separately records
prompt preparation, prefill, native generation, sampling, remaining frontend
time, summed turn time and conversation wall time. Prefill throughput uses
only tokens actually evaluated, never the reused history length.

Diagnostics report disjoint engine regions and, separately, overlapping
participant GEMV compute plus controller completion-wait. Worker times must
not be added to elapsed time. Completion-wait is only the period after the
caller finishes its own shard; dispatch and notification overhead remain
inside projection wall time. Hardware traffic, cache misses, energy and
temperature are not measured; these data cannot establish DRAM saturation.

## Attempts excluded from final speed evidence

- The initial fidelity scaffold used different prompts under a historical
  label. It was interrupted, the prompt constants corrected and regression
  tested; the full canonical run passed. The interrupted log is
  [stage3-fidelity.log](stage3-fidelity.log). No final benchmark pair used it.
- The first frontend smoke attempt failed because the harness quoted a Windows
  launcher path incorrectly with a quoted prompt. The unchanged launcher
  passed after the harness used its repository-relative path. Both
  [failed](frontend-smoke.json) and [passing](frontend-smoke-pass.json) reports
  are retained.
- One-pair short warmups of the benchmark harness were explicitly exploratory,
  stored outside the final results directory and never included in medians.
- Pytest temporary-directory permission failures were resolved by executing
  the same tests with approved permissions, not by skipping tests.

The report's medians are comparison-specific. Do not pool different workloads
or different S1 reference sets into a fabricated paired estimate. Historical
`t0-affinity-v1` evidence is preserved unchanged.
