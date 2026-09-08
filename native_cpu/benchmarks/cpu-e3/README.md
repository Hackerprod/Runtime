# CPU-E3: verify four known tokens

This experiment tests the necessary condition for model-free speculative
decoding: verifying four known tokens from a prepared KV state faster than four
ordinary one-token decode calls.

The pinned A artifact is the CPU-E2 baseline
`3ff9418181d29ad9f7761e2a48866f06759027b4599ef1d024f24ac4cd3acfd5`.
The candidate exposes only the test-only `mm_verify_x4` hook; production
`mm_eval`, sampling and chat are unchanged. Context preparation is outside the
timed interval.

Run after building the candidate DLL:

```powershell
python native_cpu/benchmarks/cpu-e3/benchmark.py `
  --candidate native_cpu/build-e3/Release/minimind_cpu.dll `
  --out native_cpu/benchmarks/cpu-e3/results
```

