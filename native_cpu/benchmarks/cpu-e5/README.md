# CPU-E5 — exact FP16 attention projection viability

This is an isolated feasibility experiment for the eight MiniMind-3 layers of
`q_proj`, `k_proj`, `v_proj`, and `o_proj`. It does **not** modify the native
runtime or connect to `step()`, prefill, chat, or speculative decode.

The driver first verifies every checkpoint value by an exact FP16→FP32→FP16
round-trip and checks that the existing `MMCPU001` FP32 artifact contains the
same values. It then writes a temporary payload for the C++ harness. The
payload is removed after the benchmark; the checkpoint and model artifact are
never modified.

## Reproduce

```powershell
python native_cpu/tools/build.py --build-dir native_cpu/build-e5
python native_cpu/benchmarks/cpu-e5/benchmark.py `
  --out native_cpu/benchmarks/cpu-e5/results `
  --pairs 3 --repetitions 8
python native_cpu/benchmarks/cpu-e5/summarize.py
```

The harness pins its single participant to CPU 0 and records raw JSONL-derived
receipts. Each row covers all eight matrices of one projection family:

- x1: `gemv_f32` versus `gemv_f16`;
- x4: four independent `gemv_f16` calls versus `gemv_f16_x4`.

Preparation, allocations, verification, and warmups are outside timed
intervals. Exact output parity is checked with `memcmp` before and after each
timed pair. A favorable isolated result is not a production integration or a
model-level tokens/s claim.
