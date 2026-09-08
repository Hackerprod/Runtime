# CPU-E7: LM-head FP16 viability

This is an isolated experiment for `model.embed_tokens.weight` only. It does
not modify the runtime, embedding lookup, decode, prefill, `verify_x4`, chat,
or speculative decoding.

The driver verifies the complete `6400x768` tensor from the original
MiniMind-3 checkpoint by an exact FP16→FP32→FP16 round trip and against the
existing `MMCPU001` FP32 artifact. The C++ harness then compares:

- x1: `gemv_f32` versus `gemv_f16`;
- x4: four independent `gemv_f32` calls versus `gemv_f16_x4`;
- diagnostic: four independent `gemv_f16` calls.

All outputs are checked with `memcmp`. The harness uses two warmups, eight
timed repetitions, three alternating pairs, and one participant pinned to
CPU 0. Allocation, payload preparation, and parity checks are outside the
timed intervals.

## Reproduce

```powershell
python native_cpu/tools/build.py --build-dir native_cpu/build-e7
python native_cpu/benchmarks/cpu-e7/benchmark.py `
  --out native_cpu/benchmarks/cpu-e7/results `
  --pairs 3 --repetitions 8
```

The expected compact representation is **9,830,400 bytes** (about 9.375 MiB).
This experiment does not add it to the production runtime.
