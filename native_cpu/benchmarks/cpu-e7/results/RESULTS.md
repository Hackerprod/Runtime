# CPU-E7: LM-head FP16 viability

Status: **PASS (isolated experiment; not integrated)**

- Source commit: `77b56f0c006d4a46e21b23534ab37fb5dd891451`
- Checkpoint SHA-256: `3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8`
- MMCPU001 artifact SHA-256: `bcab52e67825061ffc3bd4480dbda1d5067bc3a0e51aba386a236c7131766f33`
- Tensor: `model.embed_tokens.weight`, **6400×768**, **4,915,200 values**.
- FP16 round-trip mismatches: **0**; artifact FP32 mismatches: **0**.
- CPU 0, one participant, two warmups, eight repetitions, three alternating pairs.
- Expected additional FP16 storage: **9,830,400 bytes** (~9.375 MiB).

## Kernel comparison

| Path | Reference median | Candidate median | Paired reduction | Favorable |
|---|---:|---:|---:|---:|
| x1 `gemv_f32` → `gemv_f16` | 3.8311 ms | **2.7044 ms** | **29.41%** | 3/3 |
| x4 four `gemv_f32` → `gemv_f16_x4` | 13.9760 ms | **4.4194 ms** | **68.80%** | 3/3 |
| x4 four `gemv_f16` → `gemv_f16_x4` (diagnostic) | 11.4377 ms | **4.4194 ms** | **61.36%** | 3/3 |

The three raw pairs are stored in `pair01.json`–`pair03.json`. Every x1 and
x4 output passed exact `memcmp`; the independent FP16 diagnostic also matched
the shared x4 output bit for bit. Preparation and verification were outside
the timed intervals.

## Identity

- Benchmark executable SHA-256: `64c8c51201fa39a3a2bdda080ede9c539a268dda0588d607f53682df9db31d10`.
- F16C was available at runtime.
- This delivery does **not** modify `step()`, `prefill_x4`, `verify_x4`, the
  embedding lookup, chat, or speculative decoding.

## Decision

The LM-head tensor is exactly reconstructible and both x1 and x4 kernels are
faster in all three pairs. CPU-E7 is retained as the candidate for the next
single integration experiment; no production route was changed here.
