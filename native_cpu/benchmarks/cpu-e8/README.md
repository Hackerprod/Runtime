# CPU-E8: exact FP16 LM-head integration

This benchmark compares the CPU-E6 DLL against the CPU-E8 candidate. The
candidate keeps FP32 embedding lookup and adds an automatically prepared,
bit-exact FP16 copy of `model.embed_tokens.weight` for LM-head GEMV.

The driver fails closed unless the CPU-E6 baseline has SHA-256
`ab464cc1c91aedd3b33bf995fef82657e2316123979e6f1d04d15bb4eace72fd`. It uses
one participant pinned to CPU 0, profiler off, and three alternating pairs for
forced 256/1792-token prefixes, the repetitive speculative workload, and the
canonical prompt. Preparation and the 9,830,400 compact bytes are reported
separately from inference timings.

Run after building `native_cpu/build-e8`:

```text
python native_cpu/benchmarks/cpu-e8/benchmark.py --out native_cpu/benchmarks/cpu-e8/results
```

The raw JSON receipts and `RESULTS.md` are generated under `results/`.

The recorded CPU-E8 trial is **rejected for production**: CPU-E6 remains the
only runtime baseline because the candidate regressed the long forced decode,
canonical output throughput, and the repetitive workload. The implementation
commit is retained in Git only as the auditable experiment; it is not part of
the current production runtime.
