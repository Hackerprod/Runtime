# CPU-E4.1: long-context no-hit guard

This bounded guard checks the cost of CPU-E4's fixed 12-to-4 lookup at a
1,792-token context when no proposed block is accepted. A natural deterministic
prefix is finished with a unique paragraph and poisoned followers so the
candidate can exercise lookup and first-position rejection without calling
`verify_x4`. Context preparation is outside the timed interval.

The exact CPU-E2 DLL is A and the published CPU-E4 baseline DLL is B. Both use
one participant pinned to CPU 0, profiler off, the existing sampling policy and
32 generated-token slots. Three AB/BA pairs are written to `results/`.

```powershell
python native_cpu/benchmarks/cpu-e4.1/benchmark.py `
  --out native_cpu/benchmarks/cpu-e4.1/results
```

