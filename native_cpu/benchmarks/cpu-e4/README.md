# CPU-E4: fixed n-gram speculative decoding

CPU-E4 is the accepted automatic proposer layered on the CPU-E2 runtime. It
uses the last 12 confirmed token IDs as a fixed key, searches backward for the
most recent earlier non-overlapping match with four followers, and proposes
exactly four tokens. The target samples every proposal position with the
existing sampler and RNG; a first mismatch falls back to `step()`, while later
mismatches and EOS roll back the unaccepted KV positions. The final sampled
token is never evaluated, matching the existing `max_new_tokens` boundary.

There is no user-facing flag. Runtimes without the optional `mm_verify_x4`
symbol report the capability as unavailable and use the ordinary CPU-E2 decode.
Lookup, verification, acceptance and rollback counters are diagnostic metrics
inside each response; they do not alter sampling or model state.

The candidate was compared with the exact CPU-E2 DLL on one participant pinned
to CPU 0, profiler disabled, three AB/BA pairs per workload. Raw receipts are
under `results/`; the candidate baseline DLL is kept under `baseline/`.

Run the benchmark after building the candidate:

```powershell
python native_cpu/benchmarks/cpu-e4/benchmark.py `
  --candidate native_cpu/build-e4/Release/minimind_cpu.dll `
  --out native_cpu/benchmarks/cpu-e4/results
```

