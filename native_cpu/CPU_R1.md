# CPU-R1: measured reductions in redundant FP32 work

CPU-R1 keeps the dense MiniMind model, weights, tokenizer, chat template,
sampling policy and FP32 accumulation kernels unchanged. The serial launcher
remains the default; experimental behavior is opt-in.

## Review sequence

1. Explicit turn/phase accounting and optional native diagnostics.
2. Selective vocabulary projections, with the all-heads reference retained.
3. Exact-token-prefix KV reuse and safe logical truncation.
4. Paired benchmarks and publicly accessible raw evidence.

The pre-change identity is recorded in
[`benchmarks/cpu-r1/baseline-identity.json`](benchmarks/cpu-r1/baseline-identity.json).
The historical `t0-affinity-v1` results are not overwritten.

## Metrics contract

Model reset changes KV state; `begin_turn()` resets statistics. Prefill and
decode phases are explicit, never inferred from how many `eval()` calls have
already occurred. Generated tokens and evaluated tokens are distinct: EOS is
sampled but not evaluated, and the final token at the generation limit is not
fed back into the model.

Per-turn JSON reports full prompt length, genuinely reused prefix length,
evaluated prefill/decode tokens, emitted tokens, sampled IDs including EOS,
phase-specific vocabulary projection counts, cache positions/action/reason,
and prompt preparation, prefill, generation, sampling and total wall times.
Unsupported counters from old libraries or other backends are unknown, not
fabricated measurements.

Native-call wall timing includes the Python/ctypes boundary and output copy;
it is not a hardware-counter measurement. Buffered prefill timing is not an
observed streaming time to first token. Sampling time is reported separately.

## Diagnostic profiling

`--diagnostics` enables optional engine profiling; it is off by default.
Diagnostic snapshots separate projections, KV attention, vocabulary output,
remaining operations, participant GEMV calculation time and controller wait.
Participant times can overlap and must not be added to produce elapsed time.
Controller wait is the interval after the caller finishes its own row shard.
Counters accumulate in memory; no files or output are written in the hot loop.

Use profiling-off runs for final speed comparisons. Diagnostic overhead must be
measured separately. No measured DRAM bandwidth, energy, or cache-miss claim is
made without actual hardware counters.
