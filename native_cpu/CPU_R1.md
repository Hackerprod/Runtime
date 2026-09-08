# CPU-R1: measured reductions in redundant FP32 work

CPU-R1 keeps the dense MiniMind model, weights, tokenizer, chat template,
sampling policy and FP32 accumulation kernels unchanged. The serial launcher
remains the default; experimental behavior is opt-in.

**Completed review package:** [Report for Sol, including exact runtime commits,
results and limitations](benchmarks/cpu-r1/REPORT_FOR_SOL.md).

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

## Selective logits (opt-in)

Pass `--selective-logits` to either native Python frontend. For a nonempty
`eval(ids)` call, every token still executes all transformer layers and commits
KV at its original position, but only the final token produces vocabulary
logits. A single-token generation call still produces one vocabulary head.
The default all-heads path remains available in the same DLL for attribution.

The C API exposes `mm_configure_selective_logits` and `mm_selective_logits`;
Python exposes `configure_selective_logits(bool)` and `selective_logits`.
An old DLL permits the disabled/default mode but rejects enabling the feature.
This does not change the model architecture or promise faster decode.

Discarded positions validate newly committed key/value state and final normalized
hidden state without calculating an unused vocabulary projection. Requested
logits retain the finite-output check. A reset or execution failure invalidates
logits, and failed batches retain the previous logical cache position. The
skipped projection itself is not evaluated for overflow at discarded positions;
the internal state, rather than a nonexistent vocabulary output, is validated.

## Persistent KV (opt-in)

Pass `--reuse-kv` to the native frontend; combine it with `--selective-logits`
to remove both repeated prompt evaluation and discarded vocabulary heads.
Both flags remain **off by default**. The original PyTorch backend rejects
native-only options rather than pretending to implement them.

`cached_token_ids` records only successfully evaluated IDs, with
`runtime.position == len(cached_token_ids)`. Every turn still renders the full
canonical prompt and applies the existing whole-pair history eviction policy.
Only its exact token-ID common prefix at identical positions may be reused.
Generated text is not assumed to re-tokenize identically. EOS and the final
length-limited token are not added to the cache unless actually evaluated.

The new `mm_truncate`/`NativeRuntime.truncate` operation only moves backward
within evaluated state. It invalidates logits and advances the cache mutation
epoch, without needing to zero discarded storage: attention respects the new
logical boundary. Even a same-position truncation invalidates logits. When
the pending prompt suffix is empty, the session rewinds one token and evaluates
the canonical last token again. It never requests empty `eval()` or stale logits.

The native handle and mutation epoch detect external cache changes, including
changes that end at the same position. Replacing the runtime invalidates reuse.
Old DLLs without truncation safely reset and evaluate the complete prompt;
metrics explicitly report zero reuse and the fallback reason.

Invalid prompts are rejected before cache/history/RNG mutation. Failures after
execution starts invalidate bookkeeping and attempt a reset; if reset also
fails, the cache remains untrusted and cannot be reused. History is committed
only after the complete response succeeds. Recovery does not add sampler calls
or rewind the RNG. `/clear` preserves the original system-message and seeded
RNG-reset behavior.

### Local comparison

From the repository root, the unchanged launcher forwards optional flags:

```bat
compare_cpu\Native_CPP.cmd
compare_cpu\Native_CPP.cmd --selective-logits --reuse-kv
```

Use the first command as the reference. Use `--diagnostics` only for profiling,
not final speed comparisons. These optimizations preserve responses; they do
not improve the checkpoint's language knowledge or conversational quality.

### Stage 3 verification and rollback

`python -m native_cpu.tools.build` builds Release and runs CTest. Run Python
regressions with `python -m pytest native_cpu/tests -q -p no:cacheprovider`.
Native tests cover scalar/AVX2, 1/2/4 participants, selective on/off, truncation,
divergent replay, invalid preflight input and post-start failures. Session tests
cover exact/short/divergent prefixes, re-tokenization, limits 0/1/normal,
immediate EOS, history eviction, clear, legacy DLLs and transactional recovery.

Stage 3 can be rolled back independently by removing cache reuse, truncation
and epoch bindings plus their tests; stage 1 diagnostics and stage 2 selective
logits remain independent. Disabling `--reuse-kv` retains the all-prompt path
without reverting any code.
