# CPU-E4: end-to-end n-gram speculative decode

**Status: PASS — accepted as the new baseline**

The functional implementation is commit `9678375354408b98a35235fd6e81b016abdd38f6`.
The candidate DLL was built with MSVC Release from that source and has SHA-256
`b410e642e2249a8982a9a83b134f3d7f5e1f37c677f11d10900ed0e213cb97ab`. The A
reference was checked before every measurement and is the exact CPU-E2 hash
`3ff9418181d29ad9f7761e2a48866f06759027b4599ef1d024f24ac4cd3acfd5`.

Conditions were identical for both DLLs: the pinned MiniMind FP32 model and
tokenizer, one native participant on CPU 0, profiler off, temperature 0.9,
top-k 50, top-p 0.85, seed 0, and 32 generated-token slots. Each workload has
three alternating pairs. Decode time includes lookup, sampling, verification,
rollback and ordinary steps; prompt preparation and prefill are reported
separately.

## Paired results

| Workload | A decode median | B decode median | Median decode-time reduction | Median total-time reduction | Favorable decode pairs |
|---|---:|---:|---:|---:|---:|
| ordinary (two existing prompts) | 0.444353 s | **0.418251 s** | **+4.331%** | **+3.845%** | 2/3 |
| repetitive (three repeated prompts) | 0.672724 s | **0.606834 s** | **+10.521%** | **+4.728%** | 3/3 |

The ordinary paired decode-time reductions are +4.331%, −2.367% and +5.874%.
The repetitive reductions are +10.521%, +10.854% and +5.444%. The ordinary
sample contains one small adverse pair but no reproducible regression; the
repetitive workload improves in all three pairs. The corresponding median
decode-throughput changes are +5.116% and +12.622%.

Candidate speculative telemetry totals:

| Workload | Proposed blocks | Proposed tokens | Accepted tokens | Verify calls | First-position failures | Mean accepted length |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 3 | 12 | 0 | 0 | 3 | 0.00 |
| repetitive | 18 | 72 | 48 | 12 | 6 | 2.67 |

All six pair comparisons have exact parity for prompt IDs, sampled IDs, text,
finish reason, generated-token count and final KV position. The focused suite
passes the no-match, 4/4 acceptance, mismatches at positions 1–4, EOS rollback,
generation limit, old-runtime fallback and RNG-state checks. The complete
Python/native regression suite passes **131 tests and 106 subtests**; CTest is
**2/2**.

The accepted candidate baseline is `baseline/minimind_cpu.dll`, with the same
hash above. Previous CPU-E2 evidence and binaries remain intact as historical
references.
