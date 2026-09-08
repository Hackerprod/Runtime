# CPU-E3: four-token verification viability

Status: **PASS as an experiment; CPU-E2 remains the production baseline**

The candidate was built from commit `bc5f304ee8b5ecf24790ffae64b0272b27ad89e3`.
Its DLL SHA-256 is
`fa02d0b58ec08904b70aaedd24099a578186d921c26058460780094c43c8e1f2`.
The A artifact was checked before running and is exactly the CPU-E2 baseline
`3ff9418181d29ad9f7761e2a48866f06759027b4599ef1d024f24ac4cd3acfd5`.

## Protocol

- Three alternating pairs for each prepared context: 256 and 1,792 tokens.
- One participant pinned to CPU 0, row weight 1, profiler disabled.
- Context preparation is outside the measured interval.
- A: four one-token `step()` evaluations. B: the test-only `mm_verify_x4`
  hook, returning four complete vocabulary-logit vectors.
- No n-gram lookup, sampler change, chat integration or production selector.

## Results

| Prepared context | A: four steps | B: verify_x4 | Paired reduction | Favorable pairs |
|---:|---:|---:|---:|---:|
| 256 | 39.732 ms | **17.457 ms** | **56.73%** | 3/3 |
| 1,792 | 80.296 ms | **62.542 ms** | **24.87%** | 3/3 |

Per-pair reductions were 28.11%, 70.68%, 56.73% at 256 and 22.11%, 41.86%,
24.87% at 1,792. The medians are calculated from the raw pair receipts and
are not ratios of independently sampled medians.

All six pairs passed exact hashes for each of the four logits, token IDs and
final positions. The focused test additionally compared continuation logits
after verification and after truncating the speculative state at 1, 2 and 3
accepted tokens. The six raw pair JSON files and derived summaries are in
[`results/`](results/); build identity is in [`build-info.json`](build-info.json).

## Decision

The necessary condition is satisfied: verifying four known tokens is faster than
four sequential decode steps in both tested contexts, with exact parity. This
does **not** yet implement n-gram lookup or speculative sampling. CPU-E3 remains
an internal verification capability only; CPU-E2 is still the sole production
baseline until an end-to-end proposer/acceptance experiment is completed.

