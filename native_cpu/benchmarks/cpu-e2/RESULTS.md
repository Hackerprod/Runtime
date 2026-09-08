# CPU-E2: prefill x4 result

Status: **PASS — integrated as the next production baseline**

The candidate was built from commit `085f16ff47b552e239bc4ffb735c796eb89106d1`.
The candidate DLL SHA-256 is
`3ff9418181d29ad9f7761e2a48866f06759027b4599ef1d024f24ac4cd3acfd5`.
The A reference was checked before measuring and is the exact CPU-U1.2 artifact:
`5956ff064ad8e8fd1ff502d18626ce19240682faa55c584cb3d6eaad1775bb3a`.
Build details are in [`build-info.json`](build-info.json).

## Protocol

- Three alternating A/B pairs for each prefix (256 and 1,792 tokens).
- One native participant pinned to CPU 0, row weight 1.
- Same model, forced token IDs, 32-token continuation and warm-up as CPU-U1.2.
- Profiler disabled for speed; loading and FP16 preparation are outside the
  inference timing.
- A complete group of four prefill positions uses `gemv_f16_x4` per FFN
  projection. Remainders of 1–3 positions and every one-token decode use the
  existing `step()` route. No public selector was added.

## Results

| Prefix | A prefill | B prefill | Paired prefill reduction | A decode | B decode | Paired decode reduction | Prefill pairs | Decode pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.918316 s | 1.042605 s | **45.91%** | 0.292342 s | 0.306594 s | **−4.38%** | 3/3 | 1/3 |
| 1,792 | 23.628830 s | 16.362637 s | **30.67%** | 0.666301 s | 0.677980 s | **−1.85%** | 3/3 | 0/3 |

The medians above are computed from the three per-pair receipts; the paired
reductions are the median of each pair's reduction, not a ratio of medians.
Prefill throughput increased from 133.45 to 245.54 tok/s at 256 and from 75.84
to 109.52 tok/s at 1,792. Decode was not changed: the small 0.014 s / 0.012 s
slower medians are treated as run-to-run noise, not an optimization claim, and
remain explicitly reported rather than hidden.

All six pairs produced identical FP32 prefill logits, identical 32 decode-logit
hash sequences, identical token IDs and final positions. The focused exact test
also covers blocks of 1, 2, 3, 4, 5, 7, 8, 9, 16, 32, 64 and 256 tokens plus
KV truncation/restart. Raw receipts are the six `*-pairNN.json` files in this
directory; `summary.json` contains their identities and derived values.

## Decision

The prefill improvement is large and reproducible in both requested contexts,
with exact parity. CPU-E2 is therefore kept as the automatic runtime behavior
and replaces CPU-U1.2 as the current baseline. Decode remains the unchanged
single-token path; no claim of decode acceleration is made.

