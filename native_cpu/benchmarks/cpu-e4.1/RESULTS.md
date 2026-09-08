# CPU-E4.1: long-context no-hit guard

**Status: PASS — no clear reproducible lookup regression**

The harness is commit `ce74d74f20140aab003966f2b6396945f16ec7f0`. It checks A
against the exact CPU-E2 SHA-256
`3ff9418181d29ad9f7761e2a48866f06759027b4599ef1d024f24ac4cd3acfd5` and B
against the CPU-E4 SHA-256
`b410e642e2249a8982a9a83b134f3d7f5e1f37c677f11d10900ed0e213cb97ab`.

The workload uses a 1,792-token natural prefix with a unique final key and
poisoned followers, then generates 32 tokens with temperature 0.9, top-k 50,
top-p 0.85 and seed 0. Both sides use one participant pinned to CPU 0 and the
profiler is disabled. Context preparation is excluded from decode timing.

## Results

| Metric | CPU-E2 A | CPU-E4 B |
|---|---:|---:|
| Decode median | 0.464022 s | 0.477225 s |
| Context preparation median (outside interval) | 11.669906 s | 11.710683 s |
| Native generation median | 0.446933 s | 0.457302 s |

Paired decode-time changes (positive means B is faster) were **−1.949%**,
**+7.679%** and **−18.290%**; one of three pairs favored B and the median was
**−1.949%**. The large adverse pair is not repeated in the other two pairs and
is far larger than the measured lookup cost, so this short guard does not show
a reproducible lookup penalty. It is a regression guard, not a speed claim.

The candidate performed **0 verify calls and accepted 0 tokens in all three
pairs**. Each run made 9 first-position rejections (36 proposed tokens), and
the complete lookup time was 2.857, 2.918 and 3.060 ms respectively. Thus the
long-context no-hit path exercised the linear lookup without entering
speculative verification.

All sampled IDs, generated IDs, initial logits, continuation-probe logits and
final positions matched exactly in every pair. Each run produced all 32
sampled tokens and ended at position 1,823, preserving the existing final-token
boundary. Raw JSON receipts are in `results/`.

No runtime, model, sampler, result from CPU-E4, hash or protocol was modified;
CPU-E4 remains the candidate ready to replace CPU-E2 as the single baseline
after audit.

