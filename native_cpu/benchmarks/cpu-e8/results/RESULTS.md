# CPU-E8: exact FP16 LM-head integration

Status: **REJECTED**

- Source commit: `b420725d00f39117c905dfc3b33db9d40a35e492`
- CPU-E6 baseline DLL SHA-256: `ab464cc1c91aedd3b33bf995fef82657e2316123979e6f1d04d15bb4eace72fd`
- CPU-E8 candidate DLL SHA-256: `7099f5e6a763f2071d2109bfb3a818be0e5ae1687fc1457261da345f63510b63`
- CPU 0, one participant, profiler off; three alternating pairs per workload.

## Forced-prefix results

| Prefix | A prefill (s) | B prefill (s) | A decode tok/s | B decode tok/s | Parity |
|---:|---:|---:|---:|---:|:---:|
| 1792 | 14.208 | 13.752 | 42.91 | 44.43 | PASS |
| 1792 | 13.408 | 13.717 | 46.14 | 41.47 | PASS |
| 1792 | 13.239 | 13.905 | 43.94 | 42.71 | PASS |
| 256 | 0.703 | 0.705 | 96.62 | 112.61 | PASS |
| 256 | 0.672 | 0.659 | 125.35 | 137.10 | PASS |
| 256 | 0.691 | 0.676 | 118.40 | 118.92 | PASS |

## Repetitive speculative workload

| A decode tok/s | B decode tok/s | A verify calls | B verify calls | A accepted | B accepted | Parity |
|---:|---:|---:|---:|---:|---:|:---:|
| 109.57 | 103.68 | 4 | 4 | 16 | 16 | PASS |
| 107.78 | 104.37 | 4 | 4 | 16 | 16 | PASS |
| 102.90 | 105.68 | 4 | 4 | 16 | 16 | PASS |

## Canonical A/B pairs

| Pair | A output tok/s | B output tok/s | A verify | B verify | Parity |
|---:|---:|---:|---:|---:|:---:|
| 1 | 92.13 | 89.66 | 0 | 0 | PASS |
| 2 | 97.85 | 89.66 | 0 | 0 | PASS |
| 3 | 98.46 | 95.36 | 0 | 0 | PASS |

LM-head compact representation: 9830400 bytes; preparation is reported separately and excluded from timed inference.

## Verdict

- Accepted for production: **no**
- Criteria: `{"canonical_median_output_improves": false, "forced_1792_decode_improves": false, "forced_256_decode_improves": true, "parity": true, "repetitive_no_reproducible_regression": false}`

A rejected result preserves the raw receipts but does not change the CPU-E6 production baseline.
