# CPU-E6: exact FP16 attention storage integration

Status: **PASS**

- Source commit: `90bff5df79f8e1a2cdcb6ced97c94f97f6399219`
- CPU-E4 baseline DLL SHA-256: `b410e642e2249a8982a9a83b134f3d7f5e1f37c677f11d10900ed0e213cb97ab`
- CPU-E6 candidate DLL SHA-256: `ab464cc1c91aedd3b33bf995fef82657e2316123979e6f1d04d15bb4eace72fd`
- CPU 0, one participant, profiler off; three alternating pairs per workload.
- Candidate compact attention storage: **28,311,552 bytes**; preparation is reported outside timed inference (about 45–48 ms in these runs).

## Forced-prefix results

| Prefix | A prefill (s) | B prefill (s) | A decode tok/s | B decode tok/s | Parity |
|---:|---:|---:|---:|---:|:---:|
| 256 | 0.910 | 0.597 | 140.70 | 146.38 | PASS |
| 256 | 0.849 | 0.576 | 99.81 | 149.15 | PASS |
| 256 | 0.820 | 0.581 | 142.01 | 149.37 | PASS |
| 1792 | 12.846 | 10.541 | 57.58 | 63.93 | PASS |
| 1792 | 12.941 | 10.567 | 63.13 | 65.10 | PASS |
| 1792 | 12.762 | 10.478 | 62.93 | 65.00 | PASS |

Paired medians: prefill improved **32.08%** at 256 tokens and **17.94%** at 1,792 tokens. Decode time improved **4.93%** and **3.19%**, respectively; all six forced pairs favored the candidate.

## Repetitive CPU-E4 workload

| A output decode tok/s | B output decode tok/s | A verify calls | B verify calls | Parity |
|---:|---:|---:|---:|:---:|
| 158.43 | 148.45 | 4 | 4 | PASS |
| 141.94 | 134.82 | 4 | 4 | PASS |
| 161.31 | 165.12 | 4 | 4 | PASS |

The repetitive workload kept identical speculative telemetry and exact chat parity. Its paired decode timing was mixed (the candidate was slower in two of three pairs and faster in one), so this short guard does not establish a reproducible decode regression; it remains a watch item. The forced-prefix decode pairs all favored the candidate.

## Canonical guard

The A and B runs both require the canonical sampled-ID and text hashes.

| Side | Median output tok/s | Generated tokens | Verify calls |
|:---|---:|---:|---:|
| A | 140.69 | 256 | 0 |
| B | 142.20 | 256 | 0 |

FP16 preparation and compact-memory values are recorded in every raw receipt and excluded from timed inference.
