# PERF-G1: canonical output throughput guard

CPU-E4 exact baseline; three fresh processes; profiler off; one participant pinned to CPU 0.

- Prompt: `Could you tell me about programming languages?`
- DLL SHA-256: `b410e642e2249a8982a9a83b134f3d7f5e1f37c677f11d10900ed0e213cb97ab`

| Run | Output tok/s | Legacy target tok/s | Prefill (s) | Decode (s) | Generated |
|---:|---:|---:|---:|---:|---:|
| 1 | 139.39 | 138.84 | 0.184 | 1.837 | 256 |
| 2 | 136.31 | 135.78 | 0.182 | 1.878 | 256 |
| 3 | 141.14 | 140.59 | 0.180 | 1.814 | 256 |

| Run | Sampled IDs SHA-256 | Text SHA-256 | Verify calls | Accepted tokens |
|---:|---|---|---:|---:|
| 1 | `de27dbd287b02a36f6116b22152359f4b5d1e13c9d0cf3f454512d40a6dcb8b6` | `bbab7af92f7db6781fe097645898ee98d3409f9367a9d764883637489bace770` | 0 | 0 |
| 2 | `de27dbd287b02a36f6116b22152359f4b5d1e13c9d0cf3f454512d40a6dcb8b6` | `bbab7af92f7db6781fe097645898ee98d3409f9367a9d764883637489bace770` | 0 | 0 |
| 3 | `de27dbd287b02a36f6116b22152359f4b5d1e13c9d0cf3f454512d40a6dcb8b6` | `bbab7af92f7db6781fe097645898ee98d3409f9367a9d764883637489bace770` | 0 | 0 |

The visible `decode_tokens_per_second` now equals output tokens divided by wall-clock decode time. Legacy target throughput is retained only as a secondary diagnostic.
