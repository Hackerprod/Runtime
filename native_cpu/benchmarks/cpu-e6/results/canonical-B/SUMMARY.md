# PERF-G1: canonical output throughput guard

CPU-E4 exact baseline; three fresh processes; profiler off; one participant pinned to CPU 0.

- Prompt: `Could you tell me about programming languages?`
- DLL SHA-256: `ab464cc1c91aedd3b33bf995fef82657e2316123979e6f1d04d15bb4eace72fd`

| Run | Output tok/s | Legacy target tok/s | Prefill (s) | Decode (s) | Generated |
|---:|---:|---:|---:|---:|---:|
| 1 | 142.20 | 141.64 | 0.131 | 1.800 | 256 |
| 2 | 144.07 | 143.51 | 0.128 | 1.777 | 256 |
| 3 | 141.84 | 141.29 | 0.129 | 1.805 | 256 |

| Run | Sampled IDs SHA-256 | Text SHA-256 | Verify calls | Accepted tokens |
|---:|---|---|---:|---:|
| 1 | `de27dbd287b02a36f6116b22152359f4b5d1e13c9d0cf3f454512d40a6dcb8b6` | `bbab7af92f7db6781fe097645898ee98d3409f9367a9d764883637489bace770` | 0 | 0 |
| 2 | `de27dbd287b02a36f6116b22152359f4b5d1e13c9d0cf3f454512d40a6dcb8b6` | `bbab7af92f7db6781fe097645898ee98d3409f9367a9d764883637489bace770` | 0 | 0 |
| 3 | `de27dbd287b02a36f6116b22152359f4b5d1e13c9d0cf3f454512d40a6dcb8b6` | `bbab7af92f7db6781fe097645898ee98d3409f9367a9d764883637489bace770` | 0 | 0 |

The visible `decode_tokens_per_second` now equals output tokens divided by wall-clock decode time. Legacy target throughput is retained only as a secondary diagnostic.
