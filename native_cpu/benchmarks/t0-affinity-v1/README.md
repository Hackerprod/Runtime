# T0 affinity benchmark

This directory contains the curated public summary for the opt-in T0-style native CPU threading profile. The raw detailed trial reports, full generations, archived/candidate DLLs, and private process logs remain local and ignored.

## Result

Four sequential AB/BA pairs on one Ryzen AI 5 330 Windows/MSVC Release setup produced:

| Metric (median) | Archived baseline | T0 profile | Change |
|---|---:|---:|---:|
| Decode throughput | 84.6423 tok/s | 87.7577 tok/s | +3.68% |
| Total inference time | 26.6943 s | 25.2992 s | -5.23% |

All 48 conversation turns matched in text and token IDs, with 733 decode evaluation steps per complete conversation. The conservative promotion gate required both at least 5% decode improvement and at least 5% latency reduction; it **failed**, so T0 remains opt-in and the serial path remains the default.

The curated [`summary.json`](summary.json) records configuration, medians, parity, gate status, and hashes. Its evidence hashes refer to detailed local reports that are intentionally not included in this fork.

## Reproduce

Follow the build, checkpoint-download, conversion, and validation instructions in [`native_cpu/README.md`](../../README.md). A fresh clone must build its own runtime and download the pinned checkpoint; it must not assume that archived baseline DLLs or generated artifacts exist. The archived-baseline command in [`compare_cpu/README.md`](../../../compare_cpu/README.md) applies only when that archive is available locally.

The native test suite previously reported **77 tests and 103 subtests**; the CTest suite reported **2/2 passing**. These are recorded results, not a claim that generated binaries are shipped here.