# CPU-E2: four-token prefill experiment

This bounded experiment compares the automatic four-token prefill path against
the pinned CPU-U1.2 DLL. It uses the existing FP16/FP32 `gemv_f16_x4` kernel
only for complete groups of four prefill positions; one-token decode remains on
the ordinary `step()` path. There is no public selector.

The harness refuses to run unless the reference artifact is exactly
`5956ff064ad8e8fd1ff502d18626ce19240682faa55c584cb3d6eaad1775bb3a`.
It records three alternating pairs for prefixes 256 and 1,792 with one
participant pinned to CPU 0 and profiling disabled. Raw pair receipts and the
derived summaries are written to this directory. Preparation and loading are
reported separately from inference.

Run after building the candidate DLL:

```powershell
python native_cpu/benchmarks/cpu-e2/benchmark.py `
  --candidate native_cpu/build-e2/Release/minimind_cpu.dll `
  --out native_cpu/benchmarks/cpu-e2/results
```

