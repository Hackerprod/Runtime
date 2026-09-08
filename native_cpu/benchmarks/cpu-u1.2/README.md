# CPU-U1.2 reproducible baseline

CPU-U1.2 freezes the consolidated CPU-U1 runtime without comparing it to CPU-R6. The tracked baseline DLL was rebuilt from source commit `06694632cfe4ee5b97013d6a5b2e6fec9fe98f50` and has SHA-256:

```text
5956ff064ad8e8fd1ff502d18626ce19240682faa55c584cb3d6eaad1775bb3a
```

The build identity and compiler flags are in `build-info.json`; the exact DLL is kept under `baseline/minimind_cpu.dll` so this baseline cannot disappear like the historical CPU-R6 artifact.

Run the two forced-token cases with three repetitions each, using one participant pinned to CPU 0 and profiling disabled:

```powershell
python native_cpu/benchmarks/cpu-u1.2/benchmark.py --out native_cpu/benchmarks/cpu-u1.2/results --repeats 3
python native_cpu/benchmarks/cpu-u1.2/summarize.py
```

The harness validates the source commit recorded in `build-info.json`, the DLL hash, fixed-token output hashes across repetitions, and the automatic CPU-U1 effective route before accepting the report.
