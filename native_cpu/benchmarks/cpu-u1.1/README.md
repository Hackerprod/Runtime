# CPU-U1.1 reference recheck

This harness compares the consolidated CPU-U1 runtime against the accepted CPU-R6 DLL only when both immutable SHA-256 identities match:

- Reference A: `4d598dd21493d2821a10783c4c97671c96ea038b98c3cc82c30172182fdb3672`
- Candidate B: `494eb6bc48afe1ecd41f532b6a4c3a2310ef2b8e73408fc8809b32bcf6ebca2a`

Place the accepted reference binary at:

```text
native_cpu/validation/cpu-u1.1/reference/cpu-r6-accepted.dll
```

Run the bounded protocol with three alternating pairs for forced prefixes 256 and 1792:

```powershell
python native_cpu/benchmarks/cpu-u1.1/benchmark.py --out native_cpu/benchmarks/cpu-u1.1/results --pairs 3
python native_cpu/benchmarks/cpu-u1.1/summarize.py
```

The benchmark checks both DLL hashes before creating the output directory or loading the model. `PREFLIGHT.log` records the fail-closed check against the locally available 5261fef reference; that binary is not an accepted CPU-R6 artifact and was not measured.
