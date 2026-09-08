# CPU-R4.1 results — statistics ABI compatibility

Status: **accepted**.

CPU-R4 remains the accepted shared-K Q·K optimization. CPU-R4.1 changes only
the statistics boundary; it does not change attention computation, kernels,
weights, defaults, or historical CPU-R4 measurements.

## Compatibility fix

`MmRuntimeStats` is restored to its pre-CPU-R4 layout and size (1,136 bytes on
x64). The Q·K timer is kept as an internal counter and is exposed through the
optional `mm_get_attention_qk_ns(void*, uint64_t*)` function, which returns
zero on success and `-1` on invalid arguments/runtime. The Python binding adds
the value when that symbol exists and reports `attention_qk_ns: null` when it
does not. A DLL that exposes CPU-R4 shared-K symbols but lacks the new query
symbol is rejected before `mm_get_stats()` is called.

## Validation

| Check | Result |
|---|---|
| C++ legacy-layout canary (old client buffer, corrected DLL) | PASS; canaries intact and historical offsets unchanged |
| Q·K query, invalid arguments, and reset | PASS |
| Corrected client + corrected DLL | PASS; Q·K read/reset |
| Corrected client + frozen compatible DLL (`cpu-r1/frozen`) | PASS; Q·K unavailable, not zero |
| Corrected client + published incompatible CPU-R4 DLL | PASS; explicit ABI rejection |
| Brief corrected/frozen inference parity | PASS; exact logits, max abs error 0.0 |
| CTest | PASS; 2/2 |
| Python regression suite | PASS; 119 tests, 103 subtests |

The published incompatible DLL used for the negative check was only a local
validation input (SHA-256
`0b0fcba402bd609ff3979f2a4ba3a4ce29a5b3c0cd8f9f13812da360fb542204`) and is
not part of this commit. The corrected Release DLL identity is recorded in
[`build-info.json`](build-info.json).

Raw validation output is kept in [`build.log`](build.log),
[`pytest.log`](pytest.log), and [`compatibility.log`](compatibility.log).
