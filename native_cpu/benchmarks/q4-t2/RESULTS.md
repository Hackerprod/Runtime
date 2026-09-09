# Q4-T2 validation

## Scope

Q4-T2 adds an all-Torch fake-quant/STE path for dense MiniMind-3 FFN
`gate_proj`, `up_proj`, and `down_proj`.  It does not load a checkpoint, run
training, or modify `model_minimind.py`, the trainer, CPU-E6, or the native
runtime.  The production model is unchanged; QAT is applied only by the
explicit `apply_spec_q4_qat(model)` helper.

## Identity

- SPEC-Q4-v1 parent: `0591d6a5cccf987ab41dc11ae00eb836c6a534cc`
- implementation commit: `29dd3beb9d76bdf4d288f214eb1c35fd0c1c4343`
- CUDA harness commit: `2e7bb8f`
- evidence commit: recorded by the commit containing this directory

## Correctness

- Torch CPU effective weights match the Q4-T1 NumPy codec bit-for-bit for all
  golden vectors, deterministic random shapes, RTNE boundaries, zero blocks,
  signs, outliers, tails, K=768, and K=2432.
- The MiniMind-3 dense wrapper selects exactly 24 projections and
  44,826,624 weights; attention, embedding, LM head, and norms remain normal
  modules.
- `state_dict` keys remain unchanged and strict loading passes.  The original
  `Parameter` objects are reused, so no duplicate trainable master weights are
  introduced.
- STE forward bits equal fake-dequant bits and the gradient is exactly an
  identity tensor.  QATLinear backward is finite.
- CUDA validation passed on an NVIDIA GeForce RTX 3090 (`torch=2.1.0+cu118`,
  CUDA 11.8): golden vectors, ten deterministic random/tail shapes, RTNE
  boundaries, STE gradient, and QATLinear backward.  CUDA and CPU codes,
  scales, and effective FP32 values were byte-identical.

## Test receipts

```
python -m pytest native_cpu/tests/test_q4_t2.py -q
17 passed, 2 skipped (CUDA skipped on local CPU-only host)

python -m pytest native_cpu/tests -q
213 passed, 2 skipped, 106 subtests passed

ctest --test-dir native_cpu/build-e8 -C Release --output-on-failure
100% tests passed, 0 tests failed out of 2
```

The raw logs are `pytest-q4-t2.log`, `pytest-all.log`, and `ctest.log`.

## CUDA diagnostic timing

The harness uses three warmups and twenty CUDA-event repetitions per shape;
allocation and verification are outside each event.  It is diagnostic only,
not a performance gate:

| Shape | Median fake-dequant | Peak allocated |
| --- | ---: | ---: |
| 2432x768 | 2.526720 ms | 273,627,648 B |
| 768x2432 | 2.528768 ms | 273,627,648 B |

Raw samples and the exact command are in `gpu-validation.log`; the complete
harness is `gpu_validation.py`.

## Environment note

The RunPod image initially had no pytest/Transformers and shipped NumPy 2.x,
which is incompatible with the image's Torch 2.1 NumPy bridge.  Validation
installed `pytest`, `transformers==4.40.2`, and `numpy==1.26.4` inside the
temporary pod only.  No repository dependency files were changed.

## Decision

Q4-T2 passes its requested CPU and CUDA correctness criteria.  This is a QAT
building block only: no parent checkpoint, optimizer, dataset, SFT, or
training result is claimed.  The temporary RunPod instance was used only for
the bounded validation and is stopped/deleted after collection of this
evidence.
