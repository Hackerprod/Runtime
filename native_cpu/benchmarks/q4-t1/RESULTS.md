# Q4-T1 evidence

Scope: SPEC-Q4-v1 tensor mathematics and canonical packing only.

## Included

- OCP E2M1 codebook with canonical -0 -> +0, RTNE and saturation.
- E8M0 shared exponent (scale = 2^(byte - 127)), with `0xff` rejected.
- Block size 32, K-axis quantization, low-nibble-even/high-nibble-odd packing.
- Explicit logical-tail handling and deterministic tensor metadata.
- Scalar dequantization to FP32 and deterministic golden hashes.

No checkpoint, training script, QAT module, exporter, runtime, CPU kernel, or model container was changed.

## Validation

Targeted codec suite: **53 passed** (see pytest-spec-q4.log).

Full native CPU regression suite: **196 passed, 106 subtests passed** (see pytest-all.log). The only warning is pytest cache creation being denied by the host environment; it does not affect test results.

Coverage includes all 16 E2M1 codes, RTNE boundaries, saturation, every tail length 1..31, block boundaries 32/33/63/64/65, K=768 and K=2432 layouts, NaN/Inf rejection, E8M0 range and reserved byte, zero blocks, PyTorch-like input, packing, and deterministic golden vectors.

Golden vector fixture SHA-256: `419348a9e8668f9dbb7f231a009d7e2831ff030f1537c124f94009d2baf60405`

The golden vector file is a fixture, not a complete model format. Its canonical hash helper is explicitly not an ABI or production container.
