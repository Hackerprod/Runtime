"""Portable quantization codecs used by the MiniMind training tools.

The Q4 implementation in this package is deliberately independent from the
native CPU runtime. It defines the tensor format and its scalar reference
implementation; CPU-specific kernels can consume the resulting arrays later.
"""

from .mxfp4 import (
    BLOCK_SIZE,
    E8M0_BIAS,
    E8M0_MAX_EXPONENT,
    E8M0_MIN_EXPONENT,
    E8M0_RESERVED,
    FP4_CODEBOOK,
    FP4_CODES,
    Mxfp4Error,
    decode_e2m1,
    decode_e8m0,
    encode_e2m1,
    encode_e8m0,
    quantize_block,
)
from .spec_q4 import (
    SPEC_Q4_METADATA,
    SPEC_Q4_VERSION,
    SpecQ4Error,
    SpecQ4Tensor,
    dequantize,
    dequantize_tensor,
    quantize,
    quantize_tensor,
)

__all__ = [
    "BLOCK_SIZE", "E8M0_BIAS", "E8M0_MAX_EXPONENT", "E8M0_MIN_EXPONENT",
    "E8M0_RESERVED", "FP4_CODEBOOK", "FP4_CODES",
    "Mxfp4Error", "SPEC_Q4_METADATA", "SPEC_Q4_VERSION", "SpecQ4Error",
    "SpecQ4Tensor", "decode_e2m1", "decode_e8m0", "dequantize",
    "dequantize_tensor", "encode_e2m1", "encode_e8m0", "quantize",
    "quantize_block", "quantize_tensor",
]
