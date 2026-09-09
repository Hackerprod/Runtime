"""Scalar, CPU-independent OCP MXFP4 building blocks.

This module is the executable definition of the mathematical parts of
SPEC-Q4-v1. It intentionally does not contain a model serializer or a CPU
kernel. All public decode results are ``float32`` and all quantization
decisions are made with a deterministic scalar nearest-even rule.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

BLOCK_SIZE = 32
E8M0_BIAS = 127
E8M0_RESERVED = 0xFF
E8M0_MIN_EXPONENT = -127
E8M0_MAX_EXPONENT = 127

# The low three bits select the positive magnitude. Bit 3 is the sign bit.
# Code 0x8 (-0) is accepted by the codec but canonicalized to code 0x0.
FP4_CODES = np.arange(16, dtype=np.uint8)
_FP4_MAGNITUDES = np.asarray((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), dtype=np.float32)
FP4_CODEBOOK = np.asarray(
    tuple(_FP4_MAGNITUDES) + tuple(-_FP4_MAGNITUDES), dtype=np.float32
)


class Mxfp4Error(ValueError):
    """Raised when a value cannot be represented by the canonical codec."""


def _as_float32(value: float | np.floating) -> np.float32:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.float32(value)
    if not np.isfinite(result):
        raise Mxfp4Error("FP4 values and scales must be finite float32 values")
    return result


def _floor_log2_float32(value: np.float32) -> int:
    """Return floor(log2(value)) using the float32 bit representation.

    Using exponent bits rather than libm makes the shared-exponent decision
    stable across platforms, including for subnormal float32 inputs.
    """

    value = _as_float32(value)
    if value <= np.float32(0.0):
        raise Mxfp4Error("floor(log2) requires a positive value")
    bits = int(np.asarray(value, dtype=np.float32).view(np.uint32).item())
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent:
        return exponent - 127
    # A subnormal is mantissa * 2^-149. Its highest set bit determines the
    # floor logarithm without a platform-dependent floating-point operation.
    return mantissa.bit_length() - 1 - 149


def encode_e8m0(exponent: int) -> np.uint8:
    """Encode an unbiased E8M0 exponent (``scale = 2**exponent``).

    Raw byte 0xff is reserved. Out-of-range exponents are rejected rather
    than clamped so that an invalid block cannot silently change its scale.
    """

    exponent = int(exponent)
    if not E8M0_MIN_EXPONENT <= exponent <= E8M0_MAX_EXPONENT:
        raise Mxfp4Error(
            f"E8M0 exponent {exponent} is outside [{E8M0_MIN_EXPONENT}, "
            f"{E8M0_MAX_EXPONENT}]"
        )
    return np.uint8(exponent + E8M0_BIAS)


def decode_e8m0(raw: int | np.integer) -> np.float32:
    """Decode an E8M0 byte, rejecting the reserved ``0xff`` value."""

    raw = int(raw)
    if not 0 <= raw <= 0xFF:
        raise Mxfp4Error(f"E8M0 byte must be in [0, 255], got {raw}")
    if raw == E8M0_RESERVED:
        raise Mxfp4Error("E8M0 byte 0xff is reserved")
    return np.ldexp(np.float32(1.0), raw - E8M0_BIAS).astype(np.float32)


def decode_e2m1(code: int | np.integer) -> np.float32:
    """Decode one OCP E2M1 code and canonicalize negative zero."""

    code = int(code)
    if not 0 <= code <= 0xF:
        raise Mxfp4Error(f"FP4 code must be in [0, 15], got {code}")
    magnitude = _FP4_MAGNITUDES[code & 0x7]
    if magnitude == 0.0:
        return np.float32(0.0)
    return np.float32(-magnitude if code & 0x8 else magnitude)


def _nearest_magnitude_index(magnitude: np.float32) -> int:
    """Nearest E2M1 magnitude, with ties resolved by even code index."""

    magnitude = _as_float32(magnitude)
    if magnitude <= np.float32(0.0):
        return 0
    value = float(magnitude)
    best = 0
    best_distance = abs(value - float(_FP4_MAGNITUDES[0]))
    for index in range(1, len(_FP4_MAGNITUDES)):
        distance = abs(value - float(_FP4_MAGNITUDES[index]))
        if distance < best_distance or (
            distance == best_distance and index % 2 == 0 and best % 2 == 1
        ):
            best = index
            best_distance = distance
    return best


def encode_e2m1(value: float | np.floating) -> np.uint8:
    """Quantize one finite value to E2M1 using RTNE and saturation.

    The sign occupies bit 3 and the magnitude code occupies bits 0..2. A
    magnitude that rounds to zero always uses the canonical positive-zero code.
    """

    value = _as_float32(value)
    magnitude = np.float32(abs(float(value)))
    index = _nearest_magnitude_index(magnitude)
    if index == 0:
        return np.uint8(0)
    return np.uint8(index | (0x8 if value < np.float32(0.0) else 0))


def _quantize_scaled(value: np.float32, scale: np.float32) -> np.uint8:
    normalized = np.float32(np.float32(value) / np.float32(scale))
    return encode_e2m1(normalized)


def quantize_block(values: Iterable[float] | np.ndarray) -> tuple[np.ndarray, np.uint8]:
    """Quantize at most one 32-value block.

    Returns ``(codes, scale_byte)``. A short block is logically padded with
    canonical +0 for quantization, but only the supplied values are returned.
    ``amax`` never observes the padding.
    """

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size == 0 or array.size > BLOCK_SIZE:
        raise Mxfp4Error("a block must contain between 1 and 32 values")
    if not np.isfinite(array).all():
        raise Mxfp4Error("FP4 inputs must be finite")
    amax = np.float32(np.max(np.abs(array)))
    if amax == np.float32(0.0):
        scale_byte = encode_e8m0(0)
        scale = np.float32(1.0)
    else:
        shared_exponent = _floor_log2_float32(amax) - 2
        scale_byte = encode_e8m0(shared_exponent)
        scale = decode_e8m0(scale_byte)
    codes = np.fromiter(
        (_quantize_scaled(value, scale) for value in array),
        dtype=np.uint8,
        count=array.size,
    )
    return codes, scale_byte
