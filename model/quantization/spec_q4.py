"""Canonical tensor representation for SPEC-Q4-v1.

Only the portable tensor representation lives here. This module does not
serialize a complete model and does not know anything about CPU instruction
sets or inference scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .mxfp4 import (
    BLOCK_SIZE,
    E8M0_RESERVED,
    decode_e2m1,
    decode_e8m0,
    quantize_block,
)

SPEC_Q4_VERSION = "SPEC-Q4-v1"
SPEC_Q4_METADATA = {
    "spec": SPEC_Q4_VERSION,
    "element_format": "OCP-E2M1",
    "scale_format": "E8M0",
    "block_size": BLOCK_SIZE,
    "quant_axis": "K",
    "symmetric": True,
    "zero_point": None,
    "scale_algorithm": "specq4_ocp_floor_rtne_v1",
    "packing": "even_low_nibble_odd_high_nibble",
    "padding_code": 0,
}


class SpecQ4Error(ValueError):
    """Raised when a tensor violates the SPEC-Q4-v1 contract."""


def _logical_shape(shape: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    result = tuple(int(dimension) for dimension in shape)
    if not result or any(dimension <= 0 for dimension in result):
        raise SpecQ4Error(f"logical_shape must contain positive dimensions, got {shape!r}")
    return result


def _row_shape(logical_shape: tuple[int, ...]) -> tuple[int, int, int]:
    rows = int(np.prod(logical_shape[:-1], dtype=np.int64)) if len(logical_shape) > 1 else 1
    columns = logical_shape[-1]
    return rows, columns, (columns + 1) // 2


@dataclass(frozen=True)
class SpecQ4Tensor:
    """Portable packed weights plus their logical shape and metadata."""

    packed_weights: np.ndarray
    scales: np.ndarray
    logical_shape: tuple[int, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        logical_shape = _logical_shape(self.logical_shape)
        packed = np.asarray(self.packed_weights)
        scales = np.asarray(self.scales)
        if packed.dtype != np.uint8 or scales.dtype != np.uint8:
            raise SpecQ4Error("packed_weights and scales must have dtype uint8")
        _, columns, packed_columns = _row_shape(logical_shape)
        expected_packed = (*logical_shape[:-1], packed_columns)
        expected_scales = (*logical_shape[:-1], (columns + BLOCK_SIZE - 1) // BLOCK_SIZE)
        if packed.shape != expected_packed:
            raise SpecQ4Error(f"packed_weights shape {packed.shape} != {expected_packed}")
        if scales.shape != expected_scales:
            raise SpecQ4Error(f"scales shape {scales.shape} != {expected_scales}")
        if np.any(scales == E8M0_RESERVED):
            raise SpecQ4Error("scales contain reserved E8M0 byte 0xff")
        supplied = dict(self.metadata)
        for key, value in SPEC_Q4_METADATA.items():
            if supplied.get(key) != value:
                raise SpecQ4Error(f"metadata[{key!r}] must be {value!r}")
        object.__setattr__(self, "logical_shape", logical_shape)
        object.__setattr__(self, "packed_weights", np.ascontiguousarray(packed))
        object.__setattr__(self, "scales", np.ascontiguousarray(scales))
        object.__setattr__(self, "metadata", supplied)

    @property
    def q4_weight_count(self) -> int:
        return int(np.prod(self.logical_shape, dtype=np.int64))

    @property
    def packed_bytes(self) -> int:
        return int(self.packed_weights.nbytes)

    @property
    def scale_bytes(self) -> int:
        return int(self.scales.nbytes)

    def metadata_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata including this tensor's shape."""

        result = dict(self.metadata)
        result["logical_shape"] = list(self.logical_shape)
        return result

    def canonical_bytes(self) -> bytes:
        """Return deterministic bytes for hashing golden tensor fixtures.

        This is a fixture/hash helper, not a model container or ABI. The
        metadata header is canonical JSON followed by packed weights and
        scales in their specified row-major order.
        """

        header = json.dumps(
            self.metadata_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return (
            len(header).to_bytes(4, "little")
            + header
            + self.packed_weights.tobytes()
            + self.scales.tobytes()
        )

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _as_float32_tensor(tensor: Any) -> np.ndarray:
    """Convert NumPy/PyTorch-like input to a C-contiguous float32 array."""

    source = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    if source.ndim < 1:
        raise SpecQ4Error("a Q4 tensor must have at least one dimension")
    # Check before and after conversion: float64 values outside float32 range
    # must fail instead of becoming infinities during an implicit cast.
    if not np.isfinite(source).all():
        raise SpecQ4Error("Q4 input tensor contains NaN or Inf")
    result = np.asarray(source, dtype=np.float32, order="C")
    if not np.isfinite(result).all():
        raise SpecQ4Error("Q4 input tensor cannot be represented as finite float32")
    return result


def _unpack_code(packed_row: np.ndarray, column: int) -> int:
    byte = int(packed_row[column // 2])
    return byte & 0xF if column % 2 == 0 else byte >> 4


def quantize_tensor(tensor: Any) -> SpecQ4Tensor:
    """Quantize a tensor along its final (K) dimension to SPEC-Q4-v1."""

    array = _as_float32_tensor(tensor)
    logical_shape = tuple(int(dimension) for dimension in array.shape)
    rows, columns, _ = _row_shape(logical_shape)
    flat = array.reshape(rows, columns)
    packed = np.zeros((rows, (columns + 1) // 2), dtype=np.uint8)
    scale_count = (columns + BLOCK_SIZE - 1) // BLOCK_SIZE
    scales = np.zeros((rows, scale_count), dtype=np.uint8)
    for row in range(rows):
        for group in range(scale_count):
            begin = group * BLOCK_SIZE
            end = min(begin + BLOCK_SIZE, columns)
            codes, scale_byte = quantize_block(flat[row, begin:end])
            scales[row, group] = scale_byte
            for offset, code in enumerate(codes.tolist()):
                column = begin + offset
                if column & 1:
                    packed[row, column // 2] |= np.uint8(code << 4)
                else:
                    packed[row, column // 2] |= np.uint8(code)
    packed = packed.reshape((*logical_shape[:-1], packed.shape[-1]))
    scales = scales.reshape((*logical_shape[:-1], scales.shape[-1]))
    return SpecQ4Tensor(packed, scales, logical_shape, dict(SPEC_Q4_METADATA))


def dequantize_tensor(qtensor: SpecQ4Tensor) -> np.ndarray:
    """Decode a SPEC-Q4 tensor through the scalar reference path."""

    if not isinstance(qtensor, SpecQ4Tensor):
        raise TypeError("dequantize_tensor expects a SpecQ4Tensor")
    rows, columns, _ = _row_shape(qtensor.logical_shape)
    packed = qtensor.packed_weights.reshape(rows, -1)
    scales = qtensor.scales.reshape(rows, -1)
    result = np.empty((rows, columns), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            code = _unpack_code(packed[row], column)
            scale = decode_e8m0(int(scales[row, column // BLOCK_SIZE]))
            result[row, column] = np.float32(decode_e2m1(code) * scale)
    return result.reshape(qtensor.logical_shape)


quantize = quantize_tensor
dequantize = dequantize_tensor
