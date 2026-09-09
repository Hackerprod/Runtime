"""Exhaustive SPEC-Q4-v1 codec and canonical representation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from model.quantization import (
    BLOCK_SIZE,
    E8M0_RESERVED,
    FP4_CODEBOOK,
    SPEC_Q4_METADATA,
    SpecQ4Error,
    SpecQ4Tensor,
    decode_e2m1,
    decode_e8m0,
    dequantize_tensor,
    encode_e2m1,
    encode_e8m0,
    quantize_block,
    quantize_tensor,
)


ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "model" / "quantization" / "golden_vectors.json"


def test_all_e2m1_codes_and_negative_zero_are_canonical():
    expected = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
    for code, value in enumerate(expected):
        decoded = decode_e2m1(code)
        assert decoded == np.float32(value)
        if code in (0, 8):
            assert not np.signbit(decoded)
    assert encode_e2m1(np.float32(-0.0)) == 0
    np.testing.assert_array_equal(FP4_CODEBOOK, np.asarray(expected, dtype=np.float32))


def test_rtne_boundaries_use_even_magnitude_code():
    # amax=6 makes the block scale exactly one. The values below are halfway
    # between adjacent codebook magnitudes.
    values = np.asarray([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], dtype=np.float32)
    codes, scale = quantize_block(values)
    assert int(scale) == 127
    assert codes.tolist() == [0, 2, 2, 4, 4, 6, 6, 7]
    negative, _ = quantize_block(-values)
    assert negative.tolist() == [0, 10, 10, 12, 12, 14, 14, 15]


def test_e2m1_saturates_when_normalized_value_exceeds_maximum():
    assert int(encode_e2m1(6.0001)) == 7
    assert int(encode_e2m1(-100.0)) == 15


def test_e8m0_roundtrip_and_reserved_byte():
    for exponent in range(-127, 128):
        raw = encode_e8m0(exponent)
        assert int(raw) != E8M0_RESERVED
        assert decode_e8m0(raw) == np.ldexp(np.float32(1), exponent)
        assert int(encode_e8m0(int(raw) - 127)) == int(raw)
    with pytest.raises(ValueError):
        decode_e8m0(E8M0_RESERVED)
    with pytest.raises(ValueError):
        encode_e8m0(128)
    with pytest.raises(ValueError):
        encode_e8m0(-128)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_inputs_are_rejected(bad):
    with pytest.raises(SpecQ4Error):
        quantize_tensor(np.asarray([[bad]], dtype=np.float32))
    with pytest.raises(ValueError):
        quantize_block(np.asarray([bad], dtype=np.float32))


@pytest.mark.parametrize("columns", [1, *range(1, 32), 32, 33, 63, 64, 65])
def test_tail_padding_and_block_boundaries(columns):
    values = (np.arange(columns, dtype=np.float32) - columns / 3.0) / 7.0
    tensor = quantize_tensor(values.reshape(1, columns))
    assert tensor.logical_shape == (1, columns)
    assert tensor.packed_weights.shape == (1, (columns + 1) // 2)
    assert tensor.scales.shape == (1, (columns + BLOCK_SIZE - 1) // BLOCK_SIZE)
    if columns & 1:
        assert int(tensor.packed_weights[0, -1] >> 4) == 0
    decoded = dequantize_tensor(tensor)
    assert decoded.shape == (1, columns)
    assert np.isfinite(decoded).all()


@pytest.mark.parametrize("shape", [(1, 768), (1, 2432), (2, 768), (2, 2432)])
def test_real_ffn_shapes_have_canonical_layout(shape):
    values = np.linspace(-6.0, 6.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    tensor = quantize_tensor(values)
    assert tensor.q4_weight_count == int(np.prod(shape))
    assert tensor.packed_bytes == shape[0] * ((shape[1] + 1) // 2)
    assert tensor.scale_bytes == shape[0] * ((shape[1] + 31) // 32)
    assert dequantize_tensor(tensor).dtype == np.float32


def test_all_zero_block_uses_unit_scale_and_positive_zero_codes():
    tensor = quantize_tensor(np.zeros((1, 32), dtype=np.float32))
    assert tensor.scales.tolist() == [[127]]
    assert tensor.packed_weights.tobytes() == bytes(16)
    decoded = dequantize_tensor(tensor)
    assert decoded.tobytes() == np.zeros((1, 32), dtype=np.float32).tobytes()


def test_reserved_scale_is_rejected_by_tensor_validation():
    with pytest.raises(SpecQ4Error):
        SpecQ4Tensor(
            np.zeros((1, 1), dtype=np.uint8),
            np.asarray([[E8M0_RESERVED]], dtype=np.uint8),
            (1, 1),
            dict(SPEC_Q4_METADATA),
        )


def test_torch_like_input_is_converted_without_runtime_dependency():
    torch = pytest.importorskip("torch")
    source = torch.tensor([[0.0, 1.0, -2.0, 3.0]], dtype=torch.float16)
    tensor = quantize_tensor(source)
    assert tensor.logical_shape == (1, 4)
    assert dequantize_tensor(tensor).dtype == np.float32


def test_scale_underflow_is_fail_closed_but_float32_range_is_supported():
    # The smallest subnormal would require an E8M0 exponent below -127.
    with pytest.raises(ValueError):
        quantize_block(np.asarray([np.nextafter(np.float32(0), np.float32(1))]))
    # The largest finite float32 still produces a valid shared exponent.
    codes, scale = quantize_block(np.asarray([np.finfo(np.float32).max]))
    assert int(scale) == 252  # shared exponent 125
    assert codes.tolist() == [7]


def test_golden_vectors_are_stable():
    vectors = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert vectors["spec"] == "SPEC-Q4-v1"
    for vector in vectors["vectors"]:
        values = np.asarray(vector["values"], dtype=np.float32).reshape(vector["shape"])
        tensor = quantize_tensor(values)
        assert tensor.packed_weights.tobytes().hex() == vector["packed_hex"]
        assert tensor.scales.tobytes().hex() == vector["scales_hex"]
        assert tensor.canonical_sha256() == vector["canonical_sha256"]
        assert hashlib.sha256(dequantize_tensor(tensor).tobytes()).hexdigest() == vector["decoded_sha256"]
