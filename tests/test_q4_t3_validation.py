import json

import numpy as np
import pytest

from trainer.q4_t3_validation import (
    build_spec_q4_validation_payload,
    compute_spec_q4_storage,
    evaluate_quality_gate,
    expected_qat_dense_shapes,
    validate_spec_q4_parity,
    write_quality_json,
    write_spec_q4_validation_json,
)


def _matrix(shape, offset=0):
    values = np.arange(np.prod(shape), dtype=np.float32) + offset
    return (np.sin(values * np.float32(0.013)) * np.float32(2.0)).reshape(shape)


def test_spec_q4_torch_numpy_parity_for_named_projection_cases():
    result = validate_spec_q4_parity({
        "gate_proj": _matrix((3, 768)),
        "up_proj": _matrix((2, 768), offset=17),
        "down_proj": _matrix((3, 2432), offset=31),
    })

    assert result["spec"] == "SPEC-Q4-v1"
    assert result["parity_status"] == "PASS"
    assert result["tensor_count"] == 3
    assert all(tensor["mismatch_count"] == 0 for tensor in result["tensors"])
    assert {tensor["name"] for tensor in result["tensors"]} == {"gate_proj", "up_proj", "down_proj"}


def test_spec_q4_storage_uses_logical_shape_arithmetic():
    odd = compute_spec_q4_storage({"odd": (2, 5)})
    assert odd["weight_count"] == 10
    assert odd["packed_e2m1_bytes"] == 6
    assert odd["scale_e8m0_bytes"] == 2
    assert odd["total_bytes"] == 8

    full = compute_spec_q4_storage(expected_qat_dense_shapes(), require_full_model=True)
    assert full == {
        "spec": "SPEC-Q4-v1",
        "tensor_count": 24,
        "weight_count": 44_826_624,
        "packed_e2m1_bytes": 22_413_312,
        "scale_e8m0_bytes": 1_400_832,
        "total_bytes": 23_814_144,
        "full_model_validated": True,
    }


@pytest.mark.parametrize("control, qat", [(0, 1), (float("nan"), 1), (1, float("inf")), (-1, 1)])
def test_quality_gate_rejects_nonfinite_or_nonpositive_perplexity(control, qat):
    with pytest.raises(ValueError, match="finite positive"):
        evaluate_quality_gate(control, qat)


def test_quality_gate_uses_immutable_threshold_boundaries():
    assert evaluate_quality_gate(100.0, 102.0)["classification"] == "PASS candidate"
    assert evaluate_quality_gate(100.0, 105.0)["classification"] == "AMBIGUOUS"
    assert evaluate_quality_gate(100.0, 106.0)["classification"] == "FAIL"
    assert evaluate_quality_gate(100.0, 102.0)["relative_degradation"] == 2.0
    assert evaluate_quality_gate(100.0, 105.0)["relative_degradation"] == 5.0


def test_validation_json_is_compact_and_contains_no_tensor_arrays(tmp_path):
    parity = validate_spec_q4_parity({"gate_proj": _matrix((2, 768))})
    payload = build_spec_q4_validation_payload(parity)
    path = tmp_path / "spec_q4_validation.json"
    written = write_spec_q4_validation_json(path, parity)
    assert written == payload
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    encoded = json.dumps(payload)
    assert "packed_weights" not in encoded
    assert "scales" not in encoded
    assert "codes" not in encoded
    assert not any(isinstance(value, np.ndarray) for value in _walk(payload))

    quality_path = tmp_path / "quality.json"
    quality = write_quality_json(quality_path, 100.0, 102.0)
    assert json.loads(quality_path.read_text(encoding="utf-8")) == quality
    assert "threshold_version" in quality


def _walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk(child)
