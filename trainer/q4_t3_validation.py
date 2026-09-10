"""Q4-T3 Slice D validation helpers.

Validation runs outside training and campaign hot loops. Codec and model
imports stay inside the APIs that need them so pure storage/quality helpers
remain importable without Transformers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SPEC_Q4_VALIDATION_SCHEMA = "q4-t3-spec-q4-validation"
SPEC_Q4_VALIDATION_VERSION = 1
QUALITY_SCHEMA = "q4-t3-quality"
QUALITY_SCHEMA_VERSION = 1
QUALITY_THRESHOLD_VERSION = "q4-t3-quality-thresholds-v1"

QAT_LAYER_COUNT = 8
QAT_TENSOR_COUNT = 24
QAT_WEIGHT_COUNT = 44_826_624
QAT_HIDDEN_SIZE = 768
QAT_INTERMEDIATE_SIZE = 2432
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_QAT_KEY_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>[0-9]+)\.mlp\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def _finite_positive(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isfinite(float(value)) and value > 0


def _shape_tuple(shape: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(dimension) for dimension in shape)
    if not result or any(dimension <= 0 for dimension in result):
        raise ValueError(f"logical shape must contain positive dimensions: {shape!r}")
    return result


def expected_qat_dense_shapes() -> dict[str, tuple[int, int]]:
    """Return exact logical shapes for the 24 dense MiniMind FFN tensors."""

    shapes: dict[str, tuple[int, int]] = {}
    for layer in range(QAT_LAYER_COUNT):
        for projection in _PROJECTIONS:
            shape = (
                (QAT_INTERMEDIATE_SIZE, QAT_HIDDEN_SIZE)
                if projection in {"gate_proj", "up_proj"}
                else (QAT_HIDDEN_SIZE, QAT_INTERMEDIATE_SIZE)
            )
            shapes[f"layer.{layer}.mlp.{projection}.weight"] = shape
    return shapes


def _named_shapes(
    shapes: Mapping[str, Sequence[int]] | Sequence[Sequence[int]],
) -> list[tuple[str, tuple[int, ...]]]:
    if isinstance(shapes, Mapping):
        return [(str(name), _shape_tuple(shape)) for name, shape in shapes.items()]
    return [(f"tensor_{index}", _shape_tuple(shape)) for index, shape in enumerate(shapes)]


def compute_spec_q4_storage(
    shapes: Mapping[str, Sequence[int]] | Sequence[Sequence[int]],
    *,
    require_full_model: bool = False,
) -> dict[str, Any]:
    """Compute packed E2M1 and E8M0 storage from logical tensor shapes."""

    named = _named_shapes(shapes)
    tensor_count = len(named)
    weight_count = sum(math.prod(shape) for _, shape in named)
    packed_bytes = sum(
        math.prod(shape[:-1], start=1) * ((shape[-1] + 1) // 2)
        for _, shape in named
    )
    scale_bytes = sum(
        math.prod(shape[:-1], start=1) * ((shape[-1] + 31) // 32)
        for _, shape in named
    )
    full_model_validated = False
    if require_full_model:
        expected = sorted(expected_qat_dense_shapes().values())
        actual = sorted(shape for _, shape in named)
        if tensor_count != QAT_TENSOR_COUNT or weight_count != QAT_WEIGHT_COUNT:
            raise ValueError(
                f"unexpected dense QAT count: tensors={tensor_count}, weights={weight_count}"
            )
        if actual != expected:
            raise ValueError("logical shapes do not match exact dense MiniMind FFN shapes")
        full_model_validated = True
    return {
        "spec": "SPEC-Q4-v1",
        "tensor_count": tensor_count,
        "weight_count": weight_count,
        "packed_e2m1_bytes": packed_bytes,
        "scale_e8m0_bytes": scale_bytes,
        "total_bytes": packed_bytes + scale_bytes,
        "full_model_validated": full_model_validated,
    }


def _float32_sha256(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def compare_spec_q4_tensor(name: str, weight: Any) -> dict[str, Any]:
    """Compare Torch fake-dequant output with canonical NumPy reference output."""

    import numpy as np
    import torch

    from model.quantization.qat import spec_q4_fake_dequant
    from model.quantization.spec_q4 import dequantize_tensor, quantize_tensor

    torch_weight = torch.as_tensor(weight, dtype=torch.float32)
    torch_output = spec_q4_fake_dequant(torch_weight).detach().cpu().numpy()
    reference_output = dequantize_tensor(quantize_tensor(weight))
    torch_output = np.asarray(torch_output, dtype=np.float32)
    reference_output = np.asarray(reference_output, dtype=np.float32)
    same_shape = torch_output.shape == reference_output.shape
    equal = same_shape and np.array_equal(torch_output, reference_output)
    mismatch_count = 0
    first_mismatch: dict[str, Any] | None = None
    if same_shape:
        mismatch_indices = np.flatnonzero(torch_output != reference_output)
        mismatch_count = int(mismatch_indices.size)
        if mismatch_count:
            flat_index = int(mismatch_indices[0])
            index = tuple(int(value) for value in np.unravel_index(flat_index, torch_output.shape))
            first_mismatch = {
                "index": list(index),
                "torch": float(torch_output[index]),
                "numpy": float(reference_output[index]),
            }
    else:
        first_mismatch = {
            "index": [],
            "torch_shape": list(torch_output.shape),
            "numpy_shape": list(reference_output.shape),
        }
    return {
        "name": str(name),
        "shape": list(torch_output.shape),
        "weight_count": int(torch_output.size),
        "parity_status": "PASS" if equal else "FAIL",
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
        "torch_sha256": _float32_sha256(torch_output),
        "numpy_sha256": _float32_sha256(reference_output),
    }


def validate_spec_q4_parity(cases: Mapping[str, Any]) -> dict[str, Any]:
    """Run exact parity for named representative tensors and return compact evidence."""

    if not cases:
        raise ValueError("SPEC-Q4 parity requires at least one named tensor")
    results = [compare_spec_q4_tensor(name, weight) for name, weight in cases.items()]
    storage = compute_spec_q4_storage({result["name"]: result["shape"] for result in results})
    return {
        "schema": SPEC_Q4_VALIDATION_SCHEMA,
        "schema_version": SPEC_Q4_VALIDATION_VERSION,
        "spec": "SPEC-Q4-v1",
        "parity_status": "PASS" if all(result["parity_status"] == "PASS" for result in results) else "FAIL",
        "tensor_count": len(results),
        "weight_count": sum(result["weight_count"] for result in results),
        "storage": storage,
        "tensors": results,
    }


def _checkpoint_state_dict(source: Any) -> Mapping[str, Any]:
    if hasattr(source, "state_dict") and callable(source.state_dict):
        source = source.state_dict()
    elif isinstance(source, (str, bytes, os.PathLike)):
        import torch

        source = torch.load(source, map_location="cpu")
    if not isinstance(source, Mapping):
        raise ValueError("QAT source must be a model, state_dict, or checkpoint path")
    if all(hasattr(value, "dtype") for value in source.values()):
        return source
    for key in ("state_dict", "model"):
        nested = source.get(key)
        if isinstance(nested, Mapping) and all(hasattr(value, "dtype") for value in nested.values()):
            return nested
    raise ValueError("checkpoint does not contain a tensor state_dict")


def _qat_keys(state_dict: Mapping[str, Any]) -> list[str]:
    found: dict[tuple[int, str], str] = {}
    for key in state_dict:
        match = _QAT_KEY_RE.search(str(key))
        if match:
            identity = (int(match.group("layer")), match.group("projection"))
            if identity in found:
                raise ValueError(f"duplicate QAT projection key: {identity}")
            found[identity] = str(key)
    expected = {
        (layer, projection)
        for layer in range(QAT_LAYER_COUNT)
        for projection in _PROJECTIONS
    }
    if set(found) != expected:
        raise ValueError(f"QAT state_dict must contain exact 24 dense FFN keys, got {len(found)}")
    return [found[(layer, projection)] for layer in range(QAT_LAYER_COUNT) for projection in _PROJECTIONS]


def validate_qat_checkpoint_parity(
    source: Any,
    *,
    require_full_model: bool = True,
) -> dict[str, Any]:
    """Validate exact SPEC-Q4 parity for QAT master weights outside training."""

    import torch

    state_dict = _checkpoint_state_dict(source)
    keys = _qat_keys(state_dict) if require_full_model else sorted(
        str(key) for key in state_dict if _QAT_KEY_RE.search(str(key))
    )
    if not keys:
        raise ValueError("QAT state_dict contains no dense FFN projection keys")
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    cases: dict[str, Any] = {}
    for key in keys:
        tensor = state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"QAT key is not a torch.Tensor: {key}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"QAT master tensor must be float32: {key}")
        shape = tuple(int(dimension) for dimension in tensor.shape)
        tensor_shapes[key] = shape
        cases[key] = tensor
    storage = compute_spec_q4_storage(tensor_shapes, require_full_model=require_full_model)
    parity = validate_spec_q4_parity(cases)
    parity["source"] = "model" if hasattr(source, "state_dict") else "state_dict_or_checkpoint"
    parity["storage"] = storage
    return parity


def evaluate_quality_gate(control_ppl: Any, qat_ppl: Any) -> dict[str, Any]:
    """Classify immutable QAT holdout perplexity degradation thresholds."""

    if not _finite_positive(control_ppl) or not _finite_positive(qat_ppl):
        raise ValueError("CONTROL and QAT perplexity must be finite positive numbers")
    relative_degradation = (float(qat_ppl) / float(control_ppl) - 1.0) * 100.0
    for boundary in (2.0, 5.0):
        if math.isclose(relative_degradation, boundary, rel_tol=0.0, abs_tol=1e-12):
            relative_degradation = boundary
    if relative_degradation <= 2.0:
        classification = "PASS candidate"
    elif relative_degradation <= 5.0:
        classification = "AMBIGUOUS"
    else:
        classification = "FAIL"
    return {
        "schema": QUALITY_SCHEMA,
        "schema_version": QUALITY_SCHEMA_VERSION,
        "control_ppl": float(control_ppl),
        "qat_ppl": float(qat_ppl),
        "relative_degradation": relative_degradation,
        "classification": classification,
        "threshold_version": QUALITY_THRESHOLD_VERSION,
    }


def build_spec_q4_validation_payload(
    parity: Mapping[str, Any],
    storage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build compact JSON-ready SPEC-Q4 validation evidence."""

    result = dict(parity)
    if storage is not None:
        result["storage"] = dict(storage)
    return result


def write_spec_q4_validation_json(
    path: str | Path,
    parity: Mapping[str, Any],
    storage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_spec_q4_validation_payload(parity, storage)
    _write_json(path, payload)
    return payload


def write_quality_json(path: str | Path, control_ppl: Any, qat_ppl: Any) -> dict[str, Any]:
    payload = evaluate_quality_gate(control_ppl, qat_ppl)
    _write_json(path, payload)
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


__all__ = [
    "QUALITY_SCHEMA",
    "QUALITY_THRESHOLD_VERSION",
    "SPEC_Q4_VALIDATION_SCHEMA",
    "build_spec_q4_validation_payload",
    "compare_spec_q4_tensor",
    "compute_spec_q4_storage",
    "evaluate_quality_gate",
    "expected_qat_dense_shapes",
    "validate_qat_checkpoint_parity",
    "validate_spec_q4_parity",
    "write_quality_json",
    "write_spec_q4_validation_json",
]
