"""Bounded Q4-T3 Slice E evidence assembly.

This module only consumes explicit evidence objects or JSON paths. It never
loads checkpoints or datasets and writes only the requested evidence folder.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


EVIDENCE_SCHEMA = "q4-t3-evidence"
EVIDENCE_VERSION = 1
EVIDENCE_RELATIVE_PATH = Path("native_cpu") / "benchmarks" / "q4-t3"
MAX_INPUT_BYTES = 128 * 1024
MAX_FILE_BYTES = 128 * 1024
MAX_TOTAL_BYTES = 512 * 1024
MAX_LOG_BYTES = 32 * 1024
MAX_ARRAY_ITEMS = 64
MAX_ARRAY_ITEMS_TOTAL = 256
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT = re.compile(r"^[0-9a-fA-F]{7,64}$")
_SENSITIVE = re.compile(
    r"(?:api[_-]?(?:key|token|secret)|access[_-]?token|auth[_-]?token|runpod[_-]?token|"
    r"(?:^|[_-])(?<!valid_)token(?:$|[_-])|secret|password|credential|"
    r"private[_-]?key|ssh[_-]?key)",
    re.I,
)
_ABSOLUTE = re.compile(r"(?:^[A-Za-z]:[\\/]|^\\\\|^/)")
_SECRET_VALUE = re.compile(r"(?:ghp_|github_pat_|sk-[A-Za-z0-9]|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+)", re.I)
_USER_HOST = re.compile(r"\b\S+@\S+\b")
_USER_HOST_ASSIGNMENT = re.compile(r"(?i)\b(?:user(?:name)?|host(?:name)?|server|node)\s*[=:]\s*\S+")
_INLINE_SECRET = re.compile(
    r"(?i)--?(?:api[_-]?(?:key|token|secret)|access[_-]?token|auth[_-]?token|"
    r"runpod[_-]?token|token|secret|password|credential|private[_-]?key)(?:=|\s+)\S+"
)

_RUNNER_KEYS = {
    "schema", "mode", "smoke_success", "config", "sft", "parent", "stages", "comparisons",
    "device", "max_steps", "epochs", "batch_size", "accumulation_steps", "seed",
    "holdout_seed", "holdout_size", "dataset_sha256", "dataset_bytes", "raw_record_count",
    "plan_sha256", "holdout_sha256", "train_seed", "name", "command", "exit_code", "metrics",
    "checkpoint", "exists", "verified", "strict_reload", "dtype", "slowdown_vs_control",
    "slowdown_unavailable_reason", "memory_overhead_vs_control", "memory_overhead_unavailable_reason",
    "method", "sample_count", "mean_abs", "max_abs", "parameters_changed",
    "master_changed_vs_parent", "matrix_count", "key_count", "rms",
}
_METRIC_KEYS = {
    "metrics_schema_version", "stage", "config", "initial_loss", "final_loss",
    "first_10_percent_loss_median", "last_10_percent_loss_median", "optimizer_steps",
    "samples_processed", "tokens_processed", "median_step_ms", "tokens_per_second",
    "duration_seconds", "peak_allocated_vram_bytes", "peak_reserved_vram_bytes",
    "finite_loss", "finite_gradients", "training_failed", "status", "rank",
    "accounting_scope", "evidence_authority", "median_aggregation",
    "parameter_update_probe", "master_vs_parent",
    "method", "sample_count", "mean_abs", "max_abs", "parameters_changed",
    "master_changed_vs_parent", "matrix_count", "key_count", "rms",
    "first_loss_exact", "last_loss_exact", "step_duration_exact", "capacity",
}
_HOLDOUT_KEYS = {
    "plan_sha256", "holdout_sha256", "holdout_count", "batch_size", "cross_entropy",
    "perplexity", "valid_token_count", "sample_count",
}
_QUALITY_KEYS = {
    "schema", "schema_version", "control_ppl", "qat_ppl", "relative_degradation",
    "classification", "threshold_version",
}
_SPEC_KEYS = {
    "schema", "schema_version", "spec", "parity_status", "tensor_count", "weight_count",
    "storage", "tensors", "source", "packed_e2m1_bytes", "scale_e8m0_bytes", "total_bytes",
    "full_model_validated", "name", "shape", "mismatch_count", "first_mismatch",
    "torch_sha256", "numpy_sha256", "index", "torch", "numpy", "torch_shape", "numpy_shape",
}
_DIAGNOSTIC_KEYS = {
    "schema", "version", "informational", "prompts_sha256", "policy_sha256", "branches",
    "branch", "seed", "outputs", "id", "text", "metrics", "top1_agreement",
    "mean_logit_kl_control_to_qat", "status",
}
_ROL9_REFERENCES = {
    "torch": "2.1.0/cu118",
    "cuda": "cu118",
    "gpu": "RTX 3090",
}
_DYNAMIC_CHILD_CONTAINERS = {"branches", "holdout_results"}
_EXPECTED_STAGES = ("pretrain", "control", "qat")


class EvidenceError(ValueError):
    """Raised when evidence is incomplete, unsafe, or exceeds bounded limits."""


def _scan(value: Any, *, key: str = "", counters: list[int] | None = None) -> None:
    counters = counters or [0]
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            child_key = str(child_key)
            if _SENSITIVE.search(child_key):
                raise EvidenceError(f"sensitive field is not allowed: {child_key}")
            _scan(child_value, key=child_key, counters=counters)
    elif isinstance(value, (list, tuple)):
        counters[0] += len(value)
        if len(value) > MAX_ARRAY_ITEMS or counters[0] > MAX_ARRAY_ITEMS_TOTAL:
            raise EvidenceError("evidence contains oversized array data")
        for child in value:
            _scan(child, key=key, counters=counters)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceError(f"non-finite evidence value: {key}")
    elif isinstance(value, str):
        if _ABSOLUTE.search(value) or _SECRET_VALUE.search(value) or _USER_HOST.search(value) or _USER_HOST_ASSIGNMENT.search(value):
            raise EvidenceError(f"unsafe evidence value: {key}")
        if key not in {"command", "text"} and _INLINE_SECRET.search(value):
            raise EvidenceError(f"inline secret is not allowed: {key}")
        if key and _SENSITIVE.search(key):
            raise EvidenceError(f"sensitive evidence value: {key}")


def _load_json(source: Any, label: str) -> Mapping[str, Any]:
    if isinstance(source, (str, bytes, Path)):
        path = Path(source)
        parts = {part.lower() for part in path.parts}
        if parts & {"dataset", "datasets", "checkpoint", "checkpoints"}:
            raise EvidenceError(f"{label} may not read dataset/checkpoint paths")
        try:
            if path.stat().st_size > MAX_INPUT_BYTES:
                raise EvidenceError(f"{label} exceeds input size limit")
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except OSError as exc:
            raise EvidenceError(f"cannot read {label}") from exc
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"{label} must be JSON") from exc
    else:
        value = source
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} must contain an object")
    _scan(value)
    return value


def _compact(source: Mapping[str, Any], allowed: set[str]) -> dict[str, Any]:
    def visit(value: Any, parent_key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            result = {}
            for key, child in value.items():
                key = str(key)
                if parent_key is None or parent_key in _DYNAMIC_CHILD_CONTAINERS or key in allowed:
                    result[key] = visit(child, key)
            return result
        if isinstance(value, list):
            return [visit(child, parent_key) for child in value]
        if isinstance(value, str) and parent_key in {"command", "text"}:
            return sanitize_log(value)
        return value

    return visit(source)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _same_hash(value: Any, expected: str, label: str) -> None:
    if value != expected:
        raise EvidenceError(f"{label} does not match supplied identity")


def _validate_metric_payload(metrics: Mapping[str, Any], stage: str) -> None:
    if metrics.get("stage") != stage or metrics.get("status") != "success":
        raise EvidenceError(f"{stage} metrics must report status success for matching stage")
    if metrics.get("training_failed") is not False:
        raise EvidenceError(f"{stage} metrics report training failure")
    if metrics.get("finite_loss") is not True or metrics.get("finite_gradients") is not True:
        raise EvidenceError(f"{stage} metrics report non-finite values")
    if not _positive_int(metrics.get("optimizer_steps")):
        raise EvidenceError(f"{stage} optimizer_steps must be positive")
    for key in ("duration_seconds", "tokens_per_second", "median_step_ms"):
        if not _finite_number(metrics.get(key)) or metrics[key] <= 0:
            raise EvidenceError(f"{stage} metric {key} must be finite and positive")
    for key in ("samples_processed", "tokens_processed"):
        if not _positive_int(metrics.get(key)):
            raise EvidenceError(f"{stage} metric {key} must be positive")
    for key in ("peak_allocated_vram_bytes", "peak_reserved_vram_bytes"):
        if not _nonnegative_int(metrics.get(key)):
            raise EvidenceError(f"{stage} metric {key} must be non-negative")
    for key in (
        "initial_loss", "final_loss", "first_10_percent_loss_median", "last_10_percent_loss_median",
    ):
        if not _finite_number(metrics.get(key)):
            raise EvidenceError(f"{stage} metric {key} must be finite")
    if stage in {"pretrain", "control"}:
        probe = metrics.get("parameter_update_probe")
        if not isinstance(probe, Mapping) or probe.get("parameters_changed") is not True:
            raise EvidenceError(f"{stage} parameter update probe is missing or unsuccessful")
        if probe.get("method") is not None and not isinstance(probe.get("method"), str):
            raise EvidenceError(f"{stage} parameter probe method must be a string")
        if not _positive_int(probe.get("sample_count")):
            raise EvidenceError(f"{stage} parameter probe sample_count must be positive")
        for key in ("mean_abs", "max_abs"):
            if key not in probe or not _finite_number(probe.get(key)) or probe[key] < 0:
                raise EvidenceError(f"{stage} parameter probe {key} must be finite and non-negative")
        if "rms" in probe and (not _finite_number(probe.get("rms")) or probe["rms"] < 0):
            raise EvidenceError(f"{stage} parameter probe rms must be finite and non-negative")
        for key in ("mean_abs", "max_abs"):
            if probe[key] <= 0:
                raise EvidenceError(f"{stage} parameter probe {key} must be finite and positive")
    else:
        delta = metrics.get("master_vs_parent")
        if not isinstance(delta, Mapping) or delta.get("master_changed_vs_parent") is not True:
            raise EvidenceError("qat master-vs-parent evidence is missing or unsuccessful")
        if delta.get("method") is not None and not isinstance(delta.get("method"), str):
            raise EvidenceError("qat parent delta method must be a string")
        if delta.get("key_count") != 24 or delta.get("matrix_count") != 24:
            raise EvidenceError("qat master-vs-parent evidence must cover 24 tensors")
        for key in ("sample_count",):
            if key in delta and not _positive_int(delta.get(key)):
                raise EvidenceError(f"qat parent delta {key} must be a positive integer")
        for key in ("mean_abs", "max_abs", "rms"):
            if key not in delta or not _finite_number(delta.get(key)) or delta[key] < 0:
                raise EvidenceError(f"qat parent delta {key} must be finite and non-negative")
        for key in ("mean_abs", "max_abs"):
            if delta[key] <= 0:
                raise EvidenceError(f"qat parent delta {key} must be finite and positive")


def _validate_runner_manifest(
    runner: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    if runner.get("schema") != "q4-t3-slice-c" or runner.get("mode") not in {"smoke", "full"}:
        raise EvidenceError("runner manifest has unsupported schema or mode")
    if runner.get("smoke_success") is not True:
        raise EvidenceError("runner manifest must report successful smoke/full execution")
    config = runner.get("config")
    if not isinstance(config, Mapping):
        raise EvidenceError("runner manifest configuration is missing")
    if runner["mode"] == "smoke":
        if not _positive_int(config.get("max_steps")):
            raise EvidenceError("smoke runner manifest requires positive max_steps")
    elif config.get("max_steps") is not None:
        raise EvidenceError("full runner manifest must not contain max_steps")
    sft = runner.get("sft")
    if not isinstance(sft, Mapping):
        raise EvidenceError("runner manifest SFT binding is missing")
    _same_hash(sft.get("plan_sha256"), identity["plan_sha256"], "runner plan_sha256")
    _same_hash(sft.get("holdout_sha256"), identity["holdout_sha256"], "runner holdout_sha256")
    parent = runner.get("parent")
    if not isinstance(parent, Mapping):
        raise EvidenceError("runner manifest Parent binding is missing")
    _same_hash(parent.get("sha256"), identity["parent"]["sha256"], "runner Parent SHA-256")
    if parent.get("dtype") != identity["parent"]["dtype"]:
        raise EvidenceError("runner Parent dtype does not match supplied identity")
    _same_hash(sft.get("dataset_sha256"), identity["dataset_manifest_sha256"], "runner dataset SHA-256")
    stages = runner.get("stages")
    if not isinstance(stages, list) or [stage.get("name") for stage in stages if isinstance(stage, Mapping)] != list(_EXPECTED_STAGES):
        raise EvidenceError("runner manifest must contain pretrain, control, qat stages in order")
    for stage_name, stage in zip(_EXPECTED_STAGES, stages):
        if not isinstance(stage, Mapping) or stage.get("exit_code") != 0:
            raise EvidenceError(f"runner {stage_name} stage is missing or failed")
        metrics = stage.get("metrics")
        if not isinstance(metrics, Mapping):
            raise EvidenceError(f"runner {stage_name} metrics are missing")
        _validate_metric_payload(metrics, stage_name)
        checkpoint = stage.get("checkpoint")
        expected_dtype = "float32" if stage_name == "qat" else "float16"
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("exists") is not True
            or checkpoint.get("verified") is not True
            or checkpoint.get("strict_reload") is not True
            or checkpoint.get("dtype") != expected_dtype
        ):
            raise EvidenceError(f"runner {stage_name} checkpoint verification is incomplete")


def _validate_holdout_result(result: Mapping[str, Any], branch: str, identity: Mapping[str, Any]) -> None:
    _same_hash(result.get("plan_sha256"), identity["plan_sha256"], f"{branch} holdout plan_sha256")
    _same_hash(result.get("holdout_sha256"), identity["holdout_sha256"], f"{branch} holdout holdout_sha256")
    if result.get("holdout_count") != 128 or result.get("sample_count") != 128:
        raise EvidenceError(f"{branch} holdout must contain exactly 128 samples")
    if not _positive_int(result.get("valid_token_count")) or not _positive_int(result.get("batch_size")):
        raise EvidenceError(f"{branch} holdout accounting is incomplete")
    if not _finite_number(result.get("cross_entropy")) or result["cross_entropy"] < 0:
        raise EvidenceError(f"{branch} holdout cross_entropy must be finite and non-negative")
    if not _finite_number(result.get("perplexity")) or result["perplexity"] <= 0:
        raise EvidenceError(f"{branch} holdout perplexity must be finite and positive")
    expected_perplexity = math.exp(float(result["cross_entropy"]))
    if not math.isclose(result["perplexity"], expected_perplexity, rel_tol=1e-6, abs_tol=1e-6):
        raise EvidenceError(f"{branch} holdout perplexity contradicts cross_entropy")


def _validate_quality(quality: Mapping[str, Any], holdouts: Mapping[str, Mapping[str, Any]]) -> None:
    if quality.get("schema") != "q4-t3-quality" or quality.get("schema_version") != 1:
        raise EvidenceError("quality payload has unsupported schema")
    if quality.get("threshold_version") != "q4-t3-quality-thresholds-v1":
        raise EvidenceError("quality threshold version is not Slice D v1")
    if not _finite_number(quality.get("control_ppl")) or not _finite_number(quality.get("qat_ppl")) or not _finite_number(quality.get("relative_degradation")):
        raise EvidenceError("quality payload contains non-finite values")
    if not math.isclose(quality["control_ppl"], holdouts["control"]["perplexity"], rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(quality["qat_ppl"], holdouts["qat"]["perplexity"], rel_tol=1e-12, abs_tol=1e-12):
        raise EvidenceError("quality perplexity inputs contradict holdout results")
    from trainer.q4_t3_validation import evaluate_quality_gate

    expected = evaluate_quality_gate(quality["control_ppl"], quality["qat_ppl"])
    if quality.get("classification") != expected["classification"] or not math.isclose(quality.get("relative_degradation"), expected["relative_degradation"], rel_tol=1e-12, abs_tol=1e-12):
        raise EvidenceError("quality classification or degradation contradicts Slice D formula")


def _validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("schema") != "q4-t3-spec-q4-validation" or spec.get("schema_version") != 1 or spec.get("parity_status") != "PASS":
        raise EvidenceError("SPEC-Q4 validation payload is incomplete or failed")
    storage = spec.get("storage")
    if not isinstance(storage, Mapping):
        raise EvidenceError("SPEC-Q4 storage summary is missing")
    from trainer.q4_t3_validation import compute_spec_q4_storage, expected_qat_dense_shapes

    expected = compute_spec_q4_storage(expected_qat_dense_shapes(), require_full_model=True)
    for key in ("tensor_count", "weight_count"):
        if spec.get(key) != expected[key]:
            raise EvidenceError(f"SPEC-Q4 full-model {key} does not match D contract")
    for key in ("tensor_count", "weight_count", "packed_e2m1_bytes", "scale_e8m0_bytes", "total_bytes"):
        if storage.get(key) != expected[key]:
            raise EvidenceError(f"SPEC-Q4 storage field {key} contradicts D arithmetic")
    tensors = spec.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != expected["tensor_count"]:
        raise EvidenceError("SPEC-Q4 tensor parity details are incomplete or failed")
    expected_shapes_by_name = expected_qat_dense_shapes()
    expected_shapes = sorted(expected_shapes_by_name.values())
    actual_shapes: list[tuple[int, ...]] = []
    normalized_tensors: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for tensor in tensors:
        if not isinstance(tensor, Mapping) or tensor.get("parity_status") != "PASS":
            raise EvidenceError("SPEC-Q4 tensor parity details are incomplete or failed")
        name = tensor.get("name")
        shape_value = tensor.get("shape")
        if not isinstance(name, str) or not name or name in seen_names:
            raise EvidenceError("SPEC-Q4 tensor identifiers must be unique strings")
        if not isinstance(shape_value, (list, tuple)) or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in shape_value):
            raise EvidenceError(f"SPEC-Q4 tensor shape is invalid: {name}")
        shape = tuple(shape_value)
        if name not in expected_shapes_by_name or shape != expected_shapes_by_name[name]:
            raise EvidenceError(f"SPEC-Q4 tensor shape does not match dense MiniMind contract: {name}")
        actual_shapes.append(shape)
        tensor_storage = compute_spec_q4_storage({name: shape})
        if tensor.get("weight_count") != tensor_storage["weight_count"]:
            raise EvidenceError(f"SPEC-Q4 tensor weight count contradicts shape: {name}")
        for key in ("packed_e2m1_bytes", "scale_e8m0_bytes", "total_bytes"):
            if key in tensor and tensor[key] != tensor_storage[key]:
                raise EvidenceError(f"SPEC-Q4 tensor {key} contradicts shape: {name}")
        normalized = dict(tensor)
        normalized.update({key: tensor_storage[key] for key in ("packed_e2m1_bytes", "scale_e8m0_bytes", "total_bytes")})
        normalized_tensors.append(normalized)
        seen_names.add(name)
    if sorted(actual_shapes) != expected_shapes:
        raise EvidenceError("SPEC-Q4 tensor shapes do not match full dense MiniMind contract")
    if seen_names != set(expected_shapes_by_name):
        raise EvidenceError("SPEC-Q4 tensor names do not match full dense MiniMind contract")
    normalized_spec = dict(spec)
    normalized_spec["storage"] = dict(expected)
    normalized_spec["tensors"] = normalized_tensors
    return normalized_spec


def _validate_diagnostics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    if (
        diagnostics.get("schema") != "q4-t3-diagnostics"
        or diagnostics.get("version") != "v1"
        or diagnostics.get("informational") is not True
    ):
        raise EvidenceError("diagnostics payload has unsupported schema or status")
    from trainer.q4_t3_diagnostics import load_diagnostic_fixture

    fixture = load_diagnostic_fixture()
    if diagnostics.get("prompts_sha256") != fixture["prompts_sha256"] or diagnostics.get("policy_sha256") != fixture["policy_sha256"]:
        raise EvidenceError("diagnostics prompt or policy hash does not match v1 fixture")
    branches = diagnostics.get("branches")
    if not isinstance(branches, Mapping) or set(branches) != {"control", "qat"}:
        raise EvidenceError("diagnostics must contain CONTROL and QAT branches")
    expected_ids = [item["id"] for item in fixture["prompts"]]
    expected_seed = fixture["sampling_policy"]["seed"]
    for branch_name in ("control", "qat"):
        branch = branches[branch_name]
        if not isinstance(branch, Mapping) or branch.get("branch") != branch_name or branch.get("seed") != expected_seed:
            raise EvidenceError(f"diagnostics {branch_name} branch identity is invalid")
        outputs = branch.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(expected_ids):
            raise EvidenceError(f"diagnostics {branch_name} must contain exactly 32 outputs")
        for item, expected_id in zip(outputs, expected_ids):
            if not isinstance(item, Mapping) or item.get("id") != expected_id or not isinstance(item.get("text"), str):
                raise EvidenceError(f"diagnostics {branch_name} output list is invalid")
            if len(item["text"].encode("utf-8", errors="replace")) > MAX_LOG_BYTES:
                raise EvidenceError(f"diagnostics {branch_name} output is oversized")
    return dict(diagnostics)


def _validate_runner_metric_agreement(runner: Mapping[str, Any], metrics: Mapping[str, Mapping[str, Any]]) -> None:
    for stage, runner_stage in zip(_EXPECTED_STAGES, runner["stages"]):
        runner_metrics = runner_stage["metrics"]
        supplied_metrics = metrics[stage]
        for key in (
            "status", "training_failed", "finite_loss", "finite_gradients", "optimizer_steps",
            "samples_processed", "tokens_processed", "initial_loss", "final_loss",
            "first_10_percent_loss_median", "last_10_percent_loss_median",
        ):
            if runner_metrics.get(key) != supplied_metrics.get(key):
                raise EvidenceError(f"runner {stage} metric contradicts supplied {stage} metrics: {key}")
        for nested_key in ("parameter_update_probe", "master_vs_parent"):
            runner_nested = runner_metrics.get(nested_key)
            supplied_nested = supplied_metrics.get(nested_key)
            if isinstance(runner_nested, Mapping) and isinstance(supplied_nested, Mapping):
                for key, value in runner_nested.items():
                    if key in supplied_nested and supplied_nested[key] != value:
                        raise EvidenceError(f"runner {stage} nested metric contradicts supplied metrics: {nested_key}.{key}")


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"missing required identity field: {label}")
    if (
        _ABSOLUTE.search(value)
        or _SECRET_VALUE.search(value)
        or _USER_HOST.search(value)
        or _USER_HOST_ASSIGNMENT.search(value)
    ):
        raise EvidenceError(f"unsafe identity value: {label}")
    return value


def _required_sha(value: Any, label: str) -> str:
    value = _required_text(value, label)
    if not _HEX64.fullmatch(value):
        raise EvidenceError(f"{label} must be a SHA-256 hex digest")
    return value.lower()


def validate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate required runtime identity and explicit Rol §9 deviations."""

    _scan(identity)
    commit = _required_text(identity.get("harness_git_commit"), "harness_git_commit")
    if not _COMMIT.fullmatch(commit):
        raise EvidenceError("harness_git_commit must be a hexadecimal commit identifier")
    parent = identity.get("parent")
    if not isinstance(parent, Mapping):
        raise EvidenceError("missing required identity field: parent")
    parent_payload = {
        "sha256": _required_sha(parent.get("sha256"), "parent.sha256"),
        "dtype": _required_text(parent.get("dtype"), "parent.dtype"),
    }
    environment = identity.get("environment")
    if not isinstance(environment, Mapping):
        raise EvidenceError("missing required identity field: environment")
    environment_payload = {
        "gpu": _required_text(environment.get("gpu"), "environment.gpu"),
        "vram_gb": environment.get("vram_gb"),
        "cuda": _required_text(environment.get("cuda"), "environment.cuda"),
        "driver": _required_text(environment.get("driver"), "environment.driver"),
        "price_per_hour": environment.get("price_per_hour"),
    }
    if not isinstance(environment_payload["vram_gb"], (int, float)) or isinstance(environment_payload["vram_gb"], bool) or not math.isfinite(float(environment_payload["vram_gb"])) or environment_payload["vram_gb"] <= 0:
        raise EvidenceError("environment.vram_gb must be finite and positive")
    if not isinstance(environment_payload["price_per_hour"], (int, float)) or isinstance(environment_payload["price_per_hour"], bool) or not math.isfinite(float(environment_payload["price_per_hour"])) or environment_payload["price_per_hour"] < 0:
        raise EvidenceError("environment.price_per_hour must be finite and non-negative")
    deviations = identity.get("deviations")
    if not isinstance(deviations, Mapping):
        raise EvidenceError("missing required identity field: deviations")
    deviation_payload: dict[str, Any] = {}
    for key in ("torch", "cuda", "gpu"):
        item = deviations.get(key)
        if not isinstance(item, Mapping):
            raise EvidenceError(f"missing required deviation: {key}")
        reference = _required_text(item.get("reference"), f"deviations.{key}.reference")
        if reference != _ROL9_REFERENCES[key]:
            raise EvidenceError(f"deviations.{key}.reference must match Rol §9")
        deviation_payload[key] = {
            "reference": reference,
            "actual": _required_text(item.get("actual"), f"deviations.{key}.actual"),
        }
    return {
        "schema": "q4-t3-evidence-identity",
        "schema_version": EVIDENCE_VERSION,
        "harness_git_commit": commit,
        "parent": parent_payload,
        "plan_sha256": _required_sha(identity.get("plan_sha256"), "plan_sha256"),
        "holdout_sha256": _required_sha(identity.get("holdout_sha256"), "holdout_sha256"),
        "dataset_manifest_sha256": _required_sha(identity.get("dataset_manifest_sha256"), "dataset_manifest_sha256"),
        "environment": environment_payload,
        "deviations": deviation_payload,
    }


def sanitize_log(text: str, *, max_bytes: int = MAX_LOG_BYTES) -> str:
    """Redact paths, secrets, tokens, and host identity from bounded logs."""

    if len(text.encode("utf-8", errors="replace")) > max_bytes:
        raise EvidenceError("stage log exceeds size limit")
    value = re.sub(
        r"(?i)(--?[A-Za-z0-9_-]*(?:token|secret|password|credential|api[_-]?key|private[_-]?key)[A-Za-z0-9_-]*)(?:=|\s+)\S+",
        r"\1=<redacted>",
        text,
    )
    value = re.sub(r"(?i)\b(?:AWS|HF|RUNPOD|SSH)_[A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD)\s*=\s*\S+", "<redacted-env>", value)
    value = re.sub(r"(?i)\b(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]+\b", "<redacted-token>", value)
    value = re.sub(r"(?i)\b\S+@\S+\b", "<redacted-user-host>", value)
    value = re.sub(r"(?:(?:[A-Za-z]:[\\/])|(?:\\\\)|/(?:[^\s,;]+/)?[^\s,;]+)[^\s,;]*", "<redacted-path>", value)
    value = re.sub(
        r"(?i)(?:--?)?\b(?:user(?:name)?|hostname|host|server|node)\s*(?:[=:]|\s+)\s*\S+",
        "<redacted-user-host>",
        value,
    )
    if len(value.encode("utf-8", errors="replace")) > max_bytes:
        raise EvidenceError("sanitized stage log exceeds size limit")
    return value


def _read_log(source: Any, label: str, repo_root: Path) -> str:
    if not isinstance(source, (str, os.PathLike)):
        raise EvidenceError(f"{label} must be an approved runtime log path")
    path = Path(source)
    if not path.is_absolute():
        raise EvidenceError(f"{label} must use an absolute runtime log path")
    if path.is_symlink():
        raise EvidenceError(f"{label} symlink sources are not allowed")
    if path.suffix.lower() != ".log":
        raise EvidenceError(f"{label} log source must use .log extension")
    resolved = path.resolve()
    approved_roots = (
        (repo_root / "logs").resolve(),
        (repo_root / "runtime_logs").resolve(),
        (repo_root / EVIDENCE_RELATIVE_PATH).resolve(),
    )
    if not any(resolved == root or root in resolved.parents for root in approved_roots):
        raise EvidenceError(f"{label} log source is not an approved runtime log location")
    try:
        source_stat = path.stat()
    except OSError as exc:
        raise EvidenceError(f"{label} log source is unreadable") from exc
    if not stat.S_ISREG(source_stat.st_mode):
        raise EvidenceError(f"{label} log source must be a regular file")
    parts = {part.lower() for part in path.parts}
    if parts & {"dataset", "datasets", "checkpoint", "checkpoints", "out"}:
        raise EvidenceError(f"{label} log source is outside approved runtime locations")
    if source_stat.st_size > MAX_LOG_BYTES:
        raise EvidenceError(f"{label} exceeds log size limit")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise EvidenceError(f"{label} log source is unreadable") from exc
    return sanitize_log(text, max_bytes=MAX_LOG_BYTES)


def _json_bytes(payload: Any) -> bytes:
    try:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EvidenceError("evidence contains non-JSON values") from exc
    if len(encoded) > MAX_FILE_BYTES:
        raise EvidenceError("evidence file exceeds size limit")
    return encoded


def _target_dir(repo_root: str | Path, output_dir: str | Path | None) -> Path:
    root = Path(repo_root).resolve()
    expected = root / EVIDENCE_RELATIVE_PATH
    target = expected if output_dir is None else Path(output_dir)
    if not target.is_absolute():
        target = Path.cwd() / target
    if os.path.normcase(os.path.normpath(str(target))) != os.path.normcase(os.path.normpath(str(expected))):
        raise EvidenceError("evidence output must be native_cpu/benchmarks/q4-t3")
    current = target
    while True:
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise EvidenceError("evidence output path may not contain symlinks")
        except FileNotFoundError:
            pass
        if current == root or current.parent == current:
            break
        current = current.parent
    return target


def _validate_target_state(target: Path, allowed_names: set[str]) -> None:
    if not target.exists():
        return
    if target.is_symlink() or not target.is_dir():
        raise EvidenceError("evidence output directory must be a regular directory")
    for child in target.iterdir():
        if child.name not in allowed_names:
            raise EvidenceError(f"unexpected stale evidence artifact: {child.name}")
        if child.is_symlink() or child.is_dir() or not child.is_file():
            raise EvidenceError(f"unsafe existing evidence artifact: {child.name}")


def _results_markdown(identity: Mapping[str, Any], quality: Mapping[str, Any], runner: Mapping[str, Any]) -> str:
    stage_status: dict[str, str] = {}
    for stage in runner.get("stages", []):
        if isinstance(stage, Mapping):
            checkpoint = stage.get("checkpoint", {})
            verified = isinstance(checkpoint, Mapping) and checkpoint.get("strict_reload") is True
            stage_status[str(stage.get("name"))] = f"exit={stage.get('exit_code')}, strict_reload={verified}"
    rows = [
        "# Q4-T3 Slice E Evidence",
        "",
        f"Outcome: **{quality.get('classification', 'UNCLASSIFIED')}**",
        f"Relative degradation: `{quality.get('relative_degradation')}` percent.",
        "",
        "| Artifact | Identity | Verification |",
        "| --- | --- | --- |",
        f"| `identity.json` | commit `{identity['harness_git_commit']}` | required runtime identity validated |",
        f"| `datasets.json` | plan `{identity['plan_sha256']}` | holdout `{identity['holdout_sha256']}` |",
        f"| `pretrain_metrics.json` | Parent `{identity['parent']['sha256']}` | {stage_status.get('pretrain', 'not supplied')} |",
        f"| `control_metrics.json` | CONTROL | {stage_status.get('control', 'not supplied')} |",
        f"| `qat_metrics.json` | QAT | {stage_status.get('qat', 'not supplied')} |",
        "| `quality.json` | Slice D quality schema | informational classification, no new gate |",
        "| `spec_q4_validation.json` | SPEC-Q4-v1 | parity/storage evidence supplied |",
        "| `diagnostics.json` | CONTROL/QAT prompt identity | informational only |",
        "",
        "Runtime logs and diagnostic outputs are bounded and sanitized. No datasets, checkpoints, credentials, absolute paths, usernames, or hostnames are included.",
        "",
    ]
    return "\n".join(rows)


def assemble_q4_t3_evidence(
    repo_root: str | Path,
    *,
    runner_manifest: Any,
    pretrain_metrics: Any,
    control_metrics: Any,
    qat_metrics: Any,
    holdout_results: Mapping[str, Any],
    spec_q4_validation: Any,
    quality: Any,
    identity: Mapping[str, Any],
    stage_logs: Mapping[str, Any] | None = None,
    diagnostics: Any | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble compact Slice E evidence under the exact requested directory."""

    target = _target_dir(repo_root, output_dir)
    if not isinstance(holdout_results, Mapping) or set(holdout_results) != {"parent", "control", "qat"}:
        raise EvidenceError("holdout_results must contain parent, control, and qat")
    identity_payload = validate_identity(identity)
    runner_raw = _load_json(runner_manifest, "runner_manifest")
    _validate_runner_manifest(runner_raw, identity_payload)
    raw_metrics = {
        "pretrain": _load_json(pretrain_metrics, "pretrain_metrics"),
        "control": _load_json(control_metrics, "control_metrics"),
        "qat": _load_json(qat_metrics, "qat_metrics"),
    }
    for stage, metrics in raw_metrics.items():
        _validate_metric_payload(metrics, stage)
    _validate_runner_metric_agreement(runner_raw, raw_metrics)
    metric_payloads = {
        "pretrain_metrics.json": _compact(raw_metrics["pretrain"], _METRIC_KEYS),
        "control_metrics.json": _compact(raw_metrics["control"], _METRIC_KEYS),
        "qat_metrics.json": _compact(raw_metrics["qat"], _METRIC_KEYS),
    }
    holdout_payload: dict[str, Any] = {}
    raw_holdouts: dict[str, Mapping[str, Any]] = {}
    for branch, source in holdout_results.items():
        holdout = _load_json(source, f"{branch}_holdout")
        _validate_holdout_result(holdout, branch, identity_payload)
        raw_holdouts[branch] = holdout
        holdout_payload[branch] = _compact(holdout, _HOLDOUT_KEYS)
    raw_quality = _load_json(quality, "quality")
    _validate_quality(raw_quality, raw_holdouts)
    quality_payload = _compact(raw_quality, _QUALITY_KEYS)
    raw_spec = _load_json(spec_q4_validation, "spec_q4_validation")
    spec_payload = _compact(_validate_spec(raw_spec), _SPEC_KEYS)
    runner_payload = _compact(runner_raw, _RUNNER_KEYS)
    datasets_payload = {
        "schema": "q4-t3-evidence-datasets",
        "runner_sft": runner_payload.get("sft", {}),
        "holdout_results": holdout_payload,
    }
    payloads: dict[str, Any] = {
        "identity.json": identity_payload,
        "datasets.json": datasets_payload,
        **metric_payloads,
        "quality.json": quality_payload,
        "spec_q4_validation.json": spec_payload,
    }
    if diagnostics is None:
        payloads["diagnostics.json"] = {
            "schema": "q4-t3-diagnostics",
            "version": "v1",
            "informational": True,
            "status": "absent",
        }
    else:
        raw_diagnostics = _load_json(diagnostics, "diagnostics")
        payloads["diagnostics.json"] = _compact(_validate_diagnostics(raw_diagnostics), _DIAGNOSTIC_KEYS)
    log_payloads: dict[str, str] = {}
    for name, source in (stage_logs or {}).items():
        safe_name = Path(str(name)).name
        if safe_name != str(name) or safe_name in {"", ".", ".."}:
            raise EvidenceError("stage log names must be simple filenames")
        if safe_name in {*payloads, "RESULTS.md"}:
            raise EvidenceError("stage log would overwrite required evidence")
        log_payloads[safe_name] = _read_log(source, safe_name, Path(repo_root).resolve())

    markdown = _results_markdown(identity_payload, quality_payload, runner_payload)
    encoded_payloads = {name: _json_bytes(payload) for name, payload in payloads.items()}
    encoded_logs = {name: value.encode("utf-8") for name, value in log_payloads.items()}
    markdown_bytes = markdown.encode("utf-8")
    if len(markdown_bytes) > MAX_FILE_BYTES:
        raise EvidenceError("RESULTS.md exceeds size limit")
    total_bytes = sum(len(value) for value in encoded_payloads.values()) + sum(len(value) for value in encoded_logs.values()) + len(markdown_bytes)
    if total_bytes > MAX_TOTAL_BYTES:
        raise EvidenceError("evidence output exceeds total size limit")
    allowed_names = {*payloads, *log_payloads, "RESULTS.md"}
    _validate_target_state(target, allowed_names)
    target.mkdir(parents=True, exist_ok=True)
    _validate_target_state(target, allowed_names)
    for name, data in encoded_payloads.items():
        (target / name).write_bytes(data)
    for name, data in encoded_logs.items():
        (target / name).write_bytes(data)
    (target / "RESULTS.md").write_bytes(markdown_bytes)
    return {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_VERSION,
        "output_relative_path": str(EVIDENCE_RELATIVE_PATH).replace("\\", "/"),
        "files": sorted([*payloads, *log_payloads, "RESULTS.md"]),
        "total_bytes": total_bytes,
    }


__all__ = [
    "EVIDENCE_RELATIVE_PATH",
    "EvidenceError",
    "assemble_q4_t3_evidence",
    "sanitize_log",
    "validate_identity",
]
