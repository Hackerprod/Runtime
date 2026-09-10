"""Q4-T3 Slice C campaign orchestration.

This module keeps model and Transformers imports lazy. Pure validation helpers
can therefore run locally; stage execution is fail-closed behind CUDA and
explicit smoke/full mode gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from trainer.sft_plan import build_sft_index_plan


DEFAULT_SMOKE_MAX_STEPS = 500
CAMPAIGN_SCHEMA = "q4-t3-slice-c"
_PATH_FLAGS = {
    "--data_path",
    "--parent_path",
    "--save_dir",
    "--metrics_path",
    "--tokenizer_path",
    "--manifest_path",
    "--smoke_manifest",
}
_SENSITIVE_PARTS = ("token", "secret", "password", "credential", "api_key", "private")


class CampaignError(RuntimeError):
    """Raised when campaign preflight or evidence validation fails."""


@dataclass(frozen=True)
class StageExecution:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    sanitized_command: tuple[str, ...]


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def count_jsonl_records(data_path: str | os.PathLike[str]) -> int:
    """Count non-empty JSONL records without constructing a tokenizer dataset."""

    path = Path(data_path)
    if not path.is_file():
        raise CampaignError(f"dataset is missing: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignError(f"invalid JSONL record at line {line_number}") from exc
            if not isinstance(record, Mapping):
                raise CampaignError(f"JSONL record at line {line_number} must be an object")
            count += 1
    if count <= 0:
        raise CampaignError("dataset must contain at least one JSONL record")
    return count


def build_sft_plan_binding(
    data_path: str | os.PathLike[str],
    *,
    epochs: int = 1,
    batch_size: int = 16,
    accumulation_steps: int = 1,
    train_seed: int = 42,
    holdout_seed: int = 4242,
    holdout_size: int = 128,
) -> dict[str, Any]:
    """Build one portable SFT plan binding from raw record count and file identity."""

    path = Path(data_path)
    sample_count = count_jsonl_records(path)
    plan = build_sft_index_plan(
        path,
        sample_count,
        train_seed=train_seed,
        holdout_seed=holdout_seed,
        holdout_size=holdout_size,
        epochs=epochs,
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
    )
    if not plan.plan_sha256 or not plan.holdout_sha256:
        raise CampaignError("SFT plan and holdout bindings are required")
    return {
        "dataset_sha256": plan.dataset_sha256,
        "dataset_bytes": plan.dataset_bytes,
        "raw_record_count": sample_count,
        "plan_sha256": plan.plan_sha256,
        "holdout_sha256": plan.holdout_sha256,
        "train_seed": plan.train_seed,
        "holdout_seed": plan.holdout_seed,
        "holdout_size": plan.holdout_size,
        "epochs": plan.epochs,
        "batch_size": plan.batch_size,
        "accumulation_steps": plan.accumulation_steps,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Q4-T3 Slice C campaign runner")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true", help="run bounded smoke campaign")
    modes.add_argument("--full", action="store_true", help="run full campaign after smoke approval")
    parser.add_argument("--smoke-approved", action="store_true", help="require matching prior smoke manifest")
    parser.add_argument("--run-pretrain", action="store_true", help="create fresh Parent P before SFT branches")
    parser.add_argument("--max_steps", type=int, default=None, help="smoke steps per invocation budget")
    parser.add_argument("--sft_data", default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--pretrain_data", default="dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--parent_path", default=None, help="validated existing Parent P for smoke without pretraining")
    parser.add_argument("--output_dir", default="out/q4_t3_campaign")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--smoke_manifest", default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser


def validate_cli_options(args: argparse.Namespace) -> argparse.Namespace:
    """Validate mode contract and return a detached normalized namespace."""

    normalized = argparse.Namespace(**vars(args))
    if bool(getattr(normalized, "smoke", False) == getattr(normalized, "full", False)):
        raise CampaignError("select exactly one of --smoke or --full")
    if normalized.smoke:
        if normalized.smoke_approved:
            raise CampaignError("--smoke-approved is valid only with --full")
        normalized.max_steps = (
            DEFAULT_SMOKE_MAX_STEPS if normalized.max_steps is None else normalized.max_steps
        )
        if normalized.max_steps <= 0:
            raise CampaignError("--smoke requires positive --max_steps")
        if normalized.run_pretrain and normalized.parent_path:
            raise CampaignError("--run-pretrain creates Parent P; do not also supply --parent_path")
    else:
        if not normalized.smoke_approved:
            raise CampaignError("--full requires --smoke-approved")
        if not normalized.run_pretrain:
            raise CampaignError("--full requires --run-pretrain for a fresh full Parent P")
        if normalized.max_steps is not None:
            raise CampaignError("--max_steps is smoke-only")
        if normalized.parent_path:
            raise CampaignError("--full cannot reuse an existing or smoke Parent P")
    if not str(normalized.device).startswith("cuda"):
        raise CampaignError("campaign stages require a CUDA device")
    return normalized


def sanitize_command(command: Sequence[str]) -> list[str]:
    """Redact path and secret values while retaining compact stage arguments."""

    if isinstance(command, str):
        return ["<redacted-command>"]
    sanitized: list[str] = []
    redact_next = False
    for value in command:
        text = os.fspath(value)
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        lowered = text.lower()
        if text.startswith("--") and "=" in text:
            flag, _ = text.split("=", 1)
            flag_lowered = flag.lower()
            if flag in _PATH_FLAGS or any(part in flag_lowered for part in _SENSITIVE_PARTS):
                sanitized.append(f"{flag}=<redacted>")
                continue
        if text in _PATH_FLAGS:
            sanitized.append(text)
            redact_next = True
            continue
        if text.startswith("--") and any(part in lowered for part in _SENSITIVE_PARTS):
            sanitized.append(text)
            redact_next = True
            continue
        if any(part in lowered for part in _SENSITIVE_PARTS):
            sanitized.append("<redacted>")
            continue
        if os.path.isabs(text) or re.match(r"^[A-Za-z]:[\\/]", text):
            sanitized.append("<redacted>")
        elif text == sys.executable:
            sanitized.append(os.path.basename(text))
        else:
            sanitized.append(text)
    return sanitized


def _checkpoint_binding(path: str | os.PathLike[str], expected_dtype: str) -> dict[str, Any]:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise CampaignError(f"required checkpoint is missing: {checkpoint}")
    checkpoint_bytes, checkpoint_sha256 = _sha256_file(checkpoint)
    state_dict = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise CampaignError("checkpoint must contain a state_dict mapping")
    floating_dtypes = sorted({
        str(value.dtype).replace("torch.", "")
        for value in state_dict.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    })
    actual_dtype = floating_dtypes[0] if len(floating_dtypes) == 1 else "mixed"
    if actual_dtype != expected_dtype:
        raise CampaignError(
            f"checkpoint dtype mismatch: expected {expected_dtype}, got {floating_dtypes}"
        )
    return {
        "sha256": checkpoint_sha256,
        "bytes": checkpoint_bytes,
        "dtype": actual_dtype,
    }


def validate_smoke_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    plan_binding: Mapping[str, Any],
    config_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        raise CampaignError("matching prior smoke manifest is required")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, Mapping):
        raise CampaignError("smoke manifest must contain an object")
    if (
        manifest.get("schema") != CAMPAIGN_SCHEMA
        or manifest.get("mode") != "smoke"
        or manifest.get("smoke_success") is not True
    ):
        raise CampaignError("prior smoke manifest is not successful Slice C evidence")
    stages = manifest.get("stages", [])
    if not isinstance(stages, list):
        raise CampaignError("prior smoke stages must be a list")
    stage_names = {stage.get("name") for stage in stages if isinstance(stage, Mapping)}
    if {"pretrain", "control", "qat"} - stage_names:
        raise CampaignError("prior smoke manifest lacks successful stage records")
    if len(stages) != 3:
        raise CampaignError("prior smoke manifest must contain exactly three stage records")
    for stage in stages:
        if not isinstance(stage, Mapping) or stage.get("name") not in {"pretrain", "control", "qat"}:
            raise CampaignError("prior smoke manifest contains an invalid stage record")
        if stage.get("exit_code") != 0 or not isinstance(stage.get("metrics"), Mapping):
            raise CampaignError("prior smoke manifest contains failed stage records")
        _validate_stage_payload(
            stage["metrics"],
            stage=stage["name"],
            require_update_evidence=True,
            require_loss_decrease=True,
        )
        checkpoint = stage.get("checkpoint")
        expected_dtype = "float32" if stage["name"] == "qat" else "float16"
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("exists") is not True
            or checkpoint.get("verified") is not True
            or checkpoint.get("strict_reload") is not True
            or checkpoint.get("dtype") != expected_dtype
        ):
            raise CampaignError("prior smoke manifest lacks strict checkpoint verification")
    prior_parent = manifest.get("parent")
    if (
        not isinstance(prior_parent, Mapping)
        or not isinstance(prior_parent.get("sha256"), str)
        or len(prior_parent.get("sha256", "")) != 64
        or prior_parent.get("dtype") != "float16"
    ):
        raise CampaignError("prior smoke manifest lacks validated Parent P binding")
    prior_plan = manifest.get("sft", {})
    if not isinstance(prior_plan, Mapping):
        raise CampaignError("prior smoke plan binding must be an object")
    for key in plan_binding:
        if prior_plan.get(key) != plan_binding.get(key):
            raise CampaignError(f"prior smoke binding mismatch: {key}")
    if config_binding is not None:
        prior_config = manifest.get("config", {})
        if not isinstance(prior_config, Mapping):
            raise CampaignError("prior smoke configuration must be an object")
        for key, value in config_binding.items():
            if prior_config.get(key) != value:
                raise CampaignError(f"prior smoke configuration mismatch: {key}")
    prior_comparisons = manifest.get("comparisons", {})
    if not isinstance(prior_comparisons, Mapping):
        raise CampaignError("prior smoke comparisons must be an object")
    _validate_comparisons(prior_comparisons)
    return manifest


def compute_runner_comparisons(
    control_metrics: Mapping[str, Any], qat_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    control_step = control_metrics.get("median_step_ms")
    qat_step = qat_metrics.get("median_step_ms")
    if _finite_positive(control_step) and _finite_positive(qat_step):
        try:
            ratio = qat_step / control_step
        except (OverflowError, ZeroDivisionError):
            ratio = None
        if _finite_positive(ratio):
            result["slowdown_vs_control"] = ratio
            result["slowdown_unavailable_reason"] = None
        else:
            result["slowdown_vs_control"] = None
            result["slowdown_unavailable_reason"] = "computed ratio is not finite and positive"
    else:
        result["slowdown_vs_control"] = None
        result["slowdown_unavailable_reason"] = "missing or non-positive median_step_ms"

    control_memory = control_metrics.get("peak_allocated_vram_bytes")
    qat_memory = qat_metrics.get("peak_allocated_vram_bytes")
    if _finite_positive(control_memory) and _finite_positive(qat_memory):
        try:
            ratio = qat_memory / control_memory
        except (OverflowError, ZeroDivisionError):
            ratio = None
        if _finite_positive(ratio):
            result["memory_overhead_vs_control"] = ratio
            result["memory_overhead_unavailable_reason"] = None
        else:
            result["memory_overhead_vs_control"] = None
            result["memory_overhead_unavailable_reason"] = "computed ratio is not finite and positive"
    else:
        result["memory_overhead_vs_control"] = None
        result["memory_overhead_unavailable_reason"] = "missing or non-positive peak_allocated_vram_bytes"
    return result


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _finite_positive(value: Any) -> bool:
    return _finite_number(value) and value > 0


def _validate_comparisons(comparisons: Mapping[str, Any]) -> None:
    for value_key, reason_key in (
        ("slowdown_vs_control", "slowdown_unavailable_reason"),
        ("memory_overhead_vs_control", "memory_overhead_unavailable_reason"),
    ):
        if value_key not in comparisons:
            continue
        value = comparisons[value_key]
        reason = comparisons.get(reason_key)
        if value is None:
            if not isinstance(reason, str) or not reason:
                raise CampaignError(f"{value_key} null requires explicit reason")
        elif not _finite_positive(value) or reason is not None:
            raise CampaignError(f"{value_key} must be finite positive or null with reason")


def _validate_stage_payload(
    metrics: Mapping[str, Any],
    *,
    stage: str,
    require_update_evidence: bool,
    require_loss_decrease: bool,
) -> None:
    if metrics.get("status") != "success":
        raise CampaignError(f"{stage} metrics status is not success")
    if metrics.get("training_failed") is not False:
        raise CampaignError(f"{stage} metrics report training failure")
    if metrics.get("finite_loss") is not True or metrics.get("finite_gradients") is not True:
        raise CampaignError(f"{stage} metrics report non-finite values")
    optimizer_steps = metrics.get("optimizer_steps")
    if not isinstance(optimizer_steps, int) or isinstance(optimizer_steps, bool) or optimizer_steps <= 0:
        raise CampaignError(f"{stage} metric optimizer_steps must be a positive integer")
    for key in ("duration_seconds", "tokens_per_second", "median_step_ms"):
        if not _finite_positive(metrics.get(key)):
            raise CampaignError(f"{stage} metric {key} must be finite and positive")
    for key in ("samples_processed", "tokens_processed"):
        value = metrics.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise CampaignError(f"{stage} metric {key} must be a positive integer")
    for key in ("peak_allocated_vram_bytes", "peak_reserved_vram_bytes"):
        value = metrics.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CampaignError(f"{stage} metric {key} must be a non-negative integer")
    loss_values = {
        key: metrics.get(key)
        for key in (
            "initial_loss",
            "final_loss",
            "first_10_percent_loss_median",
            "last_10_percent_loss_median",
        )
    }
    if any(not _finite_number(value) for value in loss_values.values()):
        raise CampaignError(f"{stage} losses and loss medians must be finite numbers")
    if require_loss_decrease and (
        loss_values["last_10_percent_loss_median"]
        > loss_values["first_10_percent_loss_median"]
    ):
        raise CampaignError(f"{stage} median loss increased")
    if not require_update_evidence:
        return
    if stage in {"pretrain", "control"}:
        probe = metrics.get("parameter_update_probe")
        if not isinstance(probe, Mapping) or probe.get("parameters_changed") is not True:
            raise CampaignError(f"{stage} lacks nonzero parameter update evidence")
        if (
            not isinstance(probe.get("sample_count"), int)
            or isinstance(probe.get("sample_count"), bool)
            or probe.get("sample_count") <= 0
        ):
            raise CampaignError(f"{stage} parameter probe sample_count is invalid")
        if (
            not _finite_positive(probe.get("max_abs"))
            or not _finite_positive(probe.get("mean_abs"))
        ):
            raise CampaignError(f"{stage} parameter probe deltas are invalid")
    else:
        delta = metrics.get("master_vs_parent")
        if not isinstance(delta, Mapping) or delta.get("master_changed_vs_parent") is not True:
            raise CampaignError("qat lacks nonzero master-vs-Parent evidence")
        if delta.get("key_count") != 24 or delta.get("matrix_count") != 24:
            raise CampaignError("qat Parent delta must cover exactly 24 keys")
        if (
            not _finite_positive(delta.get("max_abs"))
            or not _finite_positive(delta.get("mean_abs"))
        ):
            raise CampaignError("qat Parent delta values are invalid")


def validate_stage_metrics(
    metrics_path: str | os.PathLike[str],
    *,
    stage: str,
    require_loss_decrease: bool,
) -> dict[str, Any]:
    """Validate compact stage metrics; smoke requires non-increasing loss medians."""

    path = Path(metrics_path)
    if not path.is_file():
        raise CampaignError(f"{stage} metrics JSON is missing")
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, Mapping):
        raise CampaignError(f"{stage} metrics JSON must contain an object")
    if metrics.get("stage") != stage:
        raise CampaignError(f"{stage} metrics stage mismatch")
    _validate_stage_payload(
        metrics,
        stage=stage,
        require_update_evidence=require_loss_decrease,
        require_loss_decrease=require_loss_decrease,
    )
    return metrics


def validate_stage_binding(output: str, plan_binding: Mapping[str, Any]) -> None:
    expected_plan = str(plan_binding["plan_sha256"])
    expected_holdout = str(plan_binding["holdout_sha256"])
    plan_values = re.findall(r"index plan SHA-256:\s*([0-9a-fA-F]{64})", output)
    holdout_values = re.findall(r"holdout SHA-256:\s*([0-9a-fA-F]{64})", output)
    if not plan_values or any(value.lower() != expected_plan.lower() for value in plan_values):
        raise CampaignError("stage returned mismatched or missing SFT plan SHA-256")
    if not holdout_values or any(value.lower() != expected_holdout.lower() for value in holdout_values):
        raise CampaignError("stage returned mismatched or missing SFT holdout SHA-256")


def reload_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    variant: str,
    device: str,
) -> None:
    """Strictly reload one stage checkpoint with variant-specific dtype rules."""

    expected_dtype = "float32" if variant == "qat" else "float16"
    _checkpoint_binding(checkpoint_path, expected_dtype)
    from trainer.q4_t3_campaign import _load_eval_model, validate_parent_checkpoint

    if variant == "parent":
        validate_parent_checkpoint(checkpoint_path, device=device)
    elif variant in {"control", "qat"}:
        model = _load_eval_model(checkpoint_path, variant, device)
        del model
    else:
        raise CampaignError(f"unknown checkpoint variant: {variant}")


def _stage_paths(output_dir: Path, mode: str) -> dict[str, Path]:
    prefix = "smoke" if mode == "smoke" else "full"
    return {
        "parent_dir": output_dir / f"{prefix}_parent",
        "control_dir": output_dir / f"{prefix}_control",
        "qat_dir": output_dir / f"{prefix}_qat",
        "pretrain_metrics": output_dir / f"{prefix}_pretrain_metrics.json",
        "control_metrics": output_dir / f"{prefix}_control_metrics.json",
        "qat_metrics": output_dir / f"{prefix}_qat_metrics.json",
        "pretrain_log": output_dir / f"{prefix}_pretrain.log",
        "control_log": output_dir / f"{prefix}_control.log",
        "qat_log": output_dir / f"{prefix}_qat.log",
    }


def build_stage_command(
    stage: str,
    *,
    args: argparse.Namespace,
    paths: Mapping[str, Path],
    plan_binding: Mapping[str, Any],
    python_executable: str | None = None,
) -> list[str]:
    executable = python_executable or sys.executable
    if stage == "pretrain":
        command = [
            executable,
            "train_pretrain.py",
            "--save_dir", str(paths["parent_dir"]),
            "--save_weight", "pretrain",
            "--epochs", "1",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--seed", "42",
            "--dtype", "bfloat16",
            "--batch_size", "16",
            "--accumulation_steps", "1",
            "--max_seq_len", "768",
            "--learning_rate", "5e-4",
            "--from_weight", "none",
            "--data_path", str(args.pretrain_data),
            "--device", args.device,
            "--metrics_path", str(paths["pretrain_metrics"]),
        ]
    elif stage == "control":
        command = [
            executable,
            "train_full_sft.py",
            "--save_dir", str(paths["control_dir"]),
            "--save_weight", "control",
            "--epochs", "1",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--seed", "42",
            "--dtype", "bfloat16",
            "--batch_size", "16",
            "--accumulation_steps", "1",
            "--max_seq_len", "768",
            "--learning_rate", "1e-5",
            "--holdout_seed", "4242",
            "--holdout_size", "128",
            "--from_weight", "pretrain",
            "--parent_path", str(paths["parent_path"]),
            "--data_path", str(args.sft_data),
            "--device", args.device,
            "--metrics_path", str(paths["control_metrics"]),
            "--expected_plan_sha256", str(plan_binding["plan_sha256"]),
            "--expected_holdout_sha256", str(plan_binding["holdout_sha256"]),
        ]
    elif stage == "qat":
        command = [
            executable,
            "train_qat_sft.py",
            "--save_dir", str(paths["qat_dir"]),
            "--save_weight", "qat_sft",
            "--epochs", "1",
            "--hidden_size", "768",
            "--num_hidden_layers", "8",
            "--use_moe", "0",
            "--seed", "42",
            "--dtype", "bfloat16",
            "--batch_size", "16",
            "--accumulation_steps", "1",
            "--max_seq_len", "768",
            "--learning_rate", "1e-5",
            "--holdout_seed", "4242",
            "--holdout_size", "128",
            "--parent_path", str(paths["parent_path"]),
            "--data_path", str(args.sft_data),
            "--device", args.device,
            "--metrics_path", str(paths["qat_metrics"]),
            "--expected_plan_sha256", str(plan_binding["plan_sha256"]),
            "--expected_holdout_sha256", str(plan_binding["holdout_sha256"]),
        ]
    else:
        raise CampaignError(f"unknown stage: {stage}")
    if args.smoke:
        command.extend(("--max_steps", str(args.max_steps)))
    return command


def run_stage_live(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    popen_factory: Any = subprocess.Popen,
) -> StageExecution:
    """Run stage with merged line-buffered output sent to console and log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = popen_factory(
        list(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        shell=False,
    )
    output: list[str] = []
    if process.stdout is None:
        raise CampaignError("stage process did not provide stdout pipe")
    with log_path.open("w", encoding="utf-8") as log_handle:
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)
            sys.stdout.flush()
            log_handle.write(line)
            log_handle.flush()
            output.append(line)
    process.stdout.close()
    return StageExecution(
        args=tuple(os.fspath(value) for value in command),
        returncode=int(process.wait()),
        stdout="".join(output),
        sanitized_command=tuple(sanitize_command(command)),
    )


def _compact_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        key: metrics.get(key)
        for key in (
            "status", "training_failed", "finite_loss", "finite_gradients",
            "optimizer_steps", "duration_seconds", "tokens_per_second", "median_step_ms",
            "samples_processed", "tokens_processed", "peak_allocated_vram_bytes",
            "peak_reserved_vram_bytes", "initial_loss", "final_loss",
            "first_10_percent_loss_median", "last_10_percent_loss_median",
            "parameter_update_probe", "master_vs_parent",
        )
    }


def _compact_checkpoint(checkpoint: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(checkpoint, Mapping):
        return {}
    return {
        key: checkpoint.get(key)
        for key in ("exists", "verified", "strict_reload", "dtype")
        if key in checkpoint
    }


def _compact_plan_binding(plan_binding: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "dataset_sha256", "dataset_bytes", "raw_record_count", "plan_sha256",
        "holdout_sha256", "train_seed", "holdout_seed", "holdout_size",
        "epochs", "batch_size", "accumulation_steps",
    )
    return {key: plan_binding.get(key) for key in keys if key in plan_binding}


def _compact_parent_binding(parent_binding: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(parent_binding, Mapping):
        return None
    return {
        key: parent_binding.get(key)
        for key in ("sha256", "bytes", "dtype")
        if key in parent_binding
    }


def build_manifest(
    args: argparse.Namespace,
    *,
    plan_binding: Mapping[str, Any],
    parent_binding: Mapping[str, Any] | None,
    stages: Sequence[Mapping[str, Any]],
    comparisons: Mapping[str, Any] | None,
    smoke_success: bool,
) -> dict[str, Any]:
    comparison_keys = (
        "slowdown_vs_control", "slowdown_unavailable_reason",
        "memory_overhead_vs_control", "memory_overhead_unavailable_reason",
    )
    comparison_payload = {
        key: comparisons.get(key)
        for key in comparison_keys
        if isinstance(comparisons, Mapping) and key in comparisons
    }
    _validate_comparisons(comparison_payload)
    return {
        "schema": CAMPAIGN_SCHEMA,
        "mode": "smoke" if args.smoke else "full",
        "smoke_success": bool(smoke_success),
        "config": {
            "device": str(args.device),
            "max_steps": args.max_steps if args.smoke else None,
            "epochs": 1,
            "batch_size": 16,
            "accumulation_steps": 1,
            "seed": 42,
            "holdout_seed": 4242,
            "holdout_size": 128,
        },
        "sft": _compact_plan_binding(plan_binding),
        "parent": _compact_parent_binding(parent_binding),
        "stages": [
            {
                "name": stage.get("name"),
                "command": sanitize_command(stage.get("command", ())),
                "exit_code": stage.get("exit_code"),
                "metrics": _compact_metrics(stage.get("metrics")),
                "checkpoint": _compact_checkpoint(stage.get("checkpoint")),
            }
            for stage in stages
        ],
        "comparisons": comparison_payload,
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def preflight(args: argparse.Namespace, *, repo_root: Path | None = None) -> dict[str, Any]:
    args = validate_cli_options(args)
    if not torch.cuda.is_available():
        raise CampaignError("CUDA is unavailable; no campaign subprocess launched")
    root = repo_root or Path(__file__).resolve().parents[1]
    sft_data = Path(args.sft_data)
    if not sft_data.is_absolute():
        sft_data = (root / sft_data).resolve()
    args.sft_data = str(sft_data)
    if args.run_pretrain:
        pretrain_data = Path(args.pretrain_data)
        if not pretrain_data.is_absolute():
            pretrain_data = (root / pretrain_data).resolve()
        if not pretrain_data.is_file():
            raise CampaignError("pretrain dataset is missing")
        args.pretrain_data = str(pretrain_data)
    elif not args.parent_path:
        raise CampaignError("CONTROL/QAT require validated --parent_path when --run-pretrain is absent")
    elif not Path(args.parent_path).is_absolute():
        args.parent_path = str((root / args.parent_path).resolve())
    plan_binding = build_sft_plan_binding(args.sft_data)
    if args.full:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (root / output_dir).resolve()
        smoke_manifest = Path(args.smoke_manifest) if args.smoke_manifest else output_dir / "smoke_manifest.json"
        if not smoke_manifest.is_absolute():
            smoke_manifest = (root / smoke_manifest).resolve()
        validate_smoke_manifest(
            smoke_manifest,
            plan_binding=plan_binding,
            config_binding={
                "epochs": 1,
                "batch_size": 16,
                "accumulation_steps": 1,
                "seed": 42,
                "holdout_seed": 4242,
                "holdout_size": 128,
            },
        )
    elif not args.run_pretrain:
        _checkpoint_binding(args.parent_path, "float16")
    return {"args": args, "repo_root": root, "plan": plan_binding}


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    prepared = preflight(args)
    args = prepared["args"]
    repo_root: Path = prepared["repo_root"]
    plan_binding = prepared["plan"]
    mode = "smoke" if args.smoke else "full"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    paths = _stage_paths(output_dir, mode)
    paths = dict(paths)
    paths["parent_path"] = (
        paths["parent_dir"] / "pretrain_768.pth"
        if args.run_pretrain else Path(args.parent_path).resolve()
    )
    trainer_cwd = repo_root / "trainer"
    records: list[dict[str, Any]] = []
    parent_binding: dict[str, Any] | None = None

    def execute(stage: str) -> dict[str, Any]:
        command = build_stage_command(stage, args=args, paths=paths, plan_binding=plan_binding)
        execution = run_stage_live(
            command,
            cwd=trainer_cwd,
            log_path=paths[f"{stage}_log"],
        )
        record: dict[str, Any] = {
            "name": stage,
            "command": sanitize_command(command),
            "exit_code": execution.returncode,
        }
        if execution.returncode != 0:
            records.append(record)
            raise CampaignError(f"{stage} failed with exit code {execution.returncode}")
        if stage in {"control", "qat"}:
            validate_stage_binding(execution.stdout, plan_binding)
        metric_path = paths[f"{stage}_metrics"]
        metrics = validate_stage_metrics(metric_path, stage, require_loss_decrease=args.smoke)
        variant = {"pretrain": "parent", "control": "control", "qat": "qat"}[stage]
        checkpoint = (
            paths["parent_path"] if stage == "pretrain"
            else paths[f"{stage}_dir"] / ("control_768.pth" if stage == "control" else "qat_sft_768.pth")
        )
        reload_checkpoint(checkpoint, variant=variant, device=args.device)
        record["checkpoint"] = {
            "exists": True,
            "verified": True,
            "strict_reload": True,
            "dtype": "float32" if variant == "qat" else "float16",
        }
        record["exit_code"] = execution.returncode
        record["metrics"] = metrics
        records.append(record)
        return metrics

    if args.run_pretrain:
        execute("pretrain")
        parent_binding = _checkpoint_binding(paths["parent_path"], "float16")
        reload_checkpoint(paths["parent_path"], variant="parent", device=args.device)
    else:
        parent_binding = _checkpoint_binding(paths["parent_path"], "float16")
        reload_checkpoint(paths["parent_path"], variant="parent", device=args.device)

    control_metrics = None
    qat_metrics = None
    for stage in ("control", "qat"):
        current_parent = _checkpoint_binding(paths["parent_path"], "float16")
        if current_parent["sha256"] != parent_binding["sha256"]:
            raise CampaignError("Parent P changed before SFT branch")
        metrics = execute(stage)
        if stage == "control":
            control_metrics = metrics
        else:
            qat_metrics = metrics

    comparisons = compute_runner_comparisons(control_metrics, qat_metrics)
    manifest = build_manifest(
        args,
        plan_binding=plan_binding,
        parent_binding=parent_binding,
        stages=records,
        comparisons=comparisons,
        smoke_success=True,
    )
    manifest_path = Path(args.manifest_path or output_dir / ("smoke_manifest.json" if args.smoke else "full_manifest.json"))
    _write_manifest(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_campaign(args)
    except (CampaignError, OSError, ValueError) as exc:
        print(f"Q4-T3 campaign blocked: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
