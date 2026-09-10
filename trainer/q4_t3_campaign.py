"""Pure Q4-T3 Slice A bindings and deterministic holdout evaluation.

This module deliberately keeps Transformers, MiniMind, SFTDataset, and QAT
imports inside runtime functions so plan and metric tests stay portable.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping
from typing import Any

import torch
from torch.nn import functional as F

from trainer.sft_plan import build_dataset_identity, build_deterministic_sft_loader


_SENSITIVE_KEY_PARTS = ("secret", "token", "password", "private", "credential", "api_key")


def _json_safe(value: Any, key: str = "") -> Any:
    lowered_key = key.lower()
    if any(part in lowered_key for part in _SENSITIVE_KEY_PARTS):
        return None
    if isinstance(value, Mapping):
        return {
            str(child_key): safe_value
            for child_key, child_value in value.items()
            if (safe_value := _json_safe(child_value, str(child_key))) is not None
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(child, key) for child in value]
    if "path" in lowered_key or lowered_key.endswith("dir"):
        try:
            return os.path.basename(os.fspath(value))
        except TypeError:
            return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_campaign_identity(
    dataset_path: str | os.PathLike[str],
    *,
    git_commit: str | None = None,
    environment: Mapping[str, Any] | None = None,
    parent_sha256: str | None = None,
    parent_dtype: str | None = None,
    parent_basename: str | None = None,
    plan_sha256: str | None = None,
    holdout_sha256: str | None = None,
) -> dict[str, Any]:
    """Return compact JSON-safe identity fields for future identity.json."""

    identity: dict[str, Any] = {
        "schema": "q4-t3-slice-a",
        "dataset": build_dataset_identity(dataset_path),
    }
    if git_commit is not None:
        identity["git_commit"] = str(git_commit)
    if environment is not None:
        identity["environment"] = _json_safe(environment)
    parent = {
        key: value
        for key, value in {
            "sha256": parent_sha256,
            "dtype": parent_dtype,
            "basename": parent_basename,
        }.items()
        if value is not None
    }
    if parent:
        identity["parent"] = parent
    sft = {
        key: value
        for key, value in {
            "plan_sha256": plan_sha256,
            "holdout_sha256": holdout_sha256,
        }.items()
        if value is not None
    }
    if sft:
        identity["sft"] = sft
    return identity


def summarize_state_dict_dtypes(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize actual checkpoint tensor dtypes without importing model code."""

    dtypes = sorted({str(value.dtype).replace("torch.", "") for value in state_dict.values() if isinstance(value, torch.Tensor)})
    floating_dtypes = sorted({
        str(value.dtype).replace("torch.", "")
        for value in state_dict.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    })
    return {
        "state_dict_dtypes": dtypes,
        "floating_dtypes": floating_dtypes,
        "parent_dtype": floating_dtypes[0] if len(floating_dtypes) == 1 else "mixed",
    }


def _checkpoint_sha256(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with open(path, "rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


@contextmanager
def _preserve_global_rng():
    python_state = random.getstate()
    torch_state = torch.get_rng_state()
    numpy_state = None
    try:
        import numpy as np
        numpy_state = np.random.get_state()
    except ImportError:
        np = None
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def validate_parent_checkpoint(checkpoint_path: str | os.PathLike[str], *, device: str = "cpu") -> dict[str, Any]:
    """Strict-load post-pretrain Parent P and report its actual tensor dtypes."""

    path = os.path.abspath(os.fspath(checkpoint_path))
    checkpoint_bytes, checkpoint_sha256 = _checkpoint_sha256(path)
    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Parent P checkpoint must contain a state_dict mapping")
    dtype_summary = summarize_state_dict_dtypes(state_dict)
    if dtype_summary["parent_dtype"] != "float16":
        raise ValueError(
            f"Parent P floating tensors must be float16, got {dtype_summary['floating_dtypes']}"
        )

    with _preserve_global_rng():
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
        model = MiniMindForCausalLM(config)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
    return {
        "checkpoint_basename": os.path.basename(path),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_sha256": checkpoint_sha256,
        "strict_load": True,
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "use_moe": False,
        **dtype_summary,
    }


def validate_holdout_binding(
    plan,
    *,
    expected_plan_sha256: str,
    expected_holdout_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless plan hashes and the exact 128-index holdout match."""

    if not expected_plan_sha256 or plan.plan_sha256 != expected_plan_sha256:
        raise ValueError("SFT plan SHA-256 mismatch")
    if not expected_holdout_sha256 or plan.holdout_sha256 != expected_holdout_sha256:
        raise ValueError("SFT holdout SHA-256 mismatch")
    indices = tuple(plan.holdout_indices)
    if len(indices) != 128 or len(set(indices)) != 128:
        raise ValueError("Q4-T3 holdout must contain exactly 128 unique indices")
    if any(index < 0 or index >= plan.sample_count for index in indices):
        raise ValueError("Q4-T3 holdout contains an out-of-range raw index")
    return {
        "plan_sha256": plan.plan_sha256,
        "holdout_sha256": plan.holdout_sha256,
        "holdout_count": len(indices),
    }


def _token_weighted_ce_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [batch, sequence, vocab] and labels [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels sequence shapes must match")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        raise ValueError("holdout batch contains no valid target tokens")
    loss_sum = F.cross_entropy(
        shifted_logits[valid],
        shifted_labels[valid],
        reduction="sum",
    )
    return loss_sum, valid_count


def compute_token_weighted_ce(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    """Compute CE as total valid-token loss divided by valid-token count."""

    loss_sum, valid_token_count = _token_weighted_ce_sum(logits, labels)
    cross_entropy = float((loss_sum / valid_token_count).item())
    return {
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(cross_entropy),
        "valid_token_count": valid_token_count,
    }


def evaluate_holdout_model(
    model,
    loader,
    plan,
    *,
    batch_size: int,
    expected_plan_sha256: str,
    expected_holdout_sha256: str,
) -> dict[str, Any]:
    """Evaluate one already-loaded model on plan holdout in stored order."""

    binding = validate_holdout_binding(
        plan,
        expected_plan_sha256=expected_plan_sha256,
        expected_holdout_sha256=expected_holdout_sha256,
    )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    expected_batches = [
        list(plan.holdout_indices[start:start + batch_size])
        for start in range(0, len(plan.holdout_indices), batch_size)
    ]
    actual_batches = [list(batch) for batch in getattr(loader, "batch_sampler", ())]
    if actual_batches != expected_batches:
        raise ValueError("holdout loader does not match plan holdout order")
    if getattr(loader, "sft_num_padded_samples", None) != 0:
        raise ValueError("holdout loader must not pad examples")
    if getattr(loader, "sft_num_source_samples", None) != 128:
        raise ValueError("holdout loader must contain exactly 128 source examples")

    model.eval()
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = None
    total_loss = 0.0
    valid_token_count = 0
    sample_count = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            if model_device is not None:
                input_ids = input_ids.to(model_device)
                labels = labels.to(model_device)
            outputs = model(input_ids, labels=None)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            batch_loss, batch_tokens = _token_weighted_ce_sum(logits, labels)
            total_loss += float(batch_loss.item())
            valid_token_count += batch_tokens
            sample_count += int(input_ids.shape[0])
    if sample_count != 128:
        raise ValueError("holdout evaluation did not process exactly 128 samples")
    cross_entropy = total_loss / valid_token_count
    return {
        **binding,
        "batch_size": batch_size,
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(cross_entropy),
        "valid_token_count": valid_token_count,
        "sample_count": sample_count,
    }


def _load_eval_model(checkpoint_path, variant: str, device: str):
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

    if variant not in {"parent", "control", "qat"}:
        raise ValueError(f"unknown checkpoint variant: {variant}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, Mapping):
        raise ValueError("evaluation checkpoint must contain a state_dict mapping")
    dtype_summary = summarize_state_dict_dtypes(state_dict)
    expected_dtype = "float32" if variant == "qat" else "float16"
    if dtype_summary["parent_dtype"] != expected_dtype:
        raise ValueError(
            f"{variant} checkpoint floating tensors must be {expected_dtype}, "
            f"got {dtype_summary['floating_dtypes']}"
        )
    model = MiniMindForCausalLM(
        MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    )
    model.load_state_dict(state_dict, strict=True)
    if variant == "qat":
        from model.quantization.qat import apply_spec_q4_qat

        apply_spec_q4_qat(model)
    return model.to(device).eval()


def evaluate_sft_holdout(
    checkpoint_path: str | os.PathLike[str],
    data_path: str | os.PathLike[str],
    tokenizer_path: str | os.PathLike[str],
    plan,
    *,
    variant: str,
    batch_size: int,
    expected_plan_sha256: str,
    expected_holdout_sha256: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load one checkpoint variant and evaluate its deterministic SFT holdout."""

    validate_holdout_binding(
        plan,
        expected_plan_sha256=expected_plan_sha256,
        expected_holdout_sha256=expected_holdout_sha256,
    )
    with _preserve_global_rng():
        from transformers import AutoTokenizer
        from dataset.lm_dataset import SFTDataset

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        dataset = SFTDataset(data_path, tokenizer, max_length=768, seed=42)
        if len(dataset) != plan.sample_count:
            raise ValueError("dataset length does not match SFT index plan")
        loader = build_deterministic_sft_loader(
            dataset,
            plan,
            split="holdout",
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
        )
        model = _load_eval_model(checkpoint_path, variant, device)
        return evaluate_holdout_model(
            model,
            loader,
            plan,
            batch_size=batch_size,
            expected_plan_sha256=expected_plan_sha256,
            expected_holdout_sha256=expected_holdout_sha256,
        )


__all__ = [
    "build_campaign_identity",
    "compute_token_weighted_ce",
    "evaluate_holdout_model",
    "evaluate_sft_holdout",
    "summarize_state_dict_dtypes",
    "validate_holdout_binding",
    "validate_parent_checkpoint",
]
