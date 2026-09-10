"""Informational Q4-T3 CONTROL/QAT diagnostic helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PROMPT_SCHEMA = "q4-t3-diagnostic-prompts"
DIAGNOSTICS_SCHEMA = "q4-t3-diagnostics"
PROMPT_VERSION = "v1"
DEFAULT_PROMPT_FIXTURE = Path(__file__).with_name("q4_t3_prompts_v1.json")
_MAX_PROMPT_COUNT = 32


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def load_diagnostic_fixture(path: str | Path | None = None) -> dict[str, Any]:
    """Load and verify the versioned prompt list and shared sampling policy."""

    fixture_path = Path(path) if path is not None else DEFAULT_PROMPT_FIXTURE
    with fixture_path.open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)
    if not isinstance(fixture, Mapping):
        raise ValueError("diagnostic fixture must contain an object")
    if fixture.get("schema") != PROMPT_SCHEMA or fixture.get("version") != PROMPT_VERSION:
        raise ValueError("unsupported diagnostic prompt fixture")
    prompts = fixture.get("prompts")
    policy = fixture.get("sampling_policy")
    if not isinstance(prompts, list) or len(prompts) != _MAX_PROMPT_COUNT:
        raise ValueError("diagnostic fixture must contain exactly 32 prompts")
    if not isinstance(policy, Mapping):
        raise ValueError("diagnostic fixture sampling policy must be an object")
    normalized_prompts = []
    prompt_ids = set()
    for prompt in prompts:
        if not isinstance(prompt, Mapping) or not isinstance(prompt.get("id"), str) or not isinstance(prompt.get("prompt"), str):
            raise ValueError("diagnostic prompts require string id and prompt")
        if prompt["id"] in prompt_ids or not prompt["prompt"].strip():
            raise ValueError("diagnostic prompt ids must be unique and non-empty")
        prompt_ids.add(prompt["id"])
        normalized_prompts.append({"id": prompt["id"], "prompt": prompt["prompt"]})
    normalized_policy = dict(policy)
    required_policy = {"do_sample", "temperature", "top_p", "top_k", "max_new_tokens", "seed"}
    if set(normalized_policy) != required_policy:
        raise ValueError("diagnostic sampling policy has unexpected fields")
    prompt_hash = _sha256(normalized_prompts)
    policy_hash = _sha256(normalized_policy)
    if fixture.get("prompts_sha256") != prompt_hash or fixture.get("policy_sha256") != policy_hash:
        raise ValueError("diagnostic fixture identity hash mismatch")
    return {
        "schema": PROMPT_SCHEMA,
        "version": PROMPT_VERSION,
        "prompts": normalized_prompts,
        "sampling_policy": normalized_policy,
        "prompts_sha256": prompt_hash,
        "policy_sha256": policy_hash,
    }


def diagnostic_prompt_identity(path: str | Path | None = None) -> dict[str, Any]:
    fixture = load_diagnostic_fixture(path)
    return {
        "version": fixture["version"],
        "prompt_count": len(fixture["prompts"]),
        "prompts_sha256": fixture["prompts_sha256"],
        "policy_sha256": fixture["policy_sha256"],
    }


def _branch_seed(fixture: Mapping[str, Any]) -> int:
    return int(fixture["sampling_policy"]["seed"])


def run_diagnostic_branch(
    model: Any,
    tokenizer: Any,
    *,
    branch: str,
    fixture: Mapping[str, Any] | None = None,
    max_output_chars: int = 4096,
) -> dict[str, Any]:
    """Generate compact outputs for one branch using an isolated deterministic RNG."""

    import torch

    fixture = dict(fixture or load_diagnostic_fixture())
    prompts = fixture["prompts"]
    policy = dict(fixture["sampling_policy"])
    seed = _branch_seed(fixture)
    generation_policy = {key: value for key, value in policy.items() if key != "seed"}
    was_training = bool(model.training)
    model.eval()
    outputs: list[dict[str, str]] = []
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for item in prompts:
                encoded = tokenizer(item["prompt"], return_tensors="pt")
                try:
                    device = next(model.parameters()).device
                except StopIteration:
                    device = None
                if device is not None:
                    encoded = {
                        key: value.to(device) if hasattr(value, "to") else value
                        for key, value in encoded.items()
                    }
                with torch.no_grad():
                    generated = model.generate(**encoded, **generation_policy)
                text = tokenizer.decode(generated[0], skip_special_tokens=True)
                outputs.append({"id": item["id"], "text": str(text)[:max_output_chars]})
    finally:
        model.train(was_training)
    return {
        "branch": branch,
        "seed": seed,
        "prompts_sha256": fixture["prompts_sha256"],
        "policy_sha256": fixture["policy_sha256"],
        "outputs": outputs,
    }


def run_control_qat_diagnostics(
    control_model: Any,
    qat_model: Any,
    tokenizer: Any,
    *,
    fixture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run identical diagnostics for CONTROL and QAT with independent RNG scopes."""

    fixture = dict(fixture or load_diagnostic_fixture())
    return {
        "schema": DIAGNOSTICS_SCHEMA,
        "version": PROMPT_VERSION,
        "prompts_sha256": fixture["prompts_sha256"],
        "policy_sha256": fixture["policy_sha256"],
        "informational": True,
        "branches": {
            "control": run_diagnostic_branch(control_model, tokenizer, branch="control", fixture=fixture),
            "qat": run_diagnostic_branch(qat_model, tokenizer, branch="qat", fixture=fixture),
        },
    }


def _logits_array(logits: Any):
    import numpy as np

    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    result = np.asarray(logits, dtype=np.float64)
    if result.ndim < 1 or not np.isfinite(result).all():
        raise ValueError("logits must be finite and have a final vocabulary dimension")
    return result.reshape(-1, result.shape[-1])


def compute_top1_agreement(control_logits: Any, qat_logits: Any) -> float:
    """Return fraction of rows whose top-1 logit index agrees."""

    control = _logits_array(control_logits)
    qat = _logits_array(qat_logits)
    if control.shape != qat.shape or control.shape[-1] == 0:
        raise ValueError("CONTROL and QAT logits must have matching non-empty shapes")
    return float((control.argmax(axis=-1) == qat.argmax(axis=-1)).mean())


def compute_mean_logit_kl(control_logits: Any, qat_logits: Any) -> float:
    """Return mean KL(softmax(CONTROL logits) || softmax(QAT logits))."""

    import numpy as np

    control = _logits_array(control_logits)
    qat = _logits_array(qat_logits)
    if control.shape != qat.shape or control.shape[-1] == 0:
        raise ValueError("CONTROL and QAT logits must have matching non-empty shapes")
    control_log_norm = np.logaddexp.reduce(control, axis=-1, keepdims=True)
    qat_log_norm = np.logaddexp.reduce(qat, axis=-1, keepdims=True)
    control_log_probs = control - control_log_norm
    qat_log_probs = qat - qat_log_norm
    control_probs = np.exp(control_log_probs)
    return float(np.mean(np.sum(control_probs * (control_log_probs - qat_log_probs), axis=-1)))


def build_diagnostic_metrics(control_logits: Any, qat_logits: Any) -> dict[str, Any]:
    return {
        "informational": True,
        "top1_agreement": compute_top1_agreement(control_logits, qat_logits),
        "mean_logit_kl_control_to_qat": compute_mean_logit_kl(control_logits, qat_logits),
    }


__all__ = [
    "DIAGNOSTICS_SCHEMA",
    "PROMPT_SCHEMA",
    "build_diagnostic_metrics",
    "compute_mean_logit_kl",
    "compute_top1_agreement",
    "diagnostic_prompt_identity",
    "load_diagnostic_fixture",
    "run_control_qat_diagnostics",
    "run_diagnostic_branch",
]
