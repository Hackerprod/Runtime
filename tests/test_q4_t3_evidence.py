import json
import copy

import pytest

from trainer.q4_t3_evidence import (
    EVIDENCE_RELATIVE_PATH,
    EvidenceError,
    assemble_q4_t3_evidence,
    sanitize_log,
    validate_identity,
)


def _identity():
    return {
        "harness_git_commit": "a" * 40,
        "parent": {"sha256": "b" * 64, "dtype": "float16"},
        "plan_sha256": "c" * 64,
        "holdout_sha256": "d" * 64,
        "dataset_manifest_sha256": "e" * 64,
        "environment": {
            "gpu": "H100 SXM",
            "vram_gb": 80,
            "cuda": "12.4",
            "driver": "550.54.15",
            "price_per_hour": 4.98,
        },
        "deviations": {
            "torch": {"reference": "2.1.0/cu118", "actual": "2.4.1/cu124"},
            "cuda": {"reference": "cu118", "actual": "cu124"},
            "gpu": {"reference": "RTX 3090", "actual": "H100 SXM"},
        },
    }


def _metrics(stage):
    metrics = {
        "stage": stage,
        "status": "success",
        "training_failed": False,
        "finite_loss": True,
        "finite_gradients": True,
        "optimizer_steps": 1,
        "initial_loss": 3.0,
        "final_loss": 2.0,
        "first_10_percent_loss_median": 3.0,
        "last_10_percent_loss_median": 2.0,
        "duration_seconds": 1.0,
        "tokens_per_second": 2.0,
        "median_step_ms": 1.0,
        "samples_processed": 1,
        "tokens_processed": 1,
        "peak_allocated_vram_bytes": 1,
        "peak_reserved_vram_bytes": 1,
    }
    if stage in {"pretrain", "control"}:
        metrics["parameter_update_probe"] = {
            "method": "test",
            "sample_count": 2,
            "mean_abs": 0.1,
            "max_abs": 0.2,
            "parameters_changed": True,
        }
    else:
        metrics["master_vs_parent"] = {
            "matrix_count": 24,
            "key_count": 24,
            "mean_abs": 0.1,
            "max_abs": 0.2,
            "rms": 0.1,
            "master_changed_vs_parent": True,
        }
    return metrics


def _inputs():
    runner = {
        "schema": "q4-t3-slice-c",
        "mode": "smoke",
        "smoke_success": True,
        "config": {"max_steps": 500},
        "parent": {"sha256": "b" * 64, "dtype": "float16"},
        "sft": {
            "dataset_sha256": "e" * 64,
            "plan_sha256": "c" * 64,
            "holdout_sha256": "d" * 64,
        },
        "stages": [
            {
                "name": stage,
                "exit_code": 0,
                "metrics": _metrics(stage),
                "checkpoint": {
                    "exists": True,
                    "verified": True,
                    "strict_reload": True,
                    "dtype": "float32" if stage == "qat" else "float16",
                },
            }
            for stage in ("pretrain", "control", "qat")
        ],
    }
    spec = {
        "schema": "q4-t3-spec-q4-validation",
        "schema_version": 1,
        "spec": "SPEC-Q4-v1",
        "parity_status": "PASS",
        "tensor_count": 24,
        "weight_count": 44826624,
        "storage": {
            "tensor_count": 24,
            "weight_count": 44826624,
            "packed_e2m1_bytes": 22413312,
            "scale_e8m0_bytes": 1400832,
            "total_bytes": 23814144,
        },
        "tensors": [
            {
                "name": f"layer.{index // 3}.mlp.{('gate_proj', 'up_proj', 'down_proj')[index % 3]}.weight",
                "shape": [768, 2432] if index % 3 == 2 else [2432, 768],
                "weight_count": 1867776,
                "parity_status": "PASS",
            }
            for index in range(24)
        ],
    }
    quality = {
        "schema": "q4-t3-quality",
        "schema_version": 1,
        "control_ppl": 1.0,
        "qat_ppl": 1.0,
        "relative_degradation": 0.0,
        "classification": "PASS candidate",
        "threshold_version": "q4-t3-quality-thresholds-v1",
    }
    diagnostics = {
        "schema": "q4-t3-diagnostics",
        "version": "v1",
        "informational": True,
        "prompts_sha256": "eb9e29430c959c83f128ca865ab0d80999b5a321ae9508d31c5d3981ebc3494c",
        "policy_sha256": "37a6573b26be01b9528449525d2160efdba773c75f478df22a9c73e5194e857d",
        "branches": {
            branch: {
                "branch": branch,
                "seed": 20250909,
                "outputs": [{"id": f"p{index:02d}", "text": "ok"} for index in range(1, 33)],
            }
            for branch in ("control", "qat")
        },
    }
    return {
        "runner_manifest": runner,
        "pretrain_metrics": _metrics("pretrain"),
        "control_metrics": _metrics("control"),
        "qat_metrics": _metrics("qat"),
        "holdout_results": {
            branch: {
                "plan_sha256": "c" * 64,
                "holdout_sha256": "d" * 64,
                "holdout_count": 128,
                "sample_count": 128,
                "batch_size": 16,
                "valid_token_count": 128,
                "cross_entropy": 0.0,
                "perplexity": 1.0,
            }
            for branch in ("parent", "control", "qat")
        },
        "spec_q4_validation": spec,
        "quality": quality,
        "identity": _identity(),
        "diagnostics": diagnostics,
    }


def test_evidence_assembler_writes_required_compact_files_and_results(tmp_path):
    inputs = _inputs()
    log_source = tmp_path / "logs" / "pretrain.log"
    log_source.parent.mkdir()
    log_source.write_text("step=1 --api_token=secret C:\\Users\\alice\\x\n", encoding="utf-8")
    result = assemble_q4_t3_evidence(
        tmp_path,
        **inputs,
        stage_logs={"pretrain.log": log_source},
    )
    target = tmp_path / EVIDENCE_RELATIVE_PATH
    required = {
        "identity.json", "datasets.json", "pretrain_metrics.json", "control_metrics.json",
        "qat_metrics.json", "quality.json", "spec_q4_validation.json", "diagnostics.json", "RESULTS.md",
        "pretrain.log",
    }
    assert required == {path.name for path in target.iterdir()}
    assert set(result["files"]) == required
    assert "PASS candidate" in (target / "RESULTS.md").read_text(encoding="utf-8")
    log = (target / "pretrain.log").read_text(encoding="utf-8")
    assert "secret" not in log
    assert "C:\\Users" not in log
    encoded = "".join(path.read_text(encoding="utf-8", errors="ignore") for path in target.iterdir())
    assert "api_token=secret" not in encoded
    assert "alice" not in encoded
    metric = json.loads((target / "pretrain_metrics.json").read_text(encoding="utf-8"))
    assert metric["parameter_update_probe"]["mean_abs"] == 0.1
    qat_metric = json.loads((target / "qat_metrics.json").read_text(encoding="utf-8"))
    assert qat_metric["master_vs_parent"]["master_changed_vs_parent"] is True


def test_identity_requires_runtime_fields_and_explicit_deviations():
    identity = _identity()
    assert validate_identity(identity)["environment"]["gpu"] == "H100 SXM"
    for field in ("harness_git_commit", "plan_sha256", "holdout_sha256", "dataset_manifest_sha256", "environment", "deviations"):
        invalid = _identity()
        invalid.pop(field)
        with pytest.raises(EvidenceError, match="required"):
            validate_identity(invalid)
    unsafe = _identity()
    unsafe["environment"]["driver"] = "hostname=worker-01"
    with pytest.raises(EvidenceError, match="unsafe"):
        validate_identity(unsafe)


def test_evidence_rejects_unsafe_inputs_and_output_paths(tmp_path):
    inputs = _inputs()
    unsafe = dict(inputs["quality"])
    unsafe["api_token"] = "secret"
    with pytest.raises(EvidenceError, match="sensitive"):
        assemble_q4_t3_evidence(tmp_path, **{**inputs, "quality": unsafe})
    with pytest.raises(EvidenceError, match="output"):
        assemble_q4_t3_evidence(tmp_path, **inputs, output_dir=tmp_path / "out" / "evidence")
    for folder in ("dataset", "checkpoints", "out"):
        unsafe_log = tmp_path / folder / "stage.log"
        unsafe_log.parent.mkdir()
        unsafe_log.write_text("safe-looking log", encoding="utf-8")
        with pytest.raises(EvidenceError, match="approved|outside"):
            assemble_q4_t3_evidence(tmp_path, **inputs, stage_logs={"stage.log": unsafe_log})
    real_log = tmp_path / "logs" / "real.log"
    real_log.parent.mkdir(exist_ok=True)
    real_log.write_text("runtime", encoding="utf-8")
    symlink_log = real_log.parent / "link.log"
    try:
        symlink_log.symlink_to(real_log)
    except OSError:
        pass
    else:
        with pytest.raises(EvidenceError, match="symlink"):
            assemble_q4_t3_evidence(tmp_path, **inputs, stage_logs={"link.log": symlink_log})
    with pytest.raises(EvidenceError, match="size"):
        sanitize_log("x" * (32 * 1024 + 1))
    oversized = dict(inputs["quality"])
    oversized["extra"] = list(range(65))
    with pytest.raises(EvidenceError, match="array"):
        assemble_q4_t3_evidence(tmp_path, **{**inputs, "quality": oversized})


def test_sanitized_log_redacts_inline_flags_env_paths_and_user_hosts():
    value = sanitize_log(
        "--api_token=secret RUNPOD_API_KEY=secret alice@example.invalid "
        "C:\\Users\\alice\\checkpoint.pth /opt/runtime/checkpoint "
        "\\\\host\\share\\run.log"
    )
    assert "secret" not in value
    assert "alice@example" not in value
    assert "C:\\Users" not in value
    assert "/opt/" not in value
    assert "\\\\host" not in value


def test_sanitized_log_redacts_generic_user_and_host_assignments():
    value = sanitize_log(
        "user=alice username=alice host=foo hostname=foo server=server-one node=node-one "
        "--user bob --hostname=bar USER=carol"
    )
    assert "alice" not in value
    assert "bob" not in value
    assert "carol" not in value
    assert "foo" not in value
    assert "bar" not in value
    assert "server-one" not in value
    assert "node-one" not in value


def test_evidence_rejects_semantically_invalid_or_contradictory_inputs(tmp_path):
    cases = []
    invalid = copy.deepcopy(_inputs())
    invalid["runner_manifest"]["mode"] = "invalid"
    cases.append((invalid, "schema or mode"))
    invalid = copy.deepcopy(_inputs())
    invalid["runner_manifest"]["stages"] = []
    cases.append((invalid, "stages"))
    invalid = copy.deepcopy(_inputs())
    invalid["control_metrics"]["status"] = "failed"
    cases.append((invalid, "status success"))
    invalid = copy.deepcopy(_inputs())
    invalid["runner_manifest"]["sft"]["plan_sha256"] = "f" * 64
    cases.append((invalid, "runner plan_sha256"))
    invalid = copy.deepcopy(_inputs())
    invalid["holdout_results"]["qat"]["holdout_sha256"] = "f" * 64
    cases.append((invalid, "qat holdout"))
    invalid = copy.deepcopy(_inputs())
    invalid["quality"]["classification"] = "FAIL"
    cases.append((invalid, "quality classification"))
    invalid = copy.deepcopy(_inputs())
    invalid["spec_q4_validation"]["parity_status"] = "FAIL"
    cases.append((invalid, "SPEC-Q4"))
    for invalid, message in cases:
        with pytest.raises(EvidenceError, match=message):
            assemble_q4_t3_evidence(tmp_path, **invalid)


def test_evidence_rejects_diagnostics_identity_or_cardinality_mismatches(tmp_path):
    for mutate in (
        lambda diagnostics: diagnostics.update(prompts_sha256="f" * 64),
        lambda diagnostics: diagnostics["branches"]["control"]["outputs"].pop(),
        lambda diagnostics: diagnostics.update(informational=False),
    ):
        inputs = _inputs()
        mutate(inputs["diagnostics"])
        with pytest.raises(EvidenceError, match="diagnostics"):
            assemble_q4_t3_evidence(tmp_path, **inputs)


def test_evidence_rejects_spec_arithmetic_and_shape_mismatches(tmp_path):
    for mutate in (
        lambda spec: spec["spec_q4_validation"]["storage"].update(packed_e2m1_bytes=0),
        lambda spec: spec["spec_q4_validation"]["tensors"][0].update(weight_count=1),
        lambda spec: spec["spec_q4_validation"]["tensors"].__setitem__(0, {**spec["spec_q4_validation"]["tensors"][0], "shape": [1, 1]}),
    ):
        inputs = _inputs()
        mutate(inputs)
        with pytest.raises(EvidenceError, match="SPEC-Q4"):
            assemble_q4_t3_evidence(tmp_path, **inputs)


def test_evidence_rejects_invalid_nested_probe_and_delta_fields(tmp_path):
    for stage, nested_key, field, value in (
        ("control", "parameter_update_probe", "rms", "not-a-number"),
        ("qat", "master_vs_parent", "rms", float("nan")),
        ("control", "parameter_update_probe", "parameters_changed", 1),
        ("qat", "master_vs_parent", "matrix_count", True),
    ):
        inputs = _inputs()
        inputs[f"{stage}_metrics"][nested_key][field] = value
        inputs["runner_manifest"]["stages"][{"pretrain": 0, "control": 1, "qat": 2}[stage]]["metrics"][nested_key][field] = value
        with pytest.raises(EvidenceError):
            assemble_q4_t3_evidence(tmp_path, **inputs)


def test_evidence_rejects_final_output_symlink(tmp_path):
    expected = tmp_path / EVIDENCE_RELATIVE_PATH
    expected.parent.mkdir(parents=True)
    real_target = tmp_path / "real-evidence"
    real_target.mkdir()
    try:
        expected.symlink_to(real_target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(EvidenceError, match="symlink"):
        assemble_q4_t3_evidence(tmp_path, **_inputs())


def test_evidence_rejects_stale_or_symlink_artifacts_and_rewrites_known_files(tmp_path):
    inputs = _inputs()
    assemble_q4_t3_evidence(tmp_path, **inputs)
    target = tmp_path / EVIDENCE_RELATIVE_PATH
    (target / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(EvidenceError, match="stale"):
        assemble_q4_t3_evidence(tmp_path, **inputs)
    (target / "stale.json").unlink()
    (target / "quality.json").write_text("stale", encoding="utf-8")
    assemble_q4_t3_evidence(tmp_path, **inputs)
    assert json.loads((target / "quality.json").read_text(encoding="utf-8"))["schema"] == "q4-t3-quality"

    link = target / "spec_q4_validation.json"
    backup = target / "spec_q4_validation.bak"
    link.rename(backup)
    try:
        link.symlink_to(backup)
    except OSError:
        backup.rename(link)
        pytest.skip("symlink creation unavailable")
    with pytest.raises(EvidenceError, match="unsafe"):
        assemble_q4_t3_evidence(tmp_path, **inputs)
