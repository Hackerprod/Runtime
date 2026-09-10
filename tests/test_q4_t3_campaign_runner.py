import json
import io
import sys

import pytest

import trainer.q4_t3_campaign_runner as runner
from trainer.q4_t3_campaign_runner import (
    CampaignError,
    build_manifest,
    build_parser,
    build_stage_command,
    build_sft_plan_binding,
    compute_runner_comparisons,
    count_jsonl_records,
    run_campaign,
    run_stage_live,
    sanitize_command,
    validate_cli_options,
    validate_stage_metrics,
)


def _dataset(path, count=256):
    path.write_text(
        "".join(json.dumps({"conversations": []}) + "\n" for _ in range(count)),
        encoding="utf-8",
    )
    return path


def _args(*argv):
    return build_parser().parse_args(list(argv))


def _valid_stage_metrics(stage):
    metrics = {
        "stage": stage,
        "status": "success",
        "training_failed": False,
        "finite_loss": True,
        "finite_gradients": True,
        "optimizer_steps": 1,
        "duration_seconds": 1.0,
        "tokens_per_second": 2.0,
        "median_step_ms": 5.0,
        "samples_processed": 1,
        "tokens_processed": 1,
        "peak_allocated_vram_bytes": 1,
        "peak_reserved_vram_bytes": 1,
        "initial_loss": 3.0,
        "final_loss": 2.0,
        "first_10_percent_loss_median": 3.0,
        "last_10_percent_loss_median": 2.0,
    }
    if stage in {"pretrain", "control"}:
        metrics["parameter_update_probe"] = {
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
            "master_changed_vs_parent": True,
        }
    return metrics


def _valid_checkpoint(stage):
    return {
        "exists": True,
        "verified": True,
        "strict_reload": True,
        "dtype": "float32" if stage == "qat" else "float16",
    }


def test_cli_requires_explicit_mode_and_smoke_budget():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
    smoke = validate_cli_options(_args("--smoke"))
    assert smoke.max_steps == runner.DEFAULT_SMOKE_MAX_STEPS
    with pytest.raises(CampaignError, match="positive"):
        validate_cli_options(_args("--smoke", "--max_steps", "0"))
    with pytest.raises(CampaignError, match="smoke-approved"):
        validate_cli_options(_args("--smoke", "--smoke-approved"))
    with pytest.raises(CampaignError, match="run-pretrain"):
        validate_cli_options(_args("--full", "--smoke-approved"))


def test_raw_jsonl_count_and_plan_binding_are_portable(tmp_path):
    data_path = _dataset(tmp_path / "sft.jsonl")
    assert count_jsonl_records(data_path) == 256
    binding = build_sft_plan_binding(data_path)
    assert binding["raw_record_count"] == 256
    assert len(binding["plan_sha256"]) == 64
    assert len(binding["holdout_sha256"]) == 64
    assert "holdout_indices" not in binding
    assert "ordered_train_indices" not in binding


def test_stage_commands_share_expected_bindings_and_smoke_budget(tmp_path):
    args = validate_cli_options(_args("--smoke", "--run-pretrain"))
    paths = {
        "parent_dir": tmp_path / "parent",
        "control_dir": tmp_path / "control",
        "qat_dir": tmp_path / "qat",
        "pretrain_metrics": tmp_path / "pretrain.json",
        "control_metrics": tmp_path / "control.json",
        "qat_metrics": tmp_path / "qat.json",
        "parent_path": tmp_path / "parent" / "pretrain_768.pth",
    }
    binding = {"plan_sha256": "p" * 64, "holdout_sha256": "h" * 64}
    command = build_stage_command(
        "control", args=args, paths=paths, plan_binding=binding, python_executable=sys.executable
    )
    assert "--expected_plan_sha256" in command
    assert "p" * 64 in command
    assert "--expected_holdout_sha256" in command
    assert "h" * 64 in command
    assert command[-2:] == ["--max_steps", str(args.max_steps)]


def test_stage_commands_pin_all_campaign_hyperparameters(tmp_path):
    args = validate_cli_options(_args("--smoke", "--run-pretrain"))
    paths = {
        "parent_dir": tmp_path / "parent",
        "control_dir": tmp_path / "control",
        "qat_dir": tmp_path / "qat",
        "pretrain_metrics": tmp_path / "pretrain.json",
        "control_metrics": tmp_path / "control.json",
        "qat_metrics": tmp_path / "qat.json",
        "parent_path": tmp_path / "parent" / "pretrain_768.pth",
    }
    binding = {"plan_sha256": "p" * 64, "holdout_sha256": "h" * 64}

    def values(command):
        return {flag: command[command.index(flag) + 1] for flag in (
            "--hidden_size", "--num_hidden_layers", "--use_moe", "--seed", "--dtype",
            "--batch_size", "--accumulation_steps", "--max_seq_len", "--learning_rate", "--epochs",
        )}

    expected_model = {
        "--hidden_size": "768", "--num_hidden_layers": "8", "--use_moe": "0",
        "--seed": "42", "--dtype": "bfloat16", "--batch_size": "16",
        "--accumulation_steps": "1",
        "--max_seq_len": "768", "--epochs": "1",
    }
    for stage in ("pretrain", "control", "qat"):
        command = build_stage_command(
            stage, args=args, paths=paths, plan_binding=binding, python_executable=sys.executable
        )
        stage_values = values(command)
        assert {key: stage_values[key] for key in expected_model} == expected_model
        assert stage_values["--learning_rate"] == ("5e-4" if stage == "pretrain" else "1e-5")
        assert command.count("--max_steps") == 1
        assert command[command.index("--max_steps") + 1] == str(args.max_steps)
        if stage != "pretrain":
            assert command[command.index("--holdout_seed") + 1] == "4242"
            assert command[command.index("--holdout_size") + 1] == "128"


def test_run_stage_live_tees_merged_output_and_returns_sanitized_command(tmp_path, capsys):
    calls = {}

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO("stdout line\nstderr line\n")

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return FakeProcess()

    log_path = tmp_path / "logs" / "smoke_control.log"
    result = run_stage_live(
        [sys.executable, "train_full_sft.py", "--metrics_path", str(tmp_path / "metrics.json")],
        cwd=tmp_path,
        log_path=log_path,
        popen_factory=fake_popen,
    )

    assert result.returncode == 0
    assert result.stdout == "stdout line\nstderr line\n"
    assert log_path.read_text(encoding="utf-8") == result.stdout
    assert capsys.readouterr().out == result.stdout
    assert calls["kwargs"] == {
        "cwd": tmp_path,
        "stdout": runner.subprocess.PIPE,
        "stderr": runner.subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
        "shell": False,
    }
    assert str(tmp_path) not in json.dumps(result.sanitized_command)


def test_sanitized_command_and_manifest_contain_no_absolute_paths_or_secrets(tmp_path):
    secret = str(tmp_path / "private-token.txt")
    raw_command = [
        sys.executable,
        "train_qat_sft.py",
        "--data_path", secret,
        f"--data_path={secret}",
        "--api_token=secret-value",
    ]
    command = sanitize_command(raw_command)
    assert str(tmp_path) not in json.dumps(command)
    assert "secret-value" not in json.dumps(command)
    assert command[-1] == "--api_token=<redacted>"

    args = validate_cli_options(_args("--smoke", "--run-pretrain"))
    manifest = build_manifest(
        args,
        plan_binding={"dataset_sha256": "d", "raw_record_count": 256, "plan_sha256": "p", "holdout_sha256": "h"},
        parent_binding={"sha256": "p" * 64, "dtype": "float16"},
        stages=[{"name": "pretrain", "command": raw_command, "exit_code": 0}],
        comparisons={},
        smoke_success=True,
    )
    assert str(tmp_path) not in json.dumps(manifest)
    assert "secret-value" not in json.dumps(manifest)
    assert "stages" in manifest


def test_runner_comparison_ratios_fail_closed_without_division_by_zero():
    result = compute_runner_comparisons(
        {"median_step_ms": 10.0, "peak_allocated_vram_bytes": 100},
        {"median_step_ms": 15.0, "peak_allocated_vram_bytes": 250},
    )
    assert result["slowdown_vs_control"] == 1.5
    assert result["memory_overhead_vs_control"] == 2.5

    missing = compute_runner_comparisons({}, {"median_step_ms": 1.0, "peak_allocated_vram_bytes": 1})
    assert missing["slowdown_vs_control"] is None
    assert missing["slowdown_unavailable_reason"]
    assert missing["memory_overhead_vs_control"] is None
    assert missing["memory_overhead_unavailable_reason"]
    non_finite = compute_runner_comparisons(
        {"median_step_ms": float("nan"), "peak_allocated_vram_bytes": 1},
        {"median_step_ms": 1.0, "peak_allocated_vram_bytes": float("inf")},
    )
    assert non_finite["slowdown_vs_control"] is None
    assert non_finite["memory_overhead_vs_control"] is None
    with pytest.raises(CampaignError, match="finite positive"):
        build_manifest(
            validate_cli_options(_args("--smoke", "--run-pretrain")),
            plan_binding={"dataset_sha256": "d", "raw_record_count": 256, "plan_sha256": "p", "holdout_sha256": "h"},
            parent_binding=None,
            stages=[],
            comparisons={"slowdown_vs_control": float("inf"), "slowdown_unavailable_reason": None},
            smoke_success=False,
        )


def test_smoke_metrics_require_success_finiteness_progress_and_decreasing_loss(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(_valid_stage_metrics("control")), encoding="utf-8")
    assert validate_stage_metrics(path, stage="control", require_loss_decrease=True)["final_loss"] == 2.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_loss"] = payload["initial_loss"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_stage_metrics(path, stage="control", require_loss_decrease=True)["final_loss"] == 3.0
    payload["final_loss"] = float("inf")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CampaignError, match="finite"):
        validate_stage_metrics(path, stage="control", require_loss_decrease=True)
    payload["final_loss"] = 4.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_stage_metrics(path, stage="control", require_loss_decrease=True)["final_loss"] == 4.0

    payload["last_10_percent_loss_median"] = 4.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CampaignError, match="median loss increased"):
        validate_stage_metrics(path, stage="control", require_loss_decrease=True)

    for key, value in (
        ("first_10_percent_loss_median", None),
        ("last_10_percent_loss_median", None),
        ("first_10_percent_loss_median", float("nan")),
        ("last_10_percent_loss_median", float("inf")),
    ):
        invalid_payload = _valid_stage_metrics("control")
        invalid_payload[key] = value
        path.write_text(json.dumps(invalid_payload), encoding="utf-8")
        with pytest.raises(CampaignError, match="finite"):
            validate_stage_metrics(path, stage="control", require_loss_decrease=True)


def test_full_approval_requires_matching_successful_smoke_manifest(tmp_path):
    data_path = _dataset(tmp_path / "sft.jsonl")
    binding = build_sft_plan_binding(data_path)
    smoke_manifest = tmp_path / "smoke_manifest.json"
    smoke_args = validate_cli_options(_args("--smoke", "--run-pretrain"))
    smoke_manifest.write_text(json.dumps(build_manifest(
        smoke_args,
        plan_binding=binding,
        parent_binding={"sha256": "p" * 64, "dtype": "float16"},
        stages=[
            {
                "name": stage,
                "command": [],
                "exit_code": 0,
                "metrics": _valid_stage_metrics(stage),
                "checkpoint": _valid_checkpoint(stage),
            }
            for stage in ("pretrain", "control", "qat")
        ],

        comparisons={},
        smoke_success=True,
    )), encoding="utf-8")
    assert runner.validate_smoke_manifest(smoke_manifest, plan_binding=binding)["mode"] == "smoke"
    with pytest.raises(CampaignError, match="requires --smoke-approved"):
        validate_cli_options(_args("--full", "--run-pretrain"))


def test_incomplete_smoke_evidence_is_rejected(tmp_path):
    data_path = _dataset(tmp_path / "sft.jsonl")
    binding = build_sft_plan_binding(data_path)
    args = validate_cli_options(_args("--smoke", "--run-pretrain"))
    manifest = build_manifest(
        args,
        plan_binding=binding,
        parent_binding={"sha256": "p" * 64, "dtype": "float16"},
        stages=[
            {
                "name": stage,
                "exit_code": 0,
                "metrics": _valid_stage_metrics(stage),
                "checkpoint": _valid_checkpoint(stage),
            }
            for stage in ("pretrain", "control", "qat")
        ],
        comparisons={},
        smoke_success=True,
    )
    manifest["stages"][1]["metrics"].pop("parameter_update_probe")
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(CampaignError, match="parameter update"):
        runner.validate_smoke_manifest(path, plan_binding=binding)


def test_manifest_preserves_update_probe_and_qat_parent_delta(tmp_path):
    args = validate_cli_options(_args("--smoke", "--run-pretrain"))
    manifest = build_manifest(
        args,
        plan_binding={"dataset_sha256": "d", "raw_record_count": 256, "plan_sha256": "p", "holdout_sha256": "h"},
        parent_binding={"sha256": "p" * 64, "dtype": "float16"},
        stages=[
            {"name": "pretrain", "command": [], "exit_code": 0, "metrics": _valid_stage_metrics("pretrain")},
            {"name": "control", "command": [], "exit_code": 0, "metrics": _valid_stage_metrics("control")},
            {"name": "qat", "command": [], "exit_code": 0, "metrics": _valid_stage_metrics("qat")},
        ],
        comparisons={},
        smoke_success=True,
    )
    assert manifest["stages"][0]["metrics"]["parameter_update_probe"]["parameters_changed"] is True
    assert manifest["stages"][2]["metrics"]["master_vs_parent"]["key_count"] == 24


def test_preflight_failure_does_not_launch_subprocess(tmp_path, monkeypatch):
    data_path = _dataset(tmp_path / "sft.jsonl")
    args = _args("--smoke", "--sft_data", str(data_path))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("subprocess launched"))
    with pytest.raises(CampaignError, match="CUDA is unavailable"):
        run_campaign(args)


def test_missing_parent_fails_closed_before_branch_execution(tmp_path, monkeypatch):
    data_path = _dataset(tmp_path / "sft.jsonl")
    args = _args("--smoke", "--sft_data", str(data_path))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    with pytest.raises(CampaignError, match="--parent_path"):
        runner.preflight(args, repo_root=tmp_path)
