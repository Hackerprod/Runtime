import json
import math

import pytest
import torch

from trainer.q4_t3_metrics import (
    MetricsCollector,
    check_finite_gradient_norm,
    check_finite_loss,
    finish_parent_delta_evidence,
    is_metrics_writer,
    summarize_parent_delta,
    summarize_weight_differences,
)


def _labels(batch_size=2):
    return torch.tensor([[0, 0, -100, -100]] * batch_size)


def test_collector_writes_compact_metrics_and_accounts_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "metrics.json"
    collector = MetricsCollector(
        "control",
        config={"batch_size": 2},
        metrics_path=path,
        planned_steps=10,
    )
    collector.start()
    for step in range(1, 11):
        collector.record_batch(float(step), labels=_labels(), duration_seconds=step / 1000)
    for _ in range(3):
        collector.record_optimizer_step()

    metrics = collector.finish(duration_seconds=2.0)
    written = json.loads(path.read_text(encoding="utf-8"))

    assert metrics == written
    assert metrics["initial_loss"] == 1.0
    assert metrics["final_loss"] == 10.0
    assert metrics["first_10_percent_loss_median"] == 1.0
    assert metrics["last_10_percent_loss_median"] == 10.0
    assert metrics["optimizer_steps"] == 3
    assert metrics["samples_processed"] == 20
    assert metrics["tokens_processed"] == 20
    assert metrics["median_step_ms"] == 5.5
    assert metrics["tokens_per_second"] == 10.0
    assert metrics["peak_allocated_vram_bytes"] == 0
    assert metrics["peak_reserved_vram_bytes"] == 0
    assert metrics["finite_loss"] is True
    assert metrics["finite_gradients"] is True
    assert "losses" not in metrics
    assert "index_plan" not in metrics


def test_collector_rejects_nonfinite_loss_and_gradient():
    collector = MetricsCollector("pretrain")
    with pytest.raises(RuntimeError, match="non-finite pretrain loss"):
        collector.record_batch(math.nan, labels=_labels())
    assert collector.finite_loss is False

    with pytest.raises(RuntimeError, match="non-finite pretrain gradient norm"):
        collector.record_gradient_norm(float("inf"))
    assert collector.finite_gradients is False


def test_collector_storage_is_bounded_and_median_sampling_is_documented():
    collector = MetricsCollector("pretrain", planned_steps=200, sample_capacity=3)
    for step in range(200):
        collector.record_batch(
            float(step),
            labels=_labels(1),
            duration_seconds=float(step),
        )

    assert len(collector._first_losses.values) == 3
    assert len(collector._last_losses.values) == 3
    assert len(collector._step_durations.values) == 3
    metrics = collector.finish(duration_seconds=1.0)
    assert metrics["median_aggregation"] == {
        "method": "deterministic_reservoir",
        "capacity": 3,
        "first_loss_exact": False,
        "last_loss_exact": False,
        "step_duration_exact": False,
    }


def test_rank_zero_is_sole_writer_and_counts_are_rank_local(tmp_path):
    assert is_metrics_writer(0)
    assert not is_metrics_writer(1)
    path = tmp_path / "rank-one.json"
    collector = MetricsCollector("control", metrics_path=path, rank=1)
    collector.record_batch(1.0, labels=_labels(1))
    metrics = collector.finish(duration_seconds=1.0)
    assert not path.exists()
    assert metrics["rank"] == 1
    assert metrics["accounting_scope"] == "rank_local"
    assert metrics["evidence_authority"] == "rank_0"


def test_finite_helpers_mark_collector_before_raising():
    collector = MetricsCollector("qat")
    with pytest.raises(RuntimeError, match="non-finite qat loss"):
        check_finite_loss(float("nan"), stage="qat", collector=collector)
    with pytest.raises(RuntimeError, match="non-finite qat gradient norm"):
        check_finite_gradient_norm(float("inf"), stage="qat", collector=collector)
    assert collector.finite_loss is False
    assert collector.finite_gradients is False


def test_optimizer_flush_does_not_add_fake_batch_timing():
    collector = MetricsCollector("control")
    collector.record_batch(1.0, labels=_labels(1), duration_seconds=0.25)
    collector.record_optimizer_step()
    metrics = collector.finish(duration_seconds=1.0)
    assert len(collector._step_durations.values) == 1
    assert metrics["median_step_ms"] == 250.0


def test_weight_difference_summary_is_global_and_parent_delta_is_key_bounded():
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[2.0, 4.0]])
    summary = summarize_weight_differences([(first, second)])
    assert summary["matrix_count"] == 1
    assert summary["mean_abs"] == 1.5
    assert math.isclose(summary["rms"], math.sqrt(2.5))
    assert summary["max_abs"] == 2.0

    parent = {"a.weight": torch.zeros(2), "ignored.weight": torch.ones(2)}
    master = {"a.weight": torch.ones(2), "ignored.weight": torch.zeros(2)}
    delta = summarize_parent_delta(master, parent, ["a.weight"])
    assert delta["key_count"] == 1
    assert delta["master_changed_vs_parent"] is True
    assert delta["mean_abs"] == 1.0


def test_parent_delta_fails_for_missing_or_empty_keys():
    with pytest.raises(ValueError, match="at least one key"):
        summarize_parent_delta({}, {}, [])
    with pytest.raises(ValueError, match="missing parent/master keys"):
        summarize_parent_delta({"a.weight": torch.ones(1)}, {}, ["a.weight"])


def test_parent_delta_evidence_is_persisted_before_unchanged_failure(tmp_path):
    path = tmp_path / "qat-metrics.json"
    collector = MetricsCollector("qat", metrics_path=path)
    weights = {"layer.weight": torch.ones(2)}
    with pytest.raises(RuntimeError, match="did not change"):
        finish_parent_delta_evidence(
            collector,
            weights,
            weights,
            ["layer.weight"],
            extra={"quantization_error_start": {"matrix_count": 24}},
            duration_seconds=2.0,
        )

    metrics = json.loads(path.read_text(encoding="utf-8"))
    assert metrics["master_vs_parent"]["master_changed_vs_parent"] is False
    assert metrics["master_vs_parent"]["mean_abs"] == 0.0
    assert metrics["master_vs_parent"]["max_abs"] == 0.0
    assert metrics["training_failed"] is True
    assert metrics["status"] == "failed"
    assert metrics["quantization_error_start"]["matrix_count"] == 24
