"""Compact, model-independent Q4-T3 training metrics.

Median fields use deterministic reservoir samples capped at
``METRICS_SAMPLE_CAPACITY``. They are exact while each sampled population fits
the cap and approximate thereafter; no per-step arrays are retained or emitted.
Counters remain rank-local. Rank zero is the sole evidence writer in DDP.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections.abc import Iterable, Mapping
from typing import Any

import torch


METRICS_SCHEMA_VERSION = 1
METRICS_SAMPLE_CAPACITY = 1024


class _DeterministicReservoir:
    """Fixed-size deterministic reservoir for bounded online statistics."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self.values: list[float] = []
        self.seen = 0
        self._state = int(seed) & 0x7FFFFFFF

    def add(self, value: float) -> None:
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
            return
        self._state = (1103515245 * self._state + 12345) & 0x7FFFFFFF
        index = self._state % self.seen
        if index < self.capacity:
            self.values[index] = value

    @property
    def exact(self) -> bool:
        return self.seen <= self.capacity


def is_metrics_writer(rank: int) -> bool:
    """Return whether rank may write authoritative metrics evidence."""

    return int(rank) == 0


def check_finite_loss(loss: torch.Tensor | float, *, stage: str, collector=None) -> None:
    if bool(torch.isfinite(torch.as_tensor(loss)).all()):
        return
    if collector is not None:
        collector.mark_loss_nonfinite()
    raise RuntimeError(f"non-finite {stage} loss")


def check_finite_gradient_norm(
    gradient_norm: torch.Tensor | float,
    *,
    stage: str,
    collector=None,
) -> None:
    if bool(torch.isfinite(torch.as_tensor(gradient_norm)).all()):
        return
    if collector is not None:
        collector.mark_gradient_nonfinite()
    raise RuntimeError(f"non-finite {stage} gradient norm")


def summarize_weight_differences(
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    """Summarize global absolute differences without materializing a concat."""

    absolute_sum = 0.0
    square_sum = 0.0
    maximum = 0.0
    count = 0
    matrix_count = 0
    for left, right in pairs:
        difference = (
            left.detach().to(dtype=torch.float32, device="cpu")
            - right.detach().to(dtype=torch.float32, device="cpu")
        ).abs()
        absolute_sum += float(difference.sum().item())
        square_sum += float((difference * difference).sum().item())
        maximum = max(maximum, float(difference.max().item()))
        count += difference.numel()
        matrix_count += 1
    if count == 0:
        raise ValueError("cannot summarize empty weight differences")
    return {
        "matrix_count": matrix_count,
        "mean_abs": absolute_sum / count,
        "rms": math.sqrt(square_sum / count),
        "max_abs": maximum,
    }


def summarize_parent_delta(
    master_weights: Mapping[str, torch.Tensor],
    parent_weights: Mapping[str, torch.Tensor],
    keys: Iterable[str],
) -> dict[str, Any]:
    """Summarize master-vs-parent deltas over exactly the supplied keys."""

    keys = tuple(keys)
    if not keys:
        raise ValueError("parent delta requires at least one key")
    missing = [key for key in keys if key not in master_weights or key not in parent_weights]
    if missing:
        raise ValueError(f"missing parent/master keys: {missing[:3]}")
    summary = summarize_weight_differences(
        (master_weights[key], parent_weights[key]) for key in keys
    )
    summary["key_count"] = len(keys)
    summary["master_changed_vs_parent"] = summary["mean_abs"] > 0.0 and summary["max_abs"] > 0.0
    return summary


def capture_parameter_probe(model: Any, max_values: int = 64) -> tuple[float, ...]:
    """Capture bounded deterministic boundary values from trainable parameters."""

    if max_values < 1:
        raise ValueError("parameter probe capacity must be positive")
    values: list[float] = []
    for parameter in model.parameters():
        if not parameter.is_floating_point() or parameter.numel() == 0:
            continue
        flat = parameter.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        indices = (0,) if flat.numel() == 1 else (0, flat.numel() - 1)
        for index in indices:
            values.append(float(flat[index].item()))
            if len(values) >= max_values:
                return tuple(values)
    if not values:
        raise ValueError("cannot capture empty parameter probe")
    return tuple(values)


def summarize_parameter_probe(
    initial: Iterable[float],
    final: Iterable[float],
) -> dict[str, Any]:
    """Summarize bounded initial/final parameter probe differences."""

    initial_values = tuple(float(value) for value in initial)
    final_values = tuple(float(value) for value in final)
    if not initial_values or len(initial_values) != len(final_values):
        raise ValueError("parameter probes must be non-empty and equally sized")
    differences = [abs(left - right) for left, right in zip(initial_values, final_values)]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("parameter probe differences must be finite")
    return {
        "method": "deterministic_parameter_boundaries_v1",
        "sample_count": len(differences),
        "mean_abs": sum(differences) / len(differences),
        "max_abs": max(differences),
        "parameters_changed": max(differences) > 0.0,
    }


def finish_parent_delta_evidence(
    collector: "MetricsCollector | None",
    master_weights: Mapping[str, torch.Tensor],
    parent_weights: Mapping[str, torch.Tensor],
    keys: Iterable[str],
    *,
    extra: Mapping[str, Any] | None = None,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Write parent-delta evidence before enforcing that training changed weights."""

    delta = summarize_parent_delta(master_weights, parent_weights, keys)
    if collector is not None:
        payload = dict(extra or {})
        payload["master_vs_parent"] = delta
        if not delta["master_changed_vs_parent"]:
            payload["training_failed"] = True
        collector.finish(extra=payload, duration_seconds=duration_seconds)
    if not delta["master_changed_vs_parent"]:
        raise RuntimeError("QAT master weights did not change from Parent P")
    return delta


class MetricsCollector:
    """Collect bounded training metrics; median_step_ms measures batch time.

    Batch time covers forward plus backward only. It excludes data-loader wait,
    device transfer, optimizer updates, logging, and checkpoint serialization.
    Final partial accumulation flushes increment optimizer_steps but is not a
    batch and therefore contributes no duration sample.
    """

    def __init__(
        self,
        stage: str,
        *,
        config: Mapping[str, Any] | None = None,
        metrics_path: str | os.PathLike[str] | None = None,
        planned_steps: int | None = None,
        sample_capacity: int = METRICS_SAMPLE_CAPACITY,
        rank: int = 0,
    ):
        self.stage = stage
        self.config = dict(config or {})
        self.metrics_path = os.fspath(metrics_path) if metrics_path is not None else None
        self.planned_steps = planned_steps
        self.sample_capacity = int(sample_capacity)
        self.rank = int(rank)
        self._write_enabled = is_metrics_writer(self.rank)
        self._started_at = None
        self._duration_seconds = None
        self._step_count = 0
        self._first_losses = _DeterministicReservoir(self.sample_capacity, seed=11)
        self._last_losses = _DeterministicReservoir(self.sample_capacity, seed=23)
        self._step_durations = _DeterministicReservoir(self.sample_capacity, seed=37)
        self._initial_loss = None
        self._final_loss = None
        self.optimizer_steps = 0
        self.samples_processed = 0
        self.tokens_processed = 0
        self.finite_loss = True
        self.finite_gradients = True

    def start(self, planned_steps: int | None = None) -> None:
        if self._started_at is not None:
            return
        if planned_steps is not None:
            self.planned_steps = int(planned_steps)
        self._started_at = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stop(self) -> float:
        if self._started_at is None:
            self.start()
        if self._duration_seconds is None:
            self._duration_seconds = time.perf_counter() - self._started_at
        return self._duration_seconds

    def _window_sizes(self) -> tuple[int, int]:
        total = max(int(self.planned_steps or self._step_count), 1)
        first = max(1, math.ceil(total * 0.1))
        last = max(1, math.ceil(total * 0.1))
        return first, last

    def record_batch(
        self,
        loss: float | torch.Tensor,
        *,
        labels: torch.Tensor,
        duration_seconds: float | None = None,
    ) -> None:
        if self._started_at is None:
            self.start()
        loss_value = float(loss.detach().item() if isinstance(loss, torch.Tensor) else loss)
        if not math.isfinite(loss_value):
            self.mark_loss_nonfinite()
            raise RuntimeError(f"non-finite {self.stage} loss")
        self._step_count += 1
        if self._initial_loss is None:
            self._initial_loss = loss_value
        self._final_loss = loss_value
        first_size, last_size = self._window_sizes()
        if self._step_count <= first_size:
            self._first_losses.add(loss_value)
        if self.planned_steps is None:
            self._last_losses.add(loss_value)
        else:
            last_start = max(1, int(self.planned_steps) - last_size + 1)
            if last_start <= self._step_count <= int(self.planned_steps):
                self._last_losses.add(loss_value)
        if duration_seconds is not None:
            self._step_durations.add(float(duration_seconds))
        self.samples_processed += int(labels.shape[0])
        self.tokens_processed += int((labels[..., 1:] != -100).sum().item())

    def record_optimizer_step(self) -> None:
        self.optimizer_steps += 1

    def mark_loss_nonfinite(self) -> None:
        self.finite_loss = False

    def mark_gradient_nonfinite(self) -> None:
        self.finite_gradients = False

    def record_gradient_norm(self, gradient_norm: torch.Tensor | float) -> None:
        finite = bool(torch.isfinite(torch.as_tensor(gradient_norm)).all())
        if not finite:
            self.mark_gradient_nonfinite()
            raise RuntimeError(f"non-finite {self.stage} gradient norm")

    def _memory_metrics(self) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {
                "peak_allocated_vram_bytes": 0,
                "peak_reserved_vram_bytes": 0,
            }
        return {
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def finish(
        self,
        *,
        extra: Mapping[str, Any] | None = None,
        duration_seconds: float | None = None,
    ) -> dict[str, Any]:
        if self._started_at is None:
            self.start()
        if duration_seconds is None:
            duration_seconds = self.stop()
        else:
            self._duration_seconds = float(duration_seconds)
        median_step_ms = statistics.median(self._step_durations.values) * 1000.0 if self._step_durations.values else None
        metrics: dict[str, Any] = {
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "stage": self.stage,
            "config": self.config,
            "initial_loss": self._initial_loss,
            "final_loss": self._final_loss,
            "first_10_percent_loss_median": statistics.median(self._first_losses.values) if self._first_losses.values else None,
            "last_10_percent_loss_median": statistics.median(self._last_losses.values) if self._last_losses.values else None,
            "optimizer_steps": self.optimizer_steps,
            "samples_processed": self.samples_processed,
            "tokens_processed": self.tokens_processed,
            "median_step_ms": median_step_ms,
            "tokens_per_second": self.tokens_processed / duration_seconds if duration_seconds > 0 else None,
            "duration_seconds": duration_seconds,
            "finite_loss": self.finite_loss,
            "finite_gradients": self.finite_gradients,
            "training_failed": False,
            "status": "success",
            "rank": self.rank,
            "accounting_scope": "rank_local",
            "evidence_authority": "rank_0",
            "median_aggregation": {
                "method": "deterministic_reservoir",
                "capacity": self.sample_capacity,
                "first_loss_exact": self._first_losses.exact,
                "last_loss_exact": self._last_losses.exact,
                "step_duration_exact": self._step_durations.exact,
            },
            **self._memory_metrics(),
        }
        if extra:
            extra_metrics = dict(extra)
            metrics.update(extra_metrics)
            if extra_metrics.get("training_failed"):
                metrics["status"] = "failed"
        if self.metrics_path and self._write_enabled:
            parent = os.path.dirname(os.path.abspath(self.metrics_path))
            os.makedirs(parent, exist_ok=True)
            with open(self.metrics_path, "w", encoding="utf-8") as output:
                json.dump(metrics, output, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return metrics


__all__ = [
    "METRICS_SCHEMA_VERSION",
    "METRICS_SAMPLE_CAPACITY",
    "MetricsCollector",
    "check_finite_gradient_norm",
    "check_finite_loss",
    "finish_parent_delta_evidence",
    "capture_parameter_probe",
    "is_metrics_writer",
    "summarize_parameter_probe",
    "summarize_parent_delta",
    "summarize_weight_differences",
]
