"""Deterministic SFT index planning and loader construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import torch


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            byte_count += len(chunk)
            digest.update(chunk)
    return byte_count, digest.hexdigest()


def _seed_worker(worker_id: int) -> None:
    """Seed worker-side libraries from DataLoader's private generator."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass


def _loader_seed(plan: "SFTIndexPlan", epoch: int, split: str, rank: int, world_size: int) -> int:
    seed = plan.train_seed if split == "train" else plan.holdout_seed
    return (
        int(seed)
        + 1_000_003 * int(epoch)
        + 10_007 * int(rank)
        + 100_003 * int(world_size)
        + (1 if split == "holdout" else 0)
    ) % (2**63)


def _dataset_identity(plan_path: str, dataset_bytes: int, dataset_sha256: str) -> dict[str, Any]:
    return {
        "dataset_sha256": dataset_sha256,
        "dataset_bytes": dataset_bytes,
        "dataset_basename": os.path.basename(plan_path),
    }


def build_dataset_identity(data_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return portable dataset identity; path itself is intentionally omitted."""

    resolved_path = str(Path(data_path).resolve())
    dataset_bytes, dataset_sha256 = _sha256_file(resolved_path)
    return _dataset_identity(resolved_path, dataset_bytes, dataset_sha256)


@dataclass(frozen=True)
class SFTIndexPlan:
    dataset_path: str
    dataset_bytes: int
    dataset_sha256: str
    sample_count: int
    train_seed: int
    holdout_seed: int
    holdout_size: int
    epochs: int
    batch_size: int
    accumulation_steps: int
    holdout_indices: tuple[int, ...]
    ordered_train_indices: tuple[int, ...]
    plan_sha256: str
    holdout_sha256: str

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "plan_sha256": self.plan_sha256,
            "holdout_sha256": self.holdout_sha256,
            **_dataset_identity(self.dataset_path, self.dataset_bytes, self.dataset_sha256),
            "dataset_sample_count": self.sample_count,
            "train_seed": self.train_seed,
            "holdout_seed": self.holdout_seed,
            "holdout_size": self.holdout_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "train_index_count": len(self.ordered_train_indices),
            "holdout_index_count": len(self.holdout_indices),
            "first_train_indices": list(self.ordered_train_indices[:16]),
            "last_train_indices": list(self.ordered_train_indices[-16:]),
            "first_holdout_indices": list(self.holdout_indices[:16]),
            "last_holdout_indices": list(self.holdout_indices[-16:]),
        }

    def epoch_indices(self, epoch: int) -> tuple[int, ...]:
        if epoch < 0 or epoch >= self.epochs:
            raise IndexError(f"epoch out of range: {epoch}")
        per_epoch = len(self.ordered_train_indices) // self.epochs
        start = epoch * per_epoch
        return self.ordered_train_indices[start:start + per_epoch]


def build_sft_index_plan(
    data_path: str | os.PathLike[str],
    sample_count: int,
    *,
    train_seed: int = 42,
    holdout_seed: int = 4242,
    holdout_size: int = 128,
    epochs: int = 1,
    batch_size: int = 16,
    accumulation_steps: int = 1,
) -> SFTIndexPlan:
    """Split raw indices, then create deterministic per-epoch train order."""

    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if holdout_size < 0 or holdout_size >= sample_count:
        raise ValueError("holdout_size must be non-negative and smaller than sample_count")
    if epochs <= 0 or batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("epochs, batch_size, and accumulation_steps must be positive")

    resolved_path = str(Path(data_path).resolve())
    dataset_bytes, dataset_sha256 = _sha256_file(resolved_path)

    holdout_generator = torch.Generator(device="cpu")
    holdout_generator.manual_seed(int(holdout_seed))
    holdout_order = torch.randperm(sample_count, generator=holdout_generator).tolist()
    holdout_indices = tuple(holdout_order[:holdout_size])
    holdout_set = set(holdout_indices)

    ordered_train_indices: list[int] = []
    for epoch in range(epochs):
        train_generator = torch.Generator(device="cpu")
        train_generator.manual_seed(int(train_seed) + epoch)
        permutation = torch.randperm(sample_count, generator=train_generator).tolist()
        ordered_train_indices.extend(index for index in permutation if index not in holdout_set)

    plan_payload = {
        **_dataset_identity(resolved_path, dataset_bytes, dataset_sha256),
        "sample_count": sample_count,
        "train_seed": int(train_seed),
        "holdout_seed": int(holdout_seed),
        "holdout_size": int(holdout_size),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "accumulation_steps": int(accumulation_steps),
        "holdout_indices": list(holdout_indices),
        "ordered_train_indices": ordered_train_indices,
    }
    holdout_payload = {
        **_dataset_identity(resolved_path, dataset_bytes, dataset_sha256),
        "sample_count": sample_count,
        "holdout_seed": int(holdout_seed),
        "holdout_size": int(holdout_size),
        "holdout_indices": list(holdout_indices),
    }
    return SFTIndexPlan(
        dataset_path=resolved_path,
        dataset_bytes=dataset_bytes,
        dataset_sha256=dataset_sha256,
        sample_count=sample_count,
        train_seed=int(train_seed),
        holdout_seed=int(holdout_seed),
        holdout_size=int(holdout_size),
        epochs=int(epochs),
        batch_size=int(batch_size),
        accumulation_steps=int(accumulation_steps),
        holdout_indices=holdout_indices,
        ordered_train_indices=tuple(ordered_train_indices),
        plan_sha256=_sha256_json(plan_payload),
        holdout_sha256=_sha256_json(holdout_payload),
    )


def build_deterministic_sft_loader(
    dataset,
    plan: SFTIndexPlan,
    *,
    epoch: int = 0,
    batch_size: int | None = None,
    split: str = "train",
    skip_batches: int = 0,
    num_workers: int = 0,
    pin_memory: bool = True,
    rank: int = 0,
    world_size: int = 1,
):
    """Build a DataLoader whose batches depend only on plan, not workers."""

    if split == "train":
        source_indices = list(plan.epoch_indices(epoch))
    elif split == "holdout":
        source_indices = list(plan.holdout_indices)
    else:
        raise ValueError(f"unknown SFT split: {split}")
    if batch_size is None:
        batch_size = plan.batch_size
    if batch_size <= 0 or skip_batches < 0:
        raise ValueError("batch_size must be positive and skip_batches non-negative")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid distributed rank")

    # Match DistributedSampler(drop_last=False): pad source order to a
    # world-size multiple, then stride-partition. Padding is explicit, so no
    # source sample is silently dropped and every rank has equal batch count.
    if source_indices:
        total_size = ((len(source_indices) + world_size - 1) // world_size) * world_size
        padding_count = total_size - len(source_indices)
        indices = source_indices + [source_indices[i % len(source_indices)] for i in range(padding_count)]
        indices = indices[rank:total_size:world_size]
    else:
        padding_count = 0
        indices = []

    batches = [
        indices[start:start + batch_size]
        for start in range(0, len(indices), batch_size)
    ]
    batches = batches[skip_batches:]

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=torch.Generator(device="cpu").manual_seed(
            _loader_seed(plan, epoch, split, rank, world_size)
        ),
        worker_init_fn=_seed_worker,
    )
    loader.sft_num_source_samples = len(source_indices)
    loader.sft_num_padded_samples = padding_count
    loader.sft_num_processed_samples = sum(len(batch) for batch in batches)
    return loader


__all__ = [
    "SFTIndexPlan",
    "build_dataset_identity",
    "build_deterministic_sft_loader",
    "build_sft_index_plan",
]
