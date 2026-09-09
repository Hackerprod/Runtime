import hashlib
import json
import os
import random

import torch
from torch.utils.data import Dataset

from dataset.lm_dataset import post_processing_chat, pre_processing_chat, rng_for_index
from trainer.sft_plan import build_deterministic_sft_loader, build_sft_index_plan


class IndexDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return index


def test_plan_is_stable_and_excludes_holdout(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    data_path.write_bytes(b"sample data\n")
    plan = build_sft_index_plan(
        data_path,
        10,
        train_seed=42,
        holdout_seed=4242,
        holdout_size=3,
        epochs=2,
        batch_size=2,
        accumulation_steps=1,
    )
    same_plan = build_sft_index_plan(
        data_path,
        10,
        train_seed=42,
        holdout_seed=4242,
        holdout_size=3,
        epochs=2,
        batch_size=2,
        accumulation_steps=1,
    )

    assert plan == same_plan
    assert len(plan.holdout_indices) == 3
    assert len(plan.ordered_train_indices) == 14
    assert set(plan.holdout_indices).isdisjoint(plan.ordered_train_indices)
    assert set(plan.ordered_train_indices[:7]) == set(range(10)) - set(plan.holdout_indices)
    assert set(plan.ordered_train_indices[7:]) == set(range(10)) - set(plan.holdout_indices)

    payload = {
        "dataset_sha256": plan.dataset_sha256,
        "dataset_bytes": plan.dataset_bytes,
        "dataset_basename": os.path.basename(plan.dataset_path),
        "sample_count": plan.sample_count,
        "train_seed": plan.train_seed,
        "holdout_seed": plan.holdout_seed,
        "holdout_size": plan.holdout_size,
        "epochs": plan.epochs,
        "batch_size": plan.batch_size,
        "accumulation_steps": plan.accumulation_steps,
        "holdout_indices": list(plan.holdout_indices),
        "ordered_train_indices": list(plan.ordered_train_indices),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert plan.plan_sha256 == hashlib.sha256(canonical).hexdigest()


def test_plan_hashes_are_portable_across_absolute_roots(tmp_path):
    left_path = tmp_path / "left" / "sft.jsonl"
    right_path = tmp_path / "right" / "sft.jsonl"
    left_path.parent.mkdir()
    right_path.parent.mkdir()
    left_path.write_bytes(b"same content\n")
    right_path.write_bytes(b"same content\n")

    left = build_sft_index_plan(left_path, 10, holdout_size=3)
    right = build_sft_index_plan(right_path, 10, holdout_size=3)

    assert left.dataset_path != right.dataset_path
    assert left.dataset_sha256 == right.dataset_sha256
    assert left.plan_sha256 == right.plan_sha256
    assert left.holdout_sha256 == right.holdout_sha256


def test_compact_plan_has_no_index_arrays(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    data_path.write_bytes(b"sample data\n")
    compact = build_sft_index_plan(data_path, 256).to_compact_dict()

    assert "ordered_train_indices" not in compact
    assert "holdout_indices" not in compact
    assert compact["dataset_basename"] == "sft.jsonl"
    assert compact["train_index_count"] == 128
    assert compact["holdout_index_count"] == 128


def test_default_holdout_is_exactly_128_and_disjoint(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    data_path.write_bytes(b"sample data\n")
    plan = build_sft_index_plan(data_path, 256)

    assert len(plan.holdout_indices) == 128
    assert len(plan.ordered_train_indices) == 128
    assert set(plan.holdout_indices).isdisjoint(plan.ordered_train_indices)


def test_loader_uses_plan_batches_for_train_and_holdout(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    data_path.write_bytes(b"sample data\n")
    plan = build_sft_index_plan(data_path, 10, holdout_size=3, batch_size=2)

    train_loader = build_deterministic_sft_loader(IndexDataset(), plan, batch_size=2)
    holdout_loader = build_deterministic_sft_loader(IndexDataset(), plan, split="holdout", batch_size=2)
    train_batches = [batch.tolist() for batch in train_loader]
    holdout_batches = [batch.tolist() for batch in holdout_loader]

    assert train_batches == [list(batch) for batch in zip(*[iter(plan.epoch_indices(0))] * 2)] + (
        [list(plan.epoch_indices(0)[-1:])] if len(plan.epoch_indices(0)) % 2 else []
    )
    assert holdout_batches == [
        list(plan.holdout_indices[start:start + 2])
        for start in range(0, len(plan.holdout_indices), 2)
    ]


def test_loader_equal_rank_partition_pads_without_dropping_source_indices(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    data_path.write_bytes(b"sample data\n")
    plan = build_sft_index_plan(data_path, 10, holdout_size=3, batch_size=2)

    loaders = [
        build_deterministic_sft_loader(
            IndexDataset(), plan, batch_size=2, rank=rank, world_size=3, pin_memory=False
        )
        for rank in range(3)
    ]
    batches = [[batch.tolist() for batch in loader] for loader in loaders]
    expected_padded = list(plan.epoch_indices(0)) + list(plan.epoch_indices(0)[:2])
    expected_rank_samples = [expected_padded[rank::3] for rank in range(3)]

    assert [len(loader) for loader in loaders] == [2, 2, 2]
    assert [loader.sft_num_padded_samples for loader in loaders] == [2, 2, 2]
    assert [loader.sft_num_processed_samples for loader in loaders] == [3, 3, 3]
    flattened_batches = [
        [item for batch in rank_batches for item in batch]
        for rank_batches in batches
    ]
    assert flattened_batches == expected_rank_samples
    assert set(sum(expected_rank_samples, [])) == set(plan.epoch_indices(0))


def test_loader_generator_does_not_consume_ambient_torch_rng(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    data_path.write_bytes(b"sample data\n")
    plan = build_sft_index_plan(data_path, 10, holdout_size=3, batch_size=2)

    torch.manual_seed(99)
    state_before = torch.get_rng_state()
    loader = build_deterministic_sft_loader(IndexDataset(), plan, pin_memory=False)
    list(loader)
    state_after = torch.get_rng_state()

    assert torch.equal(state_before, state_after)


def test_sft_rng_is_per_index_and_does_not_touch_global_random():
    conversations = [{"role": "user", "content": "hello"}]
    random.seed(123)
    state_before = random.getstate()
    first = pre_processing_chat(
        conversations,
        add_system_ratio=1.0,
        rng=rng_for_index(42, 7),
    )
    first_prompt = post_processing_chat(
        "<think>\n\n</think>\n\nanswer",
        empty_think_ratio=0.0,
        rng=rng_for_index(42, 7),
    )
    state_after = random.getstate()

    second = pre_processing_chat(
        conversations,
        add_system_ratio=1.0,
        rng=rng_for_index(42, 7),
    )
    second_prompt = post_processing_chat(
        "<think>\n\n</think>\n\nanswer",
        empty_think_ratio=0.0,
        rng=rng_for_index(42, 7),
    )

    assert state_before == state_after
    assert first == second
    assert first_prompt == second_prompt == "answer"
    assert [rng_for_index(42, 7).random() for _ in range(2)] == [
        rng_for_index(42, 7).random() for _ in range(2)
    ]
    assert rng_for_index(42, 7).random() != rng_for_index(42, 8).random()
