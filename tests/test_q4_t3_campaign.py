import json
import math
import random

import pytest
import torch
from torch.utils.data import Dataset

from trainer.q4_t3_campaign import (
    build_campaign_identity,
    compute_token_weighted_ce,
    evaluate_holdout_model,
    summarize_state_dict_dtypes,
    validate_holdout_binding,
)
from trainer.sft_plan import build_deterministic_sft_loader, build_sft_index_plan


class SyntheticHoldoutDataset(Dataset):
    def __init__(self, sample_count=256, sequence_length=4):
        self.sample_count = sample_count
        self.sequence_length = sequence_length

    def __len__(self):
        return self.sample_count

    def __getitem__(self, index):
        input_ids = torch.arange(self.sequence_length, dtype=torch.long)
        labels = torch.tensor([0, 0, -100, -100], dtype=torch.long)
        return input_ids, labels


class UniformLogitModel:
    def __init__(self, vocab_size=4):
        self.vocab_size = vocab_size
        self.was_eval = False

    def eval(self):
        self.was_eval = True
        return self

    def parameters(self):
        return iter(())

    def __call__(self, input_ids, labels=None):
        shape = (*input_ids.shape, self.vocab_size)
        return torch.zeros(shape, dtype=torch.float32)


def test_campaign_identity_is_compact_json_safe_and_portable(tmp_path):
    left = tmp_path / "left" / "sft.jsonl"
    right = tmp_path / "right" / "sft.jsonl"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_bytes(b"same dataset\n")
    right.write_bytes(b"same dataset\n")

    identity = build_campaign_identity(
        left,
        git_commit="f7b853b",
        environment={"torch": "2.4.1", "workspace_path": "C:/secret/workspace", "api_token": "do-not-save"},
        parent_sha256="parent-sha",
        parent_dtype="float16",
        plan_sha256="plan-sha",
        holdout_sha256="holdout-sha",
    )
    same_identity = build_campaign_identity(right, git_commit="f7b853b")

    json.dumps(identity)
    assert identity["dataset"] == same_identity["dataset"]
    assert identity["dataset"]["dataset_basename"] == "sft.jsonl"
    assert "api_token" not in identity["environment"]
    assert identity["environment"]["workspace_path"] == "workspace"
    assert "C:/secret/workspace" not in json.dumps(identity)
    assert identity["sft"] == {"plan_sha256": "plan-sha", "holdout_sha256": "holdout-sha"}


def test_expected_hashes_and_holdout_size_fail_closed(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    data_path.write_bytes(b"dataset\n")
    plan = build_sft_index_plan(data_path, 256)

    binding = validate_holdout_binding(
        plan,
        expected_plan_sha256=plan.plan_sha256,
        expected_holdout_sha256=plan.holdout_sha256,
    )
    assert binding["holdout_count"] == 128
    with pytest.raises(ValueError, match="plan SHA"):
        validate_holdout_binding(
            plan,
            expected_plan_sha256="wrong",
            expected_holdout_sha256=plan.holdout_sha256,
        )

    small_plan = build_sft_index_plan(data_path, 256, holdout_size=3)
    with pytest.raises(ValueError, match="exactly 128"):
        validate_holdout_binding(
            small_plan,
            expected_plan_sha256=small_plan.plan_sha256,
            expected_holdout_sha256=small_plan.holdout_sha256,
        )


def test_token_weighted_ce_uses_valid_token_total_and_perplexity():
    first_logits = torch.zeros((1, 2, 2), dtype=torch.float32)
    first_labels = torch.tensor([[0, 0]])
    second_logits = torch.tensor([[[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0]]])
    second_labels = torch.tensor([[0, 0, 0, 0]])
    first = compute_token_weighted_ce(first_logits, first_labels)
    second = compute_token_weighted_ce(second_logits, second_labels)

    weighted_ce = (
        first["cross_entropy"] * first["valid_token_count"]
        + second["cross_entropy"] * second["valid_token_count"]
    ) / (first["valid_token_count"] + second["valid_token_count"])
    assert first["valid_token_count"] == 1
    assert second["valid_token_count"] == 3
    assert weighted_ce != (first["cross_entropy"] + second["cross_entropy"]) / 2
    assert math.isclose(first["perplexity"], math.exp(first["cross_entropy"]))


def test_holdout_evaluation_is_deterministic_and_does_not_mutate_rng(tmp_path):
    data_path = tmp_path / "sft.jsonl"
    data_path.write_bytes(b"dataset\n")
    plan = build_sft_index_plan(data_path, 256, batch_size=7)
    dataset = SyntheticHoldoutDataset()
    model = UniformLogitModel()

    random.seed(123)
    torch.manual_seed(456)
    random_before = random.getstate()
    torch_before = torch.get_rng_state()
    first = evaluate_holdout_model(
        model,
        build_deterministic_sft_loader(
            dataset, plan, split="holdout", batch_size=7, pin_memory=False
        ),
        plan,
        batch_size=7,
        expected_plan_sha256=plan.plan_sha256,
        expected_holdout_sha256=plan.holdout_sha256,
    )
    second = evaluate_holdout_model(
        model,
        build_deterministic_sft_loader(
            dataset, plan, split="holdout", batch_size=7, pin_memory=False
        ),
        plan,
        batch_size=7,
        expected_plan_sha256=plan.plan_sha256,
        expected_holdout_sha256=plan.holdout_sha256,
    )

    assert first == second
    assert first["sample_count"] == 128
    assert first["valid_token_count"] == 128
    assert math.isclose(first["cross_entropy"], math.log(4), rel_tol=1e-6)
    assert model.was_eval
    assert random.getstate() == random_before
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_state_dict_dtype_summary_reports_actual_dtypes():
    summary = summarize_state_dict_dtypes({
        "weight": torch.zeros(2, dtype=torch.float16),
        "counter": torch.zeros(1, dtype=torch.int64),
    })

    assert summary == {
        "state_dict_dtypes": ["float16", "int64"],
        "floating_dtypes": ["float16"],
        "parent_dtype": "float16",
    }
