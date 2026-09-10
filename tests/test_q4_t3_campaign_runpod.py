import os

import pytest

pytestmark = pytest.mark.runpod

torch = pytest.importorskip("torch")
if not torch.__version__.startswith("2.4.1"):
    pytest.skip("Q4-T3 campaign tests require RunPod torch 2.4.1", allow_module_level=True)

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.57.6":
    pytest.skip("Q4-T3 campaign tests require transformers 4.57.6", allow_module_level=True)

from torch.nn import functional as F

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quantization.qat import QATLinear, apply_spec_q4_qat, spec_q4_fake_dequant
from trainer.q4_t3_campaign import evaluate_sft_holdout, validate_parent_checkpoint
from trainer.sft_plan import build_sft_index_plan


def test_parent_validator_strict_loads_optional_runpod_checkpoint():
    checkpoint_path = os.environ.get("Q4_T3_PARENT_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("Q4_T3_PARENT_CHECKPOINT is not configured")

    result = validate_parent_checkpoint(checkpoint_path)

    assert result["strict_load"] is True
    assert result["parent_dtype"] == "float16"
    assert result["hidden_size"] == 768
    assert result["num_hidden_layers"] == 8
    assert result["use_moe"] is False


def test_qat_eval_uses_effective_fake_dequant_weights():
    model = MiniMindForCausalLM(
        MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    )
    apply_spec_q4_qat(model)
    projection = model.model.layers[0].mlp.gate_proj
    assert isinstance(projection, QATLinear)
    inputs = torch.randn(2, 3, 768)

    model.eval()
    with torch.no_grad():
        actual = projection(inputs)
        expected = F.linear(inputs, spec_q4_fake_dequant(projection.weight))
        repeated = projection(inputs)

    assert torch.equal(actual, repeated)
    assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)


def test_optional_real_holdout_evaluation_uses_same_bindings():
    names = (
        "Q4_T3_PARENT_CHECKPOINT",
        "Q4_T3_CONTROL_CHECKPOINT",
        "Q4_T3_QAT_CHECKPOINT",
        "Q4_T3_DATASET",
        "Q4_T3_TOKENIZER",
        "Q4_T3_PLAN_SHA256",
        "Q4_T3_HOLDOUT_SHA256",
        "Q4_T3_SAMPLE_COUNT",
    )
    if any(not os.environ.get(name) for name in names):
        pytest.skip("real Q4-T3 checkpoint, dataset, tokenizer, and hash bindings are not configured")

    dataset_path = os.environ["Q4_T3_DATASET"]
    plan = build_sft_index_plan(
        dataset_path,
        int(os.environ["Q4_T3_SAMPLE_COUNT"]),
        train_seed=42,
        holdout_seed=4242,
        holdout_size=128,
        epochs=1,
        batch_size=16,
        accumulation_steps=1,
    )
    common = {
        "data_path": dataset_path,
        "tokenizer_path": os.environ["Q4_T3_TOKENIZER"],
        "plan": plan,
        "batch_size": 16,
        "expected_plan_sha256": os.environ["Q4_T3_PLAN_SHA256"],
        "expected_holdout_sha256": os.environ["Q4_T3_HOLDOUT_SHA256"],
        "device": "cuda:0",
    }
    results = [
        evaluate_sft_holdout(os.environ["Q4_T3_PARENT_CHECKPOINT"], variant="parent", **common),
        evaluate_sft_holdout(os.environ["Q4_T3_CONTROL_CHECKPOINT"], variant="control", **common),
        evaluate_sft_holdout(os.environ["Q4_T3_QAT_CHECKPOINT"], variant="qat", **common),
    ]

    assert all(result["sample_count"] == 128 for result in results)
    assert all(result["plan_sha256"] == plan.plan_sha256 for result in results)
    assert all(result["holdout_sha256"] == plan.holdout_sha256 for result in results)
