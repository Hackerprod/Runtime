import os

import pytest

pytestmark = pytest.mark.runpod

torch = pytest.importorskip("torch")
if not torch.__version__.startswith("2.4.1"):
    pytest.skip("diagnostic integration tests require RunPod torch 2.4.1", allow_module_level=True)

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.57.6":
    pytest.skip("diagnostic integration tests require transformers 4.57.6", allow_module_level=True)

from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quantization.qat import apply_spec_q4_qat
from trainer.q4_t3_diagnostics import load_diagnostic_fixture, run_control_qat_diagnostics


def test_control_qat_diagnostics_optional_checkpoint_fixture():
    control_path = os.getenv("Q4_T3_CONTROL_CHECKPOINT")
    qat_path = os.getenv("Q4_T3_QAT_CHECKPOINT")
    tokenizer_path = os.getenv("Q4_T3_TOKENIZER_PATH")
    if not control_path or not qat_path or not tokenizer_path:
        pytest.skip("set Q4_T3_CONTROL_CHECKPOINT, Q4_T3_QAT_CHECKPOINT, and Q4_T3_TOKENIZER_PATH")

    control = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False))
    qat = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False))
    control.load_state_dict(torch.load(control_path, map_location="cpu"), strict=True)
    qat.load_state_dict(torch.load(qat_path, map_location="cpu"), strict=True)
    apply_spec_q4_qat(qat)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    result = run_control_qat_diagnostics(control, qat, tokenizer, fixture=load_diagnostic_fixture())
    assert result["informational"] is True
    assert len(result["branches"]["control"]["outputs"]) == 32
    assert len(result["branches"]["qat"]["outputs"]) == 32
