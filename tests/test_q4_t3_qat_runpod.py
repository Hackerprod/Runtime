import pytest

pytestmark = pytest.mark.runpod

torch = pytest.importorskip("torch")
if not torch.__version__.startswith("2.4.1"):
    pytest.skip("QAT integration tests require RunPod torch 2.4.1", allow_module_level=True)

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.57.6":
    pytest.skip("QAT integration tests require transformers 4.57.6", allow_module_level=True)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quantization.qat import QATLinear, QAT_TENSOR_COUNT, QAT_WEIGHT_COUNT, apply_spec_q4_qat
from trainer.train_qat_sft import _save_fp32_state_dict


def test_qat_targets_and_checkpoint_are_fp32(tmp_path):
    model = MiniMindForCausalLM(
        MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False)
    )
    keys_before = set(model.state_dict())
    apply_spec_q4_qat(model)

    assert sum(isinstance(module, QATLinear) for module in model.modules()) == QAT_TENSOR_COUNT
    assert sum(module.weight.numel() for module in model.modules() if isinstance(module, QATLinear)) == QAT_WEIGHT_COUNT

    checkpoint_path = str(tmp_path / "qat_sft_768.pth")
    _save_fp32_state_dict(model, checkpoint_path)
    saved = torch.load(checkpoint_path, map_location="cpu")
    assert set(saved) == keys_before
    assert all(not value.is_floating_point() or value.dtype == torch.float32 for value in saved.values())
