import os

import pytest

pytestmark = pytest.mark.runpod

torch = pytest.importorskip("torch")
if not torch.__version__.startswith("2.4.1"):
    pytest.skip("Q4-T3 validation integration tests require RunPod torch 2.4.1", allow_module_level=True)

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.57.6":
    pytest.skip("Q4-T3 validation integration tests require transformers 4.57.6", allow_module_level=True)

from trainer.q4_t3_validation import validate_qat_checkpoint_parity


def test_qat_checkpoint_has_exact_dense_shape_and_parity_contract():
    checkpoint = os.getenv("Q4_T3_QAT_CHECKPOINT") or os.getenv("Q4_T3_QAT_CHECKPOINT_PATH")
    if not checkpoint:
        pytest.skip("set Q4_T3_QAT_CHECKPOINT to a QAT FP32 state_dict checkpoint")

    result = validate_qat_checkpoint_parity(checkpoint)
    assert result["parity_status"] == "PASS"
    assert result["tensor_count"] == 24
    assert result["weight_count"] == 44_826_624
    assert result["storage"]["total_bytes"] == 23_814_144
    assert len(result["tensors"]) == 24
    assert all(tensor["parity_status"] == "PASS" for tensor in result["tensors"])
