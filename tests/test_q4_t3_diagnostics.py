import numpy as np
import pytest

from trainer.q4_t3_diagnostics import (
    build_diagnostic_metrics,
    compute_mean_logit_kl,
    compute_top1_agreement,
    diagnostic_prompt_identity,
    load_diagnostic_fixture,
)


def test_diagnostic_fixture_has_exact_32_prompts_and_stable_policy_identity():
    fixture = load_diagnostic_fixture()
    assert fixture["schema"] == "q4-t3-diagnostic-prompts"
    assert fixture["version"] == "v1"
    assert len(fixture["prompts"]) == 32
    assert len({item["id"] for item in fixture["prompts"]}) == 32
    assert fixture["sampling_policy"] == {
        "do_sample": True,
        "temperature": 0.9,
        "top_p": 0.85,
        "top_k": 50,
        "max_new_tokens": 64,
        "seed": 20250909,
    }
    first = diagnostic_prompt_identity()
    assert first == diagnostic_prompt_identity()
    assert first["prompts_sha256"] == "eb9e29430c959c83f128ca865ab0d80999b5a321ae9508d31c5d3981ebc3494c"
    assert first["policy_sha256"] == "37a6573b26be01b9528449525d2160efdba773c75f478df22a9c73e5194e857d"


def test_diagnostic_logit_metrics_are_deterministic_and_compact():
    control = np.array([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=np.float32)
    qat = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=np.float32)
    assert compute_top1_agreement(control, qat) == 1.0
    assert compute_mean_logit_kl(control, control) == 0.0
    assert compute_mean_logit_kl(control, qat) > 0.0
    metrics = build_diagnostic_metrics(control, qat)
    assert metrics["informational"] is True
    assert set(metrics) == {"informational", "top1_agreement", "mean_logit_kl_control_to_qat"}


@pytest.mark.parametrize(
    "control, qat",
    [
        ([[1.0, 2.0]], [[1.0]]),
        ([[float("nan"), 1.0]], [[1.0, 2.0]]),
    ],
)
def test_diagnostic_logit_metrics_reject_invalid_inputs(control, qat):
    with pytest.raises(ValueError):
        compute_top1_agreement(control, qat)
