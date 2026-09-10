import os

import pytest

pytestmark = pytest.mark.runpod

torch = pytest.importorskip("torch")
if not torch.__version__.startswith("2.4.1"):
    pytest.skip("Q4-T3 runner tests require RunPod torch 2.4.1", allow_module_level=True)

transformers = pytest.importorskip("transformers")
if transformers.__version__ != "4.57.6":
    pytest.skip("Q4-T3 runner tests require transformers 4.57.6", allow_module_level=True)

from trainer.q4_t3_campaign_runner import reload_checkpoint


def test_optional_strict_reload_uses_fixture_checkpoint():
    checkpoint = os.environ.get("Q4_T3_RUNNER_RELOAD_CHECKPOINT")
    variant = os.environ.get("Q4_T3_RUNNER_RELOAD_VARIANT", "parent")
    if not checkpoint:
        pytest.skip("Q4_T3_RUNNER_RELOAD_CHECKPOINT is not configured")
    reload_checkpoint(checkpoint, variant=variant, device="cuda:0")
