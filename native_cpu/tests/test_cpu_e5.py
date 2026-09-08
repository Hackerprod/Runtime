"""Contract tests for the isolated CPU-E5 attention-weight experiment."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "native_cpu" / "benchmarks" / "cpu-e5" / "benchmark.py"
CHECKPOINT = ROOT / "checkpoints" / "minimind-3-hf" / "model.safetensors"
ARTIFACT = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"


def _module():
    spec = importlib.util.spec_from_file_location("cpu_e5_benchmark", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attention_layout_is_exactly_eight_layers_and_four_projections():
    module = _module()
    names = module.expected_names()
    assert len(names) == 32
    assert names[0] == "model.layers.0.self_attn.q_proj.weight"
    assert names[-1] == "model.layers.7.self_attn.o_proj.weight"
    assert module.PROJECTIONS == {
        "q_proj": (768, 768),
        "k_proj": (384, 768),
        "v_proj": (384, 768),
        "o_proj": (768, 768),
    }


def test_payload_writer_preserves_half_and_float32_bytes():
    module = _module()
    values = np.asarray([[1.0, -0.0], [0.5, -2.0]], dtype="<f4")
    half = values.astype("<f2")
    evidence = {"matrices": [{
        "name": "model.layers.0.self_attn.q_proj.weight",
        "rows": 2,
        "cols": 2,
        "fp16_bits": half.view("<u2").tobytes(),
        "fp32_bytes_data": values.tobytes(order="C"),
    }]}
    folder = ROOT / "native_cpu" / "validation" / "cpu-e5-test-payload"
    folder.mkdir(parents=True, exist_ok=True)
    payload = folder / "weights.bin"
    try:
        module._write_payload(payload, evidence)
        raw = payload.read_bytes()
        assert raw.startswith(module.PAYLOAD_MAGIC)
        assert half.view("<u2").tobytes() in raw
        assert values.tobytes(order="C") in raw
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.skipif(not CHECKPOINT.is_file() or not ARTIFACT.is_file(), reason="CPU-E5 source artifacts unavailable")
def test_checkpoint_attention_reconstruction_is_bit_exact():
    module = _module()
    evidence = module.inspect_checkpoint(CHECKPOINT, ARTIFACT)
    assert evidence["matrix_count"] == 32
    assert evidence["elements"] == 14_155_776
    assert evidence["roundtrip_bit_mismatches"] == 0
    assert evidence["artifact_bit_mismatches"] == 0
    assert evidence["status"] == "PASS"
