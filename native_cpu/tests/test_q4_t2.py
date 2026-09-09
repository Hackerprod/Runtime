"""PyTorch Q4-T2 parity, autograd, and MiniMind wrapping tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quantization import dequantize_tensor, quantize_tensor
from model.quantization.qat import (
    QATLinear,
    QAT_WEIGHT_COUNT,
    QAT_TENSOR_COUNT,
    SpecQ4QATError,
    apply_spec_q4_qat,
    spec_q4_codes_scales,
    spec_q4_fake_dequant,
    spec_q4_ste,
)


ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "model" / "quantization" / "golden_vectors.json"


def _torch_from_golden(vector: dict[str, object]) -> torch.Tensor:
    return torch.tensor(vector["values"], dtype=torch.float32).reshape(vector["shape"])


def _logical_codes(qtensor) -> np.ndarray:
    packed = qtensor.packed_weights.reshape(-1, qtensor.logical_shape[-1] // 2 + qtensor.logical_shape[-1] % 2)
    columns = qtensor.logical_shape[-1]
    codes = np.empty((packed.shape[0], columns), dtype=np.uint8)
    for column in range(columns):
        byte = packed[:, column // 2]
        codes[:, column] = byte & 0x0F if column % 2 == 0 else byte >> 4
    return codes.reshape(qtensor.logical_shape)


def test_torch_fake_dequant_matches_q4_t1_golden_vectors_bitwise():
    vectors = json.loads(GOLDEN.read_text(encoding="utf-8"))["vectors"]
    for vector in vectors:
        source = _torch_from_golden(vector)
        torch_effective = spec_q4_fake_dequant(source).numpy()
        numpy_effective = dequantize_tensor(quantize_tensor(source.numpy()))
        assert torch_effective.tobytes() == numpy_effective.tobytes()
        codes, scales = spec_q4_codes_scales(source)
        numpy_q4 = quantize_tensor(source.numpy())
        assert codes.detach().numpy().tobytes() == _logical_codes(numpy_q4).tobytes()
        assert scales.detach().numpy().tobytes() == numpy_q4.scales.tobytes()


@pytest.mark.parametrize("shape", [(2, 1), (2, 7), (2, 31), (2, 32), (2, 33),
                                    (2, 63), (2, 64), (2, 65), (2, 768), (2, 2432)])
def test_torch_random_and_tail_parity(shape):
    generator = torch.Generator().manual_seed(sum(shape) * 17)
    source = torch.randn(shape, generator=generator, dtype=torch.float32)
    source[0, 0] = 0.0
    if shape[-1] > 2:
        source[0, 1] = -0.0
        source[0, -1] = 6.0
    torch_codes, torch_scales = spec_q4_codes_scales(source)
    numpy_q4 = quantize_tensor(source.numpy())
    assert torch_codes.numpy().tobytes() == _logical_codes(numpy_q4).tobytes()
    assert torch_scales.numpy().tobytes() == numpy_q4.scales.tobytes()
    assert spec_q4_fake_dequant(source).numpy().tobytes() == dequantize_tensor(numpy_q4).tobytes()


def test_torch_boundaries_and_zero_block_match_scalar_reference():
    source = torch.tensor(
        [[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
          -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0]],
        dtype=torch.float32,
    )
    source = torch.cat((source, torch.zeros((1, 16))), dim=1)
    torch_codes, torch_scales = spec_q4_codes_scales(source)
    numpy_q4 = quantize_tensor(source.numpy())
    assert torch_codes.numpy().tobytes() == _logical_codes(numpy_q4).tobytes()
    assert torch_scales.numpy().tobytes() == numpy_q4.scales.tobytes()
    assert spec_q4_fake_dequant(source).numpy().tobytes() == dequantize_tensor(numpy_q4).tobytes()


def test_fake_quant_rejects_non_finite_and_unrepresentable_exponents():
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(SpecQ4QATError):
            spec_q4_fake_dequant(torch.tensor([[value]], dtype=torch.float32))
    smallest = torch.tensor([[torch.finfo(torch.float32).tiny / 2]], dtype=torch.float32)
    with pytest.raises(SpecQ4QATError):
        spec_q4_fake_dequant(smallest)


def test_ste_forward_is_fake_dequant_and_gradient_is_identity():
    weight = torch.tensor([[0.25, 1.0, -2.0, 6.0]], dtype=torch.float32, requires_grad=True)
    expected = spec_q4_fake_dequant(weight).detach()
    actual = spec_q4_ste(weight)
    assert actual.detach().numpy().tobytes() == expected.numpy().tobytes()
    actual.sum().backward()
    assert weight.grad is not None
    assert weight.grad.numpy().tobytes() == torch.ones_like(weight).numpy().tobytes()


def test_qat_linear_preserves_weight_parameter_and_backward():
    linear = nn.Linear(7, 5, bias=False)
    parameter = linear.weight
    qat = QATLinear.from_linear(linear)
    assert qat.weight is parameter
    assert list(qat.state_dict()) == ["weight"]
    x = torch.randn(3, 7, dtype=torch.float32)
    output = qat(x)
    reference = torch.nn.functional.linear(x, spec_q4_ste(parameter), bias=None)
    assert output.detach().numpy().tobytes() == reference.detach().numpy().tobytes()
    output.sum().backward()
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    assert not any(name.endswith("codes") or name.endswith("scales") for name, _ in qat.named_parameters())


def test_apply_wraps_only_24_dense_ffn_projections_and_preserves_keys():
    torch.manual_seed(123)
    model = MiniMindForCausalLM(MiniMindConfig())
    keys_before = set(model.state_dict())
    parameters_before = {
        (layer_index, name): getattr(model.model.layers[layer_index].mlp, name).weight
        for layer_index in range(8)
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    apply_spec_q4_qat(model)
    keys_after = set(model.state_dict())
    assert keys_before == keys_after
    assert sum(isinstance(module, QATLinear) for module in model.modules()) == QAT_TENSOR_COUNT
    assert sum(module.weight.numel() for module in model.modules() if isinstance(module, QATLinear)) == QAT_WEIGHT_COUNT
    for (layer_index, name), parameter in parameters_before.items():
        wrapped = getattr(model.model.layers[layer_index].mlp, name)
        assert isinstance(wrapped, QATLinear)
        assert wrapped.weight is parameter
    assert not isinstance(model.model.layers[0].self_attn.q_proj, QATLinear)
    assert not isinstance(model.model.layers[0].self_attn.k_proj, QATLinear)
    assert not isinstance(model.model.layers[0].self_attn.v_proj, QATLinear)
    assert not isinstance(model.model.layers[0].self_attn.o_proj, QATLinear)
    assert not isinstance(model.model.embed_tokens, QATLinear)
    assert not isinstance(model.lm_head, QATLinear)
    assert not isinstance(model.model.layers[0].input_layernorm, QATLinear)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    assert model.load_state_dict(state, strict=True).missing_keys == []
    with pytest.raises(SpecQ4QATError):
        apply_spec_q4_qat(model)


def test_apply_rejects_moe_and_non_minimind_shapes():
    moe = MiniMindForCausalLM(MiniMindConfig(use_moe=True))
    with pytest.raises(SpecQ4QATError, match="use_moe"):
        apply_spec_q4_qat(moe)
    small = MiniMindForCausalLM(MiniMindConfig(hidden_size=128, num_hidden_layers=2))
    with pytest.raises(SpecQ4QATError):
        apply_spec_q4_qat(small)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available locally")
@pytest.mark.parametrize("shape", [(2, 768), (2, 2432)])
def test_cuda_matches_cpu_and_numpy_bitwise(shape):
    generator = torch.Generator().manual_seed(sum(shape) * 29)
    source = torch.randn(shape, generator=generator, dtype=torch.float32)
    cpu = spec_q4_fake_dequant(source)
    gpu = spec_q4_fake_dequant(source.cuda()).cpu()
    numpy_effective = dequantize_tensor(quantize_tensor(source.numpy()))
    assert gpu.numpy().tobytes() == cpu.numpy().tobytes() == numpy_effective.tobytes()
