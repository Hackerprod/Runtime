"""PyTorch fake quantization for the SPEC-Q4-v1 QAT experiment.

The implementation mirrors the scalar reference in :mod:`spec_q4` while
keeping every operation on the input device.  It is intentionally limited to
the dense MiniMind-3 FFN projections; no runtime or checkpoint format is
involved here.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .mxfp4 import BLOCK_SIZE, E8M0_MAX_EXPONENT, E8M0_MIN_EXPONENT

QAT_LAYER_COUNT = 8
QAT_TENSOR_COUNT = 24
QAT_WEIGHT_COUNT = 44_826_624
QAT_HIDDEN_SIZE = 768
QAT_INTERMEDIATE_SIZE = 2432

_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class SpecQ4QATError(ValueError):
    """Raised when a tensor or model cannot use the Q4 QAT path."""


def _require_float_tensor(weight: Tensor) -> Tensor:
    if not isinstance(weight, Tensor):
        raise TypeError("SPEC-Q4 expects a torch.Tensor")
    if not weight.is_floating_point():
        raise SpecQ4QATError("SPEC-Q4 weights must use a floating-point dtype")
    # Quantization is explicitly defined in float32, but this conversion stays
    # on the source device and does not make a second trainable parameter.
    weight32 = weight.to(dtype=torch.float32)
    if not bool(torch.isfinite(weight32).all()):
        raise SpecQ4QATError("SPEC-Q4 weights must be finite")
    return weight32


def _floor_log2_float32(values: Tensor) -> Tensor:
    """Bit-exact floor(log2) for positive float32 values on any device."""

    bits = values.view(torch.int32).to(dtype=torch.int64)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    # mantissa is at most 23 bits; this vectorized comparison avoids log2 and
    # contains no loop over weights. The zero case is replaced by the caller.
    powers = torch.bitwise_left_shift(
        torch.ones(23, dtype=torch.int64, device=values.device),
        torch.arange(23, dtype=torch.int64, device=values.device),
    )
    highest = (mantissa.unsqueeze(-1) >= powers).sum(dim=-1, dtype=torch.int64) - 1
    return torch.where(exponent != 0, exponent - 127, highest - 149)


def _nearest_magnitude_index(magnitude: Tensor) -> Tensor:
    """Vectorized nearest code with deterministic even-index tie breaking."""

    codebook = torch.tensor(
        _FP4_MAGNITUDES, dtype=torch.float32, device=magnitude.device
    )
    distances = (magnitude.unsqueeze(-1) - codebook).abs()
    minimum = distances.amin(dim=-1, keepdim=True)
    tied = distances == minimum
    indices = torch.arange(8, dtype=torch.int64, device=magnitude.device)
    even_candidates = torch.where(
        tied & ((indices & 1) == 0), indices, torch.full_like(indices, 99)
    ).amin(dim=-1)
    any_candidate = torch.where(tied, indices, torch.full_like(indices, 99)).amin(dim=-1)
    # A unique odd code (for example exactly +6, code 7) must remain odd;
    # even-index selection applies only when the nearest set is a tie.
    return torch.where(even_candidates != 99, even_candidates, any_candidate)


def _quantize_codes_and_scales(weight: Tensor) -> tuple[Tensor, Tensor, tuple[int, ...]]:
    """Return logical FP4 codes, E8M0 scale bytes, and the original shape."""

    weight32 = _require_float_tensor(weight)
    if weight32.ndim < 1:
        raise SpecQ4QATError("SPEC-Q4 requires at least one dimension")
    logical_shape = tuple(int(size) for size in weight32.shape)
    columns = logical_shape[-1]
    rows = weight32.numel() // columns
    groups = (columns + BLOCK_SIZE - 1) // BLOCK_SIZE
    flat = weight32.reshape(rows, columns)
    padded_columns = groups * BLOCK_SIZE
    if padded_columns != columns:
        padded = F.pad(flat, (0, padded_columns - columns), value=0.0)
    else:
        padded = flat
    blocks = padded.reshape(rows, groups, BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)
    zero = amax == 0
    floor_log2 = _floor_log2_float32(amax)
    shared_exponent = torch.where(
        zero, torch.zeros_like(floor_log2), floor_log2 - 2
    )
    valid = (shared_exponent >= E8M0_MIN_EXPONENT) & (
        shared_exponent <= E8M0_MAX_EXPONENT
    )
    if not bool(valid.all()):
        raise SpecQ4QATError("SPEC-Q4 shared exponent cannot be encoded as E8M0")
    scales = torch.ldexp(
        torch.ones_like(amax), shared_exponent.to(dtype=torch.int32)
    )
    normalized = padded.reshape(rows, groups, BLOCK_SIZE) / scales.unsqueeze(-1)
    magnitude = normalized.abs()
    magnitude_index = _nearest_magnitude_index(magnitude)
    sign = (normalized < 0).to(dtype=torch.int64)
    codes_padded = magnitude_index | (sign << 3)
    # Canonicalize both explicit -0 and values that round to zero.
    codes_padded = torch.where(magnitude_index == 0, torch.zeros_like(codes_padded), codes_padded)
    codes = codes_padded.reshape(rows, padded_columns)[:, :columns].to(dtype=torch.uint8)
    scales_raw = (shared_exponent + 127).to(dtype=torch.uint8)
    return (
        codes.reshape(logical_shape),
        scales_raw.reshape((*logical_shape[:-1], groups)),
        logical_shape,
    )


def spec_q4_codes_scales(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Return detached logical FP4 codes and E8M0 scales for diagnostics."""

    with torch.no_grad():
        codes, scales, _ = _quantize_codes_and_scales(weight)
    return codes, scales


def spec_q4_fake_dequant(weight: Tensor) -> Tensor:
    """Return the float32 effective weight reconstructed from SPEC-Q4."""

    with torch.no_grad():
        codes, scales_raw, logical_shape = _quantize_codes_and_scales(weight)
        magnitudes = torch.tensor(
            _FP4_MAGNITUDES, dtype=torch.float32, device=weight.device
        )
        magnitude_index = codes.to(dtype=torch.int64) & 0x7
        result = magnitudes[magnitude_index]
        sign = (codes.to(dtype=torch.int64) & 0x8) != 0
        result = torch.where(sign, -result, result)
        scales = torch.ldexp(
            torch.ones_like(scales_raw, dtype=torch.float32),
            (scales_raw.to(dtype=torch.int32) - 127),
        )
        expanded_scales = scales.repeat_interleave(BLOCK_SIZE, dim=-1)[..., : logical_shape[-1]]
        result = result * expanded_scales
        return result.reshape(logical_shape)


def spec_q4_ste(weight: Tensor) -> Tensor:
    """Fake-quantize in forward while exposing an identity gradient."""

    effective = spec_q4_fake_dequant(weight)
    return weight + (effective - weight).detach()


class QATLinear(nn.Linear):
    """Bias-free linear layer whose weight is fake-quantized on forward."""

    def __init__(self, in_features: int, out_features: int, *, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=False, device=device, dtype=dtype)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QATLinear":
        if not isinstance(linear, nn.Linear):
            raise TypeError("QATLinear.from_linear expects nn.Linear")
        if linear.bias is not None:
            raise SpecQ4QATError("QATLinear only supports bias-free projections")
        wrapped = cls(
            linear.in_features,
            linear.out_features,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        # Reuse the exact Parameter object; this preserves optimizer/checkpoint
        # identity and leaves the state_dict key as simply ``weight``.
        wrapped.weight = linear.weight
        return wrapped

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, spec_q4_ste(self.weight), bias=None)


def _model_layers(model: nn.Module) -> Any:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        layers = getattr(model, "layers", None)
    return layers


def apply_spec_q4_qat(model: nn.Module) -> nn.Module:
    """Wrap exactly the 24 dense MiniMind-3 FFN projections in-place.

    The function is deliberately fail-closed for MoE or non-MiniMind-3
    shapes. It returns the same model object for convenient composition.
    """

    config = getattr(model, "config", None)
    if bool(getattr(config, "use_moe", False)):
        raise SpecQ4QATError("SPEC-Q4 QAT currently rejects use_moe=True")
    if config is None or int(getattr(config, "hidden_size", -1)) != QAT_HIDDEN_SIZE:
        raise SpecQ4QATError("SPEC-Q4 QAT expects MiniMind-3 hidden_size=768")
    if int(getattr(config, "intermediate_size", -1)) != QAT_INTERMEDIATE_SIZE:
        raise SpecQ4QATError("SPEC-Q4 QAT expects MiniMind-3 intermediate_size=2432")
    if int(getattr(config, "num_hidden_layers", -1)) != QAT_LAYER_COUNT:
        raise SpecQ4QATError("SPEC-Q4 QAT expects exactly 8 transformer layers")
    layers = _model_layers(model)
    if layers is None or len(layers) != QAT_LAYER_COUNT:
        raise SpecQ4QATError("model does not expose the expected MiniMind layer stack")

    targets = ("gate_proj", "up_proj", "down_proj")
    wrapped_count = 0
    weight_count = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "gate_proj"):
            raise SpecQ4QATError("dense FFN projections are missing")
        for name in targets:
            projection = getattr(mlp, name)
            if isinstance(projection, QATLinear):
                raise SpecQ4QATError("model is already wrapped for SPEC-Q4 QAT")
            if not isinstance(projection, nn.Linear) or projection.bias is not None:
                raise SpecQ4QATError(f"{name} must be a bias-free nn.Linear")
            expected = (
                (QAT_HIDDEN_SIZE, QAT_INTERMEDIATE_SIZE)
                if name in ("gate_proj", "up_proj")
                else (QAT_INTERMEDIATE_SIZE, QAT_HIDDEN_SIZE)
            )
            if (projection.in_features, projection.out_features) != expected:
                raise SpecQ4QATError(f"unexpected {name} shape")
            wrapped = QATLinear.from_linear(projection)
            setattr(mlp, name, wrapped)
            wrapped_count += 1
            weight_count += projection.weight.numel()
    if wrapped_count != QAT_TENSOR_COUNT or weight_count != QAT_WEIGHT_COUNT:
        raise SpecQ4QATError(
            f"unexpected QAT target count: tensors={wrapped_count}, weights={weight_count}"
        )
    return model


__all__ = [
    "QATLinear",
    "QAT_LAYER_COUNT",
    "QAT_TENSOR_COUNT",
    "QAT_WEIGHT_COUNT",
    "SpecQ4QATError",
    "apply_spec_q4_qat",
    "spec_q4_codes_scales",
    "spec_q4_fake_dequant",
    "spec_q4_ste",
]
