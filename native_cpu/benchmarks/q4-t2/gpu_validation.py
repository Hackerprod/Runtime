"""CUDA validation and diagnostic timing for Q4-T2.

This script is intentionally a test/evidence harness, not a training entry
point.  It checks the Torch implementation against the SPEC-Q4-v1 NumPy
reference, exercises autograd on CUDA, and reports a CUDA-event timing for
the two MiniMind FFN shapes.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from model.quantization import dequantize_tensor, quantize_tensor
from model.quantization.qat import (
    QATLinear,
    spec_q4_codes_scales,
    spec_q4_fake_dequant,
    spec_q4_ste,
)


GOLDEN = ROOT / "model" / "quantization" / "golden_vectors.json"


def _bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _assert_equal(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if _bytes(left) != _bytes(right):
        raise AssertionError(f"bitwise mismatch: {label}")


def _check_reference(source_cpu: torch.Tensor, device: torch.device, label: str) -> None:
    numpy_q4 = quantize_tensor(source_cpu.numpy())
    numpy_effective = dequantize_tensor(numpy_q4)
    cpu_effective = spec_q4_fake_dequant(source_cpu)
    gpu_source = source_cpu.to(device)
    gpu_effective = spec_q4_fake_dequant(gpu_source).cpu()
    _assert_equal(cpu_effective, torch.from_numpy(numpy_effective), f"CPU {label}")
    _assert_equal(gpu_effective, cpu_effective, f"CUDA effective {label}")
    cpu_codes, cpu_scales = spec_q4_codes_scales(source_cpu)
    gpu_codes, gpu_scales = spec_q4_codes_scales(gpu_source)
    _assert_equal(cpu_codes, gpu_codes.cpu(), f"CUDA codes {label}")
    _assert_equal(cpu_scales, gpu_scales.cpu(), f"CUDA scales {label}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    device = torch.device("cuda")
    print(f"torch={torch.__version__}")
    print(f"cuda_runtime={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(device)}")

    vectors = json.loads(GOLDEN.read_text(encoding="utf-8"))["vectors"]
    for index, vector in enumerate(vectors):
        source = torch.tensor(vector["values"], dtype=torch.float32).reshape(vector["shape"])
        _check_reference(source, device, f"golden[{index}]")

    shapes = [(2, 1), (2, 7), (2, 31), (2, 32), (2, 33), (2, 63),
              (2, 64), (2, 65), (2, 768), (2, 2432)]
    for shape in shapes:
        generator = torch.Generator().manual_seed(sum(shape) * 17)
        source = torch.randn(shape, generator=generator, dtype=torch.float32)
        source[0, 0] = 0.0
        if shape[-1] > 2:
            source[0, 1] = -0.0
            source[0, -1] = 6.0
        _check_reference(source, device, f"random{shape}")

    boundaries = torch.tensor(
        [[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
          -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0]],
        dtype=torch.float32,
    )
    _check_reference(torch.cat((boundaries, torch.zeros((1, 16))), dim=1), device, "RTNE-boundaries")

    weight = torch.tensor([[0.25, 1.0, -2.0, 6.0]], device=device, requires_grad=True)
    effective = spec_q4_fake_dequant(weight).detach()
    actual = spec_q4_ste(weight)
    _assert_equal(actual, effective, "CUDA STE forward")
    actual.sum().backward()
    _assert_equal(weight.grad, torch.ones_like(weight), "CUDA STE gradient")

    linear = nn.Linear(7, 5, bias=False, device=device)
    qat = QATLinear.from_linear(linear)
    x = torch.randn(3, 7, device=device)
    qat(x).sum().backward()
    if qat.weight.grad is None or not bool(torch.isfinite(qat.weight.grad).all()):
        raise AssertionError("CUDA QATLinear gradient is missing or non-finite")

    timing = {}
    for shape in ((2432, 768), (768, 2432)):
        weight = torch.randn(shape, dtype=torch.float32, device=device)
        for _ in range(3):
            spec_q4_fake_dequant(weight)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        samples = []
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            spec_q4_fake_dequant(weight)
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        timing[str(shape)] = {
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }

    result = {
        "status": "PASS",
        "golden_vectors": len(vectors),
        "random_shapes": [list(shape) for shape in shapes],
        "rtne_boundaries": "PASS",
        "ste_gradient": "PASS",
        "qat_linear_backward": "PASS",
        "timing": timing,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
