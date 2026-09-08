"""Exact production-runtime checks for the automatic CPU-E2 prefill path."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from native_cpu.tools.native import NativeRuntime


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
REFERENCE = ROOT / "native_cpu" / "benchmarks" / "cpu-u1.2" / "baseline" / "minimind_cpu.dll"
CANDIDATE = ROOT / "native_cpu" / "build-e2" / "Release" / "minimind_cpu.dll"
REQUIRED = (MODEL, REFERENCE, CANDIDATE)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.skipif(not all(path.is_file() for path in REQUIRED), reason="production artifacts are not built")
def test_prefill_x4_exact_blocks_and_truncation():
    assert _sha(REFERENCE) == "5956ff064ad8e8fd1ff502d18626ce19240682faa55c584cb3d6eaad1775bb3a"
    with NativeRuntime(MODEL, REFERENCE, 2048, threads=1, cpus=[0], row_weights=[1]) as baseline, \
            NativeRuntime(MODEL, CANDIDATE, 2048, threads=1, cpus=[0], row_weights=[1]) as candidate:
        assert candidate.ffn_f16_storage
        for length in (1, 2, 3, 4, 5, 7, 8, 9, 16, 32, 64, 256):
            baseline.reset()
            candidate.reset()
            ids = np.asarray([(index * 17 + 11) % baseline.vocab_size for index in range(length)], dtype=np.int32)
            np.testing.assert_array_equal(candidate.eval(ids).view(np.uint8), baseline.eval(ids).view(np.uint8))
            assert candidate.position == baseline.position == length
            continuation = np.asarray([(index * 31 + 7) % baseline.vocab_size for index in range(3)], dtype=np.int32)
            np.testing.assert_array_equal(candidate.eval(continuation).view(np.uint8), baseline.eval(continuation).view(np.uint8))
            assert candidate.position == baseline.position == length + len(continuation)

        baseline.reset()
        candidate.reset()
        prefix = np.asarray([(index * 17 + 11) % baseline.vocab_size for index in range(16)], dtype=np.int32)
        baseline.eval(prefix)
        candidate.eval(prefix)
        baseline.truncate(4)
        candidate.truncate(4)
        suffix = np.asarray([222, 239, 256, 273], dtype=np.int32)
        np.testing.assert_array_equal(candidate.eval(suffix).view(np.uint8), baseline.eval(suffix).view(np.uint8))
        assert candidate.position == baseline.position == 8

