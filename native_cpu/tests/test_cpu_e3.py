"""CPU-E3 exact verification and partial rollback checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from native_cpu.tools.native import NativeRuntime


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
BASELINE = ROOT / "native_cpu" / "benchmarks" / "cpu-e2" / "baseline" / "minimind_cpu.dll"
CANDIDATE = ROOT / "native_cpu" / "build-e3" / "Release" / "minimind_cpu.dll"
REQUIRED = (MODEL, BASELINE, CANDIDATE)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequential(runtime: NativeRuntime, ids: np.ndarray) -> np.ndarray:
    return np.stack([runtime.eval(np.asarray([int(token)], dtype=np.int32)) for token in ids])


@pytest.mark.skipif(not all(path.is_file() for path in REQUIRED), reason="CPU-E3 production artifacts are not built")
def test_verify_x4_is_exact_and_can_be_partially_truncated():
    assert _sha(BASELINE) == "3ff9418181d29ad9f7761e2a48866f06759027b4599ef1d024f24ac4cd3acfd5"
    with NativeRuntime(MODEL, BASELINE, 2048, threads=1, cpus=[0], row_weights=[1]) as baseline, \
            NativeRuntime(MODEL, CANDIDATE, 2048, threads=1, cpus=[0], row_weights=[1]) as candidate:
        prefix = np.asarray([(index * 17 + 11) % baseline.vocab_size for index in range(64)], dtype=np.int32)
        speculative = np.asarray([101, 202, 303, 404], dtype=np.int32)
        baseline.eval(prefix)
        candidate.eval(prefix)
        expected = _sequential(baseline, speculative)
        actual = candidate.verify_x4(speculative)
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
        assert candidate.position == baseline.position == 68

        continuation = np.asarray([505, 606, 707], dtype=np.int32)
        np.testing.assert_array_equal(candidate.eval(continuation).view(np.uint8), baseline.eval(continuation).view(np.uint8))
        assert candidate.position == baseline.position == 71

        for accepted in (1, 2, 3):
            baseline.reset()
            candidate.reset()
            baseline.eval(prefix)
            candidate.eval(prefix)
            _sequential(baseline, speculative[:accepted])
            candidate.verify_x4(speculative)
            candidate.truncate(len(prefix) + accepted)
            expected_tail = baseline.eval(np.asarray([808, 909], dtype=np.int32))
            actual_tail = candidate.eval(np.asarray([808, 909], dtype=np.int32))
            np.testing.assert_array_equal(actual_tail.view(np.uint8), expected_tail.view(np.uint8))
            assert candidate.position == baseline.position == len(prefix) + accepted + 2

