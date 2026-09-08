"""CPU-E6 integration parity against the frozen CPU-E4 runtime."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from native_cpu.tools.chat import ChatSession, DEFAULT_CHAT_SYSTEM, LocalTokenizer
from native_cpu.tools.native import NativeRuntime


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
BASELINE = ROOT / "native_cpu" / "benchmarks" / "cpu-e4" / "baseline" / "minimind_cpu.dll"
CANDIDATE = ROOT / "native_cpu" / "build-e6" / "Release" / "minimind_cpu.dll"
TOKENIZER = ROOT / "checkpoints" / "minimind-3-hf"
REQUIRED = (MODEL, BASELINE, CANDIDATE, TOKENIZER / "tokenizer.json")
EXPECTED_BASELINE_SHA256 = "b410e642e2249a8982a9a83b134f3d7f5e1f37c677f11d10900ed0e213cb97ab"
EXPECTED_ATTENTION_BYTES = 28_311_552


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.skipif(not all(path.is_file() for path in REQUIRED), reason="CPU-E6 artifacts are not built")
def test_attention_storage_exact_and_cache_parity():
    assert _sha(BASELINE) == EXPECTED_BASELINE_SHA256
    with NativeRuntime(MODEL, BASELINE, 2048, threads=1, cpus=[0], row_weights=[1]) as baseline, \
            NativeRuntime(MODEL, CANDIDATE, 2048, threads=1, cpus=[0], row_weights=[1]) as candidate:
        assert not baseline.attention_f16_storage
        assert candidate.attention_f16_storage
        assert candidate.attention_f16_storage_bytes == EXPECTED_ATTENTION_BYTES
        assert candidate.attention_f16_prepare_ns is not None
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


@pytest.mark.skipif(not all(path.is_file() for path in REQUIRED), reason="CPU-E6 artifacts are not built")
def test_attention_storage_preserves_chat_and_speculation():
    tokenizer = LocalTokenizer(TOKENIZER)
    with NativeRuntime(MODEL, BASELINE, 2048, threads=1, cpus=[0], row_weights=[1]) as baseline, \
            NativeRuntime(MODEL, CANDIDATE, 2048, threads=1, cpus=[0], row_weights=[1]) as candidate:
        baseline_chat = ChatSession(baseline, tokenizer, context_limit=2048, max_new_tokens=32,
                                    temperature=.9, top_k=50, top_p=.85, seed=0,
                                    system=DEFAULT_CHAT_SYSTEM, reuse_kv=True)
        candidate_chat = ChatSession(candidate, tokenizer, context_limit=2048, max_new_tokens=32,
                                     temperature=.9, top_k=50, top_p=.85, seed=0,
                                     system=DEFAULT_CHAT_SYSTEM, reuse_kv=True)
        for prompt in ("Name one benefit of tests.", "Define latency briefly.",
                       "Could you tell me about programming languages?"):
            expected = baseline_chat.turn(prompt)
            actual = candidate_chat.turn(prompt)
            assert actual.sampled_ids == expected.sampled_ids
            assert actual.text == expected.text
            assert actual.position_after == expected.position_after
            for key in ("proposed_blocks", "proposed_tokens", "accepted_tokens",
                        "verify_calls", "first_position_failures", "accepted_length_total"):
                assert actual.metrics["speculative_decode"][key] == expected.metrics["speculative_decode"][key]
