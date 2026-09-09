"""CPU-E8 LM-head FP16 integration parity against the CPU-E6 baseline."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from native_cpu.tools.chat import ChatSession, DEFAULT_CHAT_SYSTEM, LocalTokenizer
from native_cpu.tools.native import NativeRuntime


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
BASELINE = ROOT / "native_cpu" / "benchmarks" / "cpu-e6" / "baseline" / "minimind_cpu.dll"
CANDIDATE = ROOT / "native_cpu" / "build-e8" / "Release" / "minimind_cpu.dll"
TOKENIZER = ROOT / "checkpoints" / "minimind-3-hf"
REQUIRED = (MODEL, BASELINE, CANDIDATE, TOKENIZER / "tokenizer.json")
EXPECTED_BASELINE_SHA256 = "ab464cc1c91aedd3b33bf995fef82657e2316123979e6f1d04d15bb4eace72fd"
EXPECTED_LM_HEAD_BYTES = 9_830_400


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.skipif(not all(path.is_file() for path in REQUIRED), reason="CPU-E8 artifacts are not built")
def test_lm_head_storage_and_eval_parity():
    assert _sha(BASELINE) == EXPECTED_BASELINE_SHA256
    with NativeRuntime(MODEL, BASELINE, 2048, threads=1, cpus=[0], row_weights=[1]) as baseline, \
            NativeRuntime(MODEL, CANDIDATE, 2048, threads=1, cpus=[0], row_weights=[1]) as candidate:
        assert not baseline.lm_head_f16_storage
        assert candidate.lm_head_f16_storage
        assert candidate.lm_head_f16_storage_bytes == EXPECTED_LM_HEAD_BYTES
        assert candidate.lm_head_f16_prepare_ns is not None
        for length in (1, 2, 3, 4, 5, 7, 8, 9, 16, 64, 256):
            baseline.reset()
            candidate.reset()
            baseline.reset_stats()
            candidate.reset_stats()
            ids = np.asarray([(index * 17 + 11) % baseline.vocab_size for index in range(length)], dtype=np.int32)
            expected = baseline.eval(ids)
            actual = candidate.eval(ids)
            np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
            assert candidate.position == baseline.position == length
            assert candidate.lm_head_calls == baseline.lm_head_calls
            continuation = np.asarray([(index * 31 + 7) % baseline.vocab_size for index in range(3)], dtype=np.int32)
            expected = baseline.eval(continuation)
            actual = candidate.eval(continuation)
            np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
            assert candidate.position == baseline.position == length + len(continuation)


@pytest.mark.skipif(not all(path.is_file() for path in REQUIRED), reason="CPU-E8 artifacts are not built")
def test_lm_head_verify_x4_and_truncation_parity():
    assert _sha(BASELINE) == EXPECTED_BASELINE_SHA256
    with NativeRuntime(MODEL, BASELINE, 2048, threads=1, cpus=[0], row_weights=[1]) as baseline, \
            NativeRuntime(MODEL, CANDIDATE, 2048, threads=1, cpus=[0], row_weights=[1]) as candidate:
        prefix = np.asarray([(index * 29 + 5) % baseline.vocab_size for index in range(16)], dtype=np.int32)
        baseline.eval(prefix)
        candidate.eval(prefix)
        before = (baseline.position, candidate.position)
        expected = baseline.verify_x4(np.asarray([901, 902, 903, 904], dtype=np.int32))
        actual = candidate.verify_x4(np.asarray([901, 902, 903, 904], dtype=np.int32))
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
        assert before == (16, 16)
        assert baseline.position == candidate.position == 20
        assert baseline.lm_head_calls == candidate.lm_head_calls
        assert candidate.lm_head_calls >= 4
        baseline.truncate(16)
        candidate.truncate(16)
        for accepted in (1, 2, 3):
            expected = baseline.verify_x4(np.asarray([901, 902, 903, 904], dtype=np.int32))
            actual = candidate.verify_x4(np.asarray([901, 902, 903, 904], dtype=np.int32))
            np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
            baseline.truncate(16 + accepted)
            candidate.truncate(16 + accepted)
            suffix = np.asarray([1001 + accepted, 1101 + accepted], dtype=np.int32)
            expected = baseline.eval(suffix)
            actual = candidate.eval(suffix)
            np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
            assert baseline.position == candidate.position == 16 + accepted + len(suffix)
            baseline.truncate(16)
            candidate.truncate(16)


@pytest.mark.skipif(not all(path.is_file() for path in REQUIRED), reason="CPU-E8 artifacts are not built")
def test_lm_head_storage_preserves_chat_and_speculation():
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
