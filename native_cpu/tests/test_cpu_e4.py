import unittest

import numpy as np

from native_cpu.tools.chat import ChatSession, sample_logits


class _Tokenizer:
    eos_token_id = 99
    vocab_size = 128

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in ids)


class _Runtime:
    max_context = 4096
    vocab_size = 128

    def __init__(self, *, supports_verify=True, eval_tokens=()):
        self.supports_truncate = True
        self.supports_verify_x4 = supports_verify
        self._pos = 0
        self._epoch = 0
        self.eval_calls = []
        self.verify_calls = []
        self.eval_tokens = list(eval_tokens)

    @property
    def position(self):
        return self._pos

    @property
    def backend(self):
        return "fake"

    @property
    def cache_identity(self):
        return (id(self), self._epoch)

    def reset(self):
        self._pos = 0
        self._epoch += 1

    def truncate(self, position):
        self._pos = int(position)
        self._epoch += 1

    @staticmethod
    def _logit(token):
        out = np.full(128, -100.0, dtype=np.float32)
        out[int(token)] = 100.0
        return out

    def eval(self, ids):
        values = [int(x) for x in np.asarray(ids).reshape(-1)]
        self.eval_calls.append(tuple(values))
        self._pos += len(values)
        token = self.eval_tokens.pop(0) if self.eval_tokens else 99
        return self._logit(token)

    def verify_x4(self, token_ids):
        values = tuple(int(x) for x in np.asarray(token_ids).reshape(-1))
        self.verify_calls.append(values)
        self._pos += 4
        # The returned rows are logits after each verified token.  Tests set
        # these to the exact target sequence expected by the sequential path.
        return np.stack([self._logit(x) for x in (21, 22, 23, 99)])


def _history(*, match=True):
    if match:
        return list(range(1, 13)) + [20, 21, 22, 23] + list(range(1, 13))
    return list(range(1, 29))


def _session(history, samples, *, max_new_tokens=5, supports_verify=True, temperature=0.0):
    runtime = _Runtime(supports_verify=supports_verify)
    runtime._pos = len(history)
    session = ChatSession(runtime, _Tokenizer(), context_limit=4096,
                          max_new_tokens=max_new_tokens, temperature=temperature,
                          top_k=50 if temperature else 0,
                          top_p=0.85 if temperature else 1.0, seed=0)
    session.cached_token_ids = list(history)
    session._cache_identity = session._identity()
    session._cache_trusted = True
    sample_queue = list(samples)
    calls = []

    def sample(_logits):
        calls.append(1)
        if not sample_queue:
            raise AssertionError("test sampler queue exhausted")
        return sample_queue.pop(0)

    session._sample = sample
    return session, runtime, calls


class CpuE4DecodeTests(unittest.TestCase):
    def _decode(self, session, initial_token=20):
        return session._decode_tokens(_Runtime._logit(initial_token), 99)

    def test_lookup_uses_latest_non_overlapping_match(self):
        session, _, _ = _session(_history(), [99])
        self.assertEqual(session._lookup_proposal(), (20, 21, 22, 23))

    def test_no_match_falls_back_without_verification(self):
        session, runtime, _ = _session(_history(match=False), [30, 99])
        generated, sampled, _, _, _, telemetry = self._decode(session, 30)
        self.assertEqual((generated, sampled), ([30], [30, 99]))
        self.assertEqual(runtime.verify_calls, [])
        self.assertEqual(telemetry["proposed_blocks"], 0)

    def test_accepts_all_four_and_commits_x4(self):
        session, runtime, calls = _session(_history(), [20, 21, 22, 23, 99])
        generated, sampled, _, _, _, telemetry = self._decode(session)
        self.assertEqual((generated, sampled), ([20, 21, 22, 23], [20, 21, 22, 23, 99]))
        self.assertEqual(runtime.verify_calls, [(20, 21, 22, 23)])
        self.assertEqual(runtime.position, len(_history()) + 4)
        self.assertEqual(len(calls), 5)
        self.assertEqual(telemetry["accepted_tokens"], 4)
        self.assertEqual(telemetry["mean_accepted_length"], 4.0)

    def test_first_position_mismatch_does_not_verify(self):
        session, runtime, _ = _session(_history(), [30, 99])
        generated, sampled, _, _, _, telemetry = self._decode(session, 30)
        self.assertEqual((generated, sampled), ([30], [30, 99]))
        self.assertEqual(runtime.verify_calls, [])
        self.assertEqual(telemetry["proposed_blocks"], 1)
        self.assertEqual(telemetry["first_position_failures"], 1)

    def test_mismatches_at_positions_two_three_and_four_rollback(self):
        cases = (([20, 30, 99], 1, 2), ([20, 21, 30, 99], 2, 3),
                 ([20, 21, 22, 30, 99], 3, 4))
        for samples, accepted, committed in cases:
            with self.subTest(samples=samples):
                session, runtime, _ = _session(_history(), samples)
                generated, sampled, _, _, _, telemetry = self._decode(session)
                self.assertEqual(generated, samples[:-1])
                self.assertEqual(sampled, samples)
                self.assertEqual(len(runtime.verify_calls), 1)
                self.assertEqual(telemetry["accepted_tokens"], accepted)
                self.assertEqual(runtime.position, len(_history()) + committed)

    def test_eos_inside_block_is_not_evaluated(self):
        session, runtime, _ = _session(_history(), [20, 99])
        generated, sampled, evaluated, _, _, telemetry = self._decode(session)
        self.assertEqual((generated, sampled), ([20], [20, 99]))
        self.assertEqual(evaluated, 1)
        self.assertEqual(runtime.position, len(_history()) + 1)
        self.assertEqual(telemetry["accepted_tokens"], 1)

    def test_generation_limit_never_verifies_final_fourth_token(self):
        session, runtime, _ = _session(_history(), [20, 21, 22, 23], max_new_tokens=4)
        generated, sampled, evaluated, _, _, telemetry = self._decode(session)
        self.assertEqual((generated, sampled), ([20, 21, 22, 23], [20, 21, 22, 23]))
        self.assertEqual(runtime.verify_calls, [])
        self.assertEqual(evaluated, 3)
        self.assertEqual(runtime.position, len(_history()) + 3)
        self.assertEqual(telemetry["proposed_blocks"], 0)

    def test_old_runtime_falls_back_without_verify_symbol(self):
        session, runtime, _ = _session(_history(), [20, 99], supports_verify=False)
        self._decode(session)
        self.assertEqual(runtime.verify_calls, [])

    def test_speculation_preserves_rng_state_when_samples_match(self):
        history = _history()
        candidate, candidate_runtime, _ = _session(history, [], temperature=0.9)
        reference, reference_runtime, _ = _session(history, [], supports_verify=False, temperature=0.9)
        candidate_runtime.eval_tokens = [21, 22, 23, 99]
        reference_runtime.eval_tokens = [21, 22, 23, 99]
        initial = _Runtime._logit(20)
        candidate._sample = lambda logits: sample_logits(logits, candidate.rng,
                                                          temperature=candidate.temperature,
                                                          top_k=candidate.top_k, top_p=candidate.top_p)
        reference._sample = lambda logits: sample_logits(logits, reference.rng,
                                                          temperature=reference.temperature,
                                                          top_k=reference.top_k, top_p=reference.top_p)
        candidate_result = candidate._decode_tokens(initial, 99)
        reference_result = reference._decode_tokens(initial, 99)
        self.assertEqual(candidate_result[1], reference_result[1])
        self.assertEqual(candidate_result[0], reference_result[0])
        self.assertEqual(candidate_runtime.position, reference_runtime.position)
        self.assertEqual(candidate.rng.bit_generator.state, reference.rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
