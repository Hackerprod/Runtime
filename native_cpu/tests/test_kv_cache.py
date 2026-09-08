"""Contract tests for transactional ChatSession KV reuse (no model/DLL required)."""
from __future__ import annotations
import numpy as np
import pytest

from native_cpu.tools.chat import ChatError, ChatSession


class ScriptedTokenizer:
    vocab_size = 64
    eos_token_id = 2
    def __init__(self): self._ids = {"one": 10, "two": 11, "three": 12, "short": 13, "different": 14}
    def apply_chat_template(self, messages, **_):
        out = [1]
        for message in messages:
            out += [3 if message["role"] == "system" else 4]
            out += [self._ids.get(word, 20 + (sum(map(ord, word)) % 20)) for word in message["content"].split()]
            out += [2]
        out += [5]
        return out
    def decode(self, ids, skip_special_tokens=True): return " ".join(str(int(x)) for x in ids if not skip_special_tokens or x not in {1, 2, 3, 4, 5})


class FakeRuntime:
    vocab_size = 64
    max_context = 64
    backend = "fake"
    def __init__(self, truncate=True, fail=False):
        self.calls, self._tokens, self.epoch, self.fail = [], [], 0, fail
        self.supports_truncate = truncate
    @property
    def position(self): return len(self._tokens)
    @property
    def cache_identity(self): return (id(self), self.epoch)
    @property
    def logits_valid(self): return bool(self._tokens)
    def reset(self): self.calls.append(("reset", [])); self._tokens.clear(); self.epoch += 1
    def truncate(self, position):
        if not self.supports_truncate: raise RuntimeError("truncate unsupported")
        if position < 0 or position > self.position: raise ValueError("invalid truncate")
        del self._tokens[position:]; self.epoch += 1; self.calls.append(("truncate", position))
    def eval(self, ids):
        if np.asarray(ids).dtype.kind not in "iu": raise ValueError("token ids must be integer")
        values = [int(x) for x in np.asarray(ids).reshape(-1)]
        if not values: raise ValueError("empty")
        if any(x < 0 or x >= self.vocab_size for x in values): raise ValueError("token id")
        self.calls.append(("eval", values))
        self._tokens.extend(values)
        if self.fail:
            self.fail = False
            self.epoch += 1
            raise RuntimeError("injected eval failure")
        self.epoch += 1
        logits = np.full(self.vocab_size, -100.0, dtype=np.float32)
        logits[(sum(self._tokens) + self.position) % self.vocab_size] = 100.0
        return logits


def make_session(runtime, **kwargs):
    sampler = kwargs.pop("sampler", lambda _logits: 6)
    session = ChatSession(runtime, ScriptedTokenizer(), context_limit=64, max_new_tokens=kwargs.pop("max_new_tokens", 1),
                          temperature=0.0, seed=0, system="system", reuse_kv=kwargs.pop("reuse_kv", False), **kwargs)
    session._sample = sampler
    return session


def eval_batches(runtime): return [batch for op, batch in runtime.calls if op == "eval"]


def test_multiturn_reuses_exact_prefix_and_never_commits_eos():
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=True, max_new_tokens=1)
    session.turn("one"); runtime.calls.clear()
    canonical = session._bounded("two")[1]; cached = list(getattr(session, "cached_token_ids", []))
    lcp = 0
    while lcp < len(canonical) and lcp < len(cached) and canonical[lcp] == cached[lcp]: lcp += 1
    session.turn("two")
    batches = eval_batches(runtime)
    assert batches and batches[-1] == canonical[lcp:]
    assert 2 not in getattr(session, "cached_token_ids", [])[-1:]


def test_no_kv_mode_resets_each_turn():
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=False)
    session.turn("one"); runtime.calls.clear(); session.turn("two")
    assert runtime.calls[0][0] == "reset"


def test_shorter_prompt_and_divergence_truncate_before_suffix():
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=True)
    session.turn("one two"); runtime.calls.clear(); session.turn("short")
    assert any(op == "truncate" for op, _ in runtime.calls)
    assert runtime.position == len(getattr(session, "cached_token_ids", []))


def test_old_runtime_falls_back_to_full_reset():
    runtime = FakeRuntime(truncate=False); session = make_session(runtime, reuse_kv=True)
    session.turn("one"); runtime.calls.clear(); session.turn("different")
    assert runtime.calls[0][0] == "reset"


def test_clear_resets_rng_history_and_cached_ids():
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=True)
    session.turn("one"); session.clear()
    assert session.messages[0]["role"] == "system"
    assert getattr(session, "cached_token_ids", []) == []
    assert runtime.position == 0


@pytest.mark.parametrize("bad", ["", "   ", -1, 999, 1.5])
def test_invalid_user_is_rejected_transactionally(bad):
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=True)
    before = list(runtime.calls)
    with pytest.raises((ChatError, ValueError, TypeError)):
        session.turn(bad)
    assert runtime.calls == before


def test_eval_failure_invalidates_cache_and_history_state():
    runtime = FakeRuntime(fail=True); session = make_session(runtime, reuse_kv=True)
    with pytest.raises(RuntimeError): session.turn("one")
    assert runtime.position == 0
    assert getattr(session, "cached_token_ids", []) == []


def test_max_new_zero_does_not_sample_or_commit_generated_tokens():
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=True, max_new_tokens=0)
    session.turn("one")
    assert runtime.position > 0
    assert not any(x == 6 for x in getattr(session, "cached_token_ids", []))


def test_eos_is_sampled_but_not_committed_or_decoded():
    runtime = FakeRuntime(); sampled = []
    def eos(logits): sampled.append(2); return 2
    session = make_session(runtime, reuse_kv=True, max_new_tokens=4, sampler=eos)
    result = session.turn("one")
    assert sampled == [2] and result.generated_tokens == 0 and result.text == ""
    assert 2 not in getattr(session, "cached_token_ids", [])[len(getattr(session, "cached_token_ids", [])) - 1:]


def test_normal_generation_does_not_cache_last_unevaluated_token():
    runtime = FakeRuntime(); values = iter([6, 7, 2])
    session = make_session(runtime, reuse_kv=True, max_new_tokens=4, sampler=lambda _: next(values))
    result = session.turn("one")
    assert result.generated_tokens == 2
    assert getattr(session, "cached_token_ids", [])[-1:] == [7]


def test_token_id_validation_happens_before_runtime_mutation():
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=True)
    before = (runtime.position, list(runtime.calls), runtime.cache_identity, list(session.messages))
    for ids in ([-1], [999], [1.5]):
        session.tokenizer.apply_chat_template = lambda _messages, ids=ids, **kwargs: ids
        with pytest.raises((ValueError, ChatError, TypeError)):
            session.turn("one")
        assert (runtime.position, runtime.calls, runtime.cache_identity[0], session.messages) == (before[0], before[1], before[2][0], before[3])


def _fixed_ids(session, table):
    session._ids = lambda messages: list(table[messages[-1]["content"]])


@pytest.mark.parametrize("canonical", [[1, 9, 5], [1, 9], [1, 8, 5]])
def test_canonical_full_short_and_divergent_prefixes(canonical):
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=True, max_new_tokens=0)
    _fixed_ids(session, {"one": [1, 9, 5], "two": canonical})
    session.turn("one"); runtime.calls.clear(); cached = list(getattr(session, "cached_token_ids", [])); session.turn("two")
    lcp = 0
    while lcp < len(cached) and lcp < len(canonical) and cached[lcp] == canonical[lcp]: lcp += 1
    expected = canonical[lcp:] or canonical[-1:]
    assert eval_batches(runtime)[-1] == expected
    assert list(getattr(session, "cached_token_ids", [])) == canonical


def test_external_reset_and_replacement_force_safe_restart():
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=True, max_new_tokens=0)
    session.turn("one"); runtime.reset(); runtime.eval([33]); runtime.calls.clear(); session.turn("two")
    assert eval_batches(runtime)[-1]  # external state cannot be reused
    replacement = FakeRuntime(); session.runtime = replacement; session.turn("one")
    assert replacement.calls[0][0] == "reset"


def test_reference_and_candidate_sessions_match_real_sampler_three_turns():
    left, right = FakeRuntime(), FakeRuntime()
    a, b = make_session(left, reuse_kv=False, max_new_tokens=1), make_session(right, reuse_kv=True, max_new_tokens=1)
    for prompt in ("one", "two", "three"):
        ra, rb = a.turn(prompt), b.turn(prompt)
        assert (ra.text, ra.generated_tokens) == (rb.text, rb.generated_tokens)


def test_history_eviction_keeps_system_and_does_not_reuse_shifted_suffix():
    runtime = FakeRuntime(); session = make_session(runtime, reuse_kv=True, max_new_tokens=1)
    for prompt in ("one", "two", "three", "one", "two"):
        session.turn(prompt)
    assert session.messages[0]["role"] == "system"
    assert runtime.position == len(getattr(session, "cached_token_ids", []))


def test_clear_restores_seeded_rng_sequence():
    first = FakeRuntime(); session = make_session(first, reuse_kv=True, max_new_tokens=1)
    session.turn("one"); session.clear(); a = session.turn("one").text
    second = FakeRuntime(); other = make_session(second, reuse_kv=True, max_new_tokens=1)
    assert a == other.turn("one").text
