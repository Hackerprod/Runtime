"""Adversarial session transactions: exercise production code, not fake validation."""
from copy import deepcopy
import pytest
from native_cpu.tests.test_kv_cache import FakeRuntime, make_session


@pytest.mark.parametrize("failure", ["sample", "decode", "truncate", "position", "statistics"])
def test_failure_after_mutation_discards_cache_without_committing_history(failure):
    runtime = FakeRuntime()
    session = make_session(runtime, reuse_kv=True)
    session.turn("one")
    previous = deepcopy(session.messages)
    samples = []
    def fail(*args, **kwargs): raise RuntimeError("injected transaction failure")
    def sample(logits):
        samples.append(6)
        if failure == "sample": fail()
        return 6
    session._sample = sample
    if failure == "decode": session.tokenizer.decode = fail
    if failure == "truncate": runtime.truncate = fail
    if failure == "statistics": runtime.reset_stats = fail
    if failure == "position":
        original = runtime.eval
        def wrong_position(ids):
            result = original(ids)
            runtime._tokens.append(1)
            return result
        runtime.eval = wrong_position
    with pytest.raises(RuntimeError): session.turn("two")
    assert session.messages == previous
    assert session.cached_token_ids == []
    assert not session._cache_trusted
    assert runtime.position == 0
    assert len(samples) == (1 if failure in ("sample", "decode") else 0)


def test_reset_failure_never_marks_unknown_native_state_trusted():
    runtime = FakeRuntime()
    session = make_session(runtime, reuse_kv=True)
    session.turn("one")
    previous = deepcopy(session.messages)
    original_reset = runtime.reset
    def fail_reset(): raise RuntimeError("reset unavailable")
    runtime.reset = fail_reset
    runtime.fail = True
    with pytest.raises(RuntimeError): session.turn("two")
    assert not session._cache_trusted and session.cached_token_ids == []
    assert session.messages == previous
    assert runtime.position > 0  # Do not pretend the failed native reset worked.
    runtime.reset = original_reset
    result = session.turn("two")
    assert result.prefix_tokens_reused == 0
    assert runtime.position == len(session.cached_token_ids)


def test_invalid_new_prompt_preserves_existing_cache_history_and_rng():
    runtime = FakeRuntime()
    session = make_session(runtime, reuse_kv=True)
    session.turn("one")
    before = (list(runtime.calls), runtime.cache_identity, list(session.cached_token_ids),
              deepcopy(session.messages), deepcopy(session.rng.bit_generator.state))
    session.tokenizer.apply_chat_template = lambda *a, **kw: [1, -1, 5]
    with pytest.raises(ValueError): session.turn("two")
    after = (runtime.calls, runtime.cache_identity, session.cached_token_ids,
             session.messages, session.rng.bit_generator.state)
    assert after == before
