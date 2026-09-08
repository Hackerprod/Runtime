import unittest
from pathlib import Path
from collections import UserDict
from unittest.mock import patch
import ctypes
import numpy as np
from native_cpu.tools.chat import (
    ChatError,
    ChatSession,
    LocalTokenizer,
    _resolve_profile,
    _response_json,
    _template_environment,
    sample_logits,
)
from native_cpu.tools.native import NativeRuntime, NativeError

class FakeTokenizer:
    eos_token_id = 2
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        out=[]
        for m in messages: out += [10 if m['role']=='system' else 11 if m['role']=='user' else 12] + list(range(len(m['content'])))
        return out + [13]
    def decode(self, ids, skip_special_tokens=True): return ''.join(chr(65+i%26) for i in ids)

class FakeRuntime:
    max_context = 12
    def __init__(self, sequence=(3,2)): self.sequence=list(sequence); self.calls=[]; self._pos=0; self._response_index=0
    def reset(self): self._pos=0; self._response_index=0; self.calls.append(('reset',))
    @property
    def position(self): return self._pos
    def eval(self, ids):
        ids = np.asarray(ids).reshape(-1)
        self.calls.append(tuple(int(x) for x in ids)); x=np.full(32,-100.,dtype=np.float32); tok=self.sequence[min(self._response_index,len(self.sequence)-1)]; x[tok]=100.; self._response_index += 1; self._pos += len(ids); return x
    @property
    def backend(self): return 'fake'

class ChatTests(unittest.TestCase):
    def test_incompatible_stats_abi_is_not_swallowed_by_diagnostics_snapshot(self):
        class IncompatibleRuntime(FakeRuntime):
            stats_abi_incompatible = True
            @property
            def stats(self):
                raise AssertionError("mm_get_stats must not be called")

        session = ChatSession(IncompatibleRuntime(), FakeTokenizer(), context_limit=12, max_new_tokens=1)
        with self.assertRaisesRegex(NativeError, "incompatible statistics ABI"):
            session.turn("hello")

    def test_profile_phase_deltas_subtract_each_participant(self):
        before = {"lm_head_calls": 3, "participant_compute_ns": [100, 90, 0]}
        after = {"lm_head_calls": 5, "participant_compute_ns": [140, 120, 0]}
        self.assertEqual(ChatSession._stats_delta(after, before),
                         {"lm_head_calls": 2, "participant_compute_ns": [40, 30, 0]})

    def test_head_counts_use_phase_deltas_even_without_statistics_reset(self):
        class CumulativeRuntime(FakeRuntime):
            heads = 100
            @property
            def stats(self): return {"lm_head_calls": self.heads}
            def eval(self, ids):
                result = super().eval(ids)
                self.heads += len(ids)
                return result
        result = ChatSession(CumulativeRuntime(), FakeTokenizer(), context_limit=12,
                             max_new_tokens=2).turn("a")
        self.assertEqual(result.lm_head_calls, {"prefill": result.context_tokens, "decode": 1})

    def test_zero_and_single_generated_token_are_not_decode_evaluations(self):
        for maximum in (0, 1):
            rt = FakeRuntime(sequence=(3, 2))
            session = ChatSession(rt, FakeTokenizer(), context_limit=12, max_new_tokens=maximum)
            result = session.turn("a")
            self.assertEqual(result.generated_tokens, maximum)
            self.assertEqual(result.decode_tokens_evaluated, 0)
            self.assertEqual(result.sampled_ids, [] if maximum == 0 else [3])
            self.assertEqual(result.lm_head_calls, None)
            self.assertEqual(rt.position, result.prefill_tokens_evaluated)

    def test_chat_and_parity_profiles_have_explicit_defaults(self):
        from argparse import Namespace

        chat = _resolve_profile(Namespace(profile="chat", temperature=None, top_k=None, top_p=None, system=None))
        self.assertEqual((chat.temperature, chat.top_k, chat.top_p, chat.system), (0.9, 50, 0.85, "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"))
        parity = _resolve_profile(Namespace(profile="parity", temperature=None, top_k=None, top_p=None, system=None))
        self.assertEqual((parity.temperature, parity.top_k, parity.top_p, parity.system), (0.0, 0, 1.0, None))

    def test_selective_logits_flag_defaults_off(self):
        from argparse import Namespace
        self.assertFalse(getattr(Namespace(), "selective_logits", False))

    def test_response_json_records_effective_sampling_policy(self):
        from argparse import Namespace

        result = ChatSession(FakeRuntime(sequence=(2,)), FakeTokenizer(), context_limit=12, max_new_tokens=1).turn("x")
        args = Namespace(profile="chat", system="system", temperature=0.9, top_k=50, top_p=0.85, seed=0)
        payload = _response_json(result, args)
        self.assertEqual({payload[k] for k in ("profile", "system", "temperature", "top_k", "top_p", "seed")}, {"chat", "system", 0.9, 50, 0.85, 0})

    def test_template_uses_immutable_sandbox(self):
        with self.assertRaises(Exception):
            _template_environment().from_string('{{ cycler.__init__.__globals__ }}').render()

    def test_local_tokenizer_matches_upstream_chat_prefix(self):
        tok = LocalTokenizer(Path('checkpoints/minimind-3-hf'))
        ids = tok.apply_chat_template([{'role':'user','content':'Hi'}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        self.assertEqual(ids, [1, 832, 311, 234, 75, 108, 2, 234, 1, 1388, 570, 811, 234, 25, 234, 234, 26, 234, 234])

    def test_ids_accepts_flat_mapping_and_rejects_multiple_batches(self):
        class MappingTokenizer(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return UserDict(input_ids=[1, 2, 3])
        rt=FakeRuntime(); s=ChatSession(rt,MappingTokenizer(),context_limit=12,max_new_tokens=1)
        self.assertEqual(s._ids([{'role':'user','content':'x'}]), [1,2,3])
        class MultiTokenizer(MappingTokenizer):
            def apply_chat_template(self, messages, **kwargs): return UserDict(input_ids=[[1],[2]])
        with self.assertRaises(ChatError): ChatSession(rt,MultiTokenizer(),context_limit=12)._ids([{'role':'user','content':'x'}])

    def test_sampling_top_k_and_top_p_bound_candidates(self):
        rng = np.random.default_rng(4); logits = np.array([0., 1., 2., 3.], dtype=np.float32)
        self.assertIn(sample_logits(logits, rng, temperature=1., top_k=2, top_p=1.), (2, 3))
        self.assertEqual(sample_logits(logits, np.random.default_rng(1), temperature=1., top_k=0, top_p=.5), 3)

    def test_clear_resets_sampling_seed(self):
        rt=FakeRuntime(sequence=(3,2)); s=ChatSession(rt,FakeTokenizer(),context_limit=12,temperature=1.,seed=7)
        first=s._sample(np.array([.1,.2,.3,.4],dtype=np.float32)); s.clear(); second=s._sample(np.array([.1,.2,.3,.4],dtype=np.float32))
        self.assertEqual(first, second)

    def test_generation_excludes_eos_and_commits_assistant(self):
        rt=FakeRuntime(); s=ChatSession(rt,FakeTokenizer(),context_limit=12,max_new_tokens=4)
        r=s.turn('hi'); self.assertEqual(r.generated_tokens,1); self.assertEqual(r.text,'D'); self.assertEqual(s.messages[-1]['role'],'assistant'); self.assertGreaterEqual(r.ttft_seconds,0); self.assertGreaterEqual(r.decode_seconds,0)
    def test_immediate_eos_keeps_history_valid_for_next_turn(self):
        rt=FakeRuntime(sequence=(2,3,2)); s=ChatSession(rt,FakeTokenizer(),context_limit=12,max_new_tokens=1)
        s.turn('first'); self.assertEqual(s.messages[-1]['role'], 'assistant')
        s.turn('second'); self.assertEqual(s.messages[-2]['role'], 'user')

    def test_rejects_reserved_tokens_transactionally(self):
        rt=FakeRuntime(); s=ChatSession(rt,FakeTokenizer(),context_limit=12)
        with self.assertRaises(ChatError): s.turn('<|im_end|>')
        self.assertEqual(s.messages,[]); self.assertEqual(rt.calls,[])
    def test_evicts_old_complete_turns(self):
        rt=FakeRuntime(sequence=(3,2)); s=ChatSession(rt,FakeTokenizer(),context_limit=12,max_new_tokens=1)
        s.turn('a'); s.turn('b'); s.turn('c'); before=len(s.messages); s.turn('d')
        self.assertEqual(s.messages[0]['role'], 'user')
        self.assertEqual(len(s.messages), 6); self.assertEqual(s.messages[0]['content'], 'b')
    def test_oversized_current_turn_errors(self):
        rt=FakeRuntime(); s=ChatSession(rt,FakeTokenizer(),context_limit=4,max_new_tokens=1)
        with self.assertRaises(ChatError): s.turn('this is too long')
    def test_clear_preserves_system(self):
        rt=FakeRuntime(); s=ChatSession(rt,FakeTokenizer(),context_limit=12,system='be concise')
        s.clear(); self.assertEqual(s.messages,[{'role':'system','content':'be concise'}])

if __name__ == '__main__': unittest.main()


class _FakeFn:
    def __init__(self, fn): self.fn = fn; self.argtypes = None; self.restype = None
    def __call__(self, *args): return self.fn(*args)

class _FakeDLL:
    def __init__(self):
        self.mm_load = _FakeFn(lambda *args: 7); self.mm_free = _FakeFn(lambda h: None)
        self.mm_reset = _FakeFn(lambda h: 0); self.mm_vocab_size = _FakeFn(lambda h: 4)
        self.mm_position = _FakeFn(lambda h: 3); self.mm_backend = _FakeFn(lambda h: b'scalar')
        def ev(h, ids, count, out, cap, err, errcap):
            np.ctypeslib.as_array(out, shape=(cap,))[:] = np.arange(cap, dtype=np.float32); return 0
        self.mm_eval = _FakeFn(ev)

class NativeBindingTests(unittest.TestCase):
    def test_wrapper_rejects_noninteger_and_noncontiguous_ids(self):
        fake = _FakeDLL()
        with patch('native_cpu.tools.native.ctypes.CDLL', return_value=fake):
            rt = NativeRuntime('weights.bin', 'runtime.dll', max_context=16)
            with self.assertRaises(ValueError): rt.eval([1.5])
            with self.assertRaises(ValueError): rt.eval([2**40])
            values = np.arange(8, dtype=np.int32)[::2]
            rt.eval(values)
            self.assertTrue(values.flags.c_contiguous is False)
            rt.close()

    def test_exact_abi_wrapper(self):
        fake = _FakeDLL()
        with patch('native_cpu.tools.native.ctypes.CDLL', return_value=fake):
            rt = NativeRuntime('weights.bin', 'runtime.dll', max_context=16, kernel='scalar')
            self.assertEqual(rt.vocab_size, 4); self.assertEqual(rt.position, 3); self.assertEqual(rt.backend, 'scalar')
            np.testing.assert_array_equal(rt.eval([1, 2]), np.arange(4, dtype=np.float32)); rt.reset(); rt.close()
            self.assertIsNone(rt._handle)


class HuggingFaceSamplerCompatibilityTests(unittest.TestCase):
    def _finite_mask(self, values):
        return np.isfinite(np.asarray(values, dtype=np.float64))

    def test_top_k_keeps_ties_at_boundary_like_hf(self):
        import torch
        from transformers import TopKLogitsWarper

        logits = np.asarray([3.0, 2.0, 2.0, 1.0], dtype=np.float32)
        expected = TopKLogitsWarper(top_k=2)(None, torch.tensor(logits).unsqueeze(0))[0].numpy()
        class CaptureRng:
            def choice(self, size, p):
                self.probs = np.asarray(p)
                return int(np.argmax(p))

        rng = CaptureRng()
        sample_logits(logits, rng, temperature=1.0, top_k=2, top_p=1.0)
        self.assertEqual(set(np.flatnonzero(self._finite_mask(expected))), {0, 1, 2})
        self.assertEqual(set(np.flatnonzero(rng.probs > 0)), {0, 1, 2})

    def test_top_p_candidate_mask_matches_hf_random_and_boundary_inputs(self):
        import torch
        from transformers import TopPLogitsWarper

        cases = [
            np.asarray([0.0, 1.0, 2.0, 3.0, -1.0], dtype=np.float32),
            np.asarray([3.0, 2.0, 2.0, 1.0], dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        ]
        for logits in cases:
            for top_p in (0.2, 0.5, 0.85, 1.0):
                expected = TopPLogitsWarper(top_p=top_p)(None, torch.tensor(logits).unsqueeze(0))[0].numpy()
                rng = type("CaptureRng", (), {
                    "choice": lambda self, size, p: (setattr(self, "probs", np.asarray(p)) or int(np.argmax(p)))
                })()
                sample_logits(logits, rng, temperature=1.0, top_p=top_p)
                self.assertEqual(
                    set(np.flatnonzero(rng.probs > 0)),
                    set(np.flatnonzero(self._finite_mask(expected))),
                    (logits, top_p, expected, rng.probs),
                )
