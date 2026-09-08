"""Independent production regressions; fake backends, no actual model or downloads."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from native_cpu.tools import compare_chat as compare
from native_cpu.tools.chat import ChatError, ChatSession, sample_logits


def args_for(options=()):
    return compare.resolve_policy(compare.make_parser().parse_args(list(options)))


class Tokenizer:
    vocab_size, eos_token_id = 8, 2
    def apply_chat_template(self, messages, **kwargs): return [0]
    def decode(self, ids, **kwargs): return " ".join(str(x) for x in ids)


class Runtime:
    vocab_size, max_context, backend = 8, 2048, "fake-native"
    def __init__(self, eos_after=None, random_logits=False):
        self.eos_after, self.random_logits = eos_after, random_logits
        self.position = self.calls = self.closed = 0
    def reset(self): self.position = self.calls = 0
    def eval(self, ids):
        self.position += int(np.asarray(ids).size)
        self.calls += 1
        logits = np.zeros(8, dtype=np.float32)
        if self.random_logits:
            logits[2] = -1000
        else:
            logits[2 if self.eos_after is not None and self.calls > self.eos_after else 1] = 9
        return logits
    def close(self): self.closed += 1


def make_session(args, eos_after=None):
    runtime = compare.CountedRuntime(Runtime(eos_after))
    session = ChatSession(runtime, Tokenizer(), context_limit=args.context_limit,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, seed=args.seed, system=args.system)
    return session, runtime, 0.5


class ParserRegressionTests(unittest.TestCase):
    def test_profiles_and_official_system_survive_encoding_changes(self):
        expected = "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"
        self.assertEqual(compare.CHAT_SYSTEM, expected)
        chat, parity = args_for(), args_for(["--profile", "parity"])
        self.assertEqual((chat.temperature, chat.top_k, chat.top_p, chat.system), (.9, 50, .85, expected))
        self.assertEqual((parity.temperature, parity.top_k, parity.top_p, parity.system), (0, 0, 1, None))
        self.assertEqual((chat.context_limit, chat.max_new_tokens, chat.seed), (2048, 256, 0))
        for path in (chat.model, chat.library, chat.checkpoint_dir): self.assertTrue(path.is_absolute())

    def test_explicit_zero_values_are_not_replaced_by_profile_defaults(self):
        args = args_for(["--temperature", "0", "--top-k", "0", "--top-p", "1", "--seed", "7"])
        self.assertEqual((args.temperature, args.top_k, args.top_p, args.seed), (0, 0, 1, 7))

    def test_invalid_cli_values_fail_before_backend_loading(self):
        cases = [("--profile", "unknown"), ("--context-limit", "1"), ("--max-new-tokens", "-1"),
                 ("--max-new-tokens", "2049"), ("--temperature", "nan"), ("--temperature", "inf"),
                 ("--temperature", "-0.1"), ("--top-k", "-1"), ("--top-k", "1.5"),
                 ("--top-p", "0"), ("--top-p", "1.01"), ("--top-p", "nan"), ("--seed", "-1")]
        for option, value in cases:
            with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()), \
                 mock.patch.object(compare, "build_session") as build:
                with self.assertRaises(SystemExit) as caught: compare.main([option, value])
                self.assertEqual(caught.exception.code, 2)
                build.assert_not_called()


class MetricsRegressionTests(unittest.TestCase):
    def test_eos_length_and_zero_generation_account_for_actual_evals(self):
        for limit, eos_after, generated, evaluated, finish in [(4, None, 4, 3, "length"),
                (4, 2, 2, 2, "eos"), (4, 0, 0, 0, "eos"), (0, None, 0, 0, "length")]:
            with self.subTest(limit=limit, eos_after=eos_after):
                args = args_for(["--profile", "parity", "--max-new-tokens", str(limit)])
                session, runtime, loading = make_session(args, eos_after)
                result = session.turn("hello")
                metrics = compare.response_metrics(args, result, runtime, loading)
                self.assertEqual((metrics["generated_tokens"], metrics["decode_eval_steps"],
                                  metrics["decode_evaluated_tokens"], metrics["finish_reason"]),
                                 (generated, evaluated, evaluated, finish))
                self.assertEqual(metrics["loading_seconds"], .5)
                if not generated:
                    self.assertIsNone(metrics["decode_tokens_per_second"])
                else:
                    self.assertAlmostEqual(metrics["decode_tokens_per_second"], generated / result.decode_seconds)
                if not evaluated:
                    self.assertIsNone(metrics["target_tokens_per_second"])
                else:
                    self.assertAlmostEqual(metrics["target_tokens_per_second"], evaluated / result.decode_seconds)

    def test_metrics_disclose_resolved_policy_and_backend(self):
        args = args_for(["--backend", "original", "--profile", "chat", "--max-new-tokens", "1"])
        session, runtime, loading = make_session(args)
        metrics = compare.response_metrics(args, session.turn("hello"), runtime, loading)
        for name in ("profile", "temperature", "top_k", "top_p", "seed"):
            self.assertEqual(metrics[name], getattr(args, name))
        self.assertEqual(metrics["backend"], "original")
        self.assertEqual(metrics["runtime_backend"], "fake-native")
        self.assertEqual(metrics["dtype"], "float32")
        self.assertEqual(metrics["threads"], 1)
        self.assertIs(metrics["flash_attn"], False)
        json.dumps(metrics, allow_nan=False)


class InteractionRegressionTests(unittest.TestCase):
    def invoke(self, options, inputs=None):
        stdout, stderr, created = io.StringIO(), io.StringIO(), []
        def build(args):
            self.assertIn("Loading", stderr.getvalue())
            item = make_session(args)
            created.append(item)
            return item
        with mock.patch.object(compare, "build_session", side_effect=build), redirect_stdout(stdout), redirect_stderr(stderr):
            if inputs is None: code = compare.main(options)
            else:
                with mock.patch("builtins.input", side_effect=inputs): code = compare.main(options)
        return code, stdout.getvalue(), stderr.getvalue(), created

    def test_unicode_oneshot_json_and_banner_are_separate(self):
        prompt = '¿Qué significa "hola"? 世界'
        code, out, err, created = self.invoke(["--backend", "original", "--profile", "parity",
            "--prompt", prompt, "--max-new-tokens", "1", "--metrics-json"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["backend"], "original")
        self.assertIn("flash_attn=False", err)
        self.assertEqual(created[0][0].messages[0]["content"], prompt)
        self.assertEqual(created[0][1].closed, 1)

    def test_clear_rejected_turn_then_eof_preserves_session_contract(self):
        code, out, err, created = self.invoke(["--profile", "parity", "--max-new-tokens", "1", "--metrics-json"],
                                            ["hello", "/clear", "", "new prompt", EOFError()])
        self.assertEqual(code, 0)
        self.assertEqual(len(out.splitlines()), 2)
        self.assertIn("cleared", err.lower())
        self.assertIn("empty", err.lower())
        self.assertEqual(created[0][0].messages[0]["content"], "new prompt")
        self.assertEqual(created[0][1].closed, 1)

    def test_exit_quit_and_interrupt_close_exactly_once(self):
        for item, expected in [("/exit", 0), ("/quit", 0), (KeyboardInterrupt(), 130)]:
            with self.subTest(item=item):
                code, _, _, created = self.invoke(["--profile", "parity"], [item])
                self.assertEqual(code, expected)
                self.assertEqual(created[0][1].closed, 1)

    def test_clear_restores_rng_as_well_as_system_history(self):
        session = ChatSession(Runtime(random_logits=True), Tokenizer(), context_limit=128, max_new_tokens=12,
                              temperature=.9, top_k=0, top_p=1, seed=71, system="retained system")
        initial = deepcopy(session.rng.bit_generator.state)
        first = session.turn("hello").text
        self.assertNotEqual(session.rng.bit_generator.state, initial)
        session.clear()
        self.assertEqual(session.rng.bit_generator.state, initial)
        self.assertEqual(session.messages, [{"role": "system", "content": "retained system"}])
        self.assertEqual(session.turn("hello").text, first)

    def test_rejected_turn_does_not_consume_rng_or_history(self):
        session = ChatSession(Runtime(random_logits=True), Tokenizer(), context_limit=128, max_new_tokens=4,
                              temperature=.9, top_k=0, top_p=1, seed=9)
        session.turn("valid")
        state, messages = deepcopy(session.rng.bit_generator.state), deepcopy(session.messages)
        for invalid in ("", "<|im_start|>"):
            with self.assertRaises(ChatError): session.turn(invalid)
            self.assertEqual(session.rng.bit_generator.state, state)
            self.assertEqual(session.messages, messages)


class FactoryRegressionTests(unittest.TestCase):
    def test_native_factory_has_no_framework_import_or_original_fallback(self):
        from native_cpu.tools import validate_parity
        args = args_for(["--profile", "parity"])
        with mock.patch.dict(sys.modules, {"torch": None, "transformers": None, "tensorflow": None}), \
             mock.patch.object(compare, "_require_file"), mock.patch.object(compare, "LocalTokenizer", return_value=Tokenizer()), \
             mock.patch.object(validate_parity, "verify_download", return_value={}), \
             mock.patch.object(validate_parity, "verify_export") as verify, \
             mock.patch.object(compare, "NativeRuntime", return_value=Runtime()), \
             mock.patch.object(compare, "OriginalRuntime", side_effect=AssertionError("native used original")):
            _, runtime, _ = compare.build_session(args)
            self.assertEqual(verify.call_args.args[2], "fp32")
            runtime.close()

    def test_original_factory_never_loads_native_library(self):
        args = args_for(["--backend", "original", "--profile", "parity"])
        with mock.patch.object(compare, "_require_file"), mock.patch.object(compare, "LocalTokenizer", return_value=Tokenizer()), \
             mock.patch.object(compare, "OriginalRuntime", return_value=Runtime()) as original, \
             mock.patch.object(compare, "NativeRuntime", side_effect=AssertionError("original used native")):
            _, runtime, _ = compare.build_session(args)
            original.assert_called_once()
            runtime.close()

    def test_factory_closes_loaded_runtime_if_session_initialization_fails(self):
        args, runtime = args_for(["--backend", "original"]), Runtime()
        with mock.patch.object(compare, "_require_file"), mock.patch.object(compare, "LocalTokenizer", return_value=Tokenizer()), \
             mock.patch.object(compare, "OriginalRuntime", return_value=runtime), \
             mock.patch.object(compare, "ChatSession", side_effect=ValueError("session failure")):
            with self.assertRaisesRegex(ValueError, "session failure"): compare.build_session(args)
        self.assertEqual(runtime.closed, 1)

    def test_actionable_missing_file_error_and_cli_failure(self):
        with self.assertRaisesRegex(FileNotFoundError, "no download or build"):
            compare._require_file(Path(__file__).with_name("definitely-not-a-model.bin"), "model", "--model")
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(compare, "build_session", side_effect=FileNotFoundError("missing prepared model")), \
             redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(compare.main(["--prompt", "hello", "--metrics-json"]), 1)
        self.assertEqual(out.getvalue(), "")
        self.assertIn("missing prepared model", err.getvalue())


class SamplerRegressionTests(unittest.TestCase):
    class Capture:
        def choice(self, size, p):
            self.probabilities = np.asarray(p).copy()
            return int(np.argmax(p))

    def test_combined_random_policy_matches_hf_masks_and_probability_tolerance(self):
        import torch
        from transformers.generation.logits_process import (
            TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
        )
        torch.set_num_threads(1)
        rng = np.random.default_rng(319)
        for size in (8, 64, 6400):
            for top_k in (0, 1, 3, 50):
                for top_p in (.01, .5, .85, 1.):
                    with self.subTest(size=size, top_k=top_k, top_p=top_p):
                        logits = rng.normal(size=size).astype(np.float32)
                        original = logits.copy()
                        capture = self.Capture()
                        sample_logits(logits, capture, temperature=.9, top_k=top_k, top_p=top_p)
                        scores = TemperatureLogitsWarper(.9)(None, torch.from_numpy(logits)[None])
                        if top_k:
                            scores = TopKLogitsWarper(top_k)(None, scores)
                        scores = TopPLogitsWarper(top_p)(None, scores)
                        expected = torch.softmax(scores, dim=-1)[0].numpy()
                        np.testing.assert_array_equal(capture.probabilities > 0, expected > 0)
                        np.testing.assert_allclose(capture.probabilities, expected, atol=1e-7, rtol=1e-6)
                        np.testing.assert_array_equal(logits, original)

    def test_full_vocab_cutoff_is_normalized_and_shared_by_both_backends(self):
        # FP64 cumulative rounding can retain 3201 equal scores here, versus
        # 3200 with HF FP32. The policy is shared; exact HF cutoff/RNG parity
        # is deliberately not promised at floating-point boundaries or ties.
        capture = self.Capture()
        sample_logits(np.zeros(6400, dtype=np.float32), capture, temperature=1., top_p=.5)
        self.assertTrue(np.isfinite(capture.probabilities).all())
        self.assertTrue((capture.probabilities >= 0).all())
        self.assertAlmostEqual(float(capture.probabilities.sum()), 1.)
        self.assertTrue((capture.probabilities > 0).any())

        class FullTokenizer(Tokenizer):
            vocab_size = 6400
        class FullRuntime(Runtime):
            vocab_size = 6400
            def eval(self, ids):
                self.position += int(np.asarray(ids).size)
                return np.zeros(6400, dtype=np.float32)

        sessions = []
        for backend in ("fake-native", "fake-original"):
            runtime = FullRuntime()
            runtime.backend = backend
            sessions.append(ChatSession(runtime, FullTokenizer(), context_limit=64,
                max_new_tokens=8, temperature=1., top_k=0, top_p=.5, seed=17))
        first, second = (session.turn("same prompt") for session in sessions)
        self.assertEqual((first.text, first.generated_tokens), (second.text, second.generated_tokens))
        self.assertNotEqual(first.backend, second.backend)


class OriginalInputRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def make_model(self):
        torch = self.torch
        class Model:
            config = SimpleNamespace(vocab_size=8, max_position_embeddings=16)
            calls = 0
            def __call__(self, ids, **kwargs):
                self.calls += 1
                self.inference_mode = torch.is_inference_mode_enabled()
                self.last_ids = ids.tolist()
                return SimpleNamespace(logits=torch.zeros((1, 1, 8)), past_key_values=(self.calls,))
        return Model()

    def test_original_rejects_invalid_batch_before_cache_or_model_changes(self):
        model = self.make_model()
        with mock.patch.object(compare, "_load_original", return_value=model):
            runtime = compare.OriginalRuntime(Path("unused"), max_context=4)
        runtime.eval([0, 1])
        before = runtime.position, runtime._cache, model.calls
        invalid = ([], [1.0], [True], [-1], [0, 8], [2**40], np.array([2**63], dtype=np.uint64), [0, 1, 2])
        for ids in invalid:
            with self.subTest(ids=ids), self.assertRaises(ValueError): runtime.eval(ids)
            self.assertEqual((runtime.position, runtime._cache, model.calls), before)
        self.assertTrue(model.inference_mode)
        runtime.reset()
        runtime.eval(np.arange(8, dtype=np.int32)[::2])
        self.assertEqual(model.last_ids, [[0, 2, 4, 6]])
        self.assertEqual(runtime.position, 4)
        runtime.close()
        with self.assertRaisesRegex(RuntimeError, "closed"): runtime.eval([0])

    def test_original_context_is_bounded_by_model_capacity(self):
        with self.assertRaises(ValueError): compare.OriginalRuntime(Path("unused"), max_context=0)
        with mock.patch.object(compare, "_load_original", return_value=self.make_model()):
            with self.assertRaisesRegex(ValueError, "capacity"):
                compare.OriginalRuntime(Path("unused"), max_context=17)


if __name__ == "__main__":
    unittest.main()
