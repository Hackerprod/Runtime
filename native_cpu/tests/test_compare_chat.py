import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from native_cpu.tools import compare_chat as compare
from native_cpu.tools.chat import ChatError


class FakeRuntime:
    max_context = 32
    vocab_size = 8
    backend = "fake-runtime"

    def __init__(self):
        self.reset_calls = 0
        self.eval_calls = []
        self.closed = False
        self._position = 0

    def reset(self):
        self.reset_calls += 1
        self._position = 0

    @property
    def position(self):
        return self._position

    def eval(self, ids):
        self.eval_calls.append(np.asarray(ids).copy())
        self._position += int(np.asarray(ids).size)
        values = np.full(self.vocab_size, -10.0, dtype=np.float32)
        values[3] = 10.0
        return values

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, result=None, error_once=False):
        self.result = result or SimpleNamespace(
            text="¡Hola!", generated_tokens=2, context_tokens=5,
            ttft_seconds=0.01, decode_seconds=0.02, backend="fake-runtime"
        )
        self.error_once = error_once
        self.clear_calls = 0
        self.turns = []

    def clear(self):
        self.clear_calls += 1

    def turn(self, prompt):
        self.turns.append(prompt)
        if self.error_once:
            self.error_once = False
            raise ChatError("reserved control token")
        return self.result


def parser_args(**overrides):
    values = dict(backend="native", profile="chat", system=compare.CHAT_SYSTEM,
                  context_limit=32, max_new_tokens=4, temperature=0.9,
                  top_k=50, top_p=0.85, seed=0)
    values.update(overrides)
    return SimpleNamespace(**values)


class ComparePolicyTests(unittest.TestCase):
    def test_optimization_switches_are_not_launcher_options(self):
        parser = compare.make_parser()
        for flag in ("--selective-logits", "--reuse-kv", "--v-blocked-attention",
                     "--ffn-row4", "--ffn-f16-storage", "--gqa-k-shared", "--gqa-v-shared"):
            with self.assertRaises(SystemExit):
                parser.parse_args([flag])

    def test_chat_profile_defaults(self):
        args = compare.resolve_policy(compare.make_parser().parse_args([]))
        self.assertEqual(args.temperature, 0.9)
        self.assertEqual(args.top_k, 50)
        self.assertEqual(args.top_p, 0.85)
        self.assertEqual(args.system, compare.CHAT_SYSTEM)

    def test_parity_profile_preserves_greedy_defaults(self):
        args = compare.resolve_policy(compare.make_parser().parse_args(["--profile", "parity"]))
        self.assertEqual((args.temperature, args.top_k, args.top_p, args.system), (0.0, 0, 1.0, None))

    def test_explicit_policy_overrides_profile(self):
        args = compare.resolve_policy(compare.make_parser().parse_args(
            ["--temperature", "0.2", "--top-k", "3", "--top-p", "0.7", "--system", "local"]
        ))
        self.assertEqual((args.temperature, args.top_k, args.top_p, args.system), (0.2, 3, 0.7, "local"))


class CountedRuntimeTests(unittest.TestCase):
    def test_counts_prefill_separately_from_decode(self):
        counted = compare.CountedRuntime(FakeRuntime())
        counted.reset(); counted.begin_turn(); counted.set_phase("prefill")
        counted.eval([1, 2, 3])
        counted.set_phase("decode")
        counted.eval([4])
        self.assertEqual(counted.eval_calls, 2)
        self.assertEqual(counted.decode_eval_steps, 1)
        self.assertEqual(counted.decode_evaluated_tokens, 1)

    def test_reset_only_resets_model_state(self):
        runtime = FakeRuntime()
        counted = compare.CountedRuntime(runtime)
        counted.begin_turn(); counted.set_phase("prefill")
        counted.eval([1])
        counted.reset()
        self.assertEqual(runtime.reset_calls, 1)
        self.assertEqual((counted.eval_calls, counted.decode_eval_steps), (1, 0))

    def test_explicit_phase_counts_tokens_not_calls(self):
        counted = compare.CountedRuntime(FakeRuntime())
        counted.begin_turn(); counted.set_phase("prefill"); counted.eval([1, 2, 3]); counted.eval([4, 5])
        counted.set_phase("decode"); counted.eval([6]); counted.eval([7, 8])
        self.assertEqual(counted.prefill_evaluated_tokens, 5)
        self.assertEqual(counted.decode_evaluated_tokens, 3)
        self.assertEqual(counted.decode_eval_steps, 2)

    def test_eval_requires_explicit_phase(self):
        runtime = FakeRuntime(); counted = compare.CountedRuntime(runtime); counted.begin_turn()
        with self.assertRaises(RuntimeError): counted.eval([1])
        self.assertEqual(runtime.eval_calls, [])

    def test_set_phase_rejects_invalid_without_state_change(self):
        counted = compare.CountedRuntime(FakeRuntime()); counted.begin_turn()
        with self.assertRaises(ValueError): counted.set_phase("unknown")
        self.assertIsNone(counted.phase)

    def test_reset_does_not_clear_counters(self):
        counted = compare.CountedRuntime(FakeRuntime()); counted.begin_turn(); counted.set_phase("decode"); counted.eval([1, 2])
        counted.reset()
        self.assertEqual(counted.decode_evaluated_tokens, 2)


class CompareMetricsTests(unittest.TestCase):
    def test_metrics_include_policy_and_decode_accounting(self):
        args = parser_args()
        result = SimpleNamespace(text="¡Qué tal!", generated_tokens=2, context_tokens=7,
                                 ttft_seconds=0.1, decode_seconds=0.2, backend="scalar")
        counted = SimpleNamespace(decode_eval_steps=2, decode_evaluated_tokens=2)
        metrics = compare.response_metrics(args, result, counted, 0.3)
        self.assertEqual(metrics["text"], "¡Qué tal!")
        self.assertEqual(metrics["profile"], "chat")
        self.assertEqual(metrics["system"], compare.CHAT_SYSTEM)
        self.assertEqual(metrics["decode_evaluated_tokens"], 2)
        self.assertEqual(metrics["finish_reason"], "eos")
        self.assertAlmostEqual(metrics["decode_tokens_per_second"], 10.0)

    def test_print_result_preserves_unicode_json(self):
        metrics = {"text": "¡Hola!", "prefill_seconds": 0.1, "decode_seconds": 0.2,
                   "decode_tokens_per_second": 5.0, "decode_eval_steps": 1, "finish_reason": "eos"}
        output = io.StringIO()
        with patch("sys.stdout", output):
            compare._print_result(metrics, as_json=True)
        self.assertIn("¡Hola!", output.getvalue())


class BuildSessionTests(unittest.TestCase):
    def test_native_build_verifies_export_and_passes_policy(self):
        runtime = FakeRuntime()
        args = parser_args(model=compare.Path("weights.bin"), library=compare.Path("runtime.dll"),
                           checkpoint_dir=compare.Path("checkpoint"), tokenizer=compare.Path("tokenizer"))
        session = object()
        with patch.object(compare, "_require_file"), patch.object(compare, "LocalTokenizer", return_value=object()), \
             patch.object(compare, "NativeRuntime", return_value=runtime) as native, \
             patch("native_cpu.tools.validate_parity.verify_download", return_value=object()) as verify_download, \
             patch("native_cpu.tools.validate_parity.verify_export") as verify_export, \
             patch.object(compare, "ChatSession", return_value=session) as chat_session:
            actual, counted, loading = compare.build_session(args)
        self.assertIs(actual, session)
        self.assertIsInstance(counted, compare.CountedRuntime)
        self.assertGreaterEqual(loading, 0.0)
        native.assert_called_once_with(args.model, args.library, args.context_limit, kernel="auto")
        verify_download.assert_called_once_with(args.checkpoint_dir)
        verify_export.assert_called_once_with(args.model, compare.Path("weights.bin.manifest.json"), "fp32", unittest.mock.ANY)
        chat_session.assert_called_once()

    def test_original_build_requires_local_weights(self):
        runtime = FakeRuntime()
        args = parser_args(backend="original", checkpoint_dir=compare.Path("checkpoint"), tokenizer=compare.Path("tokenizer"))
        with patch.object(compare, "_require_file") as require_file, patch.object(compare, "LocalTokenizer", return_value=object()), \
             patch.object(compare, "OriginalRuntime", return_value=runtime), patch.object(compare, "ChatSession", return_value=object()):
            compare.build_session(args)
        labels = [call.args[1] for call in require_file.call_args_list]
        self.assertIn("official weights", labels)

    def test_failed_session_closes_created_runtime(self):
        runtime = FakeRuntime()
        args = parser_args(model=compare.Path("weights.bin"), library=compare.Path("runtime.dll"),
                           checkpoint_dir=compare.Path("checkpoint"), tokenizer=compare.Path("tokenizer"))
        with patch.object(compare, "_require_file"), patch.object(compare, "LocalTokenizer", return_value=object()), \
             patch.object(compare, "NativeRuntime", return_value=runtime), \
             patch("native_cpu.tools.validate_parity.verify_download", return_value=object()), \
             patch("native_cpu.tools.validate_parity.verify_export"), \
             patch.object(compare, "ChatSession", side_effect=RuntimeError("bad session")):
            with self.assertRaises(RuntimeError):
                compare.build_session(args)
        self.assertTrue(runtime.closed)


class CompareCliTests(unittest.TestCase):
    def test_one_shot_json_output_contains_unicode_and_metrics(self):
        session = FakeSession()
        counted = SimpleNamespace(backend="fake-runtime", decode_eval_steps=1,
                                   decode_evaluated_tokens=2, close=lambda: None)
        with patch.object(compare, "build_session", return_value=(session, counted, 0.01)), \
             patch("sys.stdout", new_callable=io.StringIO) as stdout, \
             patch("sys.stderr", new_callable=io.StringIO):
            rc = compare.main(["--prompt", "hola", "--metrics-json"])
        self.assertEqual(rc, 0)
        self.assertIn("¡Hola!", stdout.getvalue())
        self.assertEqual(session.turns, ["hola"])

    def test_interactive_clear_and_chat_error_continue(self):
        session = FakeSession(error_once=True)
        counted = SimpleNamespace(backend="fake-runtime", decode_eval_steps=0,
                                   decode_evaluated_tokens=0, close=lambda: None)
        with patch.object(compare, "build_session", return_value=(session, counted, 0.01)), \
             patch("builtins.input", side_effect=["/clear", "bad", "good", EOFError]), \
             patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO) as stderr:
            rc = compare.main([])
        self.assertEqual(rc, 0)
        self.assertEqual(session.clear_calls, 1)
        self.assertEqual(session.turns, ["bad", "good"])
        self.assertIn("reserved control token", stderr.getvalue())

    def test_keyboard_interrupt_returns_130(self):
        session = FakeSession()
        counted = SimpleNamespace(backend="fake-runtime", decode_eval_steps=0,
                                   decode_evaluated_tokens=0, close=lambda: None)
        with patch.object(compare, "build_session", return_value=(session, counted, 0.01)), \
             patch("builtins.input", side_effect=KeyboardInterrupt), patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(compare.main([]), 130)

    def test_invalid_sampling_is_rejected_before_loading(self):
        with patch.object(compare, "build_session") as build:
            with self.assertRaises(SystemExit):
                compare.main(["--temperature", "nan"])
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
