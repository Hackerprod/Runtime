"""PERF-G1 guard formatting and identity checks without running the model."""
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("canonical_tok_s", ROOT / "compare_cpu" / "canonical_tok_s.py")
canonical = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(canonical)


class CanonicalGuardTests(unittest.TestCase):
    def test_prompt_and_policy_identity_are_fixed(self):
        self.assertEqual(canonical.PROMPT, "Could you tell me about programming languages?")
        self.assertEqual(canonical.EXPECTED_DLL_SHA256,
                         "ab464cc1c91aedd3b33bf995fef82657e2316123979e6f1d04d15bb4eace72fd")
        self.assertEqual(canonical.DLL, ROOT / "native_cpu" / "benchmarks" / "cpu-e6" / "baseline" / "minimind_cpu.dll")
        self.assertEqual(canonical.RESULTS, ROOT / "compare_cpu" / "canonical-results" / "cpu-e6")

    def test_rates_use_the_requested_numerators(self):
        metrics = {"decode_seconds": 2.0, "generated_tokens": 8,
                   "decode_evaluated_tokens": 12}
        self.assertEqual(canonical._metric_rate(metrics, "generated_tokens"), 4.0)
        self.assertEqual(canonical._metric_rate(metrics, "decode_evaluated_tokens"), 6.0)

    def test_rates_are_unavailable_without_positive_decode_time(self):
        self.assertIsNone(canonical._metric_rate({"decode_seconds": 0, "generated_tokens": 1},
                                                  "generated_tokens"))
        self.assertIsNone(canonical._metric_rate({"decode_seconds": 1, "generated_tokens": 0},
                                                  "generated_tokens"))

    def test_json_hash_is_deterministic(self):
        self.assertEqual(canonical.sha256_json([1, 2, 3]), canonical.sha256_json([1, 2, 3]))
        self.assertNotEqual(canonical.sha256_json([1, 2, 3]), canonical.sha256_json([1, 2, 4]))


if __name__ == "__main__":
    unittest.main()
