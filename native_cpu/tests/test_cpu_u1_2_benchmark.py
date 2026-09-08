"""CPU-U1.2 baseline protocol tests; no model inference is required."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "native_cpu" / "benchmarks" / "cpu-u1.2" / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("cpu_u1_2_benchmark", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CpuU12ProtocolTests(unittest.TestCase):
    def test_protocol_contains_only_the_two_forced_prefixes(self):
        self.assertEqual(MODULE.PAIR_COUNT, 3)
        self.assertEqual(
            MODULE.make_cases(),
            [
                {"key": "cpu-u1.2-forced-256", "workload": "forced", "length": 256},
                {"key": "cpu-u1.2-forced-1792", "workload": "forced", "length": 1792},
            ],
        )

    def test_wrong_baseline_hash_fails_before_output_directory(self):
        wrong = ROOT / "native_cpu" / "validation" / "cpu-u1" / "reference" / "minimind_cpu.dll"
        output = ROOT / "native_cpu" / "benchmarks" / "cpu-u1.2" / ".test-preflight-results"
        self.assertTrue(wrong.is_file())
        self.assertFalse(output.exists())
        with mock.patch.multiple(MODULE, BASELINE_LIBRARY=wrong, EXPECTED_LIBRARY_SHA256="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "refuses baseline DLL"):
                MODULE.main(["--out", str(output), "--repeats", "3"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
