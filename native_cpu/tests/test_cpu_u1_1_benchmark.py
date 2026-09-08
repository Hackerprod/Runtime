"""CPU-U1.1 benchmark preflight tests; no model or DLL execution is required."""
import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "native_cpu" / "benchmarks" / "cpu-u1.1" / "benchmark.py"
SPEC = importlib.util.spec_from_file_location("cpu_u1_1_benchmark", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CpuU11PreflightTests(unittest.TestCase):
    def test_reference_hash_mismatch_fails_before_output_directory_or_loading(self):
        reference = ROOT / "native_cpu" / "validation" / "cpu-u1" / "reference" / "minimind_cpu.dll"
        output = ROOT / "native_cpu" / "benchmarks" / "cpu-u1.1" / ".test-preflight-results"
        self.assertTrue(reference.is_file())
        self.assertFalse(output.exists())
        with mock.patch.multiple(MODULE, FROZEN_LIBRARY=reference, EXPECTED_REFERENCE_SHA256="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "refuses reference DLL"):
                MODULE.main(["--out", str(output), "--pairs", "3"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
