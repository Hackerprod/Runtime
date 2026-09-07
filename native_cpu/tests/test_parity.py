"""Lightweight parity-harness tests; no actual checkpoint or native library is loaded."""
from pathlib import Path
from contextlib import contextmanager
from collections import UserDict
import json
import sys
import unittest
from unittest import mock
from uuid import uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from native_cpu.tools import validate_parity as parity
from native_cpu.tools.validate_parity import greedy_decode, logit_metrics, native_trace, strict_state_dict


@contextmanager
def fixture_directory():
    # Python 3.14's Windows mkdtemp private ACL is incompatible with this sandbox.
    folder = Path(__file__).resolve().parent / (".parity-" + uuid4().hex)
    folder.mkdir()
    try:
        yield folder
    finally:
        for child in folder.iterdir():
            child.unlink()
        folder.rmdir()


class LogitMetricsTests(unittest.TestCase):
    def test_exact_and_shifted_distributions(self):
        logits = np.array([[1.0, 2.0, -1.0], [4.0, 1.0, 0.0]])
        exact = logit_metrics(logits, logits)
        self.assertTrue(exact["allclose"])
        self.assertEqual(exact["rms_error"], 0)
        self.assertEqual(exact["top1_agreement"], 1)
        shifted = logit_metrics(logits, logits + 5)
        self.assertFalse(shifted["allclose"])
        self.assertAlmostEqual(shifted["mean_kl_reference_to_native"], 0)

    def test_reference_relative_tolerance_and_rms(self):
        result = logit_metrics([0.0, 100.0], [1e-5, 100.001])
        self.assertTrue(result["allclose"])
        self.assertAlmostEqual(result["rms_error"], np.sqrt((1e-10 + 1e-6) / 2))
        self.assertFalse(logit_metrics([0.0], [0.00011])["allclose"])

    def test_rejects_broadcasting_empty_and_nonfinite(self):
        for left, right in [([1, 2], [[1, 2]]), ([], []), ([1, 2], [1, np.nan]),
                            ([np.inf], [0])]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                logit_metrics(left, right)

    def test_kl_direction_and_argmax(self):
        result = logit_metrics([4.0, 0.0], [0.0, 4.0])
        self.assertGreater(result["mean_kl_reference_to_native"], 3)
        self.assertEqual(result["top1_matches"], 0)


class GreedyTests(unittest.TestCase):
    def test_eos_does_not_feed_eos_or_continue(self):
        calls = []
        def step(ids):
            calls.append(ids)
            return [0.0, 0.0, 9.0]
        tokens, decisions = greedy_decode([0.0, 9.0, 0.0], step, 8)
        self.assertEqual(tokens, [1, 2])
        self.assertEqual(calls, [[1]])
        self.assertEqual(len(decisions), 2)

    def test_last_decision_does_not_overrun_context(self):
        calls = []
        tokens, _ = greedy_decode([0.0, 9.0, 0.0], lambda ids: calls.append(ids) or [0, 9, 0], 8)
        self.assertEqual(tokens, [1] * 8)
        self.assertEqual(len(calls), 7)

    def test_native_trace_detects_cache_position_error(self):
        class BrokenNative:
            position = 0
            def reset(self): pass
            def eval(self, ids): return np.zeros(3)
        with self.assertRaisesRegex(ValueError, "cache position"):
            native_trace(BrokenNative(), [0], {}, 8, 1e-4, 1e-4)

    def test_greedy_checks_every_native_decision_for_finiteness(self):
        with self.assertRaisesRegex(ValueError, "finite nonempty"):
            greedy_decode([0, 9, 0], lambda ids: [0, float("inf"), 0], 8)


class DownloadManifestTests(unittest.TestCase):
    def test_build_info_is_bound_to_actual_library(self):
        with fixture_directory() as folder:
            path = folder / "build.json"
            data = {"schema_version": 1, "compiler": {"id": "Clang", "version": "test", "path": "test"},
                    "native_threads": 1, "build_type": "Release", "compile_flags": "-O3",
                    "library_sha256": "actual"}
            path.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(parity.verify_build_info(path, {"sha256": "actual"}), data)
            with self.assertRaisesRegex(ValueError, "library hash mismatch"):
                parity.verify_build_info(path, {"sha256": "other"})
            data["native_threads"] = 4
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "native_threads"):
                parity.verify_build_info(path, {"sha256": "actual"})

    def test_export_must_match_actual_bytes_and_official_source(self):
        with fixture_directory() as folder:
            artifact, manifest = folder / "model.bin", folder / "model.manifest.json"
            artifact.write_bytes(b"test-only-artifact")
            reference = {"files": {"model.safetensors": {"sha256": "source"},
                                    "config.json": {"sha256": "config"}},
                         "manifest": {"sha256": "download"}}
            data = {"schema": "MMCPU001", "quantization": "fp32", "input_sha256": "source",
                    "config_sha256": "config", "source_manifest_sha256": "download",
                    "output_sha256": parity.sha256(artifact), "output": {"bytes": artifact.stat().st_size}}
            manifest.write_text(json.dumps(data), encoding="utf-8")
            self.assertTrue(parity.verify_export(artifact, manifest, "fp32", reference)["source_manifest_verified"])
            with self.assertRaisesRegex(ValueError, "quantization"):
                parity.verify_export(artifact, manifest, "q4", reference)
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "output_sha256"):
                parity.verify_export(artifact, manifest, "fp32", reference)

    def test_manifest_requires_all_original_files_and_matching_hashes(self):
        with fixture_directory() as folder:
            names = ("config.json", "model.safetensors", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja")
            files = []
            for name in names:
                path = folder / name
                path.write_text(name, encoding="utf-8")
                files.append({"path": name, "bytes": path.stat().st_size, "sha256": parity.sha256(path)})
            manifest = {"repo_id": "jingyaogong/minimind-3", "revision": parity.CHECKPOINT_REVISION,
                        "files": files}
            (folder / "download-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(parity, "CHECKPOINT_SHA256", parity.sha256(folder / "model.safetensors")):
                self.assertEqual(len(parity.verify_download(folder)["files"]), 5)
                (folder / "chat_template.jinja").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                    parity.verify_download(folder)
                (folder / "chat_template.jinja").unlink()
                with self.assertRaises(FileNotFoundError):
                    parity.verify_download(folder)

    def test_manifest_rejects_path_escape_before_reading(self):
        with fixture_directory() as folder:
            manifest = {"repo_id": "jingyaogong/minimind-3", "revision": parity.CHECKPOINT_REVISION,
                        "files": [{"path": "../outside", "bytes": 0, "sha256": "unused"}]}
            (folder / "download-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                parity.verify_download(folder)


class TokenizerShapeTests(unittest.TestCase):
    class Encoding:
        ids = [1, 4, 5]

    def test_normalizes_one_prompt_without_changing_tokens(self):
        values = ([1, 4, 5], self.Encoding(), [self.Encoding()], [[1, 4, 5]],
                  np.array([[1, 4, 5]], dtype=np.int32), {"input_ids": [1, 4, 5]},
                  UserDict({"input_ids": [1, 4, 5], "attention_mask": [1, 1, 1]}))
        for value in values:
            with self.subTest(value=type(value)):
                self.assertEqual(parity.normalize_prompt_ids(value), [1, 4, 5])

    def test_never_silently_takes_first_batch_or_casts_float_tokens(self):
        for value in ([self.Encoding(), self.Encoding()], [[1, 2], [3, 4]],
                      [1.0, 4.0], [True], [], [1, -1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parity.normalize_prompt_ids(value)


class ReferenceImportTests(unittest.TestCase):
    def test_dynamic_upstream_module_is_registered_before_execution(self):
        name = "_official_minimind_parity_reference"
        previous = sys.modules.get(name)
        try:
            with fixture_directory() as folder, mock.patch.object(sys, "dont_write_bytecode", True):
                source = folder / "reference.py"
                source.write_text("import sys\nregistered = sys.modules[__name__]\n", encoding="utf-8")
                module = parity.import_reference_module(source)
                self.assertIs(module.registered, module)
                self.assertIs(sys.modules[name], module)
                source.write_text("raise RuntimeError('fixture import failed')\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "fixture import failed"):
                    parity.import_reference_module(source)
                self.assertIs(sys.modules[name], module)
        finally:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class StrictStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def setUp(self):
        torch = self.torch
        self.raw = {"model.embed_tokens.weight": torch.ones(3, 2, dtype=torch.float16),
                    "layer.weight": torch.ones(2, 2, dtype=torch.float16)}
        self.expected = {**self.raw, "lm_head.weight": torch.zeros(3, 2)}

    def test_only_verified_tied_alias_is_synthesized(self):
        result, aliases = strict_state_dict(self.raw, self.expected, True)
        self.assertEqual(len(aliases), 1)
        self.assertEqual(set(result), set(self.expected))
        self.assertTrue(all(t.dtype == self.torch.float32 for t in result.values()))
        self.assertNotIn("lm_head.weight", self.raw)

    def test_rejects_unverified_alias_and_missing_non_alias_weight(self):
        with self.assertRaisesRegex(ValueError, "verified tied"):
            strict_state_dict(self.raw, self.expected, False)
        del self.raw["layer.weight"]
        with self.assertRaisesRegex(ValueError, "state keys differ"):
            strict_state_dict(self.raw, self.expected, True)

    def test_rejects_unexpected_shape_nonfinite_and_conflicting_tie(self):
        cases = [dict(self.raw, extra=self.torch.ones(1)),
                 {**self.raw, "layer.weight": self.torch.ones(4, 1)},
                 {**self.raw, "layer.weight": self.torch.full((2, 2), float("nan"))},
                 {**self.raw, "lm_head.weight": self.torch.zeros(3, 2)}]
        for state in cases:
            with self.subTest(keys=list(state)), self.assertRaises(ValueError):
                strict_state_dict(state, self.expected, True)


if __name__ == "__main__":
    unittest.main()
