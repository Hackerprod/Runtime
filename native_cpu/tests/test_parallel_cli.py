import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from native_cpu.tools import compare_chat
from native_cpu.tools.native import NativeError, NativeRuntime


class _Fn:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, *args):
        return self.callback(*args)


class _OldLibrary:
    def __init__(self):
        self.freed = 0
        self.mm_load = _Fn(lambda *_: 1)
        self.mm_free = _Fn(lambda *_: setattr(self, "freed", self.freed + 1))
        self.mm_reset = _Fn(lambda *_: 0)
        self.mm_vocab_size = _Fn(lambda *_: 6400)
        self.mm_position = _Fn(lambda *_: 0)
        self.mm_backend = _Fn(lambda *_: b"scalar")
        self.mm_eval = _Fn(lambda *_: 0)


class _ParallelLibrary(_OldLibrary):
    def __init__(self):
        super().__init__()
        self.configured = None
        self.mm_configure_threads = _Fn(self._configure)
        self.mm_thread_count = _Fn(lambda *_: 2)
        self.mm_thread_cpu = _Fn(lambda _handle, index: (3, 5)[int(index)])
        self.mm_thread_weight = _Fn(lambda _handle, index: (2, 1)[int(index)])
    def _configure(self, _handle, threads, cpus, weights, *_):
        self.configured = (int(threads), [int(cpus[i]) for i in range(2)], [int(weights[i]) for i in range(2)])
        return 0


class ParallelValidationTests(unittest.TestCase):
    def test_native_argument_validation_happens_before_loading(self):
        with patch("native_cpu.tools.native.ctypes.CDLL") as cdll:
            for kwargs in (
                {"threads": 0}, {"threads": 65}, {"threads": 2, "cpus": [0]},
                {"threads": 2, "cpus": [0, 0]}, {"threads": 2, "cpus": [-1, 2]},
                {"threads": 2, "row_weights": [1, 0]},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    NativeRuntime("model", "library", **kwargs)
            cdll.assert_not_called()

    def test_parallel_configuration_and_getters(self):
        library = _ParallelLibrary()
        with patch("native_cpu.tools.native.ctypes.CDLL", return_value=library):
            runtime = NativeRuntime("model", "library", threads=2, cpus=[3, 5], row_weights=[2, 1])
        try:
            self.assertEqual(library.configured, (2, [3, 5], [2, 1]))
            self.assertEqual(runtime.threads, 2)
            self.assertEqual(runtime.cpus, [3, 5])
            self.assertEqual(runtime.row_weights, [2, 1])
        finally:
            runtime.close()

    def test_old_library_falls_back_for_defaults_and_rejects_explicit_parallelism(self):
        library = _OldLibrary()
        with patch("native_cpu.tools.native.ctypes.CDLL", return_value=library):
            runtime = NativeRuntime("model", "library")
        self.assertEqual((runtime.threads, runtime.cpus, runtime.row_weights), (1, None, [1]))
        runtime.close()
        with patch("native_cpu.tools.native.ctypes.CDLL", return_value=library):
            with self.assertRaises(NativeError):
                NativeRuntime("model", "library", threads=2)
        self.assertEqual(library.freed, 2)


class ParallelCliTests(unittest.TestCase):
    def test_parser_exposes_affinity_and_row_shard_options(self):
        parser = compare_chat.make_parser()
        args = compare_chat.resolve_policy(parser.parse_args([
            "--backend", "native", "--threads", "3", "--cpus", "4,8,12", "--row-weights", "2,1,3"
        ]))
        args.cpus = compare_chat._parse_uint_csv(args.cpus, "--cpus")
        args.row_weights = compare_chat._parse_uint_csv(args.row_weights, "--row-weights")
        self.assertEqual((args.threads, args.cpus, args.row_weights), (3, [4, 8, 12], [2, 1, 3]))

    def test_original_rejects_nondefault_native_options(self):
        args = SimpleNamespace(
            backend="original", tokenizer=Path("tokenizer"), checkpoint_dir=Path("checkpoint"),
            threads=2, cpus=None, row_weights=None,
        )
        with patch("native_cpu.tools.compare_chat._require_file"), patch("native_cpu.tools.compare_chat.LocalTokenizer"):
            with self.assertRaisesRegex(ValueError, "only by the native backend"):
                compare_chat.build_session(args)


if __name__ == "__main__":
    unittest.main()
