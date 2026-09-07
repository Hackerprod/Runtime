"""ctypes boundary for the MMCPU001 native MiniMind runtime."""
from __future__ import annotations
import ctypes
from pathlib import Path
import os
import operator
import numpy as np

class NativeError(RuntimeError): pass

class NativeRuntime:
    def __init__(self, model_path, library_path=None, max_context=2048, kernel="auto", *, context=None, mode=None,
                 threads=1, cpus=None, row_weights=None):
        if context is not None: max_context = context
        if mode is not None: kernel = mode
        try: max_context = operator.index(max_context)
        except TypeError as exc: raise ValueError("max_context must be an integer") from exc
        if not 1 <= max_context <= 0xffffffff: raise ValueError("max_context must be in [1, 2**32-1]")
        if kernel not in ("scalar", "auto"): raise ValueError("kernel must be scalar or auto")
        threads = _positive_uint(threads, "threads", maximum=64)
        cpus = _normalize_cpus(cpus, threads)
        row_weights = _normalize_weights(row_weights, threads)
        explicit_threads = threads != 1 or cpus is not None or row_weights is not None
        self._lib = ctypes.CDLL(str(library_path) if library_path else self._discover_library())
        self._bind()
        self._handle = None
        self._error_cap = 4096
        self._error = ctypes.create_string_buffer(self._error_cap)
        self._handle = self._lib.mm_load(os.fsencode(str(model_path)), int(max_context), 0 if kernel == "scalar" else 1, self._error, self._error_cap)
        if not self._handle:
            raise NativeError(self._message("mm_load failed"))
        self._model_path = str(model_path)
        self.max_context = int(max_context)
        self._parallel_abi = self._configure_threads is not None
        self._threads = 1
        self._cpus = None
        self._row_weights = [1]
        if self._parallel_abi:
            if self._thread_count is None or self._thread_cpu is None or self._thread_weight is None:
                self._lib.mm_free(self._handle); self._handle = None
                raise NativeError("native runtime exposes incomplete thread configuration ABI")
            if explicit_threads:
                self._configure(threads, cpus, row_weights)
            self._refresh_thread_info(expected=threads if explicit_threads else None)
        elif explicit_threads:
            self._lib.mm_free(self._handle); self._handle = None
            raise NativeError("native runtime does not support explicit thread configuration")

    @staticmethod
    def _discover_library():
        base = Path(__file__).resolve().parents[1]
        roots = [base, base / "build", base / "build" / "Release", base / "build" / "Debug", base.parent / "build"]
        names = ("minimind_cpu.dll", "mm_runtime.dll", "libminimind_cpu.so", "libmm_runtime.so", "libminimind_cpu.dylib")
        for root in roots:
            for name in names:
                p = root / name
                if p.is_file(): return str(p)
        raise FileNotFoundError("native runtime library not found; pass --library")

    def _bind(self):
        lib = self._lib
        lib.mm_load.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]; lib.mm_load.restype = ctypes.c_void_p
        lib.mm_free.argtypes = [ctypes.c_void_p]; lib.mm_free.restype = None
        lib.mm_reset.argtypes = [ctypes.c_void_p]; lib.mm_reset.restype = ctypes.c_int
        lib.mm_vocab_size.argtypes = [ctypes.c_void_p]; lib.mm_vocab_size.restype = ctypes.c_uint32
        lib.mm_position.argtypes = [ctypes.c_void_p]; lib.mm_position.restype = ctypes.c_uint32
        lib.mm_backend.argtypes = [ctypes.c_void_p]; lib.mm_backend.restype = ctypes.c_char_p
        lib.mm_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]; lib.mm_eval.restype = ctypes.c_int
        self._configure_threads = getattr(lib, "mm_configure_threads", None)
        self._thread_count = getattr(lib, "mm_thread_count", None)
        self._thread_cpu = getattr(lib, "mm_thread_cpu", None)
        self._thread_weight = getattr(lib, "mm_thread_weight", None)
        if self._configure_threads is not None:
            self._configure_threads.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                                ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                                ctypes.c_char_p, ctypes.c_size_t]
            self._configure_threads.restype = ctypes.c_int
        if self._thread_count is not None:
            self._thread_count.argtypes = [ctypes.c_void_p]; self._thread_count.restype = ctypes.c_uint32
        if self._thread_cpu is not None:
            self._thread_cpu.argtypes = [ctypes.c_void_p, ctypes.c_uint32]; self._thread_cpu.restype = ctypes.c_int32
        if self._thread_weight is not None:
            self._thread_weight.argtypes = [ctypes.c_void_p, ctypes.c_uint32]; self._thread_weight.restype = ctypes.c_uint32

    def _configure(self, threads, cpus, row_weights):
        cpu_array = None if cpus is None else (ctypes.c_uint32 * len(cpus))(*cpus)
        weight_array = None if row_weights is None else (ctypes.c_uint32 * len(row_weights))(*row_weights)
        self._error[0] = 0
        try:
            rc = self._configure_threads(self._handle, threads, cpu_array, weight_array,
                                         self._error, self._error_cap)
        except Exception as exc:
            self._lib.mm_free(self._handle); self._handle = None
            raise NativeError(f"mm_configure_threads failed: {exc}") from exc
        if int(rc) != 0:
            message = self._message("mm_configure_threads failed")
            self._lib.mm_free(self._handle); self._handle = None
            raise NativeError(message)

    def _refresh_thread_info(self, expected=None):
        count = int(self._thread_count(self._handle))
        if not 1 <= count <= 64 or (expected is not None and count != expected):
            self._lib.mm_free(self._handle); self._handle = None
            raise NativeError(f"native runtime reported invalid thread count: {count}")
        cpus = [int(self._thread_cpu(self._handle, index)) for index in range(count)]
        weights = [int(self._thread_weight(self._handle, index)) for index in range(count)]
        if any(cpu < -1 for cpu in cpus) or any(weight <= 0 for weight in weights):
            self._lib.mm_free(self._handle); self._handle = None
            raise NativeError("native runtime reported invalid thread configuration")
        self._threads, self._cpus, self._row_weights = count, cpus, weights

    def _message(self, fallback):
        raw = bytes(self._error).split(b"\0", 1)[0]
        return raw.decode("utf-8", "replace") or fallback

    def _check(self):
        if self._handle is None: raise NativeError("runtime is closed")

    @property
    def vocab_size(self): self._check(); return int(self._lib.mm_vocab_size(self._handle))
    @property
    def position(self): self._check(); return int(self._lib.mm_position(self._handle))
    @property
    def backend(self):
        self._check(); raw = self._lib.mm_backend(self._handle)
        return (raw or b"unknown").decode("utf-8", "replace")

    @property
    def threads(self): self._check(); return self._threads if not self._parallel_abi else int(self._thread_count(self._handle))
    @property
    def cpus(self):
        self._check()
        if not self._parallel_abi: return None
        return [int(self._thread_cpu(self._handle, index)) for index in range(self.threads)]
    @property
    def row_weights(self):
        self._check()
        if not self._parallel_abi: return [1]
        return [int(self._thread_weight(self._handle, index)) for index in range(self.threads)]

    def reset(self):
        self._check()
        if int(self._lib.mm_reset(self._handle)) != 0: raise NativeError("mm_reset failed")

    def eval(self, token_ids):
        self._check()
        raw = np.asarray(token_ids)
        if raw.dtype.kind not in "iu": raise ValueError("token_ids must be an integer array")
        raw = raw.reshape(-1)
        if raw.size == 0: raise ValueError("token_ids cannot be empty")
        if np.any(raw < 0) or np.any(raw > np.iinfo(np.int32).max): raise ValueError("token IDs must fit non-negative int32")
        ids = np.ascontiguousarray(raw, dtype="<i4")
        out = np.empty(self.vocab_size, dtype="<f4")
        self._error[0] = 0
        rc = self._lib.mm_eval(self._handle, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ids.size, out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out.size, self._error, self._error_cap)
        if int(rc) != 0: raise NativeError(self._message("mm_eval failed"))
        return out

    def close(self):
        if self._handle is not None:
            self._lib.mm_free(self._handle); self._handle = None
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
    def __del__(self):
        try: self.close()
        except Exception: pass


# Alias retained for parity/evaluation callers that use the concise name.
NativeModel = NativeRuntime


def _positive_uint(value, name, *, maximum=0xffffffff):
    try: value = operator.index(value)
    except TypeError as exc: raise ValueError(f"{name} must be an integer") from exc
    if value < 1 or value > maximum: raise ValueError(f"{name} must be in [1, {maximum}]")
    return int(value)


def _normalize_cpus(value, threads):
    if value is None: return None
    try: values = list(value)
    except TypeError as exc: raise ValueError("cpus must be a sequence") from exc
    if len(values) != threads: raise ValueError("cpus length must equal threads")
    normalized = []
    for cpu in values:
        try: cpu = operator.index(cpu)
        except TypeError as exc: raise ValueError("cpus must contain integers") from exc
        if cpu < 0 or cpu > 0xffffffff: raise ValueError("cpus must be non-negative uint32 values")
        normalized.append(int(cpu))
    if len(set(normalized)) != len(normalized): raise ValueError("cpus must not contain duplicates")
    return normalized


def _normalize_weights(value, threads):
    if value is None: return None
    try: values = list(value)
    except TypeError as exc: raise ValueError("row_weights must be a sequence") from exc
    if len(values) != threads: raise ValueError("row_weights length must equal threads")
    normalized = []
    for weight in values:
        try: weight = operator.index(weight)
        except TypeError as exc: raise ValueError("row_weights must contain integers") from exc
        if weight < 1 or weight > 0xffffffff: raise ValueError("row_weights must be positive uint32 values")
        normalized.append(int(weight))
    return normalized
