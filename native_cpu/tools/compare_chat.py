"""Compare local native FP32 and original upstream FP32 with the same chat policy."""
from __future__ import annotations
import argparse
import json
import operator
from pathlib import Path
import sys
import time
from collections.abc import Mapping
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from native_cpu.tools.chat import ChatError, ChatSession, LocalTokenizer, DEFAULT_CHAT_SYSTEM
from native_cpu.tools.native import NativeRuntime
DEFAULT_CHECKPOINT = ROOT / "checkpoints/minimind-3-hf"
DEFAULT_MODEL = ROOT / "native_cpu/artifacts/minimind-fp32.bin"
DEFAULT_LIBRARY = ROOT / "native_cpu/build/Release/minimind_cpu.dll"

def _load_original(checkpoint_dir):
    from native_cpu.tools.validate_parity import load_reference
    return load_reference(Path(checkpoint_dir).resolve())[0]

class OriginalRuntime:
    """Untouched upstream model with matched CPU FP32/eager inference settings."""
    def __init__(self, checkpoint_dir, max_context=2048):
        max_context = operator.index(max_context)
        if max_context < 1: raise ValueError("context must be positive")
        import torch
        self._torch = torch; torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
        self._model = _load_original(checkpoint_dir)
        if max_context > self._model.config.max_position_embeddings: raise ValueError("context exceeds the original model's position capacity")
        self.max_context = max_context; self._vocab_size = int(self._model.config.vocab_size)
        self._cache, self._position, self._closed = None, 0, False
    @property
    def threads(self): self._check(); return 1
    @property
    def cpus(self): self._check(); return None
    @property
    def row_weights(self): self._check(); return [1]
    def _check(self):
        if self._closed: raise RuntimeError("original runtime is closed")
    @property
    def vocab_size(self): self._check(); return self._vocab_size
    @property
    def position(self): self._check(); return self._position
    @property
    def backend(self): self._check(); return "torch-fp32-eager"
    def reset(self): self._check(); self._cache, self._position = None, 0
    def eval(self, token_ids):
        self._check(); ids = np.asarray(token_ids)
        if ids.dtype.kind not in "iu": raise ValueError("token IDs must be an integer array")
        ids = ids.reshape(-1)
        if not ids.size or np.any(ids < 0) or np.any(ids >= self._vocab_size): raise ValueError("token IDs must be nonempty and within the model vocabulary")
        if ids.size > self.max_context - self._position: raise ValueError("token batch exceeds the remaining context capacity")
        tensor = self._torch.from_numpy(np.ascontiguousarray(ids, dtype=np.int64)).unsqueeze(0)
        with self._torch.inference_mode():
            output = self._model(tensor, past_key_values=self._cache, use_cache=True, logits_to_keep=1)
            logits = output.logits[0, -1].detach().float().cpu().numpy().copy()
        if logits.shape != (self._vocab_size,) or not np.isfinite(logits).all(): raise RuntimeError("original model returned invalid logits")
        self._cache = output.past_key_values; self._position += int(ids.size); return logits
    def close(self): self._cache, self._model, self._closed = None, None, True

class CountedRuntime:
    def __init__(self, runtime):
        self.runtime = runtime; self.eval_calls = self.decode_eval_steps = self.decode_evaluated_tokens = 0
        self.prefill_evaluated_tokens = 0; self.phase = None
    def __getattr__(self, name): return getattr(self.runtime, name)
    def reset(self):
        self.runtime.reset()
    def begin_turn(self):
        self.eval_calls = self.decode_eval_steps = self.decode_evaluated_tokens = 0
        self.prefill_evaluated_tokens = 0; self.phase = None
        fn = getattr(self.runtime, "reset_stats", None)
        if fn is not None: fn()
    reset_stats = begin_turn
    def set_phase(self, phase):
        if phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        self.phase = phase
    def eval(self, ids):
        if self.phase not in {"prefill", "decode"}:
            raise RuntimeError("evaluation phase must be explicitly set to prefill or decode")
        logits = self.runtime.eval(ids)
        count = int(np.asarray(ids).size)
        if self.phase == "decode": self.decode_eval_steps += 1; self.decode_evaluated_tokens += count
        elif self.phase == "prefill": self.prefill_evaluated_tokens += count
        self.eval_calls += 1; return logits
    @property
    def stats(self):
        try:
            value = getattr(self.runtime, "stats", {})
            if callable(value): value = value()
            return dict(value) if isinstance(value, Mapping) else {}
        except (AttributeError, RuntimeError, ValueError):
            return {}

def _require_file(path, label, flag):
    if not Path(path).is_file(): raise FileNotFoundError(f"Missing {label}: {path}. Restore the prepared file or pass {flag}; no download or build is performed automatically.")

def build_session(args):
    started = time.perf_counter(); tokenizer_path = args.tokenizer or args.checkpoint_dir
    for name in ("tokenizer.json", "chat_template.jinja"): _require_file(tokenizer_path / name, f"local {name}", "--tokenizer")
    tokenizer = LocalTokenizer(tokenizer_path); runtime = None
    try:
        threads = getattr(args, "threads", 1)
        cpus = getattr(args, "cpus", None)
        row_weights = getattr(args, "row_weights", None)
        selective_logits = bool(getattr(args, "selective_logits", False))
        diagnostics = bool(getattr(args, "diagnostics", False))
        reuse_kv = bool(getattr(args, "reuse_kv", False))
        if args.backend == "original" and (threads != 1 or cpus is not None or row_weights is not None or selective_logits or diagnostics or reuse_kv):
            if selective_logits: raise ValueError("--selective-logits is supported only by the native backend")
            if diagnostics: raise ValueError("--diagnostics is supported only by the native backend")
            if reuse_kv: raise ValueError("--reuse-kv is supported only by the native backend")
            raise ValueError("thread affinity and row-sharding options are supported only by the native backend")
        if args.backend == "native":
            _require_file(args.model, "native FP32 model", "--model"); _require_file(args.library, "native library", "--library")
            manifest = Path(str(args.model) + ".manifest.json"); _require_file(manifest, "FP32 export manifest", "--model with its matching manifest")
            from native_cpu.tools.validate_parity import verify_download, verify_export
            reference = verify_download(args.checkpoint_dir); verify_export(args.model, manifest, "fp32", reference)
            parallel_kwargs = {}
            if threads != 1 or cpus is not None or row_weights is not None:
                parallel_kwargs = {"threads": threads, "cpus": cpus, "row_weights": row_weights}
            runtime = NativeRuntime(args.model, args.library, args.context_limit, kernel="auto", **parallel_kwargs)
        else:
            _require_file(args.checkpoint_dir / "model.safetensors", "official weights", "--checkpoint-dir")
            runtime = OriginalRuntime(args.checkpoint_dir, args.context_limit)
        configure = getattr(runtime, "configure_profile", None)
        if diagnostics and configure is None: raise ValueError("native runtime does not support diagnostics")
        if configure is not None: configure(diagnostics)
        selective = getattr(runtime, "configure_selective_logits", None)
        if selective_logits and selective is None: raise ValueError("native runtime does not support selective logits")
        if selective is not None: selective(selective_logits)
        counted = CountedRuntime(runtime)
        session = ChatSession(counted, tokenizer, context_limit=args.context_limit, max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, seed=args.seed, system=getattr(args, "system", None),
                              top_k=getattr(args, "top_k", 0), top_p=getattr(args, "top_p", 1.0), reuse_kv=reuse_kv)
        return session, counted, time.perf_counter() - started
    except BaseException:
        if runtime is not None: runtime.close()
        raise

def response_metrics(args, result, counted, loading_seconds):
    threads, cpus, row_weights = _thread_values(counted)
    return {"backend": args.backend, "profile": args.profile, "system": args.system, "runtime_backend": result.backend,
            "dtype": "float32", "threads": threads, "cpus": cpus,
            "row_weights": row_weights, "flash_attn": False if args.backend == "original" else None,
            "context_limit": args.context_limit, "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "top_k": args.top_k, "top_p": args.top_p, "seed": args.seed, "text": result.text,
            "generated_tokens": result.generated_tokens, "context_tokens": result.context_tokens,
            "loading_seconds": loading_seconds, "prefill_seconds": result.ttft_seconds, "decode_seconds": result.decode_seconds,
            "decode_eval_steps": counted.decode_eval_steps, "decode_evaluated_tokens": counted.decode_evaluated_tokens,
            "prompt_tokens_total": getattr(result, "prompt_tokens_total", result.context_tokens),
            "prefix_tokens_reused": getattr(result, "prefix_tokens_reused", 0),
            "prefill_tokens_evaluated": getattr(result, "prefill_tokens_evaluated", result.context_tokens),
            "decode_tokens_evaluated": getattr(result, "decode_tokens_evaluated", counted.decode_evaluated_tokens),
            "lm_head_calls": getattr(result, "lm_head_calls", None),
            "position_before": getattr(result, "position_before", None), "position_after": getattr(result, "position_after", None),
            "cache_action": getattr(result, "cache_action", None),
            "prompt_prepare_seconds": getattr(result, "prompt_prepare_seconds", 0.0),
            "native_generation_seconds": getattr(result, "native_generation_seconds", 0.0),
            "sampling_seconds": getattr(result, "sampling_seconds", 0.0), "total_seconds": getattr(result, "total_seconds", 0.0),
            "sampled_ids": getattr(result, "sampled_ids", None), "diagnostics": getattr(args, "diagnostics", False),
            "native_stats": getattr(counted, "stats", {}),
            "native_phase_stats": getattr(result, "native_phase_stats", None),
            "selective_logits": bool(getattr(args, "selective_logits", False)),
            "reuse_kv": bool(getattr(args, "reuse_kv", False)),
            "decode_tokens_per_second": counted.decode_evaluated_tokens / result.decode_seconds if counted.decode_evaluated_tokens and result.decode_seconds > 0 else None,
            "finish_reason": "eos" if result.generated_tokens < args.max_new_tokens else "length"}


def _thread_banner(counted):
    threads, cpus, row_weights = _thread_values(counted)
    return f"threads={threads}, cpus={cpus}, row_weights={row_weights}"


def _thread_values(counted):
    return (int(getattr(counted, "threads", 1)), getattr(counted, "cpus", None),
            getattr(counted, "row_weights", [1]))

def _print_result(metrics, as_json):
    if as_json: print(json.dumps(metrics, ensure_ascii=False, allow_nan=False), flush=True)
    else:
        print(metrics["text"], flush=True); rate = metrics["decode_tokens_per_second"]; rate_text = f"{rate:.2f}" if rate is not None else "n/a"
        print(f"[prefill={metrics['prefill_seconds']:.3f}s decode={metrics['decode_seconds']:.3f}s decode_tok/s={rate_text} eval_steps={metrics['decode_eval_steps']} finish={metrics['finish_reason']}]", file=sys.stderr, flush=True)

def make_parser():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--backend", choices=("native", "original"), default="native")
    parser.add_argument("--profile", choices=("chat", "parity"), default="chat"); parser.add_argument("--prompt"); parser.add_argument("--metrics-json", action="store_true")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL); parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY); parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT); parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--context-limit", type=int, default=2048); parser.add_argument("--max-new-tokens", type=int, default=256); parser.add_argument("--temperature", type=float); parser.add_argument("--top-k", type=int); parser.add_argument("--top-p", type=float); parser.add_argument("--system"); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1, help="native compute participants including caller (1-64)")
    parser.add_argument("--diagnostics", action="store_true", help="enable native diagnostic profiling (off by default)")
    parser.add_argument("--selective-logits", action="store_true", help="evaluate vocabulary logits only when requested")
    parser.add_argument("--reuse-kv", action="store_true", help="reuse verified KV prefixes between turns")
    parser.add_argument("--cpus", help="comma-separated Windows group-0 logical CPU indices in participant order")
    parser.add_argument("--row-weights", dest="row_weights", help="comma-separated positive native row-shard weights")
    return parser


def _parse_uint_csv(raw, name):
    if raw is None: return None
    parts = raw.split(",")
    if not parts or any(not item.strip() for item in parts): raise ValueError(f"{name} must be a comma-separated integer list")
    try: values = [int(item.strip()) for item in parts]
    except ValueError as exc: raise ValueError(f"{name} must be a comma-separated integer list") from exc
    return values

CHAT_SYSTEM = DEFAULT_CHAT_SYSTEM
def resolve_policy(args):
    defaults = {"temperature": 0.9, "top_k": 50, "top_p": 0.85, "system": CHAT_SYSTEM} if args.profile == "chat" else {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "system": None}
    for name in ("temperature", "top_k", "top_p", "system"):
        if getattr(args, name) is None: setattr(args, name, defaults[name])
    return args

def main(argv=None):
    parser = make_parser(); args = resolve_policy(parser.parse_args(argv))
    try:
        args.cpus = _parse_uint_csv(args.cpus, "--cpus")
        args.row_weights = _parse_uint_csv(args.row_weights, "--row-weights")
    except ValueError as exc: parser.error(str(exc))
    if args.context_limit < 2 or args.max_new_tokens < 0 or args.max_new_tokens > args.context_limit: parser.error("context must be at least 2 and max-new-tokens must be within [0, context]")
    if args.threads < 1 or args.threads > 64: parser.error("--threads must be in [1, 64]")
    if (not np.isfinite(args.temperature) or args.temperature < 0 or args.seed < 0 or args.top_k < 0 or not np.isfinite(args.top_p) or not 0 < args.top_p <= 1): parser.error("temperature/top-p must be finite and valid; top-k and seed must be nonnegative")
    counted = None
    try:
        print(f"Loading {args.backend} CPU FP32 model and verifying local files...", file=sys.stderr, flush=True)
        session, counted, loading = build_session(args)
        reference_note = ", flash_attn=False (matched reference, not eval_llm.py defaults)" if args.backend == "original" else ""
        print(f"backend={args.backend}/{counted.backend}, CPU FP32, {_thread_banner(counted)}, context={args.context_limit}, max_new_tokens={args.max_new_tokens}, profile={args.profile}, temperature={args.temperature:g}, top_k={args.top_k}, top_p={args.top_p:g}, seed={args.seed}, system={args.system!r}{reference_note}\nloading={loading:.3f}s; buffered output (prefill timing is not observed streaming TTFT)", file=sys.stderr, flush=True)
        if args.prompt is not None: _print_result(response_metrics(args, session.turn(args.prompt), counted, loading), args.metrics_json); return 0
        print("Commands: /clear, /exit, /quit", file=sys.stderr)
        while True:
            print("> ", end="", file=sys.stderr, flush=True)
            try: line = input()
            except EOFError: return 0
            command = line.strip()
            if command in ("/exit", "/quit"): return 0
            if command == "/clear": session.clear(); print("Conversation cleared.", file=sys.stderr); continue
            try: _print_result(response_metrics(args, session.turn(line), counted, loading), args.metrics_json)
            except ChatError as exc: print(f"error: {exc}", file=sys.stderr)
    except KeyboardInterrupt: print(file=sys.stderr); return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc: print(f"error: {exc}", file=sys.stderr); return 1
    finally:
        if counted is not None: counted.close()
if __name__ == "__main__": raise SystemExit(main())
