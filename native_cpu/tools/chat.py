"""Local-tokenizer chat frontend for the MMCPU001 runtime."""
from __future__ import annotations
import argparse, json, os, sys, time, operator
from pathlib import Path
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
from .native import NativeRuntime, NativeError

class ChatError(ValueError): pass
DEFAULT_CHAT_SYSTEM = "你是MiniMind，一个乐于助人、知识渊博的AI助手。请用完整且友好的方式回答用户问题。"

def _template_environment():
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    return ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, autoescape=False)

class LocalTokenizer:
    """Pure tokenizers/Jinja2 adapter for the upstream local files."""
    def __init__(self, path):
        try:
            from tokenizers import Tokenizer
            root = Path(path)
            self._tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
            self.vocab_size = self._tokenizer.get_vocab_size(with_added_tokens=True)
            if self.vocab_size != 6400: raise ValueError("tokenizer vocabulary must contain exactly 6400 tokens")
            template_path = root / "chat_template.jinja"
            if not template_path.is_file(): raise FileNotFoundError(template_path)
            self._template = _template_environment().from_string(template_path.read_text(encoding="utf-8"))
            self.eos_token_id = self._tokenizer.token_to_id("<|im_end|>")
            if self._tokenizer.token_to_id("<|im_start|>") != 1 or self.eos_token_id != 2:
                raise ValueError("tokenizer control-token IDs must be <|im_start|>=1 and <|im_end|>=2")
        except Exception as exc: raise ChatError(f"local tokenizer load failed: {exc}") from exc
    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=True, enable_thinking=False, tools=None):
        if tools: raise ChatError("tools are not supported by the text-only runtime")
        rendered = self._template.render(messages=messages, tools=[], add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
        if not tokenize: return rendered
        return self._tokenizer.encode(rendered, add_special_tokens=False).ids
    def decode(self, ids, skip_special_tokens=True):
        return self._tokenizer.decode([int(x) for x in ids], skip_special_tokens=skip_special_tokens)

def load_local_tokenizer(path): return LocalTokenizer(path)
_RESERVED = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>")
def validate_messages(messages):
    if not isinstance(messages, list): raise ChatError("messages must be a list")
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}: raise ChatError(f"invalid message role at index {i}")
        text = m.get("content")
        if not isinstance(text, str) or (not text.strip() and m.get("role") != "assistant"): raise ChatError(f"message {i} is empty")
        if any(marker in text for marker in _RESERVED): raise ChatError("reserved control token in message")
    if messages and messages[0]["role"] != "system" and any(m["role"] == "system" for m in messages): raise ChatError("system message must be first")
    body = [m["role"] for m in messages if m["role"] != "system"]
    if any(a == b for a, b in zip(body, body[1:])): raise ChatError("messages must alternate user and assistant")
    if body and body[0] != "user": raise ChatError("conversation must start with user")
    if not body or body[-1] != "user": raise ChatError("latest message must be user")

@dataclass
class ChatResponse:
    text: str; generated_tokens: int; context_tokens: int; ttft_seconds: float; decode_seconds: float; backend: str
    # Stage-1 telemetry is additive: callers constructing the historical
    # six-field response remain valid.
    prompt_tokens_total: int = 0
    prefix_tokens_reused: int = 0
    prefill_tokens_evaluated: int = 0
    decode_tokens_evaluated: int = 0
    lm_head_calls: object = None
    position_before: object = None
    position_after: object = None
    cache_action: object = None
    prompt_prepare_seconds: float = 0.0
    prefill_seconds: float = 0.0
    native_generation_seconds: float = 0.0
    sampling_seconds: float = 0.0
    total_seconds: float = 0.0
    sampled_ids: object = None
    metrics: object = None
    native_phase_stats: object = None

def sample_logits(logits, rng, *, temperature=0.0, top_k=0, top_p=1.0):
    # Use a widened working buffer for stable sampling math.  Candidate masks
    # are semantically equivalent to HF warpers; exact cumulative-boundary
    # ties can vary with NumPy versus Torch reduction order.
    values = np.asarray(logits, dtype=np.float64).reshape(-1).copy()
    if not values.size or not np.isfinite(values).all(): raise ChatError("invalid model logits")
    if not np.isfinite(temperature) or temperature < 0: raise ValueError("temperature must be finite and non-negative")
    if int(top_k) < 0: raise ValueError("top_k must be non-negative")
    if not np.isfinite(top_p) or not 0 < top_p <= 1: raise ValueError("top_p must be in (0, 1]")
    if temperature == 0: return int(np.argmax(values))
    values = values / float(temperature)
    if top_k > 0:
        # Match Transformers' TopKLogitsWarper: the kth score is a threshold,
        # so ties at the boundary remain eligible rather than being truncated
        # to an arbitrary count by argsort.
        keep = min(int(top_k), values.size)
        if keep < values.size:
            threshold = np.partition(values, values.size - keep)[values.size - keep]
            values[values < threshold] = -np.inf
    if top_p < 1.0:
        # TopPLogitsWarper sorts ascending and removes low-probability tokens
        # whose cumulative mass is <= (1 - p), retaining at least one token.
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        finite = np.isfinite(sorted_values)
        if not finite.any():
            return int(np.argmax(logits))
        safe = np.where(finite, sorted_values, -np.finfo(np.float64).max)
        safe -= np.max(safe)
        probs = np.exp(safe)
        probs[~finite] = 0.0
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            return int(np.argmax(values))
        cumulative = np.cumsum(probs / total)
        remove = cumulative <= (1.0 - float(top_p))
        remove[-1] = False
        values[order[remove]] = -np.inf
    safe = np.where(np.isfinite(values), values, -np.finfo(np.float64).max); safe -= np.max(safe)
    probs = np.exp(safe); probs[~np.isfinite(values)] = 0.0; total = probs.sum()
    if not np.isfinite(total) or total <= 0: return int(np.argmax(logits))
    return int(rng.choice(values.size, p=probs / total))

class ChatSession:
    def __init__(self, runtime, tokenizer, *, context_limit=256, max_new_tokens=64, temperature=0.0, seed=0, system=None, top_k=0, top_p=1.0, reuse_kv=False):
        self.runtime, self.tokenizer = runtime, tokenizer
        runtime_vocab = getattr(runtime, "vocab_size", None); tokenizer_vocab = getattr(tokenizer, "vocab_size", None)
        if tokenizer_vocab is None and hasattr(tokenizer, "get_vocab_size"): tokenizer_vocab = tokenizer.get_vocab_size()
        if runtime_vocab is not None and tokenizer_vocab is not None and int(runtime_vocab) != int(tokenizer_vocab): raise ValueError(f"runtime/tokenizer vocabulary mismatch: {runtime_vocab} != {tokenizer_vocab}")
        cap = int(context_limit); maxctx = int(getattr(runtime, "max_context", cap))
        if cap < 2 or cap > maxctx: raise ValueError("context_limit must be within runtime capacity")
        if int(max_new_tokens) < 0: raise ValueError("max_new_tokens must be non-negative")
        if not np.isfinite(float(temperature)) or float(temperature) < 0: raise ValueError("temperature must be finite and non-negative")
        if int(top_k) < 0: raise ValueError("top_k must be non-negative")
        if not np.isfinite(float(top_p)) or not 0 < float(top_p) <= 1: raise ValueError("top_p must be in (0, 1]")
        self.context_limit, self.max_new_tokens, self.temperature = cap, int(max_new_tokens), float(temperature)
        self.top_k, self.top_p, self.seed = int(top_k), float(top_p), int(seed); self.rng = np.random.default_rng(self.seed)
        self.reuse_kv = bool(reuse_kv); self.cached_token_ids = []; self._cache_identity = None; self._cache_trusted = True
        self.messages = []; self.messages.append({"role":"system", "content":system}) if system else None
    def _ids(self, messages):
        validate_messages(messages)
        try: ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        except Exception as exc: raise ChatError(f"tokenizer chat template failed: {exc}") from exc
        if isinstance(ids, Mapping): ids = ids["input_ids"]
        if hasattr(ids, "ids"): ids = ids.ids
        if hasattr(ids, "tolist"): ids = ids.tolist()
        if ids and hasattr(ids[0], "ids"):
            if len(ids) != 1: raise ChatError("tokenizer returned multiple batches")
            ids = ids[0].ids
        if ids and isinstance(ids[0], list):
            if len(ids) != 1: raise ChatError("tokenizer returned multiple batches")
            ids = ids[0]
        out = []
        for x in ids:
            try: out.append(operator.index(x))
            except TypeError as exc: raise ChatError("tokenizer returned non-integer token ID") from exc
        return out
    def _bounded(self, user):
        current = list(self.messages) + [{"role":"user", "content":user}]
        while True:
            ids = self._ids(current)
            if len(ids) + self.max_new_tokens <= self.context_limit: return current, ids
            start = 1 if current and current[0]["role"] == "system" else 0
            if start + 1 >= len(current) - 1 or current[start]["role"] != "user" or current[start + 1]["role"] != "assistant": raise ChatError("current turn does not fit context limit")
            del current[start:start + 2]
    def _sample(self, logits): return sample_logits(logits, self.rng, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)
    def _position(self):
        try: return int(self.runtime.position)
        except (AttributeError, RuntimeError, TypeError, ValueError): return None
    def _identity(self):
        try: native = self.runtime.cache_identity
        except AttributeError: native = None
        return (id(self.runtime), native)
    def _reset_cache(self):
        self.cached_token_ids = []; self._cache_trusted = False
        try:
            self.runtime.reset()
            pos = self._position()
            if pos is not None and pos != 0: raise RuntimeError("runtime reset did not return to position zero")
            self._cache_identity = self._identity(); self._cache_trusted = True
        except BaseException:
            self._cache_identity = None
            raise
    def _truncate_cache(self, position):
        truncate = getattr(self.runtime, "truncate", None)
        if truncate is None or not bool(getattr(self.runtime, "supports_truncate", False)): self._reset_cache(); return False
        truncate(int(position))
        if self._position() is not None and self._position() != int(position): raise RuntimeError("runtime truncate position mismatch")
        self.cached_token_ids = self.cached_token_ids[:position]; self._cache_identity = self._identity(); return True
    def _eval_and_commit(self, ids):
        values = list(ids); before = self._position()
        if before is not None and before != len(self.cached_token_ids): raise RuntimeError("cache position invariant violated")
        if any(x < np.iinfo(np.int32).min or x > np.iinfo(np.int32).max for x in values): raise ChatError("token ID exceeds native integer range")
        logits = self.runtime.eval(np.asarray(values, dtype=np.int32))
        after = self._position()
        if after is not None and before is not None and after != before + len(values): raise RuntimeError("runtime position did not advance by evaluated token count")
        self.cached_token_ids.extend(values); self._cache_identity = self._identity(); return logits
    def _stats_snapshot(self):
        if bool(getattr(self.runtime, "stats_abi_incompatible", False)):
            raise NativeError("native runtime exposes CPU-R4 shared-K symbols with an incompatible statistics ABI; rebuild the DLL")
        try:
            value = getattr(self.runtime, "stats", None)
            if callable(value): value = value()
            return dict(value) if isinstance(value, Mapping) else None
        except (AttributeError, RuntimeError, ValueError):
            return None
    @staticmethod
    def _stats_delta(after, before):
        if not isinstance(after, Mapping): return None
        if not isinstance(before, Mapping): return None
        out = {}
        for key, value in after.items():
            old = before.get(key, 0)
            if isinstance(value, Mapping) and isinstance(old, Mapping): out[key] = ChatSession._stats_delta(value, old)
            elif isinstance(value, list) and isinstance(old, list) and len(value) == len(old):
                out[key] = [current - previous for current, previous in zip(value, old)]
            elif isinstance(value, (int, float)) and isinstance(old, (int, float)): out[key] = value - old
            else: out[key] = value
        return out
    def turn(self, user):
        # Transaction boundary: no cache/history mutation survives any
        # runtime, sampling, or decoding failure.
        if not isinstance(user, str) or not user.strip(): raise ChatError("user message is empty")
        turn_start = time.perf_counter()
        current, ids = self._bounded(user)
        if not ids: raise ChatError("empty rendered prompt")
        vocab = getattr(self.runtime, "vocab_size", None)
        if any(x < 0 or (vocab is not None and x >= int(vocab)) or x < np.iinfo(np.int32).min or x > np.iinfo(np.int32).max for x in ids): raise ChatError("invalid token IDs")
        prepare_seconds = time.perf_counter() - turn_start
        try:
            return self._turn_impl(current, ids, turn_start, prepare_seconds)
        except BaseException:
            self.cached_token_ids = []; self._cache_identity = None; self._cache_trusted = False
            try: self.runtime.reset()
            except BaseException: pass
            raise

    def _turn_impl(self, current, ids, turn_start, prepare_seconds):
        position_before = self._position()
        cache_action = {"action": "reset", "reason": "reuse_disabled" if not self.reuse_kv else "empty_cache"}
        lcp = 0
        can_reuse = (self.reuse_kv and self._cache_trusted and bool(self.cached_token_ids)
                     and bool(getattr(self.runtime, "supports_truncate", False))
                     and self._position() == len(self.cached_token_ids)
                     and self._identity() == self._cache_identity)
        if can_reuse and self._identity() == self._cache_identity:
            lcp = next((i for i, (a,b) in enumerate(zip(self.cached_token_ids, ids)) if a != b), min(len(self.cached_token_ids), len(ids)))
            if lcp < len(self.cached_token_ids):
                if self._truncate_cache(lcp): cache_action = {"action": "truncate", "reason": "prompt_prefix_diverged"}
                else: cache_action = {"action": "reset", "reason": "truncate_unsupported"}; lcp = 0
            else: cache_action = {"action": "reuse", "reason": "exact_cached_prefix"}
        elif self.reuse_kv and self.cached_token_ids and not bool(getattr(self.runtime, "supports_truncate", False)):
            self._reset_cache(); lcp = 0; cache_action = {"action": "reset", "reason": "truncate_unsupported"}
        elif self.reuse_kv and self.cached_token_ids and self._identity() != self._cache_identity:
            self._reset_cache(); lcp = 0; cache_action = {"action": "reset", "reason": "cache_identity_changed"}
        elif self.reuse_kv and self.cached_token_ids and self._position() != len(self.cached_token_ids):
            self._reset_cache(); lcp = 0; cache_action = {"action": "reset", "reason": "cache_position_mismatch"}
        elif self.reuse_kv and self.cached_token_ids and not self._cache_trusted:
            self._reset_cache(); lcp = 0; cache_action = {"action": "reset", "reason": "cache_untrusted"}
        else:
            self._reset_cache(); lcp = 0
        # Reset counters independently from model state.  Native R1 runtimes
        # expose reset_stats; the fallback wrapper implements the same hook.
        reset_stats = getattr(self.runtime, "begin_turn", None) or getattr(self.runtime, "reset_stats", None)
        if reset_stats is not None: reset_stats()
        stats_before = self._stats_snapshot()
        set_phase = getattr(self.runtime, "set_phase", None)
        if set_phase is not None: set_phase("prefill")
        pending = ids[lcp:]
        if not pending:
            if not self.cached_token_ids: raise ChatError("empty cached prompt")
            rewind = max(0, len(self.cached_token_ids)-1)
            if self._truncate_cache(rewind):
                pending = ids[rewind:]; lcp = rewind; cache_action = {"action": "truncate", "reason": "empty_suffix_reevaluation"}
            else:
                pending = ids; lcp = 0
        prefill_started = time.perf_counter(); logits = self._eval_and_commit(pending); ttft = time.perf_counter() - prefill_started
        prefill_stats = self._stats_snapshot()
        generated = []; sampled_ids = []; eos = int(getattr(self.tokenizer, "eos_token_id", 2) or 2)
        sampling_seconds = 0.0; native_generation_seconds = 0.0; decode_evaluated = 0
        set_phase and set_phase("decode")
        decode_started = time.perf_counter()
        for index in range(self.max_new_tokens):
            sample_started = time.perf_counter()
            token = self._sample(logits)
            sampling_seconds += time.perf_counter() - sample_started
            sampled_ids.append(int(token))
            if token == eos: break
            generated.append(token)
            if index + 1 < self.max_new_tokens:
                native_started = time.perf_counter(); logits = self._eval_and_commit([token]); native_generation_seconds += time.perf_counter() - native_started; decode_evaluated += 1
        decode = time.perf_counter() - decode_started; text = self.tokenizer.decode(generated, skip_special_tokens=True)
        position_after = self._position()
        native_stats = self._stats_snapshot() or {}
        native_phase_stats = {"prefill": self._stats_delta(prefill_stats, stats_before),
                              "decode": self._stats_delta(native_stats, prefill_stats)}
        lm_heads = None
        if all(isinstance(native_phase_stats[phase], Mapping) and
               "lm_head_calls" in native_phase_stats[phase] for phase in ("prefill", "decode")):
            lm_heads = {phase: native_phase_stats[phase]["lm_head_calls"] for phase in ("prefill", "decode")}
        total = time.perf_counter() - turn_start
        metrics = {"prompt_tokens_total": len(ids), "prefix_tokens_reused": lcp,
                   "prefill_tokens_evaluated": len(pending), "decode_tokens_evaluated": decode_evaluated,
                   "generated_tokens": len(generated), "lm_head_calls": lm_heads,
                   "prompt_token_ids": list(ids),
                   "position_before": position_before, "position_after": position_after,
                   "cache_action": cache_action,
                   "prompt_prepare_seconds": prepare_seconds, "prefill_seconds": ttft,
                   "native_generation_seconds": native_generation_seconds,
                   "sampling_seconds": sampling_seconds, "total_seconds": total,
                   "sampled_ids": sampled_ids, "native": native_stats,
                   "native_phase_stats": native_phase_stats}
        response = ChatResponse(text, len(generated), len(ids), ttft, decode, self.runtime.backend,
                            len(ids), lcp, len(pending), decode_evaluated, lm_heads, position_before, position_after,
                            cache_action, prepare_seconds, ttft, native_generation_seconds,
                            sampling_seconds, total, sampled_ids, metrics, native_phase_stats)
        self.messages = current + [{"role":"assistant", "content":text}]
        return response
    def clear(self):
        self._reset_cache(); self.messages = [self.messages[0]] if self.messages and self.messages[0]["role"] == "system" else []; self.rng = np.random.default_rng(self.seed)

def _response_json(r, args):
    return {
        "text": r.text,
        "generated_tokens": r.generated_tokens,
        "context_tokens": r.context_tokens,
        "ttft_seconds": r.ttft_seconds,
        "decode_seconds": r.decode_seconds,
        "backend": r.backend,
        "profile": args.profile,
        "system": args.system,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "metrics": r.metrics,
        "prompt_tokens_total": r.prompt_tokens_total,
        "prefix_tokens_reused": r.prefix_tokens_reused,
        "prefill_tokens_evaluated": r.prefill_tokens_evaluated,
        "decode_tokens_evaluated": r.decode_tokens_evaluated,
        "lm_head_calls": r.lm_head_calls,
        "position_before": r.position_before,
        "position_after": r.position_after,
        "cache_action": r.cache_action,
        "prompt_prepare_seconds": r.prompt_prepare_seconds,
        "prefill_seconds": r.prefill_seconds,
        "native_generation_seconds": r.native_generation_seconds,
        "sampling_seconds": r.sampling_seconds,
        "total_seconds": r.total_seconds,
        "sampled_ids": r.sampled_ids,
        "native_phase_stats": r.native_phase_stats,
        "diagnostics": bool(getattr(args, "diagnostics", False)),
        "selective_logits": bool(getattr(args, "selective_logits", False)),
        "reuse_kv": bool(getattr(args, "reuse_kv", False)),
        "v_blocked_attention": bool(getattr(args, "v_blocked_attention", False)),
        "ffn_row4": bool(getattr(args, "ffn_row4", False)),
        "gqa_k_shared": bool(getattr(args, "gqa_k_shared", False)),
        "gqa_v_shared": bool(getattr(args, "gqa_v_shared", False)),
    }


def _resolve_profile(args):
    if args.profile == "chat":
        defaults = (0.9, 50, 0.85, DEFAULT_CHAT_SYSTEM)
    else:
        defaults = (0.0, 0, 1.0, None)
    if args.temperature is None:
        args.temperature = defaults[0]
    if args.top_k is None:
        args.top_k = defaults[1]
    if args.top_p is None:
        args.top_p = defaults[2]
    if args.system is None:
        args.system = defaults[3]
    return args


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--library")
    p.add_argument("--prompt")
    p.add_argument("--profile", choices=("chat", "parity"), default="chat")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--context-limit", type=int, default=2048)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-k", type=int)
    p.add_argument("--top-p", type=float)
    p.add_argument("--system")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metrics-json", action="store_true")
    p.add_argument("--diagnostics", action="store_true", help="enable native diagnostic profiling (off by default)")
    p.add_argument("--selective-logits", action="store_true", help="evaluate vocabulary logits only when requested")
    p.add_argument("--reuse-kv", action="store_true", help="reuse verified KV prefixes between turns")
    p.add_argument("--v-blocked-attention", action="store_true", help="use experimental contiguous-dimension V accumulation (off by default)")
    p.add_argument("--ffn-row4", action="store_true", help="use experimental four-row FP32 FFN GEMV (off by default)")
    p.add_argument("--gqa-k-shared", action="store_true", help="share K reads across the two GQA query heads (off by default)")
    p.add_argument("--gqa-v-shared", action="store_true", help="share V reads across the two GQA query heads (off by default)")
    args = _resolve_profile(p.parse_args(argv))
    if args.max_new_tokens < 0: p.error("--max-new-tokens must be non-negative")
    if args.context_limit < 2: p.error("--context-limit must be at least 2")
    if not np.isfinite(args.temperature) or args.temperature < 0 or args.top_k < 0 or not np.isfinite(args.top_p) or not 0 < args.top_p <= 1: p.error("sampling parameters must be finite and valid")
    try:
        os.environ.setdefault("USE_TORCH", "0")
        os.environ.setdefault("USE_TF", "0")
        tokenizer = load_local_tokenizer(args.tokenizer)
        with NativeRuntime(args.model,args.library,args.context_limit) as runtime:
            configure = getattr(runtime, "configure_profile", None)
            if configure is not None: configure(bool(args.diagnostics))
            selective = getattr(runtime, "configure_selective_logits", None)
            if args.selective_logits and selective is None: raise NativeError("native runtime does not support selective logits")
            if selective is not None: selective(bool(args.selective_logits))
            v_blocked = getattr(runtime, "configure_v_blocked_attention", None)
            if args.v_blocked_attention and v_blocked is None: raise NativeError("native runtime does not support V-blocked attention")
            if v_blocked is not None: v_blocked(bool(args.v_blocked_attention))
            ffn_row4 = getattr(runtime, "configure_ffn_row4", None)
            if args.ffn_row4 and ffn_row4 is None: raise NativeError("native runtime does not support FFN row4")
            if ffn_row4 is not None: ffn_row4(bool(args.ffn_row4))
            gqa_k_shared = getattr(runtime, "configure_gqa_k_shared", None)
            if args.gqa_k_shared and gqa_k_shared is None: raise NativeError("native runtime does not support shared-K GQA attention")
            if gqa_k_shared is not None: gqa_k_shared(bool(args.gqa_k_shared))
            gqa_v_shared = getattr(runtime, "configure_gqa_v_shared", None)
            if args.gqa_v_shared and gqa_v_shared is None: raise NativeError("native runtime does not support shared-V GQA attention")
            if gqa_v_shared is not None: gqa_v_shared(bool(args.gqa_v_shared))
            session = ChatSession(runtime, tokenizer, context_limit=args.context_limit,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_k=args.top_k,
                                  top_p=args.top_p, seed=args.seed, system=args.system, reuse_kv=args.reuse_kv)
            print(
                f"profile={args.profile}, temperature={args.temperature:g}, "
                f"top_k={args.top_k}, top_p={args.top_p:g}, seed={args.seed}, "
                f"system={args.system!r}",
                file=sys.stderr,
                flush=True,
            )
            if args.prompt is not None:
                result = session.turn(args.prompt)
                print(json.dumps(_response_json(result, args), ensure_ascii=False) if args.metrics_json else result.text)
                return 0
            while True:
                try: line=input("> ")
                except EOFError: return 0
                except KeyboardInterrupt: print(file=sys.stderr); return 130
                if line.strip() in {"/exit","/quit"}: return 0
                if line.strip()=="/clear":
                    try: session.clear()
                    except ChatError as exc: print(f"error: {exc}",file=sys.stderr)
                    continue
                try:
                    result = session.turn(line)
                    print(json.dumps(_response_json(result, args), ensure_ascii=False) if args.metrics_json else result.text)
                except ChatError as exc: print(f"error: {exc}",file=sys.stderr); continue
    except ChatError as exc: print(f"error: {exc}",file=sys.stderr); return 2
    except (NativeError,OSError,RuntimeError,ValueError) as exc: print(f"error: {exc}",file=sys.stderr); return 1
if __name__ == "__main__": raise SystemExit(main())
