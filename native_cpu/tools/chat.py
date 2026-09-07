"""Local-tokenizer chat frontend for the MMCPU001 runtime."""
from __future__ import annotations
import argparse, json, os, sys, time
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
    def __init__(self, runtime, tokenizer, *, context_limit=256, max_new_tokens=64, temperature=0.0, seed=0, system=None, top_k=0, top_p=1.0):
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
        return [int(x) for x in ids]
    def _bounded(self, user):
        current = list(self.messages) + [{"role":"user", "content":user}]
        while True:
            ids = self._ids(current)
            if len(ids) + self.max_new_tokens <= self.context_limit: return current, ids
            start = 1 if current and current[0]["role"] == "system" else 0
            if start + 1 >= len(current) - 1 or current[start]["role"] != "user" or current[start + 1]["role"] != "assistant": raise ChatError("current turn does not fit context limit")
            del current[start:start + 2]
    def _sample(self, logits): return sample_logits(logits, self.rng, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p)
    def turn(self, user):
        if not isinstance(user, str) or not user.strip(): raise ChatError("user message is empty")
        current, ids = self._bounded(user)
        if not ids: raise ChatError("empty rendered prompt")
        self.runtime.reset(); start = time.perf_counter(); logits = self.runtime.eval(np.asarray(ids, dtype=np.int32)); ttft = time.perf_counter() - start
        generated = []; eos = int(getattr(self.tokenizer, "eos_token_id", 2) or 2); start = time.perf_counter()
        for index in range(self.max_new_tokens):
            token = self._sample(logits)
            if token == eos: break
            generated.append(token)
            if index + 1 < self.max_new_tokens: logits = self.runtime.eval(np.asarray([token], dtype=np.int32))
        decode = time.perf_counter() - start; text = self.tokenizer.decode(generated, skip_special_tokens=True)
        self.messages = current + [{"role":"assistant", "content":text}]
        return ChatResponse(text, len(generated), len(ids), ttft, decode, self.runtime.backend)
    def clear(self):
        self.runtime.reset(); self.messages = [self.messages[0]] if self.messages and self.messages[0]["role"] == "system" else []; self.rng = np.random.default_rng(self.seed)

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
    args = _resolve_profile(p.parse_args(argv))
    if args.max_new_tokens < 0: p.error("--max-new-tokens must be non-negative")
    if args.context_limit < 2: p.error("--context-limit must be at least 2")
    if not np.isfinite(args.temperature) or args.temperature < 0 or args.top_k < 0 or not np.isfinite(args.top_p) or not 0 < args.top_p <= 1: p.error("sampling parameters must be finite and valid")
    try:
        os.environ.setdefault("USE_TORCH", "0")
        os.environ.setdefault("USE_TF", "0")
        tokenizer = load_local_tokenizer(args.tokenizer)
        with NativeRuntime(args.model,args.library,args.context_limit) as runtime:
            session = ChatSession(runtime, tokenizer, context_limit=args.context_limit,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_k=args.top_k,
                                  top_p=args.top_p, seed=args.seed, system=args.system)
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
