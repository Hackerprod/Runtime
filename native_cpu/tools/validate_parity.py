"""Independent, offline dense-runtime parity against the pinned upstream model.

FP32 acceptance requires logit tolerance AND exact greedy token streams. Q4
results describe drift, not exact parity or conversational quality acceptance.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.util
import json
import numbers
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "6fc918beb68a0d8c40452338df6319fe168014ba"
CHECKPOINT_REVISION = "f92512d4cd6142fa9acc0d6022375049a8974bf6"
CHECKPOINT_SHA256 = "3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8"
PROMPTS = (
    ("en", "Explain what a CPU does in two sentences."),
    ("es", "Explica qué hace una CPU en dos frases."),
    ("zh", "请用两句话解释什么是机器学习。"),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_download(directory):
    """Verify every pinned download, including unchanged tokenizer/template files."""
    directory = Path(directory).resolve()
    manifest = directory / "download-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (data.get("repo_id"), data.get("revision")) != (
            "jingyaogong/minimind-3", CHECKPOINT_REVISION):
        raise ValueError("reference must be the pinned official MiniMind-3 download")
    checked = {}
    for item in data["files"]:
        name = item["path"]
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or name in checked:
            raise ValueError("unsafe or duplicate download manifest path")
        identity = file_identity(path)
        if identity["bytes"] != item["bytes"] or identity["sha256"] != item["sha256"]:
            raise ValueError(f"download manifest mismatch: {name}")
        checked[name] = identity
    required = {"config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "chat_template.jinja"}
    if not required <= checked.keys():
        raise ValueError("download manifest is missing required reference files")
    if checked["model.safetensors"]["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("reference checkpoint differs from the pinned official weights")
    return {"manifest": file_identity(manifest), "repo_id": data["repo_id"],
            "revision": data["revision"], "files": checked}


def strict_state_dict(state, expected, tied_verified):
    """Allow only the upstream-verified missing tied lm_head alias; never fill weights."""
    import torch
    result = dict(state)
    aliases = []
    if "lm_head.weight" not in result:
        if not tied_verified or "model.embed_tokens.weight" not in result:
            raise ValueError("missing lm_head without verified tied embedding")
        result["lm_head.weight"] = result["model.embed_tokens.weight"]
        aliases.append({"target": "lm_head.weight", "source": "model.embed_tokens.weight"})
    if set(result) != set(expected):
        raise ValueError(f"state keys differ: missing={sorted(set(expected)-set(result))}; "
                         f"unexpected={sorted(set(result)-set(expected))}")
    if tied_verified and not torch.equal(result["lm_head.weight"], result["model.embed_tokens.weight"]):
        raise ValueError("conflicting tied embedding and lm_head tensors")
    for name, tensor in result.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"state shape mismatch: {name}")
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-floating or non-finite source tensor: {name}")
    return {name: value.float() for name, value in result.items()}, aliases


def verify_export(artifact, manifest, quantization, reference):
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    identity = file_identity(artifact)
    expected = {
        "schema": "MMCPU001", "quantization": quantization,
        "input_sha256": reference["files"]["model.safetensors"]["sha256"],
        "config_sha256": reference["files"]["config.json"]["sha256"],
        "source_manifest_sha256": reference["manifest"]["sha256"],
        "output_sha256": identity["sha256"],
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"export provenance mismatch: {key}")
    if data.get("output", {}).get("bytes") != identity["bytes"]:
        raise ValueError("export byte count mismatch")
    return {"artifact": identity, "manifest": file_identity(manifest),
            "quantization": quantization, "source_sha256": expected["input_sha256"],
            "config_sha256": expected["config_sha256"], "source_manifest_verified": True}


def verify_build_info(path, library_identity):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("native_threads") != 1:
        raise ValueError("build info requires schema_version=1 and native_threads=1")
    for name in ("id", "version", "path"):
        if not isinstance(data.get("compiler", {}).get(name), str) or not data["compiler"][name].strip():
            raise ValueError(f"build info is missing compiler.{name}")
    for name in ("build_type", "compile_flags"):
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise ValueError(f"build info is missing {name}")
    if data.get("library_sha256") != library_identity["sha256"]:
        raise ValueError("build info DLL/library hash mismatch")
    for name, expected in data.get("native_sources_sha256", {}).items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or sha256(source) != expected:
            raise ValueError(f"build info source hash mismatch: {name}")
    return data


def normalize_prompt_ids(value):
    """Accept one tokenized prompt, including Transformers 5.x Encoding objects."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "ids"):
        value = value.ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError("tokenizer did not return a single sequence of token IDs")
    if len(value) == 1 and hasattr(value[0], "ids"):
        value = value[0].ids
    elif len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not value or any(isinstance(x, bool) or not isinstance(x, numbers.Integral) or x < 0 for x in value):
        raise ValueError("tokenizer must return one nonempty sequence of nonnegative integer IDs")
    return [int(x) for x in value]


def import_reference_module(source):
    name = "_official_minimind_parity_reference"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    # Transformers inspects sys.modules[model_class.__module__] during construction.
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def load_reference(directory):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    provenance = verify_download(directory)
    source = ROOT / "model/model_minimind.py"
    committed = subprocess.check_output(
        ["git", "show", f"{UPSTREAM_REVISION}:model/model_minimind.py"], cwd=ROOT)
    if source.read_bytes().replace(b"\r\n", b"\n") != committed.replace(b"\r\n", b"\n"):
        raise ValueError("upstream reference source differs from the pinned commit")
    module = import_reference_module(source)
    cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (cfg.get("attention_bias", False) or cfg.get("use_sliding_window", False)
            or cfg.get("rope_scaling") is not None or cfg.get("hidden_act") != "silu"
            or cfg.get("tie_word_embeddings") is not True):
        raise ValueError("unsupported official reference configuration")
    fields = ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
              "num_key_value_heads", "head_dim", "intermediate_size", "max_position_embeddings",
              "rms_norm_eps", "rope_theta", "tie_word_embeddings", "hidden_act")
    config = module.MiniMindConfig(**{key: cfg[key] for key in fields},
                                  use_moe=False, flash_attn=False, dropout=0.0,
                                  inference_rope_scaling=False)
    model = module.MiniMindForCausalLM(config).float().cpu().eval()
    raw = load_file(str(directory / "model.safetensors"), device="cpu")
    tied = config.tie_word_embeddings and model.lm_head.weight is model.model.embed_tokens.weight
    state, aliases = strict_state_dict(raw, model.state_dict(), tied)
    model.load_state_dict(state, strict=True)
    if not model.lm_head.weight is model.model.embed_tokens.weight:
        raise ValueError("reference lost tied embedding identity")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                             trust_remote_code=False)
    if tokenizer.eos_token_id != 2:
        raise ValueError("official EOS token must be 2")
    provenance.update({"upstream_revision": UPSTREAM_REVISION,
                       "upstream_source": file_identity(source), "synthesized_aliases": aliases,
                       "source_dtypes": sorted({str(x.dtype) for x in raw.values()}),
                       "loaded_dtype": "torch.float32", "flash_attn": False,
                       "state_tensor_count": len(state)})
    return model, tokenizer, provenance


def logit_metrics(reference, actual, atol=1e-4, rtol=1e-4):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape or reference.ndim not in (1, 2) or not reference.size:
        raise ValueError("logit arrays must have equal, nonempty vector/matrix shapes")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("non-finite logits")
    reference, actual = np.atleast_2d(reference), np.atleast_2d(actual)
    delta = actual - reference
    def log_softmax(value):
        value = value - value.max(axis=-1, keepdims=True)
        return value - np.log(np.exp(value).sum(axis=-1, keepdims=True))
    logp, logq = log_softmax(reference), log_softmax(actual)
    agreement = reference.argmax(-1) == actual.argmax(-1)
    return {"positions": len(reference), "logit_values": reference.size,
            "max_absolute_error": float(np.abs(delta).max()),
            "rms_error": float(np.sqrt(np.mean(delta * delta))),
            "allclose": bool(np.all(np.abs(delta) <= atol + rtol * np.abs(reference))),
            "top1_matches": int(agreement.sum()), "top1_agreement": float(agreement.mean()),
            "mean_kl_reference_to_native": max(0.0, float(np.mean(
                np.sum(np.exp(logp) * (logp - logq), axis=-1))))}


def greedy_decode(logits, step, limit, eos=2):
    tokens, decisions = [], []
    for index in range(limit):
        logits = np.asarray(logits)
        if logits.ndim != 1 or not logits.size or not np.isfinite(logits).all():
            raise ValueError("greedy decoding requires finite nonempty logit vectors")
        decisions.append(logits.copy())
        token = int(np.argmax(logits))
        tokens.append(token)
        if token == eos or index + 1 == limit:
            break
        logits = step([token])
    return tokens, decisions


class ReferenceSession:
    def __init__(self, model):
        self.model, self.cache = model, None

    def reset(self):
        self.cache = None

    def eval(self, ids):
        import torch
        result = self.model(torch.tensor([ids], dtype=torch.long), past_key_values=self.cache,
                            use_cache=True, logits_to_keep=1)
        self.cache = result.past_key_values
        return result.logits[0, -1].detach().numpy().copy()


def reference_trace(model, ids, decode_tokens):
    import torch
    session = ReferenceSession(model)
    prefill = np.stack([session.eval([token]) for token in ids])
    generated, decisions = greedy_decode(prefill[-1], session.eval, decode_tokens)
    full = model(torch.tensor([ids], dtype=torch.long), use_cache=False).logits[0].detach().numpy()
    return {"prefill": prefill, "full": full.copy(), "generation": np.stack(decisions),
            "generated_ids": generated}


def native_trace(native, ids, reference, decode_tokens, atol, rtol):
    native.reset()
    position = 0
    def step(tokens):
        nonlocal position
        value = np.asarray(native.eval(tokens)).copy()
        position += len(tokens)
        if native.position != position:
            raise ValueError(f"native cache position {native.position} != expected {position}")
        return value
    prefill = np.stack([step([token]) for token in ids])
    forced, logits = [], prefill[-1]
    for index, token in enumerate(reference["generated_ids"]):
        forced.append(logits)
        if token != 2 and index + 1 < len(reference["generated_ids"]):
            logits = step([token])
    native.reset()
    position = 0
    full_last = step(ids)
    generated, _ = greedy_decode(full_last, step, decode_tokens)
    checks = {
        "tokenwise_prefill": logit_metrics(reference["prefill"], prefill, atol, rtol),
        "full_prefill_last": logit_metrics(reference["full"][-1], full_last, atol, rtol),
        "teacher_forced_decode": logit_metrics(reference["generation"], forced, atol, rtol),
    }
    return {"logits": checks, "generated_ids": generated,
            "greedy_token_stream_match": generated == reference["generated_ids"],
            "all_logits_within_tolerance": all(x["allclose"] for x in checks.values()),
            "eos_reached": bool(generated and generated[-1] == 2)}


def benchmark(session, ids, decode_ids, repeats):
    """Separate uninstrumented batch-prefill/decode; use identical teacher-forced IDs."""
    session.reset()
    session.eval(ids)
    for token in decode_ids:
        session.eval([token])
    prefill, decode = [], []
    for _ in range(repeats):
        session.reset()
        start = time.perf_counter()
        session.eval(ids)
        prefill.append(time.perf_counter() - start)
        start = time.perf_counter()
        for token in decode_ids:
            session.eval([token])
        decode.append(time.perf_counter() - start)
    return {"prefill_tokens": len(ids), "decode_tokens": len(decode_ids),
            "prefill_seconds": prefill, "decode_seconds": decode,
            "median_prefill_seconds": statistics.median(prefill),
            "median_decode_seconds": statistics.median(decode) if decode_ids else None,
            "median_decode_tokens_per_second": len(decode_ids) / statistics.median(decode)
            if decode_ids else None}


def hardware_identity():
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return {"cpu": cpu, "machine": platform.machine(), "platform": platform.platform(),
            "logical_cpus": os.cpu_count(), "python": platform.python_version()}


def run(args):
    import torch
    sys.path.insert(0, str(ROOT))
    from native_cpu.tools.native import NativeModel

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing parity report")
    model, tokenizer, provenance = load_reference(args.checkpoint_dir.resolve())
    library_identity = file_identity(args.library)
    build = verify_build_info(args.build_info, library_identity) if args.build_info else None
    report = {"schema_version": 1, "reference": provenance,
              "library": library_identity, "hardware": hardware_identity(),
              "build_info": build, "build_info_file": file_identity(args.build_info)
              if args.build_info else None,
              "configuration": {"atol": args.atol, "rtol": args.rtol, "eos_token_id": 2,
                                "decode_tokens": args.decode_tokens, "context": args.context,
                                "torch_threads": args.threads, "torch_interop_threads": 1,
                                "native_threads": 1,
                                "torch_version": torch.__version__, "dtype": "float32",
                                "benchmark_repeats": args.bench_repeats},
              "verification_gaps": [] if build else ["Compiler/build metadata was not supplied."],
              "cases": [], "notes": [
                  "FP32 requires allclose logits and exact greedy token streams; no language-quality gate.",
                  "Q4 drift is measured separately; exact token streams are not expected or required.",
                  "Timings include Python/native boundary overhead, exclude loading, and are not fixed speed claims.",
                  "Original pinned tokenizer and chat template are used unchanged with enable_thinking=False."]}
    variants = [("fp32_scalar", args.fp32_model, args.fp32_manifest, "scalar"),
                ("fp32_auto", args.fp32_model, args.fp32_manifest, "auto")]
    if args.q4_model:
        variants.append(("q4_auto", args.q4_model, args.q4_manifest, "auto"))
    report["artifacts"] = {"fp32": verify_export(args.fp32_model, args.fp32_manifest, "fp32", provenance)}
    if args.q4_model:
        report["artifacts"]["q4"] = verify_export(args.q4_model, args.q4_manifest, "q4", provenance)
    with torch.inference_mode():
        for language, prompt in PROMPTS:
            ids = normalize_prompt_ids(tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False))
            if not ids or len(ids) > 128 or len(ids) + args.decode_tokens > args.context:
                raise ValueError(f"{language} prompt exceeds bounded validation context; never truncate")
            reference = reference_trace(model, ids, args.decode_tokens)
            case = {"language": language, "prompt": prompt, "prompt_ids": ids,
                    "reference_generated_ids": reference["generated_ids"],
                    "reference_response": tokenizer.decode(reference["generated_ids"], skip_special_tokens=True),
                    "reference_cache_vs_full": logit_metrics(reference["full"], reference["prefill"],
                                                              args.atol, args.rtol), "variants": {}}
            if args.bench_repeats:
                case["reference_benchmark"] = benchmark(ReferenceSession(model), ids,
                    reference["generated_ids"][:-1], args.bench_repeats)
            for name, artifact, manifest, mode in variants:
                native = NativeModel(artifact, args.library, args.context, mode=mode)
                try:
                    if native.vocab_size != model.config.vocab_size:
                        raise ValueError("native vocabulary differs from reference")
                    result = native_trace(native, ids, reference, args.decode_tokens, args.atol, args.rtol)
                    result.update({"artifact": "fp32" if name.startswith("fp32") else "q4",
                                   "mode": mode, "backend": native.backend,
                                   "response": tokenizer.decode(result["generated_ids"], skip_special_tokens=True)})
                    if args.bench_repeats:
                        result["benchmark"] = benchmark(native, ids, reference["generated_ids"][:-1],
                                                        args.bench_repeats)
                    result["status"] = ("PASS" if result["all_logits_within_tolerance"] and
                        result["greedy_token_stream_match"] else "FAIL") if name.startswith("fp32") else "DRIFT_MEASURED"
                    case["variants"][name] = result
                finally:
                    native.close()
            report["cases"].append(case)
    report["status"] = "PASS" if all(
        c["reference_cache_vs_full"]["allclose"] and all(
            v["status"] == "PASS" for k, v in c["variants"].items() if k.startswith("fp32"))
        for c in report["cases"]) else "FAIL"
    if not args.q4_model:
        report["verification_gaps"].append("Q4 was not supplied or evaluated.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--fp32-manifest", type=Path, required=True)
    parser.add_argument("--q4-model", type=Path)
    parser.add_argument("--q4-manifest", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints/minimind-3-hf")
    parser.add_argument("--build-info", type=Path)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, choices=range(8, 33), default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bench-repeats", type=int, choices=range(0, 11), default=3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.q4_model) != bool(args.q4_manifest):
        parser.error("Q4 model and manifest must be supplied together")
    if args.threads < 1 or args.context < 1 or not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("threads/context must be positive and tolerances finite/nonnegative")
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "verification_gaps": result["verification_gaps"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
