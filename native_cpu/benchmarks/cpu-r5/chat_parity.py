"""Run the historical six-turn conversation against CPU-R5 W0/W1."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
LIBRARY = ROOT / "native_cpu" / "build" / "Release" / "minimind_cpu.dll"
TOKENIZER = ROOT / "checkpoints" / "minimind-3-hf"
OUTPUT = Path(__file__).resolve().parent / "chat-parity.json"


def _load_benchmark():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_benchmark_for_cpu_r5_chat", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(benchmark, profile_name: str, tokenizer):
    from native_cpu.tools.native import NativeRuntime

    config = benchmark.PROFILES[profile_name]
    with NativeRuntime(MODEL, LIBRARY, 2048, threads=config["threads"], cpus=config["cpus"], row_weights=config["row_weights"]) as runtime:
        runtime.configure_selective_logits(config["selective_logits"])
        runtime.configure_v_blocked_attention(config["v_blocked_attention"])
        runtime.configure_ffn_row4(config["ffn_row4"])
        runtime.configure_gqa_k_shared(config["gqa_k_shared"])
        runtime.configure_gqa_v_shared(config["gqa_v_shared"])
        benchmark.warmup(runtime, 1)
        data, session = benchmark.run_conversation(runtime, tokenizer, config, benchmark.HISTORICAL_PROMPTS, 256)
        data["profile"] = profile_name
        data["effective"] = {
            "threads": runtime.threads,
            "cpus": runtime.cpus,
            "row_weights": runtime.row_weights,
            "selective_logits": runtime.selective_logits,
            "reuse_kv": session.reuse_kv,
            "v_blocked_attention": runtime.v_blocked_attention,
            "ffn_row4": runtime.ffn_row4,
            "gqa_k_shared": runtime.gqa_k_shared,
            "gqa_v_shared": runtime.gqa_v_shared,
        }
        return data


def main() -> int:
    benchmark = _load_benchmark()
    from native_cpu.tools.chat import LocalTokenizer

    tokenizer = LocalTokenizer(TOKENIZER)
    profiles = {name: _run(benchmark, name, tokenizer) for name in ("W0", "W1")}
    comparison = benchmark.compare_entries(profiles["W0"], profiles["W1"], "historical")
    checks = []
    for index, (left, right) in enumerate(zip(profiles["W0"]["turns"], profiles["W1"]["turns"]), start=1):
        checks.append({
            "turn": index,
            "same_prompt_token_ids": left["prompt_token_ids"] == right["prompt_token_ids"],
            "same_sampled_ids": left["sampled_ids"] == right["sampled_ids"],
            "same_text": left["text"] == right["text"],
            "same_finish_reason": left["finish_reason"] == right["finish_reason"],
        })
    result = {
        "schema": "cpu-r5/chat-parity-v1",
        "status": "PASS" if comparison.get("parity") else "FAIL",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_sha256": _sha(MODEL),
        "library_sha256": _sha(LIBRARY),
        "profiles": {name: data["effective"] for name, data in profiles.items()},
        "checks": checks,
        "comparison": comparison,
        "reference": profiles["W0"],
        "candidate": profiles["W1"],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(f"{result['status']}: {OUTPUT}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
