"""CPU-E6: exact FP16 attention-storage integration against CPU-E4."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
TOKENIZER = ROOT / "checkpoints" / "minimind-3-hf"
BASELINE = ROOT / "native_cpu" / "benchmarks" / "cpu-e4" / "baseline" / "minimind_cpu.dll"
CANDIDATE = ROOT / "native_cpu" / "build-e6" / "Release" / "minimind_cpu.dll"
EXPECTED_BASELINE_SHA256 = "b410e642e2249a8982a9a83b134f3d7f5e1f37c677f11d10900ed0e213cb97ab"
PAIR_COUNT = 3
CONTINUATION = 32
REPETITIVE_PROMPTS = ["Name one benefit of tests."] * 3
CANONICAL_SAMPLED_IDS_SHA256 = "de27dbd287b02a36f6116b22152359f4b5d1e13c9d0cf3f454512d40a6dcb8b6"
CANONICAL_TEXT_SHA256 = "bbab7af92f7db6781fe097645898ee98d3409f9367a9d764883637489bace770"


def _load_cpu_r1_benchmark():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_for_cpu_e6", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_sha(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _runtime_options(runtime) -> dict:
    return {
        "threads": runtime.threads,
        "cpus": runtime.cpus,
        "row_weights": runtime.row_weights,
        "selective_logits": runtime.selective_logits,
        "v_blocked_attention": runtime.v_blocked_attention,
        "gqa_k_shared": runtime.gqa_k_shared,
        "gqa_v_shared": runtime.gqa_v_shared,
        "ffn_f16_storage": runtime.ffn_f16_storage,
        "attention_f16_storage": runtime.attention_f16_storage,
        "attention_f16_storage_bytes": runtime.attention_f16_storage_bytes,
        "attention_f16_prepare_ns": runtime.attention_f16_prepare_ns,
        "ffn_f16_storage_bytes": runtime.ffn_f16_storage_bytes,
        "ffn_f16_prepare_ns": runtime.ffn_f16_prepare_ns,
    }


def run_forced(library: Path, length: int, benchmark) -> dict:
    from native_cpu.tools.native import NativeRuntime

    load_started = time.perf_counter()
    with NativeRuntime(MODEL, library, 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        loading_seconds = time.perf_counter() - load_started
        runtime.configure_profile(False)
        benchmark.warmup(runtime, 1)
        prefix, suffix = benchmark.forced_ids(length, CONTINUATION, runtime.vocab_size)
        runtime.reset()
        runtime.reset_stats()
        started = time.perf_counter()
        prefill_logits = runtime.eval(prefix)
        prefill_seconds = time.perf_counter() - started
        decode_hashes = []
        decode_durations = []
        for token in suffix:
            started = time.perf_counter()
            logits = runtime.eval(np.asarray([token], dtype=np.int32))
            decode_durations.append(time.perf_counter() - started)
            decode_hashes.append(vector_sha(logits))
        decode_seconds = sum(decode_durations)
        if runtime.position != length + CONTINUATION:
            raise RuntimeError("forced position invariant failed")
        return {
            "library_sha256": sha(library),
            "prefix_token_ids": prefix.tolist(),
            "continuation_token_ids": suffix.tolist(),
            "prefill_logits_sha256": vector_sha(prefill_logits),
            "decode_logits_sha256": decode_hashes,
            "position_after": runtime.position,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": prefill_seconds + decode_seconds,
            "prefill_tokens_per_second": length / prefill_seconds,
            "output_decode_tokens_per_second": CONTINUATION / decode_seconds,
            "generated_tokens": CONTINUATION,
            "loading_seconds": loading_seconds,
            "storage_preparation_seconds": (runtime.attention_f16_prepare_ns or 0) / 1e9,
            "effective": _runtime_options(runtime),
        }


def _aggregate_speculative(turns):
    keys = ("proposed_blocks", "proposed_tokens", "accepted_tokens", "verify_calls",
            "first_position_failures", "accepted_length_total", "lookup_seconds",
            "verification_seconds")
    result = {key: sum(float(turn.get("speculative_decode", {}).get(key, 0)) for turn in turns)
              for key in keys}
    result["mean_accepted_length"] = (result["accepted_length_total"] / result["proposed_blocks"]
                                      if result["proposed_blocks"] else 0.0)
    return result


def run_repetitive(library: Path) -> dict:
    from native_cpu.tools.chat import ChatSession, DEFAULT_CHAT_SYSTEM, LocalTokenizer
    from native_cpu.tools.native import NativeRuntime

    tokenizer = LocalTokenizer(TOKENIZER)
    load_started = time.perf_counter()
    with NativeRuntime(MODEL, library, 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        loading_seconds = time.perf_counter() - load_started
        runtime.configure_profile(False)
        benchmark = _load_cpu_r1_benchmark()
        benchmark.warmup(runtime, 1)
        session = ChatSession(runtime, tokenizer, context_limit=2048, max_new_tokens=32,
                              temperature=.9, top_k=50, top_p=.85, seed=0,
                              system=DEFAULT_CHAT_SYSTEM, reuse_kv=True)
        turns = []
        started = time.perf_counter()
        for prompt in REPETITIVE_PROMPTS:
            result = session.turn(prompt)
            turns.append({
                "user": prompt,
                "prompt_token_ids": result.metrics.get("prompt_token_ids"),
                "sampled_ids": result.sampled_ids,
                "text": result.text,
                "generated_tokens": result.generated_tokens,
                "position_after": result.position_after,
                "prefill_seconds": result.prefill_seconds,
                "decode_seconds": result.decode_seconds,
                "total_seconds": result.total_seconds,
                "speculative_decode": result.metrics.get("speculative_decode", {}),
            })
        wall_seconds = time.perf_counter() - started
        decode_seconds = sum(turn["decode_seconds"] for turn in turns)
        return {
            "library_sha256": sha(library),
            "turns": turns,
            "conversation_seconds": wall_seconds,
            "prefill_seconds": sum(turn["prefill_seconds"] for turn in turns),
            "decode_seconds": decode_seconds,
            "total_seconds": sum(turn["total_seconds"] for turn in turns),
            "generated_tokens": sum(turn["generated_tokens"] for turn in turns),
            "output_decode_tokens_per_second": (sum(turn["generated_tokens"] for turn in turns) / decode_seconds
                                                 if decode_seconds else None),
            "loading_seconds": loading_seconds,
            "storage_preparation_seconds": (runtime.attention_f16_prepare_ns or 0) / 1e9,
            "effective": _runtime_options(runtime),
            "speculative_decode": _aggregate_speculative(turns),
        }


def compare_forced(left: dict, right: dict) -> dict:
    fields = ("prefix_token_ids", "continuation_token_ids", "prefill_logits_sha256",
              "decode_logits_sha256", "position_after")
    for field in fields:
        if left.get(field) != right.get(field):
            return {"parity": False, "field": field,
                    "left": left.get(field), "right": right.get(field)}
    return {"parity": True}


def compare_chat(left: dict, right: dict) -> dict:
    if len(left.get("turns", [])) != len(right.get("turns", [])):
        return {"parity": False, "field": "turn_count"}
    fields = ("prompt_token_ids", "sampled_ids", "text", "generated_tokens", "position_after")
    for index, (lhs, rhs) in enumerate(zip(left["turns"], right["turns"])):
        for field in fields:
            if lhs.get(field) != rhs.get(field):
                return {"parity": False, "turn": index, "field": field,
                        "left": lhs.get(field), "right": rhs.get(field)}
    return {"parity": True}


def run_canonical(label: str, library: Path, library_sha: str, out: Path) -> dict:
    environment = os.environ.copy()
    environment.update({"MINIMIND_NO_PAUSE": "1", "CANONICAL_LIBRARY": str(library.resolve()),
                        "CANONICAL_EXPECTED_SHA256": library_sha,
                        "CANONICAL_RESULTS": str((out / f"canonical-{label}").resolve())})
    command = ["cmd", "/c", "compare_cpu\\Canonical_TokS.cmd"]
    completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                               text=True, encoding="utf-8")
    (out / f"canonical-{label}.log").write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr,
                                                   encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"canonical {label} guard failed with code {completed.returncode}: {completed.stderr}")
    summary_path = out / f"canonical-{label}" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hashes = summary["hashes_identical_across_runs"]
    if not hashes["sampled_ids"] or not hashes["text"]:
        raise RuntimeError(f"canonical {label} output is not deterministic")
    for row in summary["runs"]:
        if row["sampled_ids_sha256"] != CANONICAL_SAMPLED_IDS_SHA256 or row["text_sha256"] != CANONICAL_TEXT_SHA256:
            raise RuntimeError(f"canonical {label} hash mismatch")
    return summary


def _identity(candidate_sha: str, source_commit: str) -> dict:
    return {
        "schema": "cpu-e6/paired-v1",
        "source_commit": source_commit,
        "model_sha256": sha(MODEL),
        "tokenizer_sha256": {name: sha(TOKENIZER / name)
                              for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
        "baseline_library_sha256": EXPECTED_BASELINE_SHA256,
        "candidate_library_sha256": candidate_sha,
        "threads": 1, "cpus": [0], "row_weights": [1], "profiler": False,
        "policy": {"temperature": .9, "top_k": 50, "top_p": .85, "seed": 0,
                   "system": "official MiniMind chat system"},
        "attention_storage": {"representations": "q_proj,k_proj,v_proj,o_proj across 8 layers",
                               "requirement": "FP32->FP16->FP32 bit exact", "expected_compact_bytes": 28_311_552},
        "pair_count": PAIR_COUNT, "forced_prefix_lengths": [256, 1792],
        "continuation": CONTINUATION, "repetitive_prompts": REPETITIVE_PROMPTS,
        "timed_operation": "prefill/decode inference only; loading and FP16 preparation reported separately",
    }


def _summary_markdown(identity: dict, records: list[dict], canonical: dict) -> str:
    lines = ["# CPU-E6: exact FP16 attention storage integration", "",
             "Status: **PASS**", "", f"- Source commit: `{identity['source_commit']}`",
             f"- CPU-E4 baseline DLL SHA-256: `{identity['baseline_library_sha256']}`",
             f"- CPU-E6 candidate DLL SHA-256: `{identity['candidate_library_sha256']}`",
             "- CPU 0, one participant, profiler off; three alternating pairs per workload.", "",
             "## Forced-prefix results", "",
             "| Prefix | A prefill (s) | B prefill (s) | A decode tok/s | B decode tok/s | Parity |", "|---:|---:|---:|---:|---:|:---:|"]
    for record in records:
        if record["workload"] != "forced":
            continue
        a, b = record["entries"]["A"], record["entries"]["B"]
        lines.append(f"| {record['length']} | {a['prefill_seconds']:.3f} | {b['prefill_seconds']:.3f} | "
                     f"{a['output_decode_tokens_per_second']:.2f} | {b['output_decode_tokens_per_second']:.2f} | "
                     f"{'PASS' if record['comparison_result']['parity'] else 'FAIL'} |")
    lines += ["", "## Repetitive CPU-E4 workload", "", "| A output decode tok/s | B output decode tok/s | A verify calls | B verify calls | Parity |", "|---:|---:|---:|---:|:---:|"]
    for record in records:
        if record["workload"] != "repetitive":
            continue
        a, b = record["entries"]["A"], record["entries"]["B"]
        lines.append(f"| {a['output_decode_tokens_per_second']:.2f} | {b['output_decode_tokens_per_second']:.2f} | "
                     f"{int(a['speculative_decode']['verify_calls'])} | {int(b['speculative_decode']['verify_calls'])} | "
                     f"{'PASS' if record['comparison_result']['parity'] else 'FAIL'} |")
    lines += ["", "## Canonical guard", "", "The A and B runs both require the canonical sampled-ID and text hashes.", "",
              "| Side | Median output tok/s | Generated tokens | Verify calls |", "|:---|---:|---:|---:|"]
    for label, summary in canonical.items():
        rates = [row["output_tokens_per_second"] for row in summary["runs"]]
        verify = [row["speculative_decode"].get("verify_calls", 0) for row in summary["runs"]]
        lines.append(f"| {label} | {statistics.median(rates):.2f} | "
                     f"{summary['runs'][0]['generated_tokens']} | {int(statistics.median(verify))} |")
    lines += ["", "FP16 preparation and compact-memory values are recorded in every raw receipt and excluded from timed inference.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    args = parser.parse_args(argv)
    if args.pairs != PAIR_COUNT:
        raise ValueError(f"CPU-E6 requires exactly {PAIR_COUNT} pairs")
    for path in (MODEL, TOKENIZER / "tokenizer.json", BASELINE, CANDIDATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_sha = sha(BASELINE)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"CPU-E6 refuses CPU-E4 baseline: expected {EXPECTED_BASELINE_SHA256}, found {baseline_sha}")
    candidate_sha = sha(CANDIDATE)
    benchmark = _load_cpu_r1_benchmark()
    args.out.mkdir(parents=True, exist_ok=True)
    identity = _identity(candidate_sha, subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    records = []
    for length in (256, 1792):
        for number in range(1, PAIR_COUNT + 1):
            order = ("A", "B") if number % 2 else ("B", "A")
            entries = {}
            for label in order:
                entries[label] = run_forced(BASELINE if label == "A" else CANDIDATE, length, benchmark)
            comparison = compare_forced(entries["A"], entries["B"])
            if not comparison["parity"]:
                raise RuntimeError(f"CPU-E6 forced parity failed for prefix {length}, pair {number}: {comparison}")
            record = {"identity": identity, "workload": "forced", "length": length,
                      "pair": number, "order": list(order), "entries": entries,
                      "comparison_result": comparison, "status": "PASS"}
            write_json(args.out / f"forced-{length}-pair{number:02d}.json", record)
            records.append(record)
    for number in range(1, PAIR_COUNT + 1):
        order = ("A", "B") if number % 2 else ("B", "A")
        entries = {}
        for label in order:
            entries[label] = run_repetitive(BASELINE if label == "A" else CANDIDATE)
        comparison = compare_chat(entries["A"], entries["B"])
        if not comparison["parity"]:
            raise RuntimeError(f"CPU-E6 repetitive parity failed for pair {number}: {comparison}")
        record = {"identity": identity, "workload": "repetitive", "pair": number,
                  "order": list(order), "entries": entries,
                  "comparison_result": comparison, "status": "PASS"}
        write_json(args.out / f"repetitive-pair{number:02d}.json", record)
        records.append(record)
    canonical = {"A": run_canonical("A", BASELINE, baseline_sha, args.out),
                 "B": run_canonical("B", CANDIDATE, candidate_sha, args.out)}
    write_json(args.out / "identity.json", identity)
    write_json(args.out / "canonical.json", canonical)
    write_json(args.out / "summary.json", {"identity": identity, "status": "PASS",
                                            "records": records, "canonical": canonical})
    (args.out / "RESULTS.md").write_text(_summary_markdown(identity, records, canonical), encoding="utf-8")
    print(json.dumps({"status": "PASS", "candidate_sha256": candidate_sha,
                      "out": str(args.out.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
