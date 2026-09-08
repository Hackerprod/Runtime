"""CPU-E4: bounded end-to-end n-gram speculative decode comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
TOKENIZER = ROOT / "checkpoints" / "minimind-3-hf"
BASELINE = ROOT / "native_cpu" / "benchmarks" / "cpu-e2" / "baseline" / "minimind_cpu.dll"
EXPECTED_BASELINE_SHA256 = "3ff9418181d29ad9f7761e2a48866f06759027b4599ef1d024f24ac4cd3acfd5"
PAIR_COUNT = 3
MAX_NEW_TOKENS = 32
ORDINARY_PROMPTS = ["Name one benefit of tests.", "Define latency briefly."]
REPETITIVE_PROMPTS = ["Name one benefit of tests."] * 3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _warmup(runtime):
    runtime.reset()
    runtime.eval(np.asarray([(i * 17 + 11) % runtime.vocab_size for i in range(64)], dtype=np.int32))
    runtime.reset()
    reset_stats = getattr(runtime, "reset_stats", None)
    if reset_stats is not None:
        reset_stats()


def run_workload(library: Path, prompts):
    from native_cpu.tools.chat import ChatSession, DEFAULT_CHAT_SYSTEM, LocalTokenizer
    from native_cpu.tools.native import NativeRuntime

    tokenizer = LocalTokenizer(TOKENIZER)
    with NativeRuntime(MODEL, library, 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        runtime.configure_profile(False)
        _warmup(runtime)
        session = ChatSession(runtime, tokenizer, context_limit=2048,
                              max_new_tokens=MAX_NEW_TOKENS, temperature=0.9,
                              top_k=50, top_p=0.85, seed=0,
                              system=DEFAULT_CHAT_SYSTEM, reuse_kv=True)
        turns = []
        started = time.perf_counter()
        for prompt in prompts:
            result = session.turn(prompt)
            row = dict(result.metrics)
            row.update(user=prompt, text=result.text,
                       sampled_ids=result.sampled_ids,
                       finish_reason=("eos" if result.sampled_ids and
                                      result.sampled_ids[-1] == tokenizer.eos_token_id else "length"),
                       generation_seconds=result.decode_seconds,
                       decode_seconds=result.decode_seconds)
            if runtime.position != len(session.cached_token_ids):
                raise RuntimeError("cache position mismatch after workload turn")
            turns.append(row)
        elapsed = time.perf_counter() - started

    numeric = ("prefill_seconds", "native_generation_seconds", "sampling_seconds",
               "prompt_prepare_seconds", "decode_seconds", "total_seconds",
               "decode_tokens_evaluated", "generated_tokens")
    totals = {key: sum(float(turn.get(key, 0)) for turn in turns) for key in numeric}
    totals["decode_tokens_per_second"] = (totals["decode_tokens_evaluated"] /
                                           totals["native_generation_seconds"]
                                           if totals["native_generation_seconds"] else None)
    totals["conversation_seconds"] = elapsed
    totals["turns"] = turns
    telemetry_keys = ("proposed_blocks", "proposed_tokens", "accepted_tokens",
                      "verify_calls", "first_position_failures", "accepted_length_total",
                      "lookup_seconds", "verification_seconds")
    telemetry = {key: sum(float(turn.get("speculative_decode", {}).get(key, 0))
                          for turn in turns) for key in telemetry_keys}
    telemetry["mean_accepted_length"] = (telemetry["accepted_length_total"] /
                                          telemetry["proposed_blocks"]
                                          if telemetry["proposed_blocks"] else 0.0)
    totals["speculative_decode"] = telemetry
    return totals


def compare(left, right):
    a, b = left["turns"], right["turns"]
    if len(a) != len(b) or not a:
        return {"parity": False, "reason": "turn count"}
    fields = ("prompt_token_ids", "sampled_ids", "text", "finish_reason",
              "generated_tokens", "position_after")
    for index, (x, y) in enumerate(zip(a, b)):
        for field in fields:
            if x.get(field) != y.get(field):
                return {"parity": False, "turn": index, "field": field,
                        "left": x.get(field), "right": y.get(field)}
    return {"parity": True}


def summarize(pairs):
    out = {"pairs": len(pairs), "parity": all(p["comparison_result"]["parity"] for p in pairs),
           "profiles": {}, "paired_differences": []}
    for label in ("A", "B"):
        rows = [p["entries"][label] for p in pairs]
        out["profiles"][label] = {
            key: {"count": len(values), "median": statistics.median(values),
                  "min": min(values), "max": max(values)}
            for key in ("prefill_seconds", "native_generation_seconds", "decode_seconds",
                        "total_seconds", "decode_tokens_per_second")
            if (values := [row[key] for row in rows]) and all(value is not None for value in values)
        }
    for pair in pairs:
        a, b = pair["entries"]["A"], pair["entries"]["B"]
        out["paired_differences"].append({
            "pair": pair["pair"],
            "decode_time_reduction": 1.0 - b["decode_seconds"] / a["decode_seconds"],
            "total_time_reduction": 1.0 - b["total_seconds"] / a["total_seconds"],
            "decode_tok_s_change": (b["decode_tokens_per_second"] /
                                     a["decode_tokens_per_second"] - 1.0),
            "candidate_speculative": b.get("speculative_decode", {}),
        })
    out["median_decode_time_reduction"] = statistics.median(
        row["decode_time_reduction"] for row in out["paired_differences"])
    out["median_total_time_reduction"] = statistics.median(
        row["total_time_reduction"] for row in out["paired_differences"])
    out["favorable_decode_pairs"] = sum(
        row["decode_time_reduction"] > 0 for row in out["paired_differences"])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (MODEL, TOKENIZER / "tokenizer.json", BASELINE, args.candidate):
        if not path.exists():
            raise FileNotFoundError(path)
    baseline_sha = sha(BASELINE)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"CPU-E4 refuses CPU-E2 baseline: expected {EXPECTED_BASELINE_SHA256}, found {baseline_sha}")
    candidate_sha = sha(args.candidate)
    identity = {
        "schema": "cpu-e4/ngram-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_sha256": sha(MODEL),
        "tokenizer_path": str(TOKENIZER.relative_to(ROOT)),
        "baseline_library_sha256": baseline_sha,
        "candidate_library_sha256": candidate_sha,
        "threads": 1, "cpus": [0], "row_weights": [1], "profiler": False,
        "max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.9, "top_k": 50,
        "top_p": 0.85, "seed": 0,
        "policy": {"key_tokens": 12, "proposal_tokens": 4,
                    "search": "most recent earlier non-overlapping match"},
        "timed_operation": "end-to-end decode including lookup, sampling, verification and rollbacks",
    }
    workloads = {"ordinary": ORDINARY_PROMPTS, "repetitive": REPETITIVE_PROMPTS}
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name, prompts in workloads.items():
        pairs = []
        for number in range(1, PAIR_COUNT + 1):
            order = ("A", "B") if number % 2 else ("B", "A")
            print(f"PAIR {name} {number}/{PAIR_COUNT} {order}", flush=True)
            entries = {}
            for label in order:
                entries[label] = run_workload(BASELINE if label == "A" else args.candidate, prompts)
                entries[label]["library_sha256"] = baseline_sha if label == "A" else candidate_sha
            comparison = compare(entries["A"], entries["B"])
            if not comparison["parity"]:
                raise RuntimeError(f"CPU-E4 parity failed for {name} pair {number}: {comparison}")
            record = {"identity": identity, "workload": name, "prompts": prompts,
                      "pair": number, "order": list(order), "entries": entries,
                      "comparison_result": comparison, "status": "PASS"}
            write_json(args.out / f"cpu-e4-{name}-pair{number:02d}.json", record)
            pairs.append(record)
        summary = {"identity": identity, "workload": name, "summary": summarize(pairs),
                   "status": "PASS"}
        write_json(args.out / f"cpu-e4-{name}-summary.json", summary)
        summaries.append(summary)
    write_json(args.out / "summary.json", {"identity": identity, "workloads": summaries, "status": "PASS"})


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
