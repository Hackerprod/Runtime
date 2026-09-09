"""CPU-E8: exact FP16 LM-head integration against the CPU-E6 baseline."""
from __future__ import annotations

import hashlib
import importlib.util
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
BASELINE = ROOT / "native_cpu" / "benchmarks" / "cpu-e6" / "baseline" / "minimind_cpu.dll"
CANDIDATE = ROOT / "native_cpu" / "build-e8" / "Release" / "minimind_cpu.dll"
EXPECTED_BASELINE_SHA256 = "ab464cc1c91aedd3b33bf995fef82657e2316123979e6f1d04d15bb4eace72fd"
PAIR_COUNT = 3
CONTINUATION = 32
LM_HEAD_BYTES = 9_830_400
CANONICAL_PROMPT = "Could you tell me about programming languages?"
CANONICAL_SAMPLED_IDS_SHA256 = "de27dbd287b02a36f6116b22152359f4b5d1e13c9d0cf3f454512d40a6dcb8b6"
CANONICAL_TEXT_SHA256 = "bbab7af92f7db6781fe097645898ee98d3409f9367a9d764883637489bace770"


def _load_cpu_r1_benchmark():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_for_cpu_e8", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_canonical_guard():
    path = ROOT / "compare_cpu" / "canonical_tok_s.py"
    spec = importlib.util.spec_from_file_location("canonical_guard_for_cpu_e8", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
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


def _lm_head(runtime) -> dict:
    return {
        "storage": bool(getattr(runtime, "lm_head_f16_storage", False)),
        "compact_bytes": getattr(runtime, "lm_head_f16_storage_bytes", None),
        "prepare_ns": getattr(runtime, "lm_head_f16_prepare_ns", None),
    }


def _warmup(runtime):
    benchmark = _load_cpu_r1_benchmark()
    benchmark.warmup(runtime, 1)


def run_forced(library: Path, length: int) -> dict:
    from native_cpu.tools.native import NativeRuntime

    benchmark = _load_cpu_r1_benchmark()
    prefix, suffix = benchmark.forced_ids(length, CONTINUATION, 6400)
    load_started = time.perf_counter()
    with NativeRuntime(MODEL, library, 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        loading_seconds = time.perf_counter() - load_started
        runtime.configure_profile(False)
        _warmup(runtime)
        runtime.reset()
        runtime.reset_stats()
        before_lm = runtime.lm_head_calls
        started = time.perf_counter()
        prefill_logits = runtime.eval(prefix)
        prefill_seconds = time.perf_counter() - started
        decode_hashes = []
        decode_seconds = 0.0
        for token in suffix:
            started = time.perf_counter()
            logits = runtime.eval(np.asarray([token], dtype=np.int32))
            decode_seconds += time.perf_counter() - started
            decode_hashes.append(vector_sha(logits))
        after_lm = runtime.lm_head_calls
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
            "lm_head_calls": after_lm - before_lm,
            "lm_head": _lm_head(runtime),
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
    prompts = ["Name one benefit of tests."] * 3
    load_started = time.perf_counter()
    with NativeRuntime(MODEL, library, 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        loading_seconds = time.perf_counter() - load_started
        runtime.configure_profile(False)
        _warmup(runtime)
        session = ChatSession(runtime, tokenizer, context_limit=2048, max_new_tokens=32,
                              temperature=.9, top_k=50, top_p=.85, seed=0,
                              system=DEFAULT_CHAT_SYSTEM, reuse_kv=True)
        turns = []
        started = time.perf_counter()
        for prompt in prompts:
            result = session.turn(prompt)
            turns.append({
                "user": prompt, "prompt_token_ids": result.metrics.get("prompt_token_ids"),
                "sampled_ids": result.sampled_ids, "text": result.text,
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
            "library_sha256": sha(library), "turns": turns,
            "conversation_seconds": wall_seconds,
            "prefill_seconds": sum(turn["prefill_seconds"] for turn in turns),
            "decode_seconds": decode_seconds,
            "total_seconds": sum(turn["total_seconds"] for turn in turns),
            "generated_tokens": sum(turn["generated_tokens"] for turn in turns),
            "output_decode_tokens_per_second": (sum(turn["generated_tokens"] for turn in turns) / decode_seconds
                                                 if decode_seconds else None),
            "loading_seconds": loading_seconds,
            "lm_head_calls": runtime.lm_head_calls,
            "lm_head": _lm_head(runtime),
            "speculative_decode": _aggregate_speculative(turns),
        }


def compare_forced(left: dict, right: dict) -> dict:
    fields = ("prefix_token_ids", "continuation_token_ids", "prefill_logits_sha256",
              "decode_logits_sha256", "position_after")
    for field in fields:
        if left.get(field) != right.get(field):
            return {"parity": False, "field": field, "left": left.get(field), "right": right.get(field)}
    return {"parity": True}


def compare_repetitive(left: dict, right: dict) -> dict:
    if len(left.get("turns", [])) != len(right.get("turns", [])):
        return {"parity": False, "field": "turn_count"}
    fields = ("prompt_token_ids", "sampled_ids", "text", "generated_tokens", "position_after")
    for index, (lhs, rhs) in enumerate(zip(left["turns"], right["turns"])):
        for field in fields:
            if lhs.get(field) != rhs.get(field):
                return {"parity": False, "turn": index, "field": field,
                        "left": lhs.get(field), "right": rhs.get(field)}
    return {"parity": True}


def _canonical_pair(number: int, order: tuple[str, str], libraries: dict[str, Path], output: Path) -> dict:
    guard = _load_canonical_guard()
    entries = {}
    for label in order:
        run_dir = output / f"canonical-pair{number:02d}-{label}"
        entries[label], _ = guard.run_once(number, libraries[label], run_dir)
        entries[label]["library_sha256"] = sha(libraries[label])
        if entries[label].get("generated_tokens") != 256:
            raise RuntimeError(f"canonical {label} generated an unexpected token count")
        if entries[label].get("sampled_ids") is None or entries[label].get("text") is None:
            raise RuntimeError(f"canonical {label} did not return deterministic output")
        ids_hash = hashlib.sha256(json.dumps(entries[label]["sampled_ids"], separators=(",", ":")).encode()).hexdigest()
        text_hash = hashlib.sha256(str(entries[label]["text"]).encode("utf-8")).hexdigest()
        if ids_hash != CANONICAL_SAMPLED_IDS_SHA256 or text_hash != CANONICAL_TEXT_SHA256:
            raise RuntimeError(f"canonical {label} output hash mismatch")
    comparison = compare_repetitive({"turns": [{"sampled_ids": entries["A"].get("sampled_ids"), "text": entries["A"].get("text"),
                                                  "generated_tokens": entries["A"].get("generated_tokens"),
                                                  "position_after": entries["A"].get("position_after"),
                                                  "prompt_token_ids": entries["A"].get("prompt_token_ids")} ]},
                                     {"turns": [{"sampled_ids": entries["B"].get("sampled_ids"), "text": entries["B"].get("text"),
                                                  "generated_tokens": entries["B"].get("generated_tokens"),
                                                  "position_after": entries["B"].get("position_after"),
                                                  "prompt_token_ids": entries["B"].get("prompt_token_ids")} ]})
    record = {"workload": "canonical", "pair": number, "order": list(order),
              "entries": entries, "comparison_result": comparison, "status": "PASS"}
    write_json(output / f"canonical-pair{number:02d}.json", record)
    if not comparison["parity"]:
        raise RuntimeError(f"canonical parity failed: {comparison}")
    return record


def _evaluate(records: list[dict], canonical: list[dict]) -> dict:
    forced = {}
    for length in (256, 1792):
        rows = [record for record in records if record["workload"] == "forced" and record["length"] == length]
        reductions = [1.0 - row["entries"]["B"]["decode_seconds"] / row["entries"]["A"]["decode_seconds"] for row in rows]
        forced[length] = {
            "pairs": len(rows), "parity": all(row["comparison_result"]["parity"] for row in rows),
            "decode_time_reductions": reductions,
            "median_decode_time_reduction": statistics.median(reductions) if reductions else None,
            "favorable_pairs": sum(value > 0.0 for value in reductions),
        }
    repetitive_rows = [record for record in records if record["workload"] == "repetitive"]
    repetitive_changes = [row["entries"]["B"]["output_decode_tokens_per_second"] /
                          row["entries"]["A"]["output_decode_tokens_per_second"] - 1.0
                          for row in repetitive_rows]
    canonical_changes = [row["entries"]["B"]["output_tokens_per_second"] /
                         row["entries"]["A"]["output_tokens_per_second"] - 1.0
                         for row in canonical]
    parity = all(row["comparison_result"]["parity"] for row in records + canonical)
    criteria = {
        "parity": parity,
        "forced_256_decode_improves": forced[256]["median_decode_time_reduction"] is not None and forced[256]["median_decode_time_reduction"] > 0.0,
        "forced_1792_decode_improves": forced[1792]["median_decode_time_reduction"] is not None and forced[1792]["median_decode_time_reduction"] > 0.0,
        "canonical_median_output_improves": bool(canonical_changes) and statistics.median(canonical_changes) > 0.0,
        "repetitive_no_reproducible_regression": bool(repetitive_changes) and statistics.median(repetitive_changes) >= 0.0,
    }
    return {"accepted": all(criteria.values()), "criteria": criteria,
            "forced": forced, "canonical_output_changes": canonical_changes,
            "repetitive_output_changes": repetitive_changes}


def _summarize(records: list[dict], canonical: list[dict], identity: dict, evaluation: dict) -> str:
    lines = ["# CPU-E8: exact FP16 LM-head integration", "", f"Status: **{'PASS' if evaluation['accepted'] else 'REJECTED'}**",
             "", f"- Source commit: `{identity['source_commit']}`",
             f"- CPU-E6 baseline DLL SHA-256: `{identity['baseline_library_sha256']}`",
             f"- CPU-E8 candidate DLL SHA-256: `{identity['candidate_library_sha256']}`",
             "- CPU 0, one participant, profiler off; three alternating pairs per workload.", "",
             "## Forced-prefix results", "",
             "| Prefix | A prefill (s) | B prefill (s) | A decode tok/s | B decode tok/s | Parity |",
             "|---:|---:|---:|---:|---:|:---:|"]
    for record in records:
        if record["workload"] != "forced":
            continue
        a, b = record["entries"]["A"], record["entries"]["B"]
        lines.append(f"| {record['length']} | {a['prefill_seconds']:.3f} | {b['prefill_seconds']:.3f} | "
                     f"{a['output_decode_tokens_per_second']:.2f} | {b['output_decode_tokens_per_second']:.2f} | "
                     f"{'PASS' if record['comparison_result']['parity'] else 'FAIL'} |")
    lines += ["", "## Repetitive speculative workload", "",
              "| A decode tok/s | B decode tok/s | A verify calls | B verify calls | A accepted | B accepted | Parity |",
              "|---:|---:|---:|---:|---:|---:|:---:|"]
    for record in records:
        if record["workload"] != "repetitive":
            continue
        a, b = record["entries"]["A"], record["entries"]["B"]
        lines.append(f"| {a['output_decode_tokens_per_second']:.2f} | {b['output_decode_tokens_per_second']:.2f} | "
                     f"{int(a['speculative_decode']['verify_calls'])} | {int(b['speculative_decode']['verify_calls'])} | "
                     f"{int(a['speculative_decode']['accepted_tokens'])} | {int(b['speculative_decode']['accepted_tokens'])} | "
                     f"{'PASS' if record['comparison_result']['parity'] else 'FAIL'} |")
    lines += ["", "## Canonical A/B pairs", "",
              "| Pair | A output tok/s | B output tok/s | A verify | B verify | Parity |",
              "|---:|---:|---:|---:|---:|:---:|"]
    for record in canonical:
        a, b = record["entries"]["A"], record["entries"]["B"]
        lines.append(f"| {record['pair']} | {a['output_tokens_per_second']:.2f} | {b['output_tokens_per_second']:.2f} | "
                     f"{int((a.get('speculative_decode') or {}).get('verify_calls', 0))} | "
                     f"{int((b.get('speculative_decode') or {}).get('verify_calls', 0))} | "
                     f"{'PASS' if record['comparison_result']['parity'] else 'FAIL'} |")
    lines += ["", f"LM-head compact representation: {LM_HEAD_BYTES} bytes; preparation is reported separately and excluded from timed inference.", "",
              "## Verdict", "", f"- Accepted for production: **{'yes' if evaluation['accepted'] else 'no'}**",
              f"- Criteria: `{json.dumps(evaluation['criteria'], sort_keys=True)}`", "",
              "A rejected result preserves the raw receipts but does not change the CPU-E6 production baseline.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (MODEL, TOKENIZER / "tokenizer.json", BASELINE, CANDIDATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_sha = sha(BASELINE)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"CPU-E8 refuses CPU-E6 baseline: expected {EXPECTED_BASELINE_SHA256}, found {baseline_sha}")
    candidate_sha = sha(CANDIDATE)
    identity = {
        "schema": "cpu-e8/lm-head-f16-v1", "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_sha256": sha(MODEL), "tokenizer_sha256": sha(TOKENIZER / "tokenizer.json"),
        "baseline_library_sha256": baseline_sha, "candidate_library_sha256": candidate_sha,
        "threads": 1, "cpus": [0], "row_weights": [1], "profiler": False,
        "policy": {"temperature": .9, "top_k": 50, "top_p": .85, "seed": 0},
        "lm_head": {"tensor": "model.embed_tokens.weight", "shape": [6400, 768],
                     "exact_roundtrip": True, "compact_bytes": LM_HEAD_BYTES,
                     "embedding_lookup": "FP32"},
        "pair_count": PAIR_COUNT, "forced_prefix_lengths": [256, 1792],
        "continuation": CONTINUATION, "timed_operation": "inference only; preparation reported separately",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    libraries = {"A": BASELINE, "B": CANDIDATE}
    records = []
    for length in (256, 1792):
        for number in range(1, PAIR_COUNT + 1):
            order = ("A", "B") if number % 2 else ("B", "A")
            entries = {label: run_forced(libraries[label], length) for label in order}
            comparison = compare_forced(entries["A"], entries["B"])
            record = {"identity": identity, "workload": "forced", "length": length,
                      "pair": number, "order": list(order), "entries": entries,
                      "comparison_result": comparison, "status": "PASS"}
            write_json(args.out / f"forced-{length}-pair{number:02d}.json", record)
            if not comparison["parity"]:
                raise RuntimeError(f"forced parity failed for {length}: {comparison}")
            records.append(record)
    for number in range(1, PAIR_COUNT + 1):
        order = ("A", "B") if number % 2 else ("B", "A")
        entries = {label: run_repetitive(libraries[label]) for label in order}
        comparison = compare_repetitive(entries["A"], entries["B"])
        record = {"identity": identity, "workload": "repetitive", "pair": number,
                  "order": list(order), "entries": entries,
                  "comparison_result": comparison, "status": "PASS"}
        write_json(args.out / f"repetitive-pair{number:02d}.json", record)
        if not comparison["parity"]:
            raise RuntimeError(f"repetitive parity failed: {comparison}")
        records.append(record)
    canonical = []
    for number in range(1, PAIR_COUNT + 1):
        order = ("A", "B") if number % 2 else ("B", "A")
        canonical.append(_canonical_pair(number, order, libraries, args.out))
    evaluation = _evaluate(records, canonical)
    identity["verdict"] = "accepted" if evaluation["accepted"] else "rejected"
    write_json(args.out / "identity.json", identity)
    write_json(args.out / "summary.json", {"identity": identity, "status": identity["verdict"],
                                            "evaluation": evaluation, "records": records, "canonical": canonical})
    (args.out / "RESULTS.md").write_text(_summarize(records, canonical, identity, evaluation), encoding="utf-8")
    print(json.dumps({"status": identity["verdict"], "candidate_sha256": candidate_sha,
                      "out": str(args.out.resolve()), "evaluation": evaluation}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
