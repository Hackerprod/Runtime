"""Paired CPU-U1.1 recheck using the accepted CPU-R6 DLL byte identity."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
TOKENIZER = ROOT / "checkpoints" / "minimind-3-hf"
FROZEN_LIBRARY = ROOT / "native_cpu" / "validation" / "cpu-u1.1" / "reference" / "cpu-r6-accepted.dll"
FROZEN_IDENTITY = ROOT / "native_cpu" / "benchmarks" / "cpu-r6" / "build-info.json"
CANDIDATE_LIBRARY = ROOT / "native_cpu" / "build" / "Release" / "minimind_cpu.dll"
PAIR_COUNT = 3
REFERENCE_RUNTIME_COMMIT = "ae90d82e6007574628d496eba898cdffce2b8096"
CANDIDATE_RUNTIME_COMMIT = "06694632cfe4ee5b97013d6a5b2e6fec9fe98f50"
EXPECTED_REFERENCE_SHA256 = "4d598dd21493d2821a10783c4c97671c96ea038b98c3cc82c30172182fdb3672"
EXPECTED_CANDIDATE_SHA256 = "494eb6bc48afe1ecd41f532b6a4c3a2310ef2b8e73408fc8809b32bcf6ebca2a"


def _load_cpu_r1_benchmark():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_benchmark_for_cpu_u1_1", path)
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


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def identity():
    measurement_tree_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "schema": "cpu-u1.1/forced-paired-v1",
        "candidate_runtime_commit": CANDIDATE_RUNTIME_COMMIT,
        "reference_runtime_commit": REFERENCE_RUNTIME_COMMIT,
        "measurement_tree_commit": measurement_tree_commit,
        "model_sha256": sha(MODEL),
        "tokenizer_sha256": {name: sha(TOKENIZER / name) for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
        "reference_library": str(FROZEN_LIBRARY),
        "reference_library_sha256": sha(FROZEN_LIBRARY),
        "reference_identity_sha256": sha(FROZEN_IDENTITY) if FROZEN_IDENTITY.is_file() else None,
        "candidate_library": str(CANDIDATE_LIBRARY),
        "candidate_library_sha256": sha(CANDIDATE_LIBRARY),
        "policy": {"temperature": 0.9, "top_k": 50, "top_p": 0.85, "seed": 0, "system": "official MiniMind chat system"},
        "threads": 1,
        "cpus": [0],
        "profiler": False,
        "candidate_options": {"production_route": True, "launcher_switches": False, "selective_logits": True, "reuse_kv": True, "v_blocked_attention": True, "gqa_k_shared": True, "gqa_v_shared": True, "ffn_f16_storage": True, "ffn_row4": False},
        "reference_options": {"frozen_cpu_r6": True, "configured_via_legacy_client": True, "selective_logits": True, "reuse_kv": True, "v_blocked_attention": True, "gqa_k_shared": True, "gqa_v_shared": True, "ffn_f16_storage": True, "ffn_row4": False},
        "pair_count": PAIR_COUNT,
        "prefix_lengths": [256, 1792],
        "continuation": 32,
    }


def _configure(runtime, options):
    started = time.perf_counter()
    runtime.configure_profile(False)
    runtime.configure_selective_logits(options["selective_logits"])
    runtime.configure_v_blocked_attention(options["v_blocked_attention"])
    runtime.configure_ffn_row4(False)
    storage_started = time.perf_counter()
    runtime.configure_ffn_f16_storage(options["ffn_f16_storage"])
    storage_preparation_seconds = time.perf_counter() - storage_started
    runtime.configure_gqa_k_shared(options["gqa_k_shared"])
    runtime.configure_gqa_v_shared(options["gqa_v_shared"])
    return time.perf_counter() - started, storage_preparation_seconds


def _run_forced_compatible(benchmark, runtime, length, continuation):
    """Run the fixed-token workload without requiring new statistics symbols."""
    import numpy as np

    prefix, suffix = benchmark.forced_ids(length, continuation, runtime.vocab_size)
    runtime.reset()
    runtime.reset_stats()
    before = runtime.stats
    started = time.perf_counter()
    logits = runtime.eval(prefix)
    prefill_seconds = time.perf_counter() - started
    prefill_hash = benchmark.vector_sha(logits)
    middle = runtime.stats
    hashes, durations = [], []
    loop_started = time.perf_counter()
    for token in suffix:
        started = time.perf_counter()
        logits = runtime.eval(np.asarray([token], dtype=np.int32))
        durations.append(time.perf_counter() - started)
        hashes.append(benchmark.vector_sha(logits))
    loop_seconds = time.perf_counter() - loop_started
    after = runtime.stats
    if runtime.position != length + continuation:
        raise RuntimeError("fixed-token position invariant failed")
    phase_stats = {
        "prefill": benchmark.ChatSession._stats_delta(middle, before),
        "decode": benchmark.ChatSession._stats_delta(after, middle),
    }
    lm_head_calls = None
    if isinstance(before, dict) and isinstance(middle, dict) and isinstance(after, dict):
        lm_head_calls = {
            "prefill": middle.get("lm_head_calls", 0) - before.get("lm_head_calls", 0),
            "decode": after.get("lm_head_calls", 0) - middle.get("lm_head_calls", 0),
        }
    decode_seconds = sum(durations)
    return {
        "prefix_token_ids": prefix.tolist(),
        "continuation_token_ids": suffix.tolist(),
        "prefill_logits_sha256": prefill_hash,
        "decode_logits_sha256": hashes,
        "position_after": runtime.position,
        "prefill_tokens_evaluated": length,
        "decode_tokens_evaluated": continuation,
        "prefill_seconds": prefill_seconds,
        "native_generation_seconds": decode_seconds,
        "decode_call_seconds": durations,
        "decode_loop_with_hashing_seconds": loop_seconds,
        "native_seconds": prefill_seconds + decode_seconds,
        "total_seconds": prefill_seconds + decode_seconds,
        "prefill_tokens_per_second": length / prefill_seconds,
        "decode_tokens_per_second": continuation / decode_seconds,
        "lm_head_calls": lm_head_calls,
        "native_phase_stats": phase_stats,
    }


def run_entry(benchmark, profile, workload, *, length=None):
    from native_cpu.tools.native import NativeRuntime

    options = profile["options"]
    load_started = time.perf_counter()
    with NativeRuntime(MODEL, profile["library"], 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        loading_seconds = time.perf_counter() - load_started
        if profile.get("legacy_configure"):
            preparation_seconds, storage_preparation_seconds = _configure(runtime, options)
        else:
            preparation_seconds = 0.0
            storage_preparation_seconds = (runtime.ffn_f16_prepare_ns or 0) / 1e9
        benchmark.warmup(runtime, 1)
        if workload != "forced":
            raise ValueError(f"CPU-U1.1 only supports forced workloads, got {workload!r}")
        data = _run_forced_compatible(benchmark, runtime, length, 32)
        data["profile"] = profile["name"]
        data["library_sha256"] = profile["library_sha256"]
        data["loading_seconds"] = loading_seconds
        data["configuration_seconds"] = preparation_seconds
        data["storage_preparation_seconds"] = storage_preparation_seconds
        data["inference_excludes_configuration"] = True
        data["configured_options"] = dict(options)
        data["effective"] = {
            "threads": runtime.threads,
            "cpus": runtime.cpus,
            "row_weights": runtime.row_weights,
            "selective_logits": runtime.selective_logits,
            "reuse_kv": "not_applicable",
            "v_blocked_attention": runtime.v_blocked_attention,
            "ffn_row4": runtime.ffn_row4,
            "ffn_f16_storage": runtime.ffn_f16_storage,
            "gqa_k_shared": runtime.gqa_k_shared,
            "gqa_v_shared": runtime.gqa_v_shared,
            "diagnostics": False,
        }
        data["ffn_f16_storage_bytes"] = runtime.ffn_f16_storage_bytes
        data["ffn_f16_prepare_ns"] = runtime.ffn_f16_prepare_ns
        return data


def compare(benchmark, left, right, workload):
    return benchmark.compare_entries(left, right, workload)


def make_cases():
    return [{"key": f"cpu-u1.1-forced-{length}", "workload": "forced", "length": length}
            for length in (256, 1792)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=PAIR_COUNT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.pairs != PAIR_COUNT:
        raise ValueError(f"CPU-U1.1 requires exactly {PAIR_COUNT} pairs")
    for path in (MODEL, TOKENIZER / "tokenizer.json", FROZEN_LIBRARY, CANDIDATE_LIBRARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference_sha = sha(FROZEN_LIBRARY)
    if reference_sha != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(f"CPU-U1.1 refuses reference DLL: expected SHA-256 {EXPECTED_REFERENCE_SHA256}, found {reference_sha}")
    candidate_sha = sha(CANDIDATE_LIBRARY)
    if candidate_sha != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(f"CPU-U1.1 refuses candidate DLL: expected SHA-256 {EXPECTED_CANDIDATE_SHA256}, found {candidate_sha}")
    args.out.mkdir(parents=True, exist_ok=True)
    benchmark = _load_cpu_r1_benchmark()
    meta = identity()
    profiles = {
        "A": {"name": "A", "library": FROZEN_LIBRARY, "library_sha256": meta["reference_library_sha256"], "options": meta["reference_options"], "legacy_configure": True},
        "B": {"name": "B", "library": CANDIDATE_LIBRARY, "library_sha256": meta["candidate_library_sha256"], "options": meta["candidate_options"], "legacy_configure": False},
    }
    summaries = []
    for case in make_cases():
        pair_records = []
        for number in range(1, PAIR_COUNT + 1):
            path = args.out / f"{case['key']}-pair{number:02d}.json"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                prior = json.loads(path.read_text(encoding="utf8"))
                if prior.get("identity") != meta or prior.get("case") != case or prior.get("pair") != number:
                    raise ValueError(f"resume identity mismatch: {path}")
                if prior.get("order") != (["A", "B"] if number % 2 else ["B", "A"]):
                    raise ValueError(f"resume order mismatch: {path}")
                if prior.get("status") != "PASS":
                    raise ValueError(f"cannot resume invalid pair: {path}")
                pair_records.append(prior)
                continue
            order = ["A", "B"] if number % 2 else ["B", "A"]
            print(f"PAIR {case['key']} {number}/{PAIR_COUNT} {order}", flush=True)
            entries = [run_entry(benchmark, profiles[label], case["workload"], length=case.get("length")) for label in order]
            comparison = compare(benchmark, entries[0], entries[1], case["workload"])
            record = {"identity": meta, "case": case, "pair": number, "order": order, "entries": entries, "comparison_result": comparison, "status": "PASS" if comparison.get("parity") else "FAIL"}
            write_json(path, record)
            pair_records.append(record)
            if not comparison.get("parity"):
                raise RuntimeError(f"parity failed; preserved {path}")
        summary = {"case": case, "pairs": len(pair_records), "parity": all(p["comparison_result"].get("parity") for p in pair_records), "orders": [p["order"] for p in pair_records]}
        summaries.append(summary)
        write_json(args.out / f"{case['key']}-summary.json", summary)
    write_json(args.out / "summary.json", {"identity": meta, "cases": summaries, "status": "PASS", "final_protocol": True})


if __name__ == "__main__":
    main()
