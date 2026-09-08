"""Bounded CPU-E2 prefill comparison against the pinned CPU-U1.2 DLL."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
REFERENCE = ROOT / "native_cpu" / "benchmarks" / "cpu-u1.2" / "baseline" / "minimind_cpu.dll"
EXPECTED_REFERENCE_SHA256 = "5956ff064ad8e8fd1ff502d18626ce19240682faa55c584cb3d6eaad1775bb3a"
PAIR_COUNT = 3


def _r1_helpers():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_helpers_for_cpu_e2", path)
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


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ordered_pair(number: int):
    return ["A", "B"] if number % 2 else ["B", "A"]


def run_entry(helpers, library: Path, label: str, length: int):
    from native_cpu.tools.native import NativeRuntime

    started = time.perf_counter()
    with NativeRuntime(MODEL, library, 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        loading_seconds = time.perf_counter() - started
        helpers.warmup(runtime, 1)
        data = helpers.run_forced(runtime, length, 32, False)
        data.update(
            profile=label,
            library_sha256=sha(library),
            loading_seconds=loading_seconds,
            storage_preparation_seconds=(runtime.ffn_f16_prepare_ns or 0) / 1e9,
            ffn_f16_storage_bytes=runtime.ffn_f16_storage_bytes,
            effective={
                "threads": runtime.threads,
                "cpus": runtime.cpus,
                "row_weights": runtime.row_weights,
                "selective_logits": runtime.selective_logits,
                "v_blocked_attention": runtime.v_blocked_attention,
                "ffn_f16_storage": runtime.ffn_f16_storage,
                "gqa_k_shared": runtime.gqa_k_shared,
                "gqa_v_shared": runtime.gqa_v_shared,
                "diagnostics": False,
            },
        )
        return data


def equal_outputs(left: dict, right: dict) -> bool:
    return all(left.get(name) == right.get(name) for name in (
        "prefix_token_ids", "continuation_token_ids", "prefill_logits_sha256",
        "decode_logits_sha256", "position_after", "prefill_tokens_evaluated",
        "decode_tokens_evaluated",
    ))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for path in (MODEL, REFERENCE, args.candidate):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference_sha256 = sha(REFERENCE)
    if reference_sha256 != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(f"CPU-E2 refuses reference DLL: expected {EXPECTED_REFERENCE_SHA256}, found {reference_sha256}")
    candidate_sha256 = sha(args.candidate)
    helpers = _r1_helpers()
    args.out.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "cpu-e2/prefill-x4-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_sha256": sha(MODEL),
        "reference_library_sha256": reference_sha256,
        "candidate_library_sha256": candidate_sha256,
        "threads": 1, "cpus": [0], "row_weights": [1], "profiler": False,
        "prefix_lengths": [256, 1792], "continuation": 32, "pairs": PAIR_COUNT,
        "route": "automatic prefill groups of four; decode remains step()",
    }
    summaries = []
    for length in (256, 1792):
        case = {"key": f"cpu-e2-prefill-x4-{length}", "length": length}
        pairs = []
        for number in range(1, PAIR_COUNT + 1):
            path = args.out / f"{case['key']}-pair{number:02d}.json"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["identity"] != identity or record["pair"] != number:
                    raise ValueError(f"resume identity mismatch: {path}")
                pairs.append(record)
                continue
            order = ordered_pair(number)
            print(f"PAIR {case['key']} {number}/{PAIR_COUNT} {order}", flush=True)
            entries = []
            for label in order:
                library = REFERENCE if label == "A" else args.candidate
                entries.append(run_entry(helpers, library, label, length))
            comparison = {"parity": equal_outputs(entries[0], entries[1])}
            if not comparison["parity"]:
                raise RuntimeError(f"CPU-E2 parity failed on pair {number}")
            record = {"identity": identity, "case": case, "pair": number,
                      "order": order, "entries": entries,
                      "comparison_result": comparison, "status": "PASS"}
            write_json(path, record)
            pairs.append(record)
        a = [next(e for e in p["entries"] if e["profile"] == "A") for p in pairs]
        b = [next(e for e in p["entries"] if e["profile"] == "B") for p in pairs]
        def median(rows, key): return statistics.median(row[key] for row in rows)
        paired = [{
            "pair": p["pair"],
            "prefill_seconds_reduction": 1.0 - next(e for e in p["entries"] if e["profile"] == "B")["prefill_seconds"] / next(e for e in p["entries"] if e["profile"] == "A")["prefill_seconds"],
            "decode_seconds_reduction": 1.0 - next(e for e in p["entries"] if e["profile"] == "B")["native_generation_seconds"] / next(e for e in p["entries"] if e["profile"] == "A")["native_generation_seconds"],
        } for p in pairs]
        summary = {
            "case": case, "pairs": PAIR_COUNT, "parity": True,
            "profiles": {label: {
                "prefill_seconds": median(rows, "prefill_seconds"),
                "native_generation_seconds": median(rows, "native_generation_seconds"),
                "total_seconds": median(rows, "total_seconds"),
                "prefill_tokens_per_second": median(rows, "prefill_tokens_per_second"),
                "decode_tokens_per_second": median(rows, "decode_tokens_per_second"),
            } for label, rows in (("A", a), ("B", b))},
            "paired": paired,
            "favorable_prefill_pairs": sum(x["prefill_seconds_reduction"] > 0 for x in paired),
            "favorable_decode_pairs": sum(x["decode_seconds_reduction"] > 0 for x in paired),
        }
        write_json(args.out / f"{case['key']}-summary.json", summary)
        summaries.append(summary)
    write_json(args.out / "summary.json", {"identity": identity, "cases": summaries, "status": "PASS"})


if __name__ == "__main__":
    main()

