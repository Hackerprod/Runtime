"""CPU-E3: bounded comparison of four-step decode and verify_x4."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
BASELINE = ROOT / "native_cpu" / "benchmarks" / "cpu-e2" / "baseline" / "minimind_cpu.dll"
EXPECTED_BASELINE_SHA256 = "3ff9418181d29ad9f7761e2a48866f06759027b4599ef1d024f24ac4cd3acfd5"
PAIR_COUNT = 3


def _helpers():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_helpers_for_cpu_e3", path)
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


def vector_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sequential(runtime, tokens: np.ndarray) -> np.ndarray:
    return np.stack([runtime.eval(np.asarray([int(token)], dtype=np.int32)) for token in tokens])


def run_entry(helpers, library: Path, label: str, length: int):
    from native_cpu.tools.native import NativeRuntime

    prefix, continuation = helpers.forced_ids(length, 4, 6400)
    started = time.perf_counter()
    with NativeRuntime(MODEL, library, 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        loading_seconds = time.perf_counter() - started
        runtime.reset()
        prepare_started = time.perf_counter()
        runtime.eval(prefix)
        preparation_seconds = time.perf_counter() - prepare_started

        # Warm the measured operation, then reconstruct the same prepared state.
        if label == "A":
            sequential(runtime, continuation)
        else:
            runtime.verify_x4(continuation)
        runtime.reset()
        runtime.eval(prefix)

        measured_started = time.perf_counter()
        if label == "A":
            logits = sequential(runtime, continuation)
        else:
            logits = runtime.verify_x4(continuation)
        measured_seconds = time.perf_counter() - measured_started
        return {
            "profile": label,
            "library_sha256": sha(library),
            "loading_seconds": loading_seconds,
            "context_preparation_seconds": preparation_seconds,
            "verification_seconds": measured_seconds,
            "logits_sha256": [vector_sha(row) for row in logits],
            "position_after": runtime.position,
            "tokens": continuation.tolist(),
            "logits_shape": list(logits.shape),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for path in (MODEL, BASELINE, args.candidate):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_sha256 = sha(BASELINE)
    if baseline_sha256 != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"CPU-E3 refuses CPU-E2 baseline: expected {EXPECTED_BASELINE_SHA256}, found {baseline_sha256}")
    candidate_sha256 = sha(args.candidate)
    helpers = _helpers()
    args.out.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "cpu-e3/verify-x4-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_sha256": sha(MODEL),
        "baseline_library_sha256": baseline_sha256,
        "candidate_library_sha256": candidate_sha256,
        "threads": 1,
        "cpus": [0],
        "row_weights": [1],
        "profiler": False,
        "prefix_lengths": [256, 1792],
        "pairs": PAIR_COUNT,
        "verified_tokens": 4,
        "timed_operation": "four complete logits and KV states; context preparation excluded",
    }
    summaries = []
    for length in (256, 1792):
        case = {"key": f"cpu-e3-verify-x4-{length}", "length": length}
        pairs = []
        for number in range(1, PAIR_COUNT + 1):
            path = args.out / f"{case['key']}-pair{number:02d}.json"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("identity") != identity or record.get("pair") != number:
                    raise ValueError(f"resume identity mismatch: {path}")
                pairs.append(record)
                continue
            order = ["A", "B"] if number % 2 else ["B", "A"]
            print(f"PAIR {case['key']} {number}/{PAIR_COUNT} {order}", flush=True)
            entries = [run_entry(helpers, BASELINE if label == "A" else args.candidate, label, length) for label in order]
            left = next(entry for entry in entries if entry["profile"] == "A")
            right = next(entry for entry in entries if entry["profile"] == "B")
            comparison = {
                "logits_parity": left["logits_sha256"] == right["logits_sha256"],
                "tokens_parity": left["tokens"] == right["tokens"],
                "position_parity": left["position_after"] == right["position_after"],
            }
            comparison["parity"] = all(comparison.values())
            if not comparison["parity"]:
                raise RuntimeError(f"CPU-E3 parity failed on pair {number}")
            record = {"identity": identity, "case": case, "pair": number,
                      "order": order, "entries": entries,
                      "comparison_result": comparison, "status": "PASS"}
            write_json(path, record)
            pairs.append(record)
        a = [next(entry for entry in pair["entries"] if entry["profile"] == "A") for pair in pairs]
        b = [next(entry for entry in pair["entries"] if entry["profile"] == "B") for pair in pairs]
        paired = [{
            "pair": pair["pair"],
            "verification_time_reduction": 1.0 - next(entry for entry in pair["entries"] if entry["profile"] == "B")["verification_seconds"] / next(entry for entry in pair["entries"] if entry["profile"] == "A")["verification_seconds"],
        } for pair in pairs]
        summary = {
            "case": case,
            "pairs": PAIR_COUNT,
            "parity": True,
            "profiles": {
                label: {"verification_seconds": statistics.median(entry["verification_seconds"] for entry in rows)}
                for label, rows in (("A", a), ("B", b))
            },
            "paired": paired,
            "median_time_reduction": statistics.median(row["verification_time_reduction"] for row in paired),
            "favorable_pairs": sum(row["verification_time_reduction"] > 0 for row in paired),
        }
        write_json(args.out / f"{case['key']}-summary.json", summary)
        summaries.append(summary)
    write_json(args.out / "summary.json", {"identity": identity, "cases": summaries, "status": "PASS"})


if __name__ == "__main__":
    main()

