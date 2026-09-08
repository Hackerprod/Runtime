"""Measure the reproducible CPU-U1 single baseline."""
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
BASELINE_LIBRARY = ROOT / "native_cpu" / "benchmarks" / "cpu-u1.2" / "baseline" / "minimind_cpu.dll"
BUILD_INFO = ROOT / "native_cpu" / "benchmarks" / "cpu-u1.2" / "build-info.json"
PAIR_COUNT = 3
EXPECTED_SOURCE_COMMIT = "06694632cfe4ee5b97013d6a5b2e6fec9fe98f50"
EXPECTED_LIBRARY_SHA256 = "5956ff064ad8e8fd1ff502d18626ce19240682faa55c584cb3d6eaad1775bb3a"


def _load_r1_benchmark():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_helpers_for_cpu_u1_2", path)
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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def identity(build_info: dict, build_info_sha256: str):
    measurement_tree_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return {
        "schema": "cpu-u1.2/single-baseline-v1",
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "measurement_tree_commit": measurement_tree_commit,
        "library": str(BASELINE_LIBRARY),
        "library_sha256": sha(BASELINE_LIBRARY),
        "build_info_sha256": build_info_sha256,
        "model_sha256": sha(MODEL),
        "tokenizer_sha256": {
            name: sha(TOKENIZER / name)
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
        },
        "build": {
            "type": build_info.get("build_type"),
            "compiler": build_info.get("compiler"),
            "compile_flags": build_info.get("compile_flags"),
            "generator": build_info.get("generator"),
            "cmake": build_info.get("cmake"),
        },
        "threads": 1,
        "cpus": [0],
        "row_weights": [1],
        "profiler": False,
        "runtime_route": "CPU-U1 automatic production route; no launcher switches",
        "repeat_count": PAIR_COUNT,
        "prefix_lengths": [256, 1792],
        "continuation": 32,
    }


def _effective(runtime):
    return {
        "threads": runtime.threads,
        "cpus": runtime.cpus,
        "row_weights": runtime.row_weights,
        "selective_logits": runtime.selective_logits,
        "v_blocked_attention": runtime.v_blocked_attention,
        "ffn_row4": runtime.ffn_row4,
        "ffn_f16_storage": runtime.ffn_f16_storage,
        "gqa_k_shared": runtime.gqa_k_shared,
        "gqa_v_shared": runtime.gqa_v_shared,
        "diagnostics": False,
    }


def run_entry(benchmark, case):
    from native_cpu.tools.native import NativeRuntime

    load_started = time.perf_counter()
    with NativeRuntime(MODEL, BASELINE_LIBRARY, 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        loading_seconds = time.perf_counter() - load_started
        benchmark.warmup(runtime, 1)
        data = benchmark.run_forced(runtime, case["length"], 32, False)
        data.update(
            profile="CPU-U1.2 baseline",
            library_sha256=EXPECTED_LIBRARY_SHA256,
            loading_seconds=loading_seconds,
            configuration_seconds=0.0,
            storage_preparation_seconds=(runtime.ffn_f16_prepare_ns or 0) / 1e9,
            inference_excludes_configuration=True,
            configured_options={"production_route": True, "launcher_switches": False},
            effective=_effective(runtime),
            ffn_f16_storage_bytes=runtime.ffn_f16_storage_bytes,
            ffn_f16_prepare_ns=runtime.ffn_f16_prepare_ns,
        )
        return data


def _outputs_equal(left: dict, right: dict) -> bool:
    return all(left.get(name) == right.get(name) for name in (
        "prefix_token_ids", "continuation_token_ids", "prefill_logits_sha256",
        "decode_logits_sha256", "position_after", "prefill_tokens_evaluated",
        "decode_tokens_evaluated",
    ))


def make_cases():
    return [
        {"key": f"cpu-u1.2-forced-{length}", "workload": "forced", "length": length}
        for length in (256, 1792)
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=PAIR_COUNT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats != PAIR_COUNT:
        raise ValueError(f"CPU-U1.2 requires exactly {PAIR_COUNT} repetitions")
    for path in (MODEL, *(TOKENIZER / name for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")), BASELINE_LIBRARY, BUILD_INFO):
        if not path.is_file():
            raise FileNotFoundError(path)
    build_info = load_json(BUILD_INFO)
    if build_info.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError(
            f"CPU-U1.2 refuses build-info source: expected {EXPECTED_SOURCE_COMMIT}, "
            f"found {build_info.get('source_commit')}"
        )
    library_sha256 = sha(BASELINE_LIBRARY)
    if library_sha256 != EXPECTED_LIBRARY_SHA256:
        raise RuntimeError(
            f"CPU-U1.2 refuses baseline DLL: expected SHA-256 {EXPECTED_LIBRARY_SHA256}, "
            f"found {library_sha256}"
        )
    if build_info.get("library_sha256") != EXPECTED_LIBRARY_SHA256:
        raise RuntimeError("CPU-U1.2 build-info DLL hash does not match the baseline artifact")
    args.out.mkdir(parents=True, exist_ok=True)
    benchmark = _load_r1_benchmark()
    build_info_sha256 = sha(BUILD_INFO)
    meta = identity(build_info, build_info_sha256)
    for case in make_cases():
        repetitions = []
        for number in range(1, PAIR_COUNT + 1):
            path = args.out / f"{case['key']}-run{number:02d}.json"
            if path.exists():
                if not args.resume:
                    raise FileExistsError(path)
                prior = load_json(path)
                if prior.get("identity") != meta or prior.get("case") != case or prior.get("run") != number:
                    raise ValueError(f"resume identity mismatch: {path}")
                if prior.get("status") != "PASS":
                    raise ValueError(f"cannot resume invalid run: {path}")
                repetitions.append(prior)
                continue
            print(f"RUN {case['key']} {number}/{PAIR_COUNT}", flush=True)
            entry = run_entry(benchmark, case)
            record = {"identity": meta, "case": case, "run": number, "entry": entry, "status": "PASS"}
            if repetitions and not _outputs_equal(repetitions[0]["entry"], entry):
                record["status"] = "FAIL"
                record["reason"] = "logits or fixed-token outputs differ between repetitions"
            write_json(path, record)
            repetitions.append(record)
            if record["status"] != "PASS":
                raise RuntimeError(f"baseline output parity failed; preserved {path}")
        summary = {
            "identity": meta,
            "case": case,
            "repetitions": len(repetitions),
            "output_parity": all(r["status"] == "PASS" for r in repetitions),
            "status": "PASS",
        }
        write_json(args.out / f"{case['key']}-summary.json", summary)
    write_json(args.out / "summary.json", {
        "identity": meta,
        "cases": [load_json(args.out / f"{case['key']}-summary.json") for case in make_cases()],
        "status": "PASS",
        "final_protocol": True,
    })


if __name__ == "__main__":
    main()
