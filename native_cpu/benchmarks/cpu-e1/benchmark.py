"""Run the isolated CPU-E1 four-input FP16 GEMV benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_EXE = ROOT / "native_cpu" / "build" / "Release" / "mm_bench_cpu_e1.exe"
PAIR_COUNT = 6
SHAPES = (("2432x768", 2432, 768), ("768x2432", 768, 2432))
SOURCE_COMMIT = "ad6e5ae94b65522b7779ef8a0b1b72dfd109c71b"


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


def identity(repetitions: int):
    return {
        "schema": "cpu-e1/isolated-v1",
        "source_commit": SOURCE_COMMIT,
        "benchmark_tree_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "benchmark_executable": str(BENCHMARK_EXE),
        "benchmark_executable_sha256": sha(BENCHMARK_EXE),
        "kernel_sources_sha256": {
            "native_cpu/include/mm_kernels.h": sha(ROOT / "native_cpu" / "include" / "mm_kernels.h"),
            "native_cpu/src/kernels.cpp": sha(ROOT / "native_cpu" / "src" / "kernels.cpp"),
            "native_cpu/benchmarks/cpu-e1/kernel_bench.cpp": sha(Path(__file__).with_name("kernel_bench.cpp")),
        },
        "shapes": [{"name": name, "rows": rows, "cols": cols} for name, rows, cols in SHAPES],
        "pairs": PAIR_COUNT,
        "repetitions": repetitions,
        "warmup_repetitions": 2,
        "cpu": 0,
        "threads": 1,
        "mode": "auto",
        "preparation_and_verification_excluded": True,
        "reference": "four independent gemv_f16 calls",
        "candidate": "one gemv_f16_x4 call sharing each converted weight fragment across four inputs",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=64)
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not BENCHMARK_EXE.is_file():
        raise FileNotFoundError(f"benchmark executable not found: {BENCHMARK_EXE}")
    args.out.mkdir(parents=True, exist_ok=True)
    meta = identity(args.repetitions)
    command = [str(BENCHMARK_EXE), "--pairs", str(PAIR_COUNT), "--repetitions", str(args.repetitions)]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    receipts = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("pair") not in range(1, PAIR_COUNT + 1) or row.get("shape") not in {shape[0] for shape in SHAPES}:
            raise ValueError(f"unexpected benchmark row: {row}")
        row["command"] = command
        row["identity"] = meta
        receipts.append(row)
    expected = {shape[0] for shape in SHAPES} | set()
    if len(receipts) != len(SHAPES) * PAIR_COUNT:
        raise RuntimeError(f"expected {len(SHAPES) * PAIR_COUNT} receipts, got {len(receipts)}")
    for shape in expected:
        rows = sorted((row for row in receipts if row["shape"] == shape), key=lambda row: row["pair"])
        if [row["pair"] for row in rows] != list(range(1, PAIR_COUNT + 1)):
            raise RuntimeError(f"incomplete pair set for {shape}")
        if not all(row.get("parity") is True and row.get("cpu") == 0 and row.get("threads") == 1 for row in rows):
            raise RuntimeError(f"parity or affinity failure for {shape}")
        for row in rows:
            path = args.out / f"cpu-e1-{shape}-pair{row['pair']:02d}.json"
            write_json(path, row)
    write_json(args.out / "identity.json", meta)
    write_json(args.out / "summary.json", {
        "identity": meta,
        "command": command,
        "status": "PASS",
        "receipts": len(receipts),
        "shapes": [{"name": shape, "pairs": PAIR_COUNT, "parity": True} for shape in expected],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
