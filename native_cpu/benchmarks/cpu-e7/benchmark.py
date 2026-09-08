"""Validate exact FP16 storage and isolated LM-head GEMV performance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "checkpoints" / "minimind-3-hf" / "model.safetensors"
MODEL_ARTIFACT = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
BENCHMARK_EXE = ROOT / "native_cpu" / "build-e7" / "Release" / "mm_bench_cpu_e7.exe"
EXPECTED_CHECKPOINT_SHA256 = "3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8"
EXPECTED_ARTIFACT_SHA256 = "bcab52e67825061ffc3bd4480dbda1d5067bc3a0e51aba386a236c7131766f33"
EXPECTED_SHAPE = (6400, 768)
EXPECTED_ELEMENTS = 4_915_200
PAYLOAD_MAGIC = b"MME7001\0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def artifact_embedding(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:8] != b"MMCPU001":
            raise ValueError("invalid MMCPU001 artifact")
        fields = struct.unpack("<8s12I2f", header)
        records = fields[12]
        for _ in range(records):
            raw = stream.read(4)
            if len(raw) != 4:
                raise ValueError("truncated tensor name length")
            name_length = struct.unpack("<I", raw)[0]
            name = stream.read(name_length).decode("ascii")
            dtype, ndim = struct.unpack("<II", stream.read(8))
            shape = struct.unpack(f"<{ndim}I", stream.read(4 * ndim))
            stream.read(4)
            data_bytes, scale_count = struct.unpack("<QQ", stream.read(16))
            data = stream.read(data_bytes)
            stream.read(scale_count * 4)
            if name == "model.embed_tokens.weight":
                if dtype != 0 or tuple(shape) != EXPECTED_SHAPE:
                    raise ValueError("MMCPU001 LM head shape/dtype mismatch")
                values = np.frombuffer(data, dtype="<f4").reshape(EXPECTED_SHAPE)
                return values.copy()
    raise ValueError("MMCPU001 artifact is missing model.embed_tokens.weight")


def inspect_checkpoint(checkpoint: Path, artifact: Path) -> dict:
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 does not match CPU-E6 source")
    if sha256(artifact) != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("MMCPU001 artifact SHA-256 does not match CPU-E6 source")
    with safe_open(str(checkpoint), framework="numpy", device="cpu") as handle:
        values = np.asarray(handle.get_tensor("model.embed_tokens.weight"))
    if values.dtype != np.float16 or tuple(values.shape) != EXPECTED_SHAPE:
        raise ValueError(f"LM head must be float16 {EXPECTED_SHAPE}, got {values.dtype} {values.shape}")
    half_bits = np.asarray(values, dtype="<f2", order="C").view("<u2")
    restored = values.astype("<f4", copy=False)
    roundtrip_bits = restored.astype("<f2").view("<u2")
    roundtrip_mismatches = int(np.count_nonzero(half_bits != roundtrip_bits))
    artifact_values = artifact_embedding(artifact)
    artifact_mismatches = int(np.count_nonzero(restored.view("<u4") != artifact_values.view("<u4")))
    if roundtrip_mismatches or artifact_mismatches:
        raise ValueError("LM-head FP16 reconstruction is not bit-exact")
    return {
        "schema": "cpu-e7/lm-head-reconstruction-v1",
        "checkpoint_sha256": sha256(checkpoint),
        "artifact_sha256": sha256(artifact),
        "tensor": "model.embed_tokens.weight",
        "shape": list(EXPECTED_SHAPE),
        "elements": EXPECTED_ELEMENTS,
        "roundtrip_bit_mismatches": roundtrip_mismatches,
        "artifact_bit_mismatches": artifact_mismatches,
        "fp16_bytes": int(values.nbytes),
        "fp32_bytes": int(restored.nbytes),
        "expected_additional_storage_bytes": int(values.nbytes),
        "status": "PASS",
        "fp16_bits": half_bits.tobytes(order="C"),
        "fp32_bytes_data": restored.tobytes(order="C"),
    }


def write_payload(path: Path, evidence: dict) -> None:
    name = evidence["tensor"].encode("ascii")
    with path.open("wb") as stream:
        stream.write(PAYLOAD_MAGIC)
        stream.write(struct.pack("<I", 1))
        stream.write(struct.pack("<IIII", len(name), EXPECTED_SHAPE[0], EXPECTED_SHAPE[1], 0))
        stream.write(name)
        stream.write(evidence["fp16_bits"])
        stream.write(evidence["fp32_bytes_data"])


def parse_jsonl(stdout: str) -> list[dict]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--artifact", type=Path, default=MODEL_ARTIFACT)
    parser.add_argument("--benchmark-exe", type=Path, default=BENCHMARK_EXE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args(argv)
    if args.pairs != 3:
        raise ValueError("CPU-E7 requires exactly three alternating pairs")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not args.benchmark_exe.is_file():
        raise FileNotFoundError(args.benchmark_exe)
    evidence = inspect_checkpoint(args.checkpoint, args.artifact)
    args.out.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".cpu-e7-", suffix=".bin", dir=ROOT)
    os.close(handle)
    payload = Path(temporary_name)
    try:
        write_payload(payload, evidence)
        command = [str(args.benchmark_exe), "--payload", str(payload), "--pairs", str(args.pairs),
                   "--warmups", "2", "--repetitions", str(args.repetitions)]
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    finally:
        payload.unlink(missing_ok=True)
    rows = parse_jsonl(completed.stdout)
    if len(rows) != args.pairs:
        raise RuntimeError(f"expected {args.pairs} rows, got {len(rows)}")
    for row in rows:
        if row.get("cpu") != 0 or row.get("threads") != 1:
            raise RuntimeError(f"affinity contract failed: {row}")
        for key in ("x1_exact_memcmp_parity", "x4_exact_memcmp_parity", "x4_independent_exact_memcmp_parity"):
            if not row.get(key):
                raise RuntimeError(f"parity failed: {row}")
    identity = {
        "schema": "cpu-e7/isolated-lm-head-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "artifact_sha256": evidence["artifact_sha256"],
        "benchmark_executable": str(args.benchmark_exe),
        "benchmark_executable_sha256": sha256(args.benchmark_exe),
        "benchmark_source_sha256": sha256(Path(__file__).with_name("kernel_bench.cpp")),
        "cmake_sha256": sha256(ROOT / "native_cpu" / "CMakeLists.txt"),
        "tensor": evidence["tensor"], "shape": list(EXPECTED_SHAPE), "elements": EXPECTED_ELEMENTS,
        "roundtrip_bit_mismatches": evidence["roundtrip_bit_mismatches"],
        "artifact_bit_mismatches": evidence["artifact_bit_mismatches"],
        "pairs": args.pairs, "repetitions": args.repetitions, "warmup_repetitions": 2,
        "cpu": 0, "threads": 1, "x1_reference": "gemv_f32", "x1_candidate": "gemv_f16",
        "x4_reference": "four independent gemv_f32 calls", "x4_candidate": "gemv_f16_x4",
        "x4_diagnostic": "four independent gemv_f16 calls",
        "preparation_and_verification_excluded": True,
        "expected_additional_storage_bytes": evidence["expected_additional_storage_bytes"],
    }
    serializable_evidence = {key: value for key, value in evidence.items() if key not in {"fp16_bits", "fp32_bytes_data"}}
    write_json(args.out / "reconstruction.json", serializable_evidence)
    write_json(args.out / "identity.json", identity)
    for index, row in enumerate(rows, 1):
        write_json(args.out / f"pair{index:02d}.json", {**row, "identity": identity})
    write_json(args.out / "summary.json", {"identity": identity, "reconstruction": serializable_evidence,
                                             "rows": len(rows), "status": "PASS"})
    print(json.dumps({"status": "PASS", "rows": len(rows), "out": str(args.out.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
