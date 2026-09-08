"""Validate exact FP16 attention weights and benchmark isolated GEMV paths.

CPU-E5 deliberately stops at the kernel boundary.  It reads the original
MiniMind safetensors checkpoint, verifies the four attention projections for
all eight layers, materialises a temporary benchmark payload, and invokes the
standalone C++ harness.  Nothing in this module loads or configures the
production runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "checkpoints" / "minimind-3-hf" / "model.safetensors"
MODEL_ARTIFACT = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
BENCHMARK_EXE = ROOT / "native_cpu" / "build-e5" / "Release" / "mm_bench_cpu_e5.exe"
EXPECTED_CHECKPOINT_SHA256 = "3adf69402b5d22e693151cabadc12528f923c4ba6bf343738aaf13f0892162e8"
EXPECTED_ARTIFACT_SHA256 = "bcab52e67825061ffc3bd4480dbda1d5067bc3a0e51aba386a236c7131766f33"
LAYERS = 8
PROJECTIONS = {
    "q_proj": (768, 768),
    "k_proj": (384, 768),
    "v_proj": (384, 768),
    "o_proj": (768, 768),
}
PAYLOAD_MAGIC = b"MME5001\0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def projection_name(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.self_attn.{projection}.weight"


def expected_names() -> tuple[str, ...]:
    return tuple(projection_name(layer, projection) for layer in range(LAYERS)
                 for projection in PROJECTIONS)


def _artifact_tensors(path: Path) -> dict[str, tuple[tuple[int, ...], bytes]]:
    """Read the small MMCPU001 record index without expanding the artifact."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) != 64 or header[:8] != b"MMCPU001":
            raise ValueError(f"invalid MMCPU001 artifact: {path}")
        fields = struct.unpack("<8s12I2f", header)
        # The final integer in the 12-I header field is the tensor count.
        record_count = fields[12]
        records: dict[str, tuple[tuple[int, ...], bytes]] = {}
        for _ in range(record_count):
            raw = stream.read(4)
            if len(raw) != 4:
                raise ValueError("truncated tensor name length")
            name_length = struct.unpack("<I", raw)[0]
            name = stream.read(name_length).decode("ascii")
            dtype, ndim = struct.unpack("<II", stream.read(8))
            shape = struct.unpack(f"<{ndim}I", stream.read(4 * ndim))
            _group = struct.unpack("<I", stream.read(4))[0]
            data_bytes, scale_count = struct.unpack("<QQ", stream.read(16))
            data = stream.read(data_bytes)
            scales = stream.read(scale_count * 4)
            if len(data) != data_bytes or len(scales) != scale_count * 4:
                raise ValueError(f"truncated tensor record: {name}")
            if dtype == 0:
                records[name] = (tuple(shape), data)
    return records


def _float_bits(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="<f4", order="C").view("<u4")


def inspect_checkpoint(checkpoint: Path, artifact: Path) -> dict:
    """Return exact reconstruction evidence and the ordered matrix payload."""
    if sha256(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA-256 does not match CPU-E4 source")
    if sha256(artifact) != EXPECTED_ARTIFACT_SHA256:
        raise ValueError("MMCPU001 artifact SHA-256 does not match CPU-E4 source")
    artifact_records = _artifact_tensors(artifact)
    matrices: list[dict] = []
    total_elements = 0
    total_mismatches = 0
    artifact_mismatches = 0
    with safe_open(str(checkpoint), framework="numpy", device="cpu") as handle:
        actual = set(handle.keys())
        required = set(expected_names())
        if not required.issubset(actual):
            missing = sorted(required - actual)
            raise ValueError(f"checkpoint missing attention projections: {missing[:3]}")
        for layer in range(LAYERS):
            for projection, shape in PROJECTIONS.items():
                name = projection_name(layer, projection)
                values = np.asarray(handle.get_tensor(name))
                if values.dtype != np.float16:
                    raise ValueError(f"{name} must be float16, got {values.dtype}")
                if tuple(values.shape) != shape:
                    raise ValueError(f"{name} shape mismatch: expected {shape}, got {values.shape}")
                half_bits = np.asarray(values, dtype="<f2", order="C").view("<u2")
                restored = values.astype("<f4", copy=False)
                roundtrip_bits = restored.astype("<f2").view("<u2")
                mismatches = int(np.count_nonzero(half_bits != roundtrip_bits))
                expected_bytes = restored.tobytes(order="C")
                artifact_shape, artifact_bytes = artifact_records.get(name, (None, None))
                if artifact_shape != shape or artifact_bytes is None:
                    raise ValueError(f"{name} is absent or malformed in MMCPU001 artifact")
                artifact_values = np.frombuffer(artifact_bytes, dtype="<f4").reshape(shape)
                artifact_mismatch = int(np.count_nonzero(_float_bits(restored) != _float_bits(artifact_values)))
                matrices.append({
                    "name": name,
                    "projection": projection,
                    "layer": layer,
                    "rows": shape[0],
                    "cols": shape[1],
                    "elements": int(values.size),
                    "roundtrip_bit_mismatches": mismatches,
                    "artifact_bit_mismatches": artifact_mismatch,
                    "fp16_bytes": int(values.nbytes),
                    "fp32_bytes": len(expected_bytes),
                    "fp16_bits": half_bits.tobytes(order="C"),
                    "fp32_bytes_data": expected_bytes,
                })
                total_elements += int(values.size)
                total_mismatches += mismatches
                artifact_mismatches += artifact_mismatch
    if total_mismatches or artifact_mismatches:
        raise ValueError("attention FP16 reconstruction is not bit-exact")
    return {
        "schema": "cpu-e5/fp16-reconstruction-v1",
        "checkpoint_sha256": sha256(checkpoint),
        "artifact_sha256": sha256(artifact),
        "matrix_count": len(matrices),
        "elements": total_elements,
        "roundtrip_bit_mismatches": total_mismatches,
        "artifact_bit_mismatches": artifact_mismatches,
        "status": "PASS",
        "matrices": matrices,
    }


def _write_payload(path: Path, evidence: dict) -> None:
    matrices = evidence["matrices"]
    with path.open("wb") as stream:
        stream.write(PAYLOAD_MAGIC)
        stream.write(struct.pack("<I", len(matrices)))
        for matrix in matrices:
            name = matrix["name"].encode("ascii")
            stream.write(struct.pack("<IIII", len(name), matrix["rows"], matrix["cols"], 0))
            stream.write(name)
            stream.write(matrix["fp16_bits"])
            stream.write(matrix["fp32_bytes_data"])


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _json_evidence(evidence: dict) -> dict:
    """Drop the binary payload fields before writing human-auditable JSON."""
    return {
        **evidence,
        "matrices": [
            {key: value for key, value in matrix.items()
             if key not in {"fp16_bits", "fp32_bytes_data"}}
            for matrix in evidence["matrices"]
        ],
    }


def _parse_jsonl(stdout: str) -> list[dict]:
    rows = []
    for line in stdout.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--artifact", type=Path, default=MODEL_ARTIFACT)
    parser.add_argument("--benchmark-exe", type=Path, default=BENCHMARK_EXE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=8)
    args = parser.parse_args(argv)
    if args.pairs != 3:
        raise ValueError("CPU-E5 requires exactly three alternating pairs")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not args.benchmark_exe.is_file():
        raise FileNotFoundError(f"benchmark executable not found: {args.benchmark_exe}")
    args.out.mkdir(parents=True, exist_ok=True)
    evidence = inspect_checkpoint(args.checkpoint, args.artifact)
    benchmark_command = [str(args.benchmark_exe), "--payload", "{temporary_payload}",
                         "--pairs", str(args.pairs), "--repetitions", str(args.repetitions)]
    descriptor, payload_name = tempfile.mkstemp(prefix=".cpu-e5-", suffix=".bin", dir=ROOT)
    os.close(descriptor)
    payload = Path(payload_name)
    try:
        _write_payload(payload, evidence)
        command = [str(args.benchmark_exe), "--payload", str(payload),
                   "--pairs", str(args.pairs), "--repetitions", str(args.repetitions)]
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    finally:
        payload.unlink(missing_ok=True)
    rows = _parse_jsonl(completed.stdout)
    expected = {"q_proj", "k_proj", "v_proj", "o_proj"}
    expected_rows = len(evidence["matrices"]) * args.pairs
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} benchmark rows, got {len(rows)}")
    seen: set[tuple[int, str, int]] = set()
    for row in rows:
        key = (row.get("layer"), row.get("projection"), row.get("pair"))
        if row.get("projection") not in expected or row.get("pair") not in range(1, args.pairs + 1) or row.get("layer") not in range(LAYERS):
            raise RuntimeError(f"unexpected benchmark row: {row}")
        if key in seen:
            raise RuntimeError(f"duplicate benchmark row: {row}")
        seen.add(key)
        if not row.get("x1_exact_memcmp_parity") or not row.get("x4_exact_memcmp_parity"):
            raise RuntimeError(f"benchmark parity failure: {row}")
        if row.get("cpu") != 0 or row.get("threads") != 1:
            raise RuntimeError(f"benchmark affinity failure: {row}")
    identity = {
        "schema": "cpu-e5/isolated-attention-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "artifact": str(args.artifact),
        "artifact_sha256": evidence["artifact_sha256"],
        "benchmark_executable": str(args.benchmark_exe),
        "benchmark_executable_sha256": sha256(args.benchmark_exe),
        "benchmark_source_sha256": sha256(Path(__file__).with_name("kernel_bench.cpp")),
        "cmake_sha256": sha256(ROOT / "native_cpu" / "CMakeLists.txt"),
        "layers": LAYERS,
        "projections": {key: list(value) for key, value in PROJECTIONS.items()},
        "matrix_count": evidence["matrix_count"],
        "elements": evidence["elements"],
        "pairs": args.pairs,
        "repetitions": args.repetitions,
        "warmup_repetitions": 2,
        "cpu": 0,
        "threads": 1,
        "preparation_and_verification_excluded": True,
        "x1_reference": "gemv_f32",
        "x1_candidate": "gemv_f16",
        "x4_reference": "four independent gemv_f16 calls",
        "x4_candidate": "gemv_f16_x4",
    }
    _write_json(args.out / "reconstruction.json", _json_evidence(evidence))
    for row in rows:
        _write_json(args.out / f"cpu-e5-layer{row['layer']}-{row['projection']}-pair{row['pair']:02d}.json", {**row, "identity": identity, "command": benchmark_command})
    _write_json(args.out / "identity.json", identity)
    _write_json(args.out / "summary.json", {"identity": identity, "reconstruction": {key: evidence[key] for key in ("matrix_count", "elements", "roundtrip_bit_mismatches", "artifact_bit_mismatches", "status")}, "rows": len(rows), "status": "PASS"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
