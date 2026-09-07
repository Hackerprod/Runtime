"""Offline exporter for the MiniMind-3 dense CPU runtime.

This module intentionally depends only on NumPy and safetensors.  It does not
import the PyTorch reference model; conversion is a deterministic serialization
step from the official local Hugging Face checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import uuid
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
from safetensors import safe_open

MAGIC = b"MMCPU001"
VERSION = 1
DTYPE_F32 = 0
DTYPE_Q4 = 1
Q4_GROUP = 32

class ExportError(ValueError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_config(config: dict) -> None:
    expected = {
        "model_type": "qwen3", "hidden_size": 768, "intermediate_size": 2432,
        "num_hidden_layers": 8, "num_attention_heads": 8,
        "num_key_value_heads": 4, "head_dim": 96, "vocab_size": 6400,
        "max_position_embeddings": 32768, "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0, "hidden_act": "silu",
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, float):
            if actual is None or not np.isclose(float(actual), value, rtol=0, atol=1e-12):
                raise ExportError(f"config.{key} must be {value!r}, got {actual!r}")
        elif actual != value:
            raise ExportError(f"config.{key} must be {value!r}, got {actual!r}")
    if config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ExportError("config.architectures must be ['Qwen3ForCausalLM']")
    if config.get("rope_scaling") is not None:
        raise ExportError("rope_scaling must be null for the CPU runtime")
    if config.get("attention_bias") not in (None, False) or config.get("mlp_bias") not in (None, False):
        raise ExportError("attention/MLP biases are not supported by the CPU runtime")
    if config.get("use_sliding_window") not in (None, False) or config.get("sliding_window") is not None:
        raise ExportError("sliding-window attention is not supported")
    if config.get("layer_types") is not None and any(x != "full_attention" for x in config["layer_types"]):
        raise ExportError("only full attention layers are supported")
    if config.get("eos_token_id") not in (None, 2):
        raise ExportError("CPU runtime contract requires eos_token_id 2")


def _expected_shape(name: str, c: dict) -> tuple[int, ...] | None:
    h, inter, q, kv, hd, v = (c["hidden_size"], c["intermediate_size"],
        c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"])
    if name == "model.embed_tokens.weight": return (v, h)
    if name == "model.norm.weight": return (h,)
    if name.startswith("model.layers."):
        tail = name.split(".", 3)[-1]
        scalar = {"input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
                  "self_attn.q_norm.weight": (hd,), "self_attn.k_norm.weight": (hd,),
                  "self_attn.q_proj.weight": (q * hd, h), "self_attn.k_proj.weight": (kv * hd, h),
                  "self_attn.v_proj.weight": (kv * hd, h), "self_attn.o_proj.weight": (h, q * hd),
                  "mlp.gate_proj.weight": (inter, h), "mlp.up_proj.weight": (inter, h),
                  "mlp.down_proj.weight": (h, inter)}
        return scalar.get(tail)
    return None


def _is_linear(name: str, arr: np.ndarray) -> bool:
    return name.endswith(".weight") and arr.ndim == 2 and "embed_tokens" not in name


def quantize_q4(array: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Quantize row-major weights; even columns occupy low nibbles."""
    a = np.asarray(array, dtype=np.float32, order="C")
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ExportError("Q4 inputs must be finite rank-2 arrays")
    rows, cols = a.shape
    groups = (cols + Q4_GROUP - 1) // Q4_GROUP
    padded_cols = groups * Q4_GROUP
    padded = np.zeros((rows, padded_cols), dtype=np.float32)
    padded[:, :cols] = a
    blocks = padded.reshape(rows, groups, Q4_GROUP)
    scales = np.max(np.abs(blocks), axis=2) / 7.0
    scales[scales == 0.0] = 1.0
    q = np.clip(np.rint(blocks / scales[:, :, None]), -8, 7).astype(np.int8) + 8
    q = q.reshape(rows, padded_cols)[:, :cols]
    packed = np.zeros((rows, (cols + 1) // 2), dtype=np.uint8)
    packed[:, : (cols + 1) // 2][:, : q[:, 0::2].shape[1]] |= q[:, 0::2].astype(np.uint8)
    if q[:, 1::2].size:
        packed[:, : q[:, 1::2].shape[1]] |= q[:, 1::2].astype(np.uint8) << 4
    if cols & 1:
        packed[:, -1] |= np.uint8(0x80)  # unused high nibble decodes to zero
    return packed.tobytes(order="C"), np.asarray(scales, dtype="<f4").reshape(-1)


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with tmp.open("xb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(tmp, path)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass


def _tensor_record(name: str, arr: np.ndarray, quantization: str) -> tuple[bytes, dict]:
    if not name.isascii(): raise ExportError(f"tensor name is not ASCII: {name!r}")
    a = np.asarray(arr)
    if not np.isfinite(a).all(): raise ExportError(f"non-finite tensor: {name}")
    if quantization == "q4" and _is_linear(name, a):
        raw, scales = quantize_q4(a)
        dtype, group = DTYPE_Q4, Q4_GROUP
    else:
        raw, scales, dtype, group = np.asarray(a, dtype="<f4", order="C").tobytes(order="C"), np.empty(0, dtype="<f4"), DTYPE_F32, 0
    name_b = name.encode("ascii")
    header = struct.pack("<I", len(name_b)) + name_b
    header += struct.pack("<II", dtype, a.ndim)
    header += struct.pack(f"<{a.ndim}I", *a.shape)
    header += struct.pack("<I", group)
    header += struct.pack("<QQ", len(raw), int(scales.size))
    record = header + raw + scales.tobytes()
    info = {"name": name, "dtype": "q4" if dtype else "f32", "shape": list(a.shape),
            "group_size": group, "data_bytes": len(raw), "scale_count": int(scales.size)}
    return record, info


def _manifest_path(output: Path, explicit: str | os.PathLike[str] | None) -> Path:
    return Path(explicit) if explicit else Path(str(output) + ".manifest.json")


def export_model(source: str | os.PathLike[str], config_path: str | os.PathLike[str], output: str | os.PathLike[str], *, quantization: str = "fp32", manifest: str | os.PathLike[str] | None = None, source_manifest: str | os.PathLike[str] | None = None) -> dict:
    source, config_path, output = Path(source), Path(config_path), Path(output)
    if quantization not in {"fp32", "q4"}: raise ExportError("quantization must be fp32 or q4")
    if not source.is_file() or not config_path.is_file(): raise FileNotFoundError("source and config must be local files")
    config = json.loads(config_path.read_text(encoding="utf-8")); validate_config(config)
    mp = _manifest_path(output, manifest)
    sm_path = Path(source_manifest) if source_manifest else source.parent / "download-manifest.json"
    resolved = {p.resolve() for p in (source, config_path, output, mp, sm_path)}
    if len(resolved) != 5:
        raise ExportError("source, config, output, manifest, and source manifest must be distinct paths")
    if output.exists() or mp.exists():
        raise FileExistsError(f"refusing to overwrite {output if output.exists() else mp}")
    source_sha = _sha256(source)
    source_meta = None
    if sm_path.is_file():
        source_meta = json.loads(sm_path.read_text(encoding="utf-8"))
        items = source_meta.get("files", [])
        source_item = next((item for item in items if item.get("path") == source.name or item.get("url", "").endswith(source.name)), None)
        config_item = next((item for item in items if item.get("path") == config_path.name or item.get("url", "").endswith(config_path.name)), None)
        if source_item is None:
            raise ExportError(f"source is absent from download manifest: {source.name}")
        expected = source_item.get("sha256")
        if expected and expected.lower() != source_sha.lower():
            raise ExportError(f"source SHA-256 mismatch: expected {expected}, got {source_sha}")
        if source_item.get("bytes") and int(source_item["bytes"]) != source.stat().st_size:
            raise ExportError("source byte count mismatch")
        if config_item is None:
            raise ExportError(f"config is absent from download manifest: {config_path.name}")
        config_sha = _sha256(config_path)
        if config_item.get("sha256") and config_item["sha256"].lower() != config_sha.lower():
            raise ExportError("config SHA-256 mismatch")
        if config_item.get("bytes") and int(config_item["bytes"]) != config_path.stat().st_size:
            raise ExportError("config byte count mismatch")
    records, layout = [], []
    with safe_open(str(source), framework="numpy", device="cpu") as handle:
        keys = sorted(handle.keys())
        if "lm_head.weight" in keys:
            raise ExportError("tied checkpoint must not contain lm_head.weight")
        expected_names = {"model.embed_tokens.weight", "model.norm.weight"}
        expected_names.update(f"model.layers.{i}.{tail}" for i in range(config["num_hidden_layers"])
                             for tail in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"))
        if set(keys) != expected_names:
            missing, extra = sorted(expected_names - set(keys)), sorted(set(keys) - expected_names)
            raise ExportError(f"tensor key layout mismatch; missing={missing[:3]} extra={extra[:3]}")
        for name in keys:
            arr = np.asarray(handle.get_tensor(name))
            shape = _expected_shape(name, config)
            if shape is None or tuple(arr.shape) != shape:
                raise ExportError(f"tensor shape mismatch for {name}: expected {shape}, got {arr.shape}")
            rec, info = _tensor_record(name, arr, quantization); records.append(rec); layout.append(info)
    header = struct.pack("<8s12I2f", MAGIC, VERSION, config["vocab_size"], config["hidden_size"], config["num_hidden_layers"], config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"], config["intermediate_size"], config["max_position_embeddings"], 2, 1, len(records), float(config["rms_norm_eps"]), float(config["rope_theta"]))
    payload = header + b"".join(records)
    binary = payload + struct.pack("<I", zlib.crc32(payload) & 0xffffffff)
    _atomic_write(output, binary)
    out_sha = hashlib.sha256(binary).hexdigest()
    config_sha = _sha256(config_path)
    layout_sha = hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    source_manifest_sha = _sha256(sm_path) if source_meta else None
    meta = {"schema": "MMCPU001", "version": VERSION, "quantization": quantization,
            "source": {"path": str(source), "sha256": source_sha, "bytes": source.stat().st_size},
            "input_sha256": source_sha, "config": {"path": str(config_path), "sha256": config_sha},
            "config_sha256": config_sha, "output": {"path": str(output), "sha256": out_sha, "bytes": len(binary)},
            "output_sha256": out_sha, "source_manifest": str(sm_path) if source_meta else None,
            "source_manifest_sha256": source_manifest_sha,
            "source_provenance": ({k: source_meta[k] for k in ("repo_id", "revision", "license") if k in source_meta} if source_meta else None),
            "tensor_count": len(layout), "key_layout": layout, "key_layout_sha256": layout_sha,
            "flags": {"tied_embeddings": True}}
    _atomic_write(mp, json.dumps(meta, indent=2, sort_keys=True).encode("utf-8"))
    return meta


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="local model.safetensors")
    p.add_argument("--config", required=True, help="local config.json")
    p.add_argument("--output", required=True, help="new MMCPU001 binary")
    p.add_argument("--quantization", choices=("fp32", "q4"), default="fp32")
    p.add_argument("--manifest")
    args = p.parse_args(argv)
    try:
        meta = export_model(args.source, args.config, args.output, quantization=args.quantization, manifest=args.manifest)
    except Exception as exc:
        p.error(str(exc))
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())
