"""Run and record the three-run canonical CPU-E4 output-throughput guard."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Could you tell me about programming languages?"
EXPECTED_DLL_SHA256 = "b410e642e2249a8982a9a83b134f3d7f5e1f37c677f11d10900ed0e213cb97ab"
DLL = ROOT / "native_cpu" / "benchmarks" / "cpu-e4" / "baseline" / "minimind_cpu.dll"
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
CHECKPOINT = ROOT / "checkpoints" / "minimind-3-hf"
RESULTS = ROOT / "compare_cpu" / "canonical-results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metric_rate(metrics: dict, numerator: str) -> float | None:
    seconds = float(metrics.get("decode_seconds", 0.0))
    count = int(metrics.get(numerator, 0))
    return count / seconds if count and seconds > 0.0 else None


def run_once(index: int) -> tuple[dict, str]:
    command = [
        sys.executable, "-m", "native_cpu.tools.compare_chat",
        "--backend", "native", "--profile", "chat", "--prompt", PROMPT,
        "--model", str(MODEL), "--library", str(DLL),
        "--checkpoint-dir", str(CHECKPOINT), "--context-limit", "2048",
        "--max-new-tokens", "256", "--temperature", "0.9", "--top-k", "50",
        "--top-p", "0.85", "--seed", "0", "--threads", "1", "--cpus", "0",
        "--metrics-json",
    ]
    environment = os.environ.copy()
    environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
                         "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    completed = subprocess.run(command, cwd=ROOT, env=environment,
                               capture_output=True, text=True, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"canonical run {index} failed with code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"canonical run {index} did not return exactly one JSON line")
    try:
        metrics = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"canonical run {index} returned invalid JSON: {lines[-1]!r}") from exc
    if not isinstance(metrics, dict):
        raise RuntimeError(f"canonical run {index} returned a non-object result")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"run-{index:02d}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / f"run-{index:02d}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    return metrics, completed.stderr


def _fmt(values: list[float | None]) -> str:
    return " / ".join("n/a" if value is None else f"{value:.2f}" for value in values)


def main() -> int:
    if not DLL.is_file():
        raise FileNotFoundError(f"missing CPU-E4 DLL: {DLL}")
    actual_sha = sha256_file(DLL)
    if actual_sha != EXPECTED_DLL_SHA256:
        raise RuntimeError(f"CPU-E4 DLL SHA-256 mismatch: expected {EXPECTED_DLL_SHA256}, found {actual_sha}")
    metrics_rows = [run_once(index) for index in range(1, 4)]
    rows = [item[0] for item in metrics_rows]
    output_rates = [_metric_rate(row, "generated_tokens") for row in rows]
    target_rates = [_metric_rate(row, "decode_evaluated_tokens") for row in rows]
    prefill = [float(row["prefill_seconds"]) for row in rows]
    generated = [int(row["generated_tokens"]) for row in rows]
    sampled_hashes = [sha256_json(row.get("sampled_ids", [])) for row in rows]
    text_hashes = [hashlib.sha256(str(row.get("text", "")).encode("utf-8")).hexdigest() for row in rows]
    speculative = [row.get("speculative_decode") or {} for row in rows]
    summary = {
        "schema": "perf-g1/canonical-output-toks-v1", "prompt": PROMPT,
        "policy": {"temperature": 0.9, "top_k": 50, "top_p": 0.85, "seed": 0,
                    "context_limit": 2048, "max_new_tokens": 256, "system": rows[0].get("system")},
        "dll": {"path": str(DLL.relative_to(ROOT)), "sha256": actual_sha},
        "runs": [
            {"index": index, "output_tokens_per_second": output_rates[index - 1],
             "legacy_target_tokens_per_second": target_rates[index - 1],
             "prefill_seconds": prefill[index - 1],
             "decode_seconds": float(rows[index - 1]["decode_seconds"]),
             "generated_tokens": generated[index - 1],
             "sampled_ids_sha256": sampled_hashes[index - 1],
             "text_sha256": text_hashes[index - 1],
             "speculative_decode": speculative[index - 1]}
            for index in range(1, 4)
        ],
        "hashes_identical_across_runs": {"sampled_ids": len(set(sampled_hashes)) == 1,
                                          "text": len(set(text_hashes)) == 1},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "SUMMARY.md").write_text(render_summary(summary), encoding="utf-8")

    print("Canonical prompt")
    print(PROMPT)
    print("\nFirst response:")
    print(rows[0].get("text", ""))
    print("\noutput tok/s: " + _fmt(output_rates))
    print(f"median output tok/s: {statistics.median(value for value in output_rates if value is not None):.2f}")
    print("legacy target tok/s: " + _fmt(target_rates))
    print("prefill: " + " / ".join(f"{value:.3f}s" for value in prefill))
    print("generated tokens: " + " / ".join(str(value) for value in generated))
    print("sampled-ids SHA256: " + " / ".join(sampled_hashes))
    print("text SHA256: " + " / ".join(text_hashes))
    for index, telemetry in enumerate(speculative, start=1):
        print(f"speculative run {index}: blocks={telemetry.get('proposed_blocks', 0)} "
              f"proposed={telemetry.get('proposed_tokens', 0)} "
              f"accepted={telemetry.get('accepted_tokens', 0)} "
              f"verify_calls={telemetry.get('verify_calls', 0)}")
    return 0


def render_summary(summary: dict) -> str:
    lines = ["# PERF-G1: canonical output throughput guard", "",
             "CPU-E4 exact baseline; three fresh processes; profiler off; one participant pinned to CPU 0.", "",
             f"- Prompt: `{PROMPT}`", f"- DLL SHA-256: `{summary['dll']['sha256']}`", "",
             "| Run | Output tok/s | Legacy target tok/s | Prefill (s) | Decode (s) | Generated |",
             "|---:|---:|---:|---:|---:|---:|"]
    for item in summary["runs"]:
        legacy = item["legacy_target_tokens_per_second"]
        lines.append(f"| {item['index']} | {item['output_tokens_per_second']:.2f} | {legacy if legacy is not None else 0.0:.2f} | "
                     f"{item['prefill_seconds']:.3f} | {item['decode_seconds']:.3f} | {item['generated_tokens']} |")
    lines += ["", "| Run | Sampled IDs SHA-256 | Text SHA-256 | Verify calls | Accepted tokens |", "|---:|---|---|---:|---:|"]
    for item in summary["runs"]:
        telemetry = item["speculative_decode"]
        lines.append(f"| {item['index']} | `{item['sampled_ids_sha256']}` | `{item['text_sha256']}` | "
                     f"{telemetry.get('verify_calls', 0)} | {telemetry.get('accepted_tokens', 0)} |")
    lines += ["", "The visible `decode_tokens_per_second` now equals output tokens divided by wall-clock decode time. "
              "Legacy target throughput is retained only as a secondary diagnostic.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
