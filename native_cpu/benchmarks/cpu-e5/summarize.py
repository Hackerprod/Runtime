"""Validate CPU-E5 receipts and render the isolated attention-kernel report."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results"
REPORT = Path(__file__).resolve().parent / "RESULTS.md"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
PAIRS = (1, 2, 3)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0


def read_receipts():
    grouped = {(projection, pair): [] for projection in PROJECTIONS for pair in PAIRS}
    issues: list[str] = []
    for path in sorted(OUT.glob("cpu-e5-layer*-pair*.json")):
        try:
            row = load(path)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{path.name}: invalid JSON ({exc})")
            continue
        key = (row.get("projection"), row.get("pair"))
        if key not in grouped:
            issues.append(f"{path.name}: unexpected projection/pair")
            continue
        if row.get("layer") not in range(8):
            issues.append(f"{path.name}: invalid layer")
        expected_order = "f32-f16-f16-x4" if row["pair"] % 2 else "f16-f32-x4-f16"
        if row.get("order") != expected_order:
            issues.append(f"{path.name}: order mismatch")
        if row.get("cpu") != 0 or row.get("threads") != 1:
            issues.append(f"{path.name}: not pinned to CPU 0 with one participant")
        if row.get("x1_exact_memcmp_parity") is not True or row.get("x4_exact_memcmp_parity") is not True:
            issues.append(f"{path.name}: exact parity was not proven")
        if row.get("x1_a_checksum") != row.get("x1_b_checksum") or row.get("x4_a_checksum") != row.get("x4_b_checksum"):
            issues.append(f"{path.name}: post-timing checksum mismatch")
        for name in ("x1_a_seconds", "x1_b_seconds", "x4_a_seconds", "x4_b_seconds"):
            if not positive(row.get(name)):
                issues.append(f"{path.name}: invalid {name}")
        grouped[key].append(row)
    for key, rows in grouped.items():
        layers = sorted(row.get("layer") for row in rows)
        if layers != list(range(8)):
            issues.append(f"{key[0]}/pair{key[1]:02d}: expected one receipt per layer")
    return grouped, issues


def aggregate(rows: list[dict]):
    first = rows[0]
    if first["pair"] % 2:
        x1_ref, x1_candidate = "x1_a_seconds", "x1_b_seconds"
        x4_ref, x4_candidate = "x4_a_seconds", "x4_b_seconds"
    else:
        x1_ref, x1_candidate = "x1_b_seconds", "x1_a_seconds"
        x4_ref, x4_candidate = "x4_b_seconds", "x4_a_seconds"
    reference_x1 = sum(float(row[x1_ref]) for row in rows)
    candidate_x1 = sum(float(row[x1_candidate]) for row in rows)
    reference_x4 = sum(float(row[x4_ref]) for row in rows)
    candidate_x4 = sum(float(row[x4_candidate]) for row in rows)
    return {
        "pair": first["pair"],
        "reference_x1": reference_x1,
        "candidate_x1": candidate_x1,
        "reference_x4": reference_x4,
        "candidate_x4": candidate_x4,
        "x1_benefit": 100.0 * (reference_x1 - candidate_x1) / reference_x1,
        "x4_benefit": 100.0 * (reference_x4 - candidate_x4) / reference_x4,
    }


def evaluate(summary: dict, grouped: dict, issues: list[str]):
    reasons = list(issues)
    reconstruction = summary.get("reconstruction", {})
    if summary.get("status") != "PASS":
        reasons.append("summary status is not PASS")
    if reconstruction.get("roundtrip_bit_mismatches") != 0 or reconstruction.get("artifact_bit_mismatches") != 0:
        reasons.append("attention weight reconstruction is not exact")
    stats = {}
    for projection in PROJECTIONS:
        pairs = []
        for pair in PAIRS:
            rows = grouped[(projection, pair)]
            if len(rows) == 8:
                pairs.append(aggregate(rows))
        if len(pairs) != len(PAIRS):
            reasons.append(f"incomplete receipts for {projection}")
            continue
        stats[projection] = {
            "pairs": pairs,
            "x1_median_benefit": statistics.median(row["x1_benefit"] for row in pairs),
            "x4_median_benefit": statistics.median(row["x4_benefit"] for row in pairs),
            "x1_favorable": sum(row["x1_benefit"] > 0.0 for row in pairs),
            "x4_favorable": sum(row["x4_benefit"] > 0.0 for row in pairs),
        }
    return {"evidence_ok": not reasons and set(stats) == set(PROJECTIONS), "reasons": reasons, "stats": stats}


def render(summary: dict, evaluation: dict) -> None:
    identity = summary.get("identity", {})
    reconstruction = summary.get("reconstruction", {})
    status = "PASS" if evaluation["evidence_ok"] else "REJECT"
    lines = [
        "# CPU-E5 results — exact FP16 attention projection viability",
        "",
        f"Evidence status: **{status}**.",
        "",
        "## Scope",
        "- The original MiniMind-3 checkpoint stores q/k/v/o projection weights as FP16. Every value from all 8 layers is checked by FP16→FP32→FP16 bit round-trip and against the existing MMCPU001 FP32 artifact.",
        "- This delivery is isolated to `gemv_f16`, four independent `gemv_f16` calls, and `gemv_f16_x4`. It does not modify or call `step()`, prefill, chat, speculative decode, workers or runtime configuration.",
        "- The x1 reference is `gemv_f32`; the x1 candidate is `gemv_f16`. The x4 reference is four independent `gemv_f16` calls, matching CPU-E1; the x4 candidate is `gemv_f16_x4`.",
        "- Each receipt benchmarks all eight matrices of one projection family in aggregate, with distinct deterministic inputs, two warmups, three alternating pairs, one participant pinned to CPU 0, and preparation/parity outside timing.",
        "",
        "## Identity",
        f"- Source commit: `{identity.get('source_commit', 'unavailable')}`.",
        f"- Checkpoint SHA-256: `{identity.get('checkpoint_sha256', 'unavailable')}`; MMCPU001 artifact SHA-256: `{identity.get('artifact_sha256', 'unavailable')}`.",
        f"- Benchmark executable SHA-256: `{identity.get('benchmark_executable_sha256', 'unavailable')}`.",
        f"- Matrices: `{reconstruction.get('matrix_count', 'unavailable')}`; elements: `{reconstruction.get('elements', 'unavailable')}`; FP16 round-trip mismatches: `{reconstruction.get('roundtrip_bit_mismatches', 'unavailable')}`; artifact mismatches: `{reconstruction.get('artifact_bit_mismatches', 'unavailable')}`.",
        f"- Repetitions: `{identity.get('repetitions', 'unavailable')}`; warmups: `{identity.get('warmup_repetitions', 'unavailable')}`; CPU: `{identity.get('cpu', 'unavailable')}`; participants: `{identity.get('threads', 'unavailable')}`.",
        "",
        "## Command",
        "```powershell",
        "python native_cpu/benchmarks/cpu-e5/benchmark.py --out native_cpu/benchmarks/cpu-e5/results --pairs 3 --repetitions 8",
        "python native_cpu/benchmarks/cpu-e5/summarize.py",
        "```",
        "",
        "## Aggregate timing — all eight matrices per projection",
        "| Projection | Shape | x1 FP32 median | x1 FP16 median | x1 median benefit | x1 favorable | x4 independent FP16 median | x4 shared FP16 median | x4 median benefit | x4 favorable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    shapes = {"q_proj": "768×768", "o_proj": "768×768", "k_proj": "384×768", "v_proj": "384×768"}
    for projection in PROJECTIONS:
        data = evaluation["stats"].get(projection)
        if not data:
            lines.append(f"| {projection} | {shapes[projection]} | unavailable | unavailable | unavailable | 0/3 | unavailable | unavailable | unavailable | 0/3 |")
            continue
        pairs = data["pairs"]
        lines.append("| {} | {} | {:.6f} s | {:.6f} s | {:.3f}% | {}/3 | {:.6f} s | {:.6f} s | {:.3f}% | {}/3 |".format(
            projection, shapes[projection],
            statistics.median(row["reference_x1"] for row in pairs), statistics.median(row["candidate_x1"] for row in pairs),
            data["x1_median_benefit"], data["x1_favorable"],
            statistics.median(row["reference_x4"] for row in pairs), statistics.median(row["candidate_x4"] for row in pairs),
            data["x4_median_benefit"], data["x4_favorable"]))
    lines += [
        "",
        "## Raw paired totals",
        "| Projection | Pair | x1 FP32 | x1 FP16 | x1 benefit | x4 independent | x4 shared | x4 benefit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for projection in PROJECTIONS:
        for row in evaluation["stats"].get(projection, {}).get("pairs", []):
            lines.append(f"| {projection} | {row['pair']} | {row['reference_x1']:.6f} s | {row['candidate_x1']:.6f} s | {row['x1_benefit']:.3f}% | {row['reference_x4']:.6f} s | {row['candidate_x4']:.6f} s | {row['x4_benefit']:.3f}% |")
    lines += [
        "",
        "## Decision",
        f"Evidence: **{status}**. This report only establishes exact reconstructibility and isolated kernel measurements; it does not integrate FP16 attention storage into production.",
        "",
        "The next integration decision must use these raw receipts: a clear, reproducible advantage in both x1 and x4 is required before connecting the compact attention weights to CPU-E4. Mixed or negative results remain experimental and are not promoted.",
        "",
        "Raw receipts: `native_cpu/benchmarks/cpu-e5/results/*.json`; reconstruction: `native_cpu/benchmarks/cpu-e5/results/reconstruction.json`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> int:
    try:
        summary = load(OUT / "summary.json")
        grouped, issues = read_receipts()
        evaluation = evaluate(summary, grouped, issues)
    except (OSError, json.JSONDecodeError) as exc:
        summary, evaluation = {}, {"evidence_ok": False, "reasons": [f"invalid evidence: {exc}"], "stats": {}}
    render(summary, evaluation)
    return 0 if evaluation["evidence_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
