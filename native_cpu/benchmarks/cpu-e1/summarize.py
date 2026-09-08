"""Validate CPU-E1 receipts and render the isolated kernel result."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parent / "results"
REPORT = Path(__file__).resolve().parent / "RESULTS.md"
PAIR_COUNT = 6
SHAPES = ("2432x768", "768x2432")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def read_pairs(identity: dict[str, Any], summary: dict[str, Any]):
    rows_by_shape: dict[str, list[dict[str, Any]]] = {shape: [] for shape in SHAPES}
    issues: list[str] = []
    for shape in SHAPES:
        for pair in range(1, PAIR_COUNT + 1):
            path = OUT / f"cpu-e1-{shape}-pair{pair:02d}.json"
            label = f"{shape}/pair{pair:02d}"
            try:
                row = load(path)
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"{label}: missing or invalid receipt ({exc})")
                continue
            if row.get("identity") != identity:
                issues.append(f"{label}: identity mismatch")
            expected_order = "reference-shared" if pair % 2 else "shared-reference"
            if row.get("order") != expected_order:
                issues.append(f"{label}: alternating order mismatch")
            if row.get("pair") != pair or row.get("shape") != shape:
                issues.append(f"{label}: shape/pair identity mismatch")
            if row.get("cpu") != 0 or row.get("threads") != 1:
                issues.append(f"{label}: participant is not pinned to CPU 0")
            if row.get("parity") is not True or row.get("exact_memcmp_parity") is not True:
                issues.append(f"{label}: exact output parity failed or was not checked")
            if row.get("reference_checksum") != row.get("shared_checksum"):
                issues.append(f"{label}: post-timing checksum mismatch")
            for name in ("reference_seconds", "shared_seconds"):
                if not finite_positive(row.get(name)):
                    issues.append(f"{label}: invalid {name}")
            if row.get("repetitions") != identity.get("repetitions"):
                issues.append(f"{label}: repetition count mismatch")
            rows_by_shape[shape].append(row)
    return rows_by_shape, issues


def stats(rows: list[dict[str, Any]]):
    deltas = [float(row["shared_seconds"]) - float(row["reference_seconds"]) for row in rows]
    benefits = [100.0 * (float(row["reference_seconds"]) - float(row["shared_seconds"])) / float(row["reference_seconds"]) for row in rows]
    return {
        "reference_median": statistics.median(float(row["reference_seconds"]) for row in rows),
        "shared_median": statistics.median(float(row["shared_seconds"]) for row in rows),
        "delta_median": statistics.median(deltas),
        "benefit_median_percent": statistics.median(benefits),
        "delta_stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "favorable": sum(1 for delta in deltas if delta < 0),
        "rows": rows,
    }


def evaluate(summary: dict[str, Any]):
    reasons: list[str] = []
    if summary.get("status") != "PASS":
        reasons.append("summary status is not PASS")
    identity = summary.get("identity")
    if not isinstance(identity, dict):
        reasons.append("identity is missing")
        identity = {}
    if identity.get("pairs") != PAIR_COUNT or identity.get("cpu") != 0 or identity.get("threads") != 1:
        reasons.append("benchmark protocol identity is invalid")
    rows_by_shape, receipt_issues = read_pairs(identity, summary)
    reasons.extend(receipt_issues)
    shape_stats = {}
    for shape, rows in rows_by_shape.items():
        if len(rows) != PAIR_COUNT:
            reasons.append(f"incomplete pair set for {shape}")
            continue
        shape_stats[shape] = stats(rows)
    evidence_ok = not reasons and set(shape_stats) == set(SHAPES)
    performance_ok = evidence_ok and all(
        shape_stats[shape]["favorable"] == PAIR_COUNT and shape_stats[shape]["benefit_median_percent"] > 0
        for shape in SHAPES
    )
    return {
        "evidence_ok": evidence_ok,
        "performance_ok": performance_ok,
        "ok": evidence_ok and performance_ok,
        "reasons": reasons,
        "rows": rows_by_shape,
        "stats": shape_stats,
    }


def render(summary: dict[str, Any], evaluation: dict[str, Any]):
    identity = summary.get("identity", {})
    status = "PASS" if evaluation["evidence_ok"] else "REJECT"
    performance = "ACCEPTED FOR NEXT INTEGRATION" if evaluation["performance_ok"] else "EXPERIMENTAL ONLY"
    lines = [
        "# CPU-E1 results — four-input FP16 GEMV",
        "",
        f"Evidence status: **{status}**.",
        f"Performance decision: **{performance}**.",
        "",
        "## Scope",
        "- The candidate kernel processes one FP16 weight row at a time, converts each eight-weight fragment once and reuses that FP32 fragment across four independent FP32 inputs.",
        "- The reference performs four independent `gemv_f16` calls. Each candidate output preserves CPU-R6's FMA order, horizontal reduction and scalar tail; the kernel is not connected to `step()`, chat or scheduling.",
        "- Preparation, allocation, parity checks and warmups are outside the timed intervals. The benchmark is pinned to one participant on CPU 0 and reports four-output wall time, not model tokens/s.",
        "",
        "## Identity",
        f"- Source commit: `{identity.get('source_commit', 'unavailable')}`; benchmark tree: `{identity.get('benchmark_tree_commit', 'unavailable')}`.",
        f"- Executable SHA-256: `{identity.get('benchmark_executable_sha256', 'unavailable')}`.",
        f"- Repetitions per timed sample: `{identity.get('repetitions', 'unavailable')}`; warmups: `{identity.get('warmup_repetitions', 'unavailable')}`; CPU: `{identity.get('cpu', 'unavailable')}`.",
        "",
        "## Command",
        "```powershell",
        "python native_cpu/benchmarks/cpu-e1/benchmark.py --out native_cpu/benchmarks/cpu-e1/results --repetitions 64",
        "python native_cpu/benchmarks/cpu-e1/summarize.py",
        "```",
        "",
        "## Summary — time for four outputs",
        "| Shape | Reference median | Shared-weight median | Median shared-reference | Median benefit | Favorable pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for shape in SHAPES:
        data = evaluation["stats"].get(shape)
        if data is None:
            lines.append(f"| {shape} | unavailable | unavailable | unavailable | unavailable | 0/6 |")
        else:
            lines.append(f"| {shape} | {data['reference_median']:.9f} s | {data['shared_median']:.9f} s | {data['delta_median']:.9f} s | {data['benefit_median_percent']:.3f}% | {data['favorable']}/6 |")
    lines += [
        "",
        "## Raw pair details",
        "| Shape | Pair | Order | Reference | Shared | Shared-reference | Benefit | Exact parity |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for shape in SHAPES:
        for row in evaluation["rows"].get(shape, []):
            reference = float(row.get("reference_seconds", 0.0))
            shared = float(row.get("shared_seconds", 0.0))
            benefit = 100.0 * (reference - shared) / reference if reference > 0 else float("nan")
            lines.append(f"| {shape} | {row.get('pair', '?')} | {row.get('order', '?')} | {reference:.9f} s | {shared:.9f} s | {shared-reference:.9f} s | {benefit:.3f}% | {row.get('exact_memcmp_parity', False)} |")
    lines += [
        "",
        "## Verification",
        "- The C++ harness compares every output buffer with `memcmp` before timing and checks the output checksums after timing.",
        "- Shapes tested: 2432×768 and 768×2432, plus small incomplete-column fixtures (3×9, 5×17, 7×31 and 4×8) in the native kernel tests.",
        "- F16C availability is recorded in every receipt; no assumption based only on AVX2 is made.",
        "",
        "## Verdict",
        f"Result: **{status}**; performance: **{performance}**." + (f" Reasons: {', '.join(evaluation['reasons'])}." if evaluation["reasons"] else ""),
        "",
        "A performance decision is reproducible only when all six pairs for both large shapes favor the shared-weight kernel. Mixed results remain documented and are not promoted.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> int:
    try:
        summary = load(OUT / "summary.json")
        evaluation = evaluate(summary)
    except (OSError, json.JSONDecodeError) as exc:
        summary = {}
        evaluation = {"evidence_ok": False, "performance_ok": False, "ok": False, "reasons": [f"invalid summary: {exc}"], "rows": {}, "stats": {}}
    render(summary, evaluation)
    return 0 if evaluation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
