"""Render the CPU-U1.2 reproducible single-baseline evidence."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "results"
REPORT = Path(__file__).resolve().parent / "RESULTS.md"
REPEAT_COUNT = 3
EXPECTED_SOURCE_COMMIT = "06694632cfe4ee5b97013d6a5b2e6fec9fe98f50"
EXPECTED_LIBRARY_SHA256 = "5956ff064ad8e8fd1ff502d18626ce19240682faa55c584cb3d6eaad1775bb3a"
REQUIRED_CASES = ("cpu-u1.2-forced-256", "cpu-u1.2-forced-1792")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _outputs_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(name) == right.get(name) for name in (
        "prefix_token_ids", "continuation_token_ids", "prefill_logits_sha256",
        "decode_logits_sha256", "position_after", "prefill_tokens_evaluated",
        "decode_tokens_evaluated",
    ))


def _read_case(case: dict[str, Any], summary_identity: dict[str, Any]):
    key = case.get("key")
    rows = []
    issues = []
    expected_case = {"key": key, "workload": "forced", "length": case.get("length")}
    for number in range(1, REPEAT_COUNT + 1):
        path = OUT / f"{key}-run{number:02d}.json"
        label = f"{key}/run{number:02d}"
        try:
            record = load(path)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{label}: invalid or missing receipt ({exc})")
            continue
        if record.get("status") != "PASS":
            issues.append(f"{label}: status is not PASS")
        if record.get("identity") != summary_identity:
            issues.append(f"{label}: identity mismatch")
        if record.get("case") != expected_case:
            issues.append(f"{label}: case mismatch")
        if record.get("run") != number:
            issues.append(f"{label}: repetition number mismatch")
        entry = record.get("entry")
        if not isinstance(entry, dict):
            issues.append(f"{label}: entry is missing")
            continue
        if entry.get("library_sha256") != EXPECTED_LIBRARY_SHA256:
            issues.append(f"{label}: baseline DLL hash mismatch")
        rows.append((number, entry))
    if len(rows) == REPEAT_COUNT:
        first = rows[0][1]
        for number, entry in rows[1:]:
            if not _outputs_equal(first, entry):
                issues.append(f"{key}/run{number:02d}: fixed-token logits or outputs differ")
    return rows, issues


def _metrics(rows: list[tuple[int, dict[str, Any]]]):
    fields = (
        "prefill_seconds", "native_generation_seconds", "total_seconds",
        "prefill_tokens_per_second", "decode_tokens_per_second",
    )
    result = {}
    for field in fields:
        values = [float(entry.get(field)) for _, entry in rows]
        if len(values) != REPEAT_COUNT or not all(_is_number(value) and value > 0 for value in values):
            result[field] = None
            continue
        result[field] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values),
        }
    return result


def evaluate(summary: dict[str, Any]):
    reasons = []
    if summary.get("status") != "PASS":
        reasons.append("summary status is not PASS")
    if summary.get("final_protocol") is not True:
        reasons.append("summary does not contain the final three-repetition protocol")
    identity = summary.get("identity")
    if not isinstance(identity, dict):
        reasons.append("summary identity is missing")
        identity = {}
    if identity.get("source_commit") != EXPECTED_SOURCE_COMMIT:
        reasons.append("source commit is not the CPU-U1 commit 06694632")
    if identity.get("library_sha256") != EXPECTED_LIBRARY_SHA256:
        reasons.append("baseline DLL hash is not the rebuilt CPU-U1 identity")
    if identity.get("threads") != 1 or identity.get("cpus") != [0] or identity.get("profiler") is not False:
        reasons.append("baseline participant or profiler policy is incorrect")
    cases = {
        case.get("case", {}).get("key"): case
        for case in summary.get("cases", [])
        if isinstance(case, dict) and isinstance(case.get("case"), dict)
    }
    if set(cases) != set(REQUIRED_CASES):
        reasons.append("summary case set does not match CPU-U1.2")
    rows_by_case = {}
    for key in REQUIRED_CASES:
        case_summary = cases.get(key)
        if case_summary is None:
            reasons.append(f"missing case summary: {key}")
            rows_by_case[key] = []
            continue
        case = case_summary.get("case", {})
        if case.get("workload") != "forced" or case.get("length") not in (256, 1792):
            reasons.append(f"invalid forced case definition: {key}")
        if case_summary.get("repetitions") != REPEAT_COUNT or case_summary.get("output_parity") is not True:
            reasons.append(f"incomplete or non-parity case summary: {key}")
        rows, issues = _read_case(case, identity)
        rows_by_case[key] = rows
        reasons.extend(issues)
        if len(rows) != REPEAT_COUNT:
            reasons.append(f"incomplete repetition set: {key}")
        metrics = _metrics(rows) if len(rows) == REPEAT_COUNT else {}
        if any(value is None for value in metrics.values()):
            reasons.append(f"missing or invalid timing evidence: {key}")
        for _, entry in rows:
            effective = entry.get("effective")
            expected = {
                "threads": 1, "cpus": [0], "row_weights": [1],
                "selective_logits": True, "v_blocked_attention": True,
                "ffn_row4": False, "ffn_f16_storage": True,
                "gqa_k_shared": True, "gqa_v_shared": True,
                "diagnostics": False,
            }
            if effective != expected:
                reasons.append(f"automatic CPU-U1 route differs from expected effective configuration: {key}")
    return {"ok": not reasons, "reasons": reasons, "rows": rows_by_case}


def _fmt(value: float | None, unit: str = "s"):
    if value is None:
        return "unavailable"
    return f"{value:.6f} {unit}" if unit == "s" else f"{value:.3f} {unit}"


def _metric_row(key: str, label: str, metric: dict[str, float] | None, unit: str):
    if metric is None:
        return f"| {key} | {label} | unavailable | unavailable | unavailable | unavailable |"
    return (f"| {key} | {label} | {_fmt(metric['median'], unit)} | {_fmt(metric['min'], unit)} | "
            f"{_fmt(metric['max'], unit)} | {_fmt(metric['stdev'], unit)} |")


def render(summary: dict[str, Any], evaluation: dict[str, Any]):
    identity = summary.get("identity", {})
    status = "PASS" if evaluation["ok"] else "REJECT"
    rows = evaluation["rows"]
    lines = [
        "# CPU-U1.2 reproducible baseline",
        "",
        f"Status: **{status}**.",
        "",
        "## Scope",
        "- This freezes CPU-U1 as a single current baseline; it does not compare against CPU-R6 and makes no historical performance claim.",
        "- The DLL was rebuilt from source commit `06694632cfe4ee5b97013d6a5b2e6fec9fe98f50` with one participant pinned to CPU 0 and profiling disabled.",
        "- Two forced-token workloads (256 and 1792 prefix tokens) use three repetitions each. CPU-E1 is not integrated.",
        "",
        "## Immutable identity",
        f"- Source commit: `{identity.get('source_commit', 'unavailable')}`.",
        f"- Measurement tree commit: `{identity.get('measurement_tree_commit', 'unavailable')}`.",
        f"- Baseline DLL SHA-256: `{identity.get('library_sha256', 'unavailable')}`.",
        f"- Build-info SHA-256: `{identity.get('build_info_sha256', 'unavailable')}`.",
        f"- Model SHA-256: `{identity.get('model_sha256', 'unavailable')}`.",
        "",
        "## Commands",
        "```powershell",
        "python native_cpu/tools/build.py --cmake \"C:\\Program Files (x86)\\Microsoft Visual Studio\\18\\BuildTools\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe\" --build-dir native_cpu/build-u1.2",
        "python native_cpu/benchmarks/cpu-u1.2/benchmark.py --out native_cpu/benchmarks/cpu-u1.2/results --repeats 3",
        "python native_cpu/benchmarks/cpu-u1.2/summarize.py",
        "```",
        "",
        "`loading_seconds`, FP16 preparation and compact-storage bytes are reported separately; sustained inference timings exclude preparation.",
        "",
        "## Sustained baseline timings",
        "| Case | Metric | Median | Minimum | Maximum | Stdev |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in REQUIRED_CASES:
        case = next((item.get("case", {}) for item in summary.get("cases", []) if item.get("case", {}).get("key") == key), {})
        metrics = _metrics(rows.get(key, [])) if len(rows.get(key, [])) == REPEAT_COUNT else {}
        specs = (
            ("prefill_seconds", "prefill", "s"),
            ("native_generation_seconds", "decode", "s"),
            ("total_seconds", "total", "s"),
            ("prefill_tokens_per_second", "prefill throughput", "tok/s"),
            ("decode_tokens_per_second", "decode throughput", "tok/s"),
        )
        for name, label, unit in specs:
            lines.append(_metric_row(key, label, metrics.get(name), unit))
    lines += [
        "",
        "## Preparation and memory",
        "| Case | Load median | Configuration median | FP16 preparation median | Compact bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in REQUIRED_CASES:
        case_rows = rows.get(key, [])
        def median_field(field):
            values = [entry.get(field) for _, entry in case_rows]
            return statistics.median(values) if values and all(_is_number(value) for value in values) else None
        compact = median_field("ffn_f16_storage_bytes")
        compact_text = "unavailable" if compact is None else f"{int(compact):,} bytes"
        lines.append(
            f"| {key} | {_fmt(median_field('loading_seconds'))} | {_fmt(median_field('configuration_seconds'))} | "
            f"{_fmt(median_field('storage_preparation_seconds'))} | {compact_text} |"
        )
    lines += [
        "",
        "## Exact output consistency",
        "- Each repetition stores FP32 prefill logits and every decode-logit SHA-256, plus the fixed token IDs and final position.",
        "- The evaluator compares those output fields across all three repetitions; any divergence rejects the report.",
        "- The effective route is automatic CPU-U1: selective logits, blocked V, shared K/V GQA and exact FP16 FFN storage; FFN row4 and diagnostics remain disabled.",
        "",
        "## Evidence verdict",
        f"Result: **{status}**" + (f" — {', '.join(evaluation['reasons'])}" if evaluation["reasons"] else "."),
        "",
        "A PASS establishes this rebuilt CPU-U1 artifact and these measurements as the reproducible baseline for CPU-E1 integration. It does not claim a gain relative to CPU-R6.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> int:
    try:
        summary = load(OUT / "summary.json")
        evaluation = evaluate(summary)
    except (OSError, json.JSONDecodeError) as exc:
        summary = {}
        evaluation = {"ok": False, "reasons": [f"invalid summary.json: {exc}"], "rows": {}}
    render(summary, evaluation)
    return 0 if evaluation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
