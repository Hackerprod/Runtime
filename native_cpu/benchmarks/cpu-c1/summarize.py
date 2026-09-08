"""Render the CPU-C1 accumulated comparison and fail closed on invalid receipts."""
from __future__ import annotations

import importlib.util
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "results"
REPORT = Path(__file__).resolve().parent / "RESULTS.md"
PAIR_COUNT = 6
REQUIRED_CASES = (
    "cpu-c1-conversation-historical",
    "cpu-c1-conversation-second",
    "cpu-c1-forced-256",
    "cpu-c1-forced-1792",
)


def _load_comparator():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_comparator_for_cpu_c1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark comparator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_entries


compare_entries = _load_comparator()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _read_pairs(case_key: str, summary_identity: dict[str, Any], case_summary: dict[str, Any]):
    rows = []
    issues = []
    expected_case = case_summary.get("case")
    for number in range(1, PAIR_COUNT + 1):
        path = OUT / f"{case_key}-pair{number:02d}.json"
        label = f"{case_key}/pair{number:02d}"
        try:
            record = load(path)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{label}: invalid or missing receipt ({exc})")
            continue
        if record.get("status") != "PASS":
            issues.append(f"{label}: pair status is not PASS")
        if record.get("identity") != summary_identity:
            issues.append(f"{label}: identity mismatch")
        case = record.get("case")
        if not isinstance(case, dict) or case.get("key") != case_key:
            issues.append(f"{label}: case identity mismatch")
            continue
        if case != expected_case:
            issues.append(f"{label}: case payload mismatch")
        expected_order = ["A", "B"] if number % 2 else ["B", "A"]
        if record.get("order") != expected_order:
            issues.append(f"{label}: order mismatch")
        entries = record.get("entries")
        by_profile = {
            entry.get("profile"): entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("profile") in {"A", "B"}
        } if isinstance(entries, list) else {}
        if set(by_profile) != {"A", "B"}:
            issues.append(f"{label}: expected A and B entries")
            continue
        workload = case.get("workload")
        try:
            direct = compare_entries(by_profile["A"], by_profile["B"], workload)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{label}: direct parity comparison failed ({exc})")
            continue
        if direct.get("parity") is not True:
            issues.append(f"{label}: direct parity mismatch ({direct.get('field', direct.get('reason', 'raw entries differ'))})")
        stored = record.get("comparison_result")
        if not isinstance(stored, dict) or stored.get("parity") != direct.get("parity"):
            issues.append(f"{label}: stored parity disagrees with direct comparison")
        rows.append((number, record.get("order", expected_order), by_profile["A"], by_profile["B"], direct))
    return rows, issues


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _metric(rows, name: str, *, higher_better: bool = False):
    values = []
    for number, order, left, right, parity in rows:
        if not _is_number(left.get(name)) or not _is_number(right.get(name)):
            return None
        a, b = float(left[name]), float(right[name])
        if a <= 0 or b <= 0:
            return None
        values.append({
            "pair": number,
            "order": order,
            "a": a,
            "b": b,
            "delta": b - a,
            "reduction_percent": 100.0 * ((b - a) if higher_better else (a - b)) / a,
            "parity": parity.get("parity") is True,
        })
    if len(values) != PAIR_COUNT:
        return None
    reductions = [row["reduction_percent"] for row in values]
    deltas = [row["delta"] for row in values]
    return {
        "pairs": values,
        "a_median": statistics.median(row["a"] for row in values),
        "b_median": statistics.median(row["b"] for row in values),
        "delta_median": statistics.median(deltas),
        "reduction_median_percent": statistics.median(reductions),
        "delta_stdev": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
        "favorable": sum(1 for row in values if (row["delta"] > 0 if higher_better else row["delta"] < 0)),
    }


def _case_metrics(case_rows: dict[str, list], workload: str):
    if workload == "forced":
        names = ("prefill_seconds", "native_generation_seconds", "total_seconds", "prefill_tokens_per_second", "decode_tokens_per_second")
    else:
        names = ("prefill_seconds", "native_generation_seconds", "total_seconds", "conversation_seconds", "decode_tokens_per_second")
    return {name: _metric(case_rows, name, higher_better=name.endswith("tokens_per_second")) for name in names}


def evaluate(summary: dict[str, Any]):
    reasons = []
    rows_by_case: dict[str, list] = {}
    if summary.get("status") != "PASS":
        reasons.append("summary status is not PASS")
    if summary.get("final_protocol") is not True:
        reasons.append("summary does not contain the final six-pair protocol")
    identity = summary.get("identity")
    if not isinstance(identity, dict):
        reasons.append("summary identity is missing")
        identity = {}
    cases = {
        case.get("case", {}).get("key"): case
        for case in summary.get("cases", [])
        if isinstance(case, dict) and isinstance(case.get("case"), dict)
    }
    if set(cases) != set(REQUIRED_CASES):
        reasons.append("summary case set does not match the four-case CPU-C1 protocol")
    for key in REQUIRED_CASES:
        case_summary = cases.get(key)
        if case_summary is None:
            reasons.append(f"missing case summary: {key}")
            rows_by_case[key] = []
            continue
        if case_summary.get("pairs") != PAIR_COUNT:
            reasons.append(f"incomplete case summary: {key}")
        rows, issues = _read_pairs(key, identity, case_summary)
        rows_by_case[key] = rows
        reasons.extend(issues)
        if len(rows) != PAIR_COUNT:
            reasons.append(f"incomplete raw pair set: {key}")
        if case_summary.get("parity") is not True:
            reasons.append(f"stored case parity failed: {key}")
        workload = case_summary.get("case", {}).get("workload")
        metrics = _case_metrics(rows, workload) if len(rows) == PAIR_COUNT else {}
        if any(value is None for value in metrics.values()):
            reasons.append(f"missing or invalid timing evidence: {key}")
    required_identity = {
        "reference_runtime_commit", "candidate_runtime_commit", "measurement_tree_commit",
        "reference_library_sha256", "candidate_library_sha256", "model_sha256", "tokenizer_sha256",
    }
    if not required_identity.issubset(identity):
        reasons.append("identity is missing immutable source or artifact identifiers")
    if identity.get("cpus") != [0] or identity.get("threads") != 1 or identity.get("profiler") is not False:
        reasons.append("CPU-C1 participant or profiler policy is not recorded correctly")
    return {"ok": not reasons, "reasons": reasons, "rows": rows_by_case}


def _fmt_seconds(value: float | None):
    return "unavailable" if value is None else f"{value:.6f} s"


def _fmt_rate(value: float | None):
    return "unavailable" if value is None else f"{value:.3f} tok/s"


def _summary_row(key: str, label: str, metric: dict[str, Any] | None, unit: str):
    if metric is None:
        return f"| {key} | {label} | unavailable | unavailable | unavailable | unavailable | 0/6 |"
    if unit == "s":
        a, b, delta = (_fmt_seconds(metric[name]) for name in ("a_median", "b_median", "delta_median"))
    else:
        a, b, delta = (_fmt_rate(metric[name]) for name in ("a_median", "b_median", "delta_median"))
    return f"| {key} | {label} | {a} | {b} | {delta} | {metric['reduction_median_percent']:.3f}% | {metric['favorable']}/6 |"


def _details(metric: dict[str, Any] | None, unit: str):
    if metric is None:
        return "unavailable"
    lines = ["| Pair | Order | A | B | B-A | A→B | Parity |", "|---:|---|---:|---:|---:|---:|---|"]
    for row in metric["pairs"]:
        if unit == "s":
            a, b, delta = f"{row['a']:.6f} s", f"{row['b']:.6f} s", f"{row['delta']:.6f} s"
        elif unit == "tok/s":
            a, b, delta = f"{row['a']:.3f} tok/s", f"{row['b']:.3f} tok/s", f"{row['delta']:.3f} tok/s"
        else:
            a, b, delta = f"{row['a']:.6f}", f"{row['b']:.6f}", f"{row['delta']:.6f}"
        lines.append(f"| {row['pair']} | {'→'.join(row['order'])} | {a} | {b} | {delta} | {row['reduction_percent']:.3f}% | {row['parity']} |")
    return "\n".join(lines)


def render(summary: dict[str, Any], evaluation: dict[str, Any]):
    identity = summary.get("identity", {})
    rows = evaluation["rows"]
    status = "PASS" if evaluation["ok"] else "REJECT"
    lines = [
        "# CPU-C1 results — original to cumulative CPU-R6",
        "",
        f"Status: **{status}**.",
        "",
        "## Scope",
        "- A is the frozen pre-CPU-R1 DLL. B is the CPU-R6 DLL with selective logits, KV reuse, V-blocked attention, shared-K GQA, shared-V GQA and exact FP16 FFN storage enabled; FFN row4 is disabled.",
        "- Both sides use the same model, tokenizer, chat template, sampling policy and seed, with one participant pinned to CPU 0. Profiling is disabled for all speed receipts.",
        "- This is an accumulated closure comparison, not a new optimization experiment. Percentages are calculated from paired A/B samples and are not summed from earlier campaigns.",
        "",
        "## Immutable identity",
        f"- Reference runtime commit: `{identity.get('reference_runtime_commit', 'unavailable')}`; DLL SHA-256: `{identity.get('reference_library_sha256', 'unavailable')}`.",
        f"- Candidate runtime commit: `{identity.get('candidate_runtime_commit', 'unavailable')}`; DLL SHA-256: `{identity.get('candidate_library_sha256', 'unavailable')}`.",
        f"- Measurement tree commit: `{identity.get('measurement_tree_commit', 'unavailable')}`; model SHA-256: `{identity.get('model_sha256', 'unavailable')}`.",
        "",
        "## Benchmark command",
        "```powershell",
        "python native_cpu/benchmarks/cpu-c1/benchmark.py --out native_cpu/benchmarks/cpu-c1/results --pairs 6",
        "python native_cpu/benchmarks/cpu-c1/summarize.py",
        "```",
        "",
        "Raw receipts are under `native_cpu/benchmarks/cpu-c1/results/`. Loading and FP16-storage preparation are recorded per entry and excluded from inference timings.",
        "",
        "## Original → cumulative final",
        "| Case | Metric | A median | B median | Median B-A | Median A→B benefit | B favorable |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    metric_specs = {
        "forced": (("prefill_seconds", "prefill", "s"), ("native_generation_seconds", "decode", "s"), ("total_seconds", "total", "s"), ("decode_tokens_per_second", "decode throughput", "tok/s")),
        "conversation": (("prefill_seconds", "prefill", "s"), ("native_generation_seconds", "decode", "s"), ("total_seconds", "total", "s"), ("conversation_seconds", "conversation wall", "s"), ("decode_tokens_per_second", "decode throughput", "tok/s")),
    }
    metric_specs["historical"] = metric_specs["conversation"]
    metric_specs["second"] = metric_specs["conversation"]
    for key in REQUIRED_CASES:
        case = next((item.get("case", {}) for item in summary.get("cases", []) if item.get("case", {}).get("key") == key), {})
        metrics = _case_metrics(rows.get(key, []), case.get("workload")) if len(rows.get(key, [])) == PAIR_COUNT else {}
        for name, label, unit in metric_specs.get(case.get("workload"), ()):
            lines.append(_summary_row(key, label, metrics.get(name), unit))
    lines += [
        "",
        "## Load and preparation (reported separately)",
        "| Case | A load median | B load median | A configuration median | B configuration median | A FP16 preparation median | B FP16 preparation median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in REQUIRED_CASES:
        case_rows = rows.get(key, [])
        def median_field(field, profile):
            values = [(left if profile == "A" else right).get(field) for _, _, left, right, _ in case_rows]
            return statistics.median(values) if values and all(_is_number(value) for value in values) else None
        lines.append(
            f"| {key} | {_fmt_seconds(median_field('loading_seconds', 'A'))} | {_fmt_seconds(median_field('loading_seconds', 'B'))} | "
            f"{_fmt_seconds(median_field('configuration_seconds', 'A'))} | {_fmt_seconds(median_field('configuration_seconds', 'B'))} | "
            f"{_fmt_seconds(median_field('storage_preparation_seconds', 'A'))} | {_fmt_seconds(median_field('storage_preparation_seconds', 'B'))} |"
        )
    lines += [
        "",
        "## Parity and availability",
        "- Forced-prefix receipts compare exact FP32 logits hashes for the prefill and every decode token, plus token IDs and final position.",
        "- Conversation receipts compare canonical prompt IDs, sampled token IDs, text, finish reason and generated-token counts for each turn. All four cases use six alternating A/B pairs.",
        "- Diagnostic counters are intentionally not required in this comparison; fields absent from the frozen DLL are reported as unavailable rather than zero.",
        "",
        "## Paired details",
    ]
    for key in REQUIRED_CASES:
        case = next((item.get("case", {}) for item in summary.get("cases", []) if item.get("case", {}).get("key") == key), {})
        metrics = _case_metrics(rows.get(key, []), case.get("workload")) if len(rows.get(key, [])) == PAIR_COUNT else {}
        lines += [f"### {key}", "", "#### Decode time", _details(metrics.get("native_generation_seconds"), "s"), "", "#### Total time", _details(metrics.get("total_seconds"), "s")]
    lines += [
        "",
        "## Evidence verdict",
        f"Result: **{status}**" + (f" — {', '.join(evaluation['reasons'])}" if evaluation["reasons"] else "."),
        "",
        "A PASS means the four-case paired evidence is complete, directly parity-checked and identity-consistent. It does not promote an additional kernel or claim that preparation cost is part of sustained decode.",
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
