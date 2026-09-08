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
    "cpu-r2-speed-V0-V1-256",
    "cpu-r2-speed-V0-V1-1792",
    "cpu-r2-diagnostics-V0-V1-256",
    "cpu-r2-diagnostics-V0-V1-1792",
)


def _load_compare_entries():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_benchmark_for_cpu_r2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark comparator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_entries


compare_entries = _load_compare_entries()


def load(name: str, out: Path = OUT):
    return json.loads((Path(out) / name).read_text(encoding="utf8"))


def pct(delta: float, base: float) -> float:
    return -100.0 * delta / base if base else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _read_pairs(
    case_key: str,
    *,
    out: Path,
    case_summary: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
) -> tuple[list[tuple[int, list[str], dict[str, Any], dict[str, Any], dict[str, Any]]], list[str]]:
    """Read pair receipts and validate their raw parity evidence.

    The summary's stored ``comparison_result`` is metadata only.  The
    comparator is rerun over the two raw entries for every pair.
    """
    rows = []
    issues: list[str] = []
    expected_case = case_summary.get("case") if case_summary else None
    for number in range(1, PAIR_COUNT + 1):
        path = Path(out) / f"{case_key}-pair{number:02d}.json"
        label = f"{case_key}/pair{number:02d}"
        try:
            record = json.loads(path.read_text(encoding="utf8"))
        except FileNotFoundError:
            issues.append(f"missing evidence: {path.name}")
            continue
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"invalid evidence {path.name}: {exc}")
            continue
        if not isinstance(record, dict):
            issues.append(f"invalid evidence {path.name}: record is not an object")
            continue
        if record.get("status") != "PASS":
            issues.append(f"{label}: pair status is not PASS")
        if identity is not None and record.get("identity") != identity:
            issues.append(f"{label}: identity mismatch")
        case = record.get("case")
        if not isinstance(case, dict) or case.get("key") != case_key:
            issues.append(f"{label}: case identity mismatch")
            continue
        if expected_case is not None and case != expected_case:
            issues.append(f"{label}: case payload mismatch")
        expected_order = ["V0", "V1"] if number % 2 else ["V1", "V0"]
        if record.get("order") != expected_order:
            issues.append(f"{label}: order mismatch")
        entries = record.get("entries")
        if not isinstance(entries, list) or len(entries) != 2:
            issues.append(f"{label}: expected exactly two entries")
            continue
        by_profile: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("profile"), str):
                issues.append(f"{label}: malformed profile entry")
                continue
            profile = entry["profile"]
            if profile in by_profile:
                issues.append(f"{label}: duplicate profile {profile}")
            by_profile[profile] = entry
        if set(by_profile) != {"V0", "V1"}:
            issues.append(f"{label}: expected V0 and V1 entries")
            continue
        try:
            direct = compare_entries(by_profile["V0"], by_profile["V1"], case.get("workload", "forced"))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{label}: direct parity comparison failed: {exc}")
            continue
        if not direct.get("parity", False):
            detail = direct.get("field") or direct.get("reason") or "raw entries differ"
            issues.append(f"{label}: direct parity mismatch ({detail})")
        stored = record.get("comparison_result")
        if not isinstance(stored, dict) or stored.get("parity") != direct.get("parity"):
            issues.append(f"{label}: stored parity disagrees with direct comparison")
        rows.append((number, record.get("order", expected_order), by_profile["V0"], by_profile["V1"], direct))
    return rows, issues


def paired(case_key: str, out: Path = OUT):
    rows, issues = _read_pairs(case_key, out=Path(out))
    if issues:
        raise ValueError("; ".join(issues))
    return rows


def metric_rows_from(rows, metric: str):
    out = []
    for i, order, left, right, cmp in rows:
        delta = right[metric] - left[metric]
        out.append((i, order, left[metric], right[metric], delta, pct(delta, left[metric]), cmp["parity"]))
    return out


def metric_rows(case_key: str, metric: str, out: Path = OUT):
    return metric_rows_from(paired(case_key, out), metric)


def attention_rows_from(rows, phase: str):
    out = []
    for i, order, left, right, cmp in rows:
        l = left["native_phase_stats"][phase]["attention_kv_ns"] / 1e6
        r = right["native_phase_stats"][phase]["attention_kv_ns"] / 1e6
        delta = r - l
        out.append((i, order, l, r, delta, pct(delta, l), cmp["parity"]))
    return out


def attention_rows(case_key: str, phase: str, out: Path = OUT):
    return attention_rows_from(paired(case_key, out), phase)


def summarize(rows):
    left = [r[2] for r in rows]
    right = [r[3] for r in rows]
    delta = [r[4] for r in rows]
    benefit = [r[5] for r in rows]
    return {
        "left_median": statistics.median(left),
        "right_median": statistics.median(right),
        "delta_median": statistics.median(delta),
        "delta_stdev": stdev(delta),
        "benefit_median_percent": statistics.median(benefit),
        "all_improved": all(x < 0 for x in delta),
    }


def _checked_metric_rows(rows, metric: str):
    values = metric_rows_from(rows, metric)
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value))
           for row in values for value in row[2:6]):
        raise ValueError(f"non-finite {metric} value")
    if metric in ("native_generation_seconds", "decode_tokens_per_second") and any(row[2] <= 0 or row[3] <= 0 for row in values):
        raise ValueError(f"non-positive {metric} value")
    return values


def _checked_attention_rows(rows, phase: str):
    values = attention_rows_from(rows, phase)
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value))
           for row in values for value in row[2:6]):
        raise ValueError(f"non-finite attention {phase} value")
    if any(row[2] < 0 or row[3] < 0 for row in values):
        raise ValueError(f"negative attention {phase} value")
    return values


def table(rows, unit: str):
    lines = [
        "| Pair | Order | V0 reference | V1 blocked | V1-V0 | Benefit | Parity |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for i, order, left, right, delta, benefit, parity in rows:
        lines.append(f"| {i} | {'→'.join(order)} | {left:.6f} {unit} | {right:.6f} {unit} | {delta:.6f} {unit} | {benefit:.3f}% | {parity} |")
    return "\n".join(lines)


def evaluate(summary: dict[str, Any], out: Path = OUT) -> dict[str, Any]:
    """Return the single verdict used by both the report and the exit code."""
    out = Path(out)
    reasons: list[str] = []
    rows_by_case: dict[str, list] = {}
    if summary.get("status") != "PASS":
        reasons.append("summary status is not PASS")
    if summary.get("final_protocol") is not True:
        reasons.append("summary does not contain the final six-pair protocol")
    case_map = {
        c.get("case", {}).get("key"): c
        for c in summary.get("cases", [])
        if isinstance(c, dict) and isinstance(c.get("case"), dict)
    }
    identity = summary.get("identity")
    for case_key in REQUIRED_CASES:
        case_summary = case_map.get(case_key)
        if case_summary is None:
            reasons.append(f"missing case summary: {case_key}")
            rows_by_case[case_key] = []
            continue
        if case_summary.get("pairs") != PAIR_COUNT:
            reasons.append(f"incomplete case summary: {case_key}")
        if case_summary.get("parity") is not True:
            reasons.append(f"stored case parity failed: {case_key}")
        rows, issues = _read_pairs(case_key, out=out, case_summary=case_summary, identity=identity)
        rows_by_case[case_key] = rows
        reasons.extend(issues)
        if len(rows) != PAIR_COUNT:
            reasons.append(f"incomplete raw pair set: {case_key}")

    for length in (256, 1792):
        speed_rows = rows_by_case.get(f"cpu-r2-speed-V0-V1-{length}", [])
        attention_case_rows = rows_by_case.get(f"cpu-r2-diagnostics-V0-V1-{length}", [])
        if len(speed_rows) == PAIR_COUNT:
            try:
                speed = _checked_metric_rows(speed_rows, "native_generation_seconds")
                _checked_metric_rows(speed_rows, "decode_tokens_per_second")
                speed_summary = summarize(speed)
                if not speed_summary["all_improved"]:
                    reasons.append(f"decode speed regression at {length}")
                if abs(speed_summary["delta_median"]) <= speed_summary["delta_stdev"]:
                    reasons.append(f"decode benefit inside dispersion at {length}")
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(f"invalid decode speed evidence at {length}: {exc}")
        if len(attention_case_rows) == PAIR_COUNT:
            try:
                _checked_attention_rows(attention_case_rows, "prefill")
                attention = _checked_attention_rows(attention_case_rows, "decode")
                if not summarize(attention)["all_improved"]:
                    reasons.append(f"attention diagnostic regression at {length}")
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(f"invalid attention evidence at {length}: {exc}")
    return {"ok": not reasons, "reasons": reasons, "rows": rows_by_case}


def _summary_line(rows, label, metric, unit):
    if len(rows) != PAIR_COUNT:
        return f"| {label} | {metric} | unavailable | unavailable | unavailable | unavailable | unavailable | False |"
    values = _checked_metric_rows(rows, metric)
    s = summarize(values)
    if metric == "decode_tokens_per_second":
        deltas = [r[3] - r[2] for r in values]
        benefits = [100.0 * (r[3] - r[2]) / r[2] for r in values]
        return f"| {label} | {metric} | {s['left_median']:.6f} {unit} | {s['right_median']:.6f} {unit} | {statistics.median(deltas):.6f} {unit} | {stdev(deltas):.6f} | {statistics.median(benefits):.3f}% | {all(v > 0 for v in deltas)} |"
    return f"| {label} | {metric} | {s['left_median']:.6f} {unit} | {s['right_median']:.6f} {unit} | {s['delta_median']:.6f} {unit} | {s['delta_stdev']:.6f} | {s['benefit_median_percent']:.3f}% | {s['all_improved']} |"


def render_report(summary: dict[str, Any], evaluation: dict[str, Any], report_path: Path, out: Path):
    rows_by_case = evaluation["rows"]
    ok = evaluation["ok"]
    lines = ["# CPU-R2 results — V accumulation locality", ""]
    if ok:
        lines.append("Status: **accepted**. The opt-in V-blocked attention route preserved bitwise logits/tokens/text and improved measured attention plus decode on both requested prefix lengths.")
    else:
        lines.append("Status: **rejected**. The supplied evidence did not satisfy the existing CPU-R2 parity, completeness, or performance criteria.")
    lines += [
        "",
        "## Scope guard",
        "- Only the post-softmax accumulation over V was changed.",
        "- Q·K, softmax, KV layout, FFN, weights, quantization, workers, affinities and defaults were not changed.",
        "- Route is opt-in with `--v-blocked-attention`; default remains the original accumulation route.",
        "",
        "## Benchmark command",
        "```powershell",
        "python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r2 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r2/results",
        "```",
        "",
        f"Benchmark status: `{summary.get('status', 'unavailable')}`; final protocol: `{summary.get('final_protocol', False)}`; raw pair files: `native_cpu/benchmarks/cpu-r2/results/*.json`.",
    ]
    chat_path = Path(out).parent / "chat-parity.json"
    if chat_path.is_file():
        try:
            chat = json.loads(chat_path.read_text(encoding="utf8"))
            lines.append(f"Chat token/text parity: `native_cpu/benchmarks/cpu-r2/chat-parity.json` = `{chat['status']}` for the historical six-turn conversation under CPU-R1 flags plus CPU 0 pinning.")
        except (OSError, json.JSONDecodeError, KeyError):
            lines.append("Chat token/text parity: unavailable (invalid chat-parity.json).")
    # Never touch metric-dependent tables after an invalid evaluation.  This
    # replaces any previous report with a fail-closed rejection instead of
    # allowing a rendering exception to leave stale `accepted` evidence.
    if not ok:
        lines += [
            "",
            "## Acceptance check",
            "Result: **REJECT** — " + ", ".join(evaluation["reasons"]),
        ]
        Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf8")
        return
    lines += [
        "",
        "## Summary",
        "| Case | Metric | V0 median | V1 median | Median V1-V0 | Δ stdev | Median benefit | All pairs improved |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for length in (256, 1792):
        speed_key = f"cpu-r2-speed-V0-V1-{length}"
        diagnostic_key = f"cpu-r2-diagnostics-V0-V1-{length}"
        lines.append(_summary_line(rows_by_case.get(speed_key, []), f"speed {length}", "native_generation_seconds", "s"))
        lines.append(_summary_line(rows_by_case.get(speed_key, []), f"speed {length}", "decode_tokens_per_second", "tok/s"))
        for phase in ("prefill", "decode"):
            rows = rows_by_case.get(diagnostic_key, [])
            if len(rows) == PAIR_COUNT:
                values = _checked_attention_rows(rows, phase)
                s = summarize(values)
                lines.append(f"| diagnostics {length} | attention_{phase}_ms | {s['left_median']:.6f} ms | {s['right_median']:.6f} ms | {s['delta_median']:.6f} ms | {s['delta_stdev']:.6f} | {s['benefit_median_percent']:.3f}% | {s['all_improved']} |")
            else:
                lines.append(f"| diagnostics {length} | attention_{phase}_ms | unavailable | unavailable | unavailable | unavailable | unavailable | False |")
    lines += ["", "## Paired speed details"]
    for length in (256, 1792):
        lines.append(f"### Prefix {length}: native decode/generation seconds")
        rows = rows_by_case.get(f"cpu-r2-speed-V0-V1-{length}", [])
        lines.append(table(_checked_metric_rows(rows, "native_generation_seconds"), "s") if len(rows) == PAIR_COUNT else "Evidence unavailable: incomplete pair set.")
        lines.append("")
    lines.append("## Paired diagnostic attention details")
    for length in (256, 1792):
        for phase in ("prefill", "decode"):
            lines.append(f"### Prefix {length}: attention {phase}")
            rows = rows_by_case.get(f"cpu-r2-diagnostics-V0-V1-{length}", [])
            lines.append(table(_checked_attention_rows(rows, phase), "ms") if len(rows) == PAIR_COUNT else "Evidence unavailable: incomplete pair set.")
            lines.append("")
    lines += ["## Acceptance check", f"Result: **{'PASS' if ok else 'REJECT'}**" + (f" — {', '.join(evaluation['reasons'])}" if evaluation["reasons"] else ".")]
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf8")


def generate(*, out: Path = OUT, report_path: Path = REPORT) -> bool:
    out = Path(out)
    report_path = Path(report_path)
    try:
        summary = load("summary.json", out)
    except (OSError, json.JSONDecodeError) as exc:
        summary = {}
        evaluation = {"ok": False, "reasons": [f"invalid summary.json: {exc}"], "rows": {}}
    else:
        evaluation = evaluate(summary, out)
    render_report(summary, evaluation, report_path, out)
    return bool(evaluation["ok"])


def main(*, out: Path | None = None, report_path: Path | None = None) -> int:
    return 0 if generate(out=OUT if out is None else out,
                         report_path=REPORT if report_path is None else report_path) else 1


if __name__ == "__main__":
    raise SystemExit(main())
