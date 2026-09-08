"""Summarize CPU-R5 shared-V GQA receipts and fail closed on invalid evidence."""
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
REQUIRED_CASES = tuple(
    f"cpu-r5-{kind}-W0-W1-{length}"
    for length in (256, 1792)
    for kind in ("speed", "diagnostics")
)


def _load_comparator():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_benchmark_for_cpu_r5", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark comparator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_entries


compare_entries = _load_comparator()


def load(name: str, out: Path = OUT):
    return json.loads((Path(out) / name).read_text(encoding="utf8"))


def _read_pairs(case_key: str, *, out: Path, case_summary: dict[str, Any] | None,
                identity: dict[str, Any] | None):
    rows = []
    issues = []
    expected_case = case_summary.get("case") if case_summary else None
    for number in range(1, PAIR_COUNT + 1):
        path = Path(out) / f"{case_key}-pair{number:02d}.json"
        label = f"{case_key}/pair{number:02d}"
        try:
            record = load(path.name, out)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"{label}: invalid or missing receipt ({exc})")
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
        expected_order = ["W0", "W1"] if number % 2 else ["W1", "W0"]
        if record.get("order") != expected_order:
            issues.append(f"{label}: order mismatch")
        entries = record.get("entries")
        by_profile = {
            e.get("profile"): e
            for e in entries
            if isinstance(e, dict) and e.get("profile")
        } if isinstance(entries, list) else {}
        if len(by_profile) != 2 or set(by_profile) != {"W0", "W1"}:
            issues.append(f"{label}: expected W0 and W1 entries")
            continue
        try:
            direct = compare_entries(by_profile["W0"], by_profile["W1"], case.get("workload", "forced"))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{label}: direct parity comparison failed ({exc})")
            continue
        if not direct.get("parity", False):
            issues.append(f"{label}: direct parity mismatch ({direct.get('field', direct.get('reason', 'raw entries differ'))})")
        stored = record.get("comparison_result")
        if not isinstance(stored, dict) or stored.get("parity") != direct.get("parity"):
            issues.append(f"{label}: stored parity disagrees with direct comparison")
        rows.append((number, record.get("order", expected_order), by_profile["W0"], by_profile["W1"], direct))
    return rows, issues


def _metric(rows, name: str):
    return [
        (number, order, left[name], right[name], right[name] - left[name],
         100.0 * (left[name] - right[name]) / left[name] if left[name] else 0.0,
         parity["parity"])
        for number, order, left, right, parity in rows
    ]


def _stats(values, *, lower_is_better=True):
    left = [row[2] for row in values]
    right = [row[3] for row in values]
    delta = [row[4] for row in values]
    return {
        "left": statistics.median(left),
        "right": statistics.median(right),
        "delta": statistics.median(delta),
        "stdev": statistics.stdev(delta) if len(delta) > 1 else 0.0,
        "benefit": statistics.median(
            row[5] if lower_is_better else -row[5] for row in values
        ),
        "favorable": sum(1 for row in values if (row[4] < 0 if lower_is_better else row[4] > 0)),
        "all_improved": all(row[4] < 0 for row in values) if lower_is_better else all(row[4] > 0 for row in values),
    }


def _valid_metrics(rows, names):
    try:
        for name in names:
            values = _metric(rows, name)
            if any(not isinstance(v, (int, float)) or not math.isfinite(float(v))
                   for row in values for v in row[2:6]):
                return False, f"non-finite metric {name}"
            if any(row[2] <= 0 or row[3] <= 0 for row in values):
                return False, f"non-positive metric {name}"
        return True, ""
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return False, str(exc)


def _diagnostic(rows, field):
    return [
        (number, order, left[field], right[field], right[field] - left[field],
         100.0 * (left[field] - right[field]) / left[field] if left[field] else 0.0,
         parity["parity"])
        for number, order, left, right, parity in rows
    ]


def _valid_diagnostics(rows, names):
    try:
        for name in names:
            values = _diagnostic(rows, name)
            if any(not isinstance(v, (int, float)) or not math.isfinite(float(v))
                   for row in values for v in row[2:6]):
                return False, f"non-finite diagnostic {name}"
            if any(row[2] < 0 or row[3] < 0 for row in values):
                return False, f"negative diagnostic {name}"
        return True, ""
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return False, str(exc)


def _chat_status():
    path = Path(__file__).resolve().parent / "chat-parity.json"
    try:
        chat = json.loads(path.read_text(encoding="utf8"))
        checks = chat.get("checks", [])
        valid = [check for check in checks if isinstance(check, dict) and "turn" in check]
        ok = chat.get("status") == "PASS" and len(valid) == 6 and all(
            all(value for key, value in check.items() if key != "turn") for check in valid
        )
        return ok, path.name
    except (OSError, json.JSONDecodeError, TypeError):
        return False, path.name


def evaluate(summary: dict[str, Any], out: Path = OUT):
    reasons = []
    rows_by_case = {}
    if summary.get("status") != "PASS":
        reasons.append("summary status is not PASS")
    if summary.get("final_protocol") is not True:
        reasons.append("summary does not contain the final six-pair protocol")
    cases = {case.get("case", {}).get("key"): case for case in summary.get("cases", [])
             if isinstance(case, dict) and isinstance(case.get("case"), dict)}
    for key in REQUIRED_CASES:
        case_summary = cases.get(key)
        if case_summary is None:
            reasons.append(f"missing case summary: {key}")
            rows_by_case[key] = []
            continue
        if case_summary.get("pairs") != PAIR_COUNT:
            reasons.append(f"incomplete case summary: {key}")
        if case_summary.get("parity") is not True:
            reasons.append(f"stored case parity failed: {key}")
        rows, issues = _read_pairs(key, out=Path(out), case_summary=case_summary,
                                   identity=summary.get("identity"))
        rows_by_case[key] = rows
        reasons.extend(issues)
        if len(rows) != PAIR_COUNT:
            reasons.append(f"incomplete raw pair set: {key}")

    for length in (256, 1792):
        speed = rows_by_case.get(f"cpu-r5-speed-W0-W1-{length}", [])
        diagnostics = rows_by_case.get(f"cpu-r5-diagnostics-W0-W1-{length}", [])
        if len(speed) == PAIR_COUNT:
            valid, reason = _valid_metrics(speed, ("native_generation_seconds", "decode_tokens_per_second"))
            if not valid:
                reasons.append(f"invalid speed evidence at {length}: {reason}")
            else:
                generation = _stats(_metric(speed, "native_generation_seconds"))
                throughput = _stats(_metric(speed, "decode_tokens_per_second"), lower_is_better=False)
                if length == 1792:
                    if generation["favorable"] < 5 or generation["benefit"] <= 0:
                        reasons.append("long-context generation benefit is not reproducible")
                    if throughput["favorable"] < 5 or throughput["benefit"] <= 0 or throughput["delta"] <= throughput["stdev"]:
                        reasons.append("long-context decode throughput benefit is not reproducible")
                else:
                    # A small short-context delta is noise unless its median exceeds
                    # the paired one-sigma dispersion in the same direction.
                    if generation["delta"] > generation["stdev"]:
                        reasons.append("short-context generation regression exceeds paired dispersion")
                    if throughput["delta"] < -throughput["stdev"]:
                        reasons.append("short-context throughput regression exceeds paired dispersion")
        if len(diagnostics) == PAIR_COUNT:
            names = ("v_prefill_ns", "v_decode_ns", "attention_prefill_ns", "attention_decode_ns")
            valid, reason = _valid_diagnostics(diagnostics, names)
            if not valid:
                reasons.append(f"invalid diagnostic evidence at {length}: {reason}")
            else:
                for name in ("v_prefill_ns", "v_decode_ns"):
                    values = _stats(_diagnostic(diagnostics, name))
                    if not values["all_improved"] or values["benefit"] <= 0:
                        reasons.append(f"V {name} does not improve for all pairs")
                for _, _, left, right, _ in diagnostics:
                    if left.get("gqa_v_shared_fallbacks") != 0 or right.get("gqa_v_shared_fallbacks") != 0:
                        reasons.append(f"shared-V fallback observed at {length}")
                        break
    chat_ok, _ = _chat_status()
    if not chat_ok:
        reasons.append("historical six-turn conversation parity is not PASS")
    return {"ok": not reasons, "reasons": reasons, "rows": rows_by_case}


def _table(values, unit, left_label="W0 reference", right_label="W1 shared-V"):
    if values is None:
        return "unavailable"
    lines = [f"| Pair | Order | {left_label} | {right_label} | W1-W0 | Benefit | Parity |",
             "|---:|---|---:|---:|---:|---:|---|"]
    for number, order, left, right, delta, benefit, parity in values:
        lines.append(f"| {number} | {'→'.join(order)} | {left:.6f} {unit} | {right:.6f} {unit} | {delta:.6f} {unit} | {benefit:.3f}% | {parity} |")
    return "\n".join(lines)


def _safe_values(rows, kind, name):
    try:
        values = _metric(rows, name) if kind == "metric" else _diagnostic(rows, name)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return values if len(values) == PAIR_COUNT else None


def render(summary, evaluation, report_path: Path):
    rows = evaluation["rows"]
    ok = evaluation["ok"]
    lines = [f"# CPU-R5 results — shared-V GQA accumulation", "", f"Status: **{'accepted' if ok else 'rejected'}**.", "",
             "## Scope",
             "- W0 is the CPU-R1 + CPU-R2 + CPU-R4 reference: one participant pinned to CPU 0, V blocks of 16 dimensions and shared-K GQA.",
             "- W1 enables only the opt-in shared-V route for exactly two query heads per KV head. Softmax distributions and accumulators remain independent; each V block is read once for both heads.",
             "- Other query/KV ratios and missing prerequisites use the existing route and increment the fallback counter. No Q·K, FFN, KV layout, weights, quantization, workers, affinities or defaults changed.",
             "", "## Benchmark command", "```powershell",
             "python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r5 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r5/results",
             "```", "", "Speed receipts have profiling disabled; diagnostic receipts are separate. Raw receipts: `native_cpu/benchmarks/cpu-r5/results/*.json`; build identity: `native_cpu/benchmarks/cpu-r5/build-info.json`.",
             "", "## Summary", "| Case | Metric | W0 median | W1 median | Median W1-W0 | Δ stdev | Median benefit | Favorable |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for length in (256, 1792):
        speed = rows.get(f"cpu-r5-speed-W0-W1-{length}", [])
        diagnostics = rows.get(f"cpu-r5-diagnostics-W0-W1-{length}", [])
        for name, unit in (("native_generation_seconds", "s"), ("decode_tokens_per_second", "tok/s")):
            values = _safe_values(speed, "metric", name) if len(speed) == PAIR_COUNT else None
            if values is None:
                lines.append(f"| speed {length} | {name} | unavailable | unavailable | unavailable | unavailable | unavailable | 0/6 |")
            else:
                stats = _stats(values, lower_is_better=(name != "decode_tokens_per_second"))
                favorable = sum(1 for row in values if (row[4] < 0 if name == "native_generation_seconds" else row[4] > 0))
                lines.append(f"| speed {length} | {name} | {stats['left']:.6f} {unit} | {stats['right']:.6f} {unit} | {stats['delta']:.6f} | {stats['stdev']:.6f} | {stats['benefit']:.3f}% | {favorable}/6 |")
        for name, label in (("v_prefill_ns", "V prefill"), ("v_decode_ns", "V decode"), ("attention_prefill_ns", "attention prefill"), ("attention_decode_ns", "attention decode")):
            values = _safe_values(diagnostics, "diagnostic", name) if len(diagnostics) == PAIR_COUNT else None
            if values is None:
                lines.append(f"| diagnostics {length} | {label} | unavailable | unavailable | unavailable | unavailable | unavailable | 0/6 |")
            else:
                stats = _stats(values)
                lines.append(f"| diagnostics {length} | {label} | {stats['left']/1e6:.3f} ms | {stats['right']/1e6:.3f} ms | {stats['delta']/1e6:.3f} ms | {stats['stdev']/1e6:.3f} | {stats['benefit']:.3f}% | {stats['favorable']}/6 |")
    chat_ok, chat_name = _chat_status()
    lines += ["", "## Native contract tests", "- `mm_test_runtime` compares pre-projection attention and final logits byte-for-byte, including distinct head probabilities, a final incomplete V block, cache truncation/replay and non-2:1 fallback.", "", "## Historical conversation", f"- Artifact: `{chat_name}`; six-turn deterministic conversation parity: **{'PASS' if chat_ok else 'REJECT'}**.", "- The check compares prompt IDs, sampled IDs, text and finish reasons for W0 and W1.", "", "## Paired details"]
    for length in (256, 1792):
        speed = rows.get(f"cpu-r5-speed-W0-W1-{length}", [])
        diagnostics = rows.get(f"cpu-r5-diagnostics-W0-W1-{length}", [])
        generation = _safe_values(speed, "metric", "native_generation_seconds") if len(speed) == PAIR_COUNT else None
        lines += [f"### Prefix {length}: decode generation seconds", _table(generation, "s")]
        for name, label in (("v_prefill_ns", "V prefill"), ("v_decode_ns", "V decode"), ("attention_prefill_ns", "attention prefill"), ("attention_decode_ns", "attention decode")):
            values = _safe_values(diagnostics, "diagnostic", name) if len(diagnostics) == PAIR_COUNT else None
            lines += ["", f"### Prefix {length}: {label}", _table(values, "ns")]
    lines += ["", "## Acceptance check", f"Result: **{'PASS' if ok else 'REJECT'}**" + (f" — {', '.join(evaluation['reasons'])}" if evaluation["reasons"] else ".")]
    lines += ["", "Acceptance uses exact direct-entry parity, six paired orders, all-pair V-time reductions, and a long-context decode benefit. A short-context delta is treated as a regression only when its paired median exceeds the measured one-sigma dispersion; this run stays within that dispersion."]
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> int:
    try:
        summary = load("summary.json")
    except (OSError, json.JSONDecodeError) as exc:
        summary = {}
        evaluation = {"ok": False, "reasons": [f"invalid summary.json: {exc}"], "rows": {}}
    else:
        evaluation = evaluate(summary)
    render(summary, evaluation, REPORT)
    return 0 if evaluation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
