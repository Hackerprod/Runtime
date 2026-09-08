"""Summarize the bounded CPU-R3 four-row FFN experiment."""
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
    "cpu-r3-speed-F0-F1-256",
    "cpu-r3-speed-F0-F1-1792",
    "cpu-r3-diagnostics-F0-F1-256",
    "cpu-r3-diagnostics-F0-F1-1792",
)


def _load_comparator():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_benchmark_for_cpu_r3", path)
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
        expected_order = ["F0", "F1"] if number % 2 else ["F1", "F0"]
        if record.get("order") != expected_order:
            issues.append(f"{label}: order mismatch")
        entries = record.get("entries")
        by_profile = {
            e.get("profile"): e
            for e in entries
            if isinstance(e, dict) and e.get("profile")
        } if isinstance(entries, list) else {}
        if len(by_profile) != 2 or set(by_profile) != {"F0", "F1"}:
            issues.append(f"{label}: expected F0 and F1 entries")
            continue
        try:
            direct = compare_entries(by_profile["F0"], by_profile["F1"], case.get("workload", "forced"))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{label}: direct parity comparison failed ({exc})")
            continue
        if not direct.get("parity", False):
            issues.append(f"{label}: direct parity mismatch ({direct.get('field', direct.get('reason', 'raw entries differ'))})")
        stored = record.get("comparison_result")
        if not isinstance(stored, dict) or stored.get("parity") != direct.get("parity"):
            issues.append(f"{label}: stored parity disagrees with direct comparison")
        rows.append((number, record.get("order", expected_order), by_profile["F0"], by_profile["F1"], direct))
    return rows, issues


def _metric(rows, name: str):
    result = []
    for number, order, left, right, parity in rows:
        l, r = left[name], right[name]
        result.append((number, order, l, r, r - l, -100.0 * (r - l) / l if l else 0.0, parity["parity"]))
    return result


def _ffn(rows, phase: str):
    result = []
    for number, order, left, right, parity in rows:
        l = left["native_phase_stats"][phase]["ffn_ns"] / 1e6
        r = right["native_phase_stats"][phase]["ffn_ns"] / 1e6
        result.append((number, order, l, r, r - l, -100.0 * (r - l) / l if l else 0.0, parity["parity"]))
    return result


def _stats(rows):
    left = [r[2] for r in rows]
    right = [r[3] for r in rows]
    delta = [r[4] for r in rows]
    return {
        "left": statistics.median(left), "right": statistics.median(right),
        "delta": statistics.median(delta), "stdev": statistics.stdev(delta) if len(delta) > 1 else 0.0,
        "benefit": statistics.median([r[5] for r in rows]), "all_improved": all(d < 0 for d in delta),
    }


def _valid(rows, fields):
    try:
        for kind, name in fields:
            values = _metric(rows, name) if kind == "metric" else _ffn(rows, name)
            if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for row in values for v in row[2:6]):
                return False, f"non-finite {kind} {name} value"
            if kind == "metric" and name in ("native_generation_seconds", "decode_tokens_per_second") and any(row[2] <= 0 or row[3] <= 0 for row in values):
                return False, f"non-positive metric {name} value"
        return True, ""
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return False, str(exc)


def evaluate(summary: dict[str, Any], out: Path = OUT):
    reasons = []
    rows_by_case = {}
    if summary.get("status") != "PASS": reasons.append("summary status is not PASS")
    if summary.get("final_protocol") is not True: reasons.append("summary does not contain the final six-pair protocol")
    cases = {c.get("case", {}).get("key"): c for c in summary.get("cases", []) if isinstance(c, dict) and isinstance(c.get("case"), dict)}
    for key in REQUIRED_CASES:
        case = cases.get(key)
        if case is None:
            reasons.append(f"missing case summary: {key}"); rows_by_case[key] = []; continue
        if case.get("pairs") != PAIR_COUNT: reasons.append(f"incomplete case summary: {key}")
        if case.get("parity") is not True: reasons.append(f"stored case parity failed: {key}")
        rows, issues = _read_pairs(key, out=Path(out), case_summary=case, identity=summary.get("identity"))
        rows_by_case[key] = rows; reasons.extend(issues)
        if len(rows) != PAIR_COUNT: reasons.append(f"incomplete raw pair set: {key}")
    for length in (256, 1792):
        speed = rows_by_case.get(f"cpu-r3-speed-F0-F1-{length}", [])
        diagnostics = rows_by_case.get(f"cpu-r3-diagnostics-F0-F1-{length}", [])
        if len(speed) == PAIR_COUNT:
            valid, reason = _valid(speed, [("metric", "native_generation_seconds"), ("metric", "decode_tokens_per_second")])
            if not valid: reasons.append(f"invalid speed evidence at {length}: {reason}")
            else:
                generation = _metric(speed, "native_generation_seconds")
                throughput = _metric(speed, "decode_tokens_per_second")
                gs = _stats(generation)
                if not gs["all_improved"]: reasons.append(f"decode speed regression at {length}")
                if abs(gs["delta"]) <= gs["stdev"]: reasons.append(f"decode benefit inside dispersion at {length}")
                if not all(row[4] > 0 for row in throughput): reasons.append(f"decode throughput regression at {length}")
        if len(diagnostics) == PAIR_COUNT:
            valid, reason = _valid(diagnostics, [("ffn", "prefill"), ("ffn", "decode")])
            if not valid: reasons.append(f"invalid FFN evidence at {length}: {reason}")
            else:
                for phase in ("prefill", "decode"):
                    if not _stats(_ffn(diagnostics, phase))["all_improved"]: reasons.append(f"FFN {phase} regression at {length}")
    return {"ok": not reasons, "reasons": reasons, "rows": rows_by_case}


def _table(rows, unit):
    lines = ["| Pair | Order | F0 reference | F1 row4 | F1-F0 | Benefit | Parity |", "|---:|---|---:|---:|---:|---:|---|"]
    for i, order, left, right, delta, benefit, parity in rows:
        lines.append(f"| {i} | {'→'.join(order)} | {left:.6f} {unit} | {right:.6f} {unit} | {delta:.6f} {unit} | {benefit:.3f}% | {parity} |")
    return "\n".join(lines)


def _safe_measure(rows, kind, name):
    """Return a complete metric table, or None when evidence is incomplete."""
    try:
        values = _metric(rows, name) if kind == "metric" else _ffn(rows, name)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return values if len(values) == PAIR_COUNT else None


def render(summary, evaluation, report_path: Path):
    rows = evaluation["rows"]; ok = evaluation["ok"]
    lines = ["# CPU-R3 results — four-row FFN FP32 GEMV", "", f"Status: **{'accepted' if ok else 'rejected'}**."]
    lines += ["", "## Scope", "- F0 is the prior FP32 GEMV; F1 enables only the opt-in four-row FFN kernel.", "- CPU-R1 selective logits, CPU-R2 V-blocked attention, one participant on CPU 0, model weights and defaults are unchanged.", "- No quantization, worker repartitioning, affinity changes, attention changes or additional optimizations were included.", "", "## Benchmark command", "```powershell", "python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r3 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r3/results", "```", "", f"Raw receipts: `native_cpu/benchmarks/cpu-r3/results/*.json`; build identity: `native_cpu/benchmarks/cpu-r3/build-info.json`.", "", "## Summary", "| Case | Metric | F0 median | F1 median | Median F1-F0 | Δ stdev | Median benefit | All improved |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for length in (256, 1792):
        speed = rows.get(f"cpu-r3-speed-F0-F1-{length}", [])
        diagnostics = rows.get(f"cpu-r3-diagnostics-F0-F1-{length}", [])
        for name, unit in (("native_generation_seconds", "s"), ("decode_tokens_per_second", "tok/s")):
            values = _safe_measure(speed, "metric", name) if len(speed) == PAIR_COUNT else None
            if values is not None:
                s = _stats(values)
                if name == "decode_tokens_per_second":
                    deltas = [r[4] for r in values]; benefits = [100.0 * r[4] / r[2] for r in values]
                    lines.append(f"| speed {length} | {name} | {s['left']:.6f} {unit} | {s['right']:.6f} {unit} | {statistics.median(deltas):.6f} {unit} | {statistics.stdev(deltas):.6f} | {statistics.median(benefits):.3f}% | {all(x > 0 for x in deltas)} |")
                else: lines.append(f"| speed {length} | {name} | {s['left']:.6f} {unit} | {s['right']:.6f} {unit} | {s['delta']:.6f} {unit} | {s['stdev']:.6f} | {s['benefit']:.3f}% | {s['all_improved']} |")
            else: lines.append(f"| speed {length} | {name} | unavailable | unavailable | unavailable | unavailable | unavailable | False |")
        for phase in ("prefill", "decode"):
            values = _safe_measure(diagnostics, "ffn", phase) if len(diagnostics) == PAIR_COUNT else None
            if values is not None:
                s = _stats(values); lines.append(f"| FFN {length} | {phase}_ms | {s['left']:.6f} ms | {s['right']:.6f} ms | {s['delta']:.6f} ms | {s['stdev']:.6f} | {s['benefit']:.3f}% | {s['all_improved']} |")
            else: lines.append(f"| FFN {length} | {phase}_ms | unavailable | unavailable | unavailable | unavailable | unavailable | False |")
    chat_path = Path(report_path).with_name("chat-parity.json")
    try:
        chat = json.loads(chat_path.read_text(encoding="utf8"))
        chat_status = chat.get("status", "UNKNOWN")
        checks = chat.get("checks", [])
        valid_checks = [check for check in checks if isinstance(check, dict) and "turn" in check]
        chat_parity = chat_status == "PASS" and len(valid_checks) == 6 and all(
            all(value for key, value in check.items() if key != "turn")
            for check in valid_checks
        )
        lines += ["", "## Historical conversation", f"- Artifact: `{chat_path.name}`; six-turn deterministic conversation parity: **{'PASS' if chat_parity else 'REJECT'}**.", "- Profiles: F0 (CPU-R1 + CPU-R2) versus F1 (the same configuration with `ffn_row4` enabled); all prompts, sampled IDs, text and finish reasons are compared."]
    except (OSError, json.JSONDecodeError, TypeError):
        lines += ["", "## Historical conversation", "- Artifact unavailable or invalid; conversation parity is not claimed."]
    lines += ["", "## Paired details"]
    for length in (256, 1792):
        speed = rows.get(f"cpu-r3-speed-F0-F1-{length}", [])
        diagnostics = rows.get(f"cpu-r3-diagnostics-F0-F1-{length}", [])
        generation = _safe_measure(speed, "metric", "native_generation_seconds") if len(speed) == PAIR_COUNT else None
        lines += [
            f"### Prefix {length}: generation seconds",
            _table(generation, "s") if generation is not None else "unavailable",
        ]
        for phase in ("prefill", "decode"):
            phase_values = _safe_measure(diagnostics, "ffn", phase) if len(diagnostics) == PAIR_COUNT else None
            lines += [
                "",
                f"### Prefix {length}: FFN {phase}",
                _table(phase_values, "ms") if phase_values is not None else "unavailable",
            ]
    lines += ["", "## Acceptance check", f"Result: **{'PASS' if ok else 'REJECT'}**" + (f" — {', '.join(evaluation['reasons'])}" if evaluation["reasons"] else ".")]
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> int:
    try: summary = load("summary.json")
    except (OSError, json.JSONDecodeError) as exc:
        summary = {}; evaluation = {"ok": False, "reasons": [f"invalid summary.json: {exc}"], "rows": {}}
    else: evaluation = evaluate(summary)
    render(summary, evaluation, REPORT)
    return 0 if evaluation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
