"""Summarize the bounded CPU-R4 shared-K GQA experiment, failing closed on bad receipts."""
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
    f"cpu-r4-{kind}-K0-K1-{length}"
    for length in (256, 1792)
    for kind in ("speed", "diagnostics")
)


def _load_comparator():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_benchmark_for_cpu_r4", path)
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
        expected_order = ["K0", "K1"] if number % 2 else ["K1", "K0"]
        if record.get("order") != expected_order:
            issues.append(f"{label}: order mismatch")
        entries = record.get("entries")
        by_profile = {
            e.get("profile"): e
            for e in entries
            if isinstance(e, dict) and e.get("profile")
        } if isinstance(entries, list) else {}
        if len(by_profile) != 2 or set(by_profile) != {"K0", "K1"}:
            issues.append(f"{label}: expected K0 and K1 entries")
            continue
        try:
            direct = compare_entries(by_profile["K0"], by_profile["K1"], case.get("workload", "forced"))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{label}: direct parity comparison failed ({exc})")
            continue
        if not direct.get("parity", False):
            issues.append(f"{label}: direct parity mismatch ({direct.get('field', direct.get('reason', 'raw entries differ'))})")
        stored = record.get("comparison_result")
        if not isinstance(stored, dict) or stored.get("parity") != direct.get("parity"):
            issues.append(f"{label}: stored parity disagrees with direct comparison")
        rows.append((number, record.get("order", expected_order), by_profile["K0"], by_profile["K1"], direct))
    return rows, issues


def _metric(rows, name: str):
    return [
        (number, order, left[name], right[name], right[name] - left[name],
         -100.0 * (right[name] - left[name]) / left[name] if left[name] else 0.0,
         parity["parity"])
        for number, order, left, right, parity in rows
    ]


def _attention(rows, phase: str, name: str):
    return [
        (number, order,
         left["native_phase_stats"][phase][name] / 1e6,
         right["native_phase_stats"][phase][name] / 1e6,
         (right["native_phase_stats"][phase][name] - left["native_phase_stats"][phase][name]) / 1e6,
         -100.0 * (right["native_phase_stats"][phase][name] - left["native_phase_stats"][phase][name]) /
         left["native_phase_stats"][phase][name] if left["native_phase_stats"][phase][name] else 0.0,
         parity["parity"])
        for number, order, left, right, parity in rows
    ]


def _stats(rows):
    left = [row[2] for row in rows]
    right = [row[3] for row in rows]
    delta = [row[4] for row in rows]
    return {
        "left": statistics.median(left), "right": statistics.median(right),
        "delta": statistics.median(delta),
        "stdev": statistics.stdev(delta) if len(delta) > 1 else 0.0,
        "benefit": statistics.median([row[5] for row in rows]),
        "all_improved": all(delta_value < 0 for delta_value in delta),
    }


def _valid(rows, fields):
    try:
        for kind, name in fields:
            values = _metric(rows, name) if kind == "metric" else _attention(rows, name[0], name[1])
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value))
                   for row in values for value in row[2:6]):
                return False, f"non-finite {kind} {name} value"
            if kind == "metric" and name in ("native_generation_seconds", "decode_tokens_per_second") \
                    and any(row[2] <= 0 or row[3] <= 0 for row in values):
                return False, f"non-positive metric {name} value"
            if kind == "attention" and any(row[2] < 0 or row[3] < 0 for row in values):
                return False, f"negative attention {name} value"
        return True, ""
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return False, str(exc)


def _chat_status(out: Path):
    path = Path(out).parent / "chat-parity.json"
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
        speed = rows_by_case.get(f"cpu-r4-speed-K0-K1-{length}", [])
        diagnostics = rows_by_case.get(f"cpu-r4-diagnostics-K0-K1-{length}", [])
        if len(speed) == PAIR_COUNT:
            valid, reason = _valid(speed, [("metric", "native_generation_seconds"),
                                            ("metric", "decode_tokens_per_second")])
            if not valid:
                reasons.append(f"invalid speed evidence at {length}: {reason}")
            else:
                generation = _metric(speed, "native_generation_seconds")
                throughput = _metric(speed, "decode_tokens_per_second")
                generation_stats = _stats(generation)
                if not generation_stats["all_improved"]:
                    reasons.append(f"decode speed regression at {length}")
                if abs(generation_stats["delta"]) <= generation_stats["stdev"]:
                    reasons.append(f"decode benefit inside dispersion at {length}")
                if not all(row[4] > 0 for row in throughput):
                    reasons.append(f"decode throughput regression at {length}")
        if len(diagnostics) == PAIR_COUNT:
            valid, reason = _valid(diagnostics, [("attention", ("prefill", "attention_qk_ns")),
                                                  ("attention", ("decode", "attention_qk_ns"))])
            if not valid:
                reasons.append(f"invalid QK evidence at {length}: {reason}")
            else:
                for phase in ("prefill", "decode"):
                    qk = _stats(_attention(diagnostics, phase, "attention_qk_ns"))
                    if not qk["all_improved"]:
                        reasons.append(f"QK {phase} regression at {length}")
                    if abs(qk["delta"]) <= qk["stdev"]:
                        reasons.append(f"QK {phase} benefit inside dispersion at {length}")
    chat_ok, _ = _chat_status(Path(out))
    if not chat_ok:
        reasons.append("historical conversation parity is not PASS")
    return {"ok": not reasons, "reasons": reasons, "rows": rows_by_case}


def _table(rows, unit, left_label="K0 reference", right_label="K1 shared-K"):
    if rows is None:
        return "unavailable"
    lines = [f"| Pair | Order | {left_label} | {right_label} | K1-K0 | Benefit | Parity |",
             "|---:|---|---:|---:|---:|---:|---|"]
    for number, order, left, right, delta, benefit, parity in rows:
        lines.append(f"| {number} | {'→'.join(order)} | {left:.6f} {unit} | {right:.6f} {unit} | {delta:.6f} {unit} | {benefit:.3f}% | {parity} |")
    return "\n".join(lines)


def _safe_measure(rows, kind, name):
    try:
        values = _metric(rows, name) if kind == "metric" else _attention(rows, name[0], name[1])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return values if len(values) == PAIR_COUNT else None


def render(summary, evaluation, report_path: Path):
    rows = evaluation["rows"]
    ok = evaluation["ok"]
    lines = ["# CPU-R4 results — shared-K GQA Q·K", "", f"Status: **{'accepted' if ok else 'rejected'}**.", "",
             "## Scope",
             "- K0 is the CPU-R1 + CPU-R2 reference: selective logits, KV reuse, V-blocked attention, one participant pinned to CPU 0.",
             "- K1 enables only the opt-in shared-K Q·K route for exactly two query heads per KV head; each head keeps independent scores, softmax and CPU-R2 V accumulation.",
             "- Other query/KV relationships use the original route. No FFN, KV layout, weights, quantization, workers, affinities or defaults changed.",
             "", "## Benchmark command", "```powershell",
             "python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r4 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r4/results",
             "```", "", "Raw receipts: `native_cpu/benchmarks/cpu-r4/results/*.json`; build identity: `native_cpu/benchmarks/cpu-r4/build-info.json`.",
             "", "## Summary", "| Case | Metric | K0 median | K1 median | Median K1-K0 | Δ stdev | Median benefit | All improved |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for length in (256, 1792):
        speed = rows.get(f"cpu-r4-speed-K0-K1-{length}", [])
        diagnostics = rows.get(f"cpu-r4-diagnostics-K0-K1-{length}", [])
        for name, unit in (("native_generation_seconds", "s"), ("decode_tokens_per_second", "tok/s")):
            values = _safe_measure(speed, "metric", name) if len(speed) == PAIR_COUNT else None
            if values is None:
                lines.append(f"| speed {length} | {name} | unavailable | unavailable | unavailable | unavailable | unavailable | False |")
            else:
                stats = _stats(values)
                benefit = statistics.median([100.0 * row[4] / row[2] for row in values])
                improved = stats["all_improved"] if name == "native_generation_seconds" else all(row[4] > 0 for row in values)
                lines.append(f"| speed {length} | {name} | {stats['left']:.6f} {unit} | {stats['right']:.6f} {unit} | {stats['delta']:.6f} {unit} | {stats['stdev']:.6f} | {benefit:.3f}% | {improved} |")
        for phase in ("prefill", "decode"):
            values = _safe_measure(diagnostics, "attention", (phase, "attention_qk_ns")) if len(diagnostics) == PAIR_COUNT else None
            if values is None:
                lines.append(f"| QK {length} | {phase}_ms | unavailable | unavailable | unavailable | unavailable | unavailable | False |")
            else:
                stats = _stats(values)
                lines.append(f"| QK {length} | {phase}_ms | {stats['left']:.6f} ms | {stats['right']:.6f} ms | {stats['delta']:.6f} ms | {stats['stdev']:.6f} | {stats['benefit']:.3f}% | {stats['all_improved']} |")
    chat_ok, chat_name = _chat_status(OUT)
    lines += ["", "## Historical conversation", f"- Artifact: `{chat_name}`; six-turn deterministic conversation parity: **{'PASS' if chat_ok else 'REJECT'}**.", "- K0 and K1 compare canonical prompt IDs, sampled IDs, text and finish reasons.", "", "## Paired details"]
    for length in (256, 1792):
        speed = rows.get(f"cpu-r4-speed-K0-K1-{length}", [])
        diagnostics = rows.get(f"cpu-r4-diagnostics-K0-K1-{length}", [])
        generation = _safe_measure(speed, "metric", "native_generation_seconds") if len(speed) == PAIR_COUNT else None
        lines += [f"### Prefix {length}: generation seconds", _table(generation, "s")]
        for phase in ("prefill", "decode"):
            qk = _safe_measure(diagnostics, "attention", (phase, "attention_qk_ns")) if len(diagnostics) == PAIR_COUNT else None
            lines += ["", f"### Prefix {length}: Q·K {phase}", _table(qk, "ms")]
    lines += ["", "## Acceptance check", f"Result: **{'PASS' if ok else 'REJECT'}**" + (f" — {', '.join(evaluation['reasons'])}" if evaluation["reasons"] else ".")]
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
