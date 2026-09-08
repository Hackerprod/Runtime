"""Summarize CPU-R6 exact FP16-storage FFN receipts and fail closed."""
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
    f"cpu-r6-{kind}-X0-X1-{length}"
    for length in (256, 1792)
    for kind in ("speed", "diagnostics")
)


def _load_comparator():
    path = ROOT / "native_cpu" / "benchmarks" / "cpu-r1" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("cpu_r1_benchmark_for_cpu_r6", path)
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
        expected_order = ["X0", "X1"] if number % 2 else ["X1", "X0"]
        if record.get("order") != expected_order:
            issues.append(f"{label}: order mismatch")
        entries = record.get("entries")
        by_profile = {e.get("profile"): e for e in entries
                      if isinstance(e, dict) and e.get("profile")} if isinstance(entries, list) else {}
        if len(by_profile) != 2 or set(by_profile) != {"X0", "X1"}:
            issues.append(f"{label}: expected X0 and X1 entries")
            continue
        try:
            direct = compare_entries(by_profile["X0"], by_profile["X1"], case.get("workload", "forced"))
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(f"{label}: direct parity comparison failed ({exc})")
            continue
        if not direct.get("parity", False):
            issues.append(f"{label}: direct parity mismatch ({direct.get('field', direct.get('reason', 'raw entries differ'))})")
        stored = record.get("comparison_result")
        if not isinstance(stored, dict) or stored.get("parity") != direct.get("parity"):
            issues.append(f"{label}: stored parity disagrees with direct comparison")
        rows.append((number, record.get("order", expected_order), by_profile["X0"], by_profile["X1"], direct))
    return rows, issues


def _values(rows, name):
    return [(number, order, left[name], right[name], right[name] - left[name],
             100.0 * (left[name] - right[name]) / left[name] if left[name] else 0.0,
             parity["parity"])
            for number, order, left, right, parity in rows]


def _stats(values, *, lower_is_better=True):
    left = [row[2] for row in values]
    right = [row[3] for row in values]
    delta = [row[4] for row in values]
    return {
        "left": statistics.median(left), "right": statistics.median(right),
        "delta": statistics.median(delta),
        "stdev": statistics.stdev(delta) if len(delta) > 1 else 0.0,
        "benefit": statistics.median(row[5] if lower_is_better else -row[5] for row in values),
        "favorable": sum(1 for row in values if (row[4] < 0 if lower_is_better else row[4] > 0)),
    }


def _valid(rows, names, *, diagnostic=False):
    try:
        for name in names:
            values = _values(rows, name)
            if any(not isinstance(v, (int, float)) or not math.isfinite(float(v))
                   for row in values for v in row[2:6]):
                return False, f"non-finite {name}"
            if any(row[2] < 0 or row[3] < 0 for row in values) if diagnostic else any(row[2] <= 0 or row[3] <= 0 for row in values):
                return False, f"invalid {name}"
        return True, ""
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return False, str(exc)


def _chat_status():
    path = Path(__file__).resolve().parent / "chat-parity.json"
    try:
        chat = json.loads(path.read_text(encoding="utf8"))
        checks = [c for c in chat.get("checks", []) if isinstance(c, dict) and "turn" in c]
        ok = chat.get("status") == "PASS" and len(checks) == 6 and all(
            all(v for k, v in check.items() if k != "turn") for check in checks)
        return ok, path.name
    except (OSError, json.JSONDecodeError, TypeError):
        return False, path.name


def _reconstruction_status():
    path = Path(__file__).resolve().parent / "reconstruction.json"
    try:
        data = json.loads(path.read_text(encoding="utf8"))
        ok = (data.get("status") == "PASS" and data.get("matrix_count") == 24
              and data.get("bit_mismatches") == 0 and data.get("elements", 0) > 0)
        return ok, path.name, data
    except (OSError, json.JSONDecodeError, TypeError):
        return False, path.name, {}


def evaluate(summary: dict[str, Any], out: Path = OUT):
    reasons = []
    rows_by_case = {}
    if summary.get("status") != "PASS": reasons.append("summary status is not PASS")
    if summary.get("final_protocol") is not True: reasons.append("summary does not contain the final six-pair protocol")
    cases = {case.get("case", {}).get("key"): case for case in summary.get("cases", [])
             if isinstance(case, dict) and isinstance(case.get("case"), dict)}
    for key in REQUIRED_CASES:
        case_summary = cases.get(key)
        if case_summary is None:
            reasons.append(f"missing case summary: {key}"); rows_by_case[key] = []; continue
        if case_summary.get("pairs") != PAIR_COUNT: reasons.append(f"incomplete case summary: {key}")
        rows, issues = _read_pairs(key, out=Path(out), case_summary=case_summary, identity=summary.get("identity"))
        rows_by_case[key] = rows; reasons.extend(issues)
        if len(rows) != PAIR_COUNT: reasons.append(f"incomplete raw pair set: {key}")
        if case_summary.get("parity") is not True: reasons.append(f"stored case parity failed: {key}")

    for length in (256, 1792):
        speed = rows_by_case.get(f"cpu-r6-speed-X0-X1-{length}", [])
        diagnostic = rows_by_case.get(f"cpu-r6-diagnostics-X0-X1-{length}", [])
        if len(speed) == PAIR_COUNT:
            valid, reason = _valid(speed, ("native_generation_seconds", "decode_tokens_per_second"))
            if not valid: reasons.append(f"invalid speed evidence at {length}: {reason}")
            else:
                generation = _stats(_values(speed, "native_generation_seconds"))
                throughput = _stats(_values(speed, "decode_tokens_per_second"), lower_is_better=False)
                if length == 1792:
                    if generation["favorable"] < 5 or generation["benefit"] <= 0: reasons.append("long-context generation benefit is not reproducible")
                    if throughput["favorable"] < 5 or throughput["benefit"] <= 0 or throughput["delta"] <= throughput["stdev"]: reasons.append("long-context decode benefit is not reproducible")
                else:
                    if generation["delta"] > generation["stdev"]: reasons.append("short-context generation regression exceeds paired dispersion")
                    if throughput["delta"] < -throughput["stdev"]: reasons.append("short-context throughput regression exceeds paired dispersion")
        if len(diagnostic) == PAIR_COUNT:
            valid, reason = _valid(diagnostic, ("ffn_prefill_ns", "ffn_decode_ns"), diagnostic=True)
            if not valid: reasons.append(f"invalid FFN diagnostic evidence at {length}: {reason}")
            else:
                for name in ("ffn_prefill_ns", "ffn_decode_ns"):
                    values = _stats(_values(diagnostic, name))
                    if values["favorable"] != PAIR_COUNT or values["benefit"] <= 0: reasons.append(f"{name} does not improve for all pairs")
    chat_ok, _ = _chat_status()
    if not chat_ok: reasons.append("historical six-turn conversation parity is not PASS")
    reconstruction_ok, _, _ = _reconstruction_status()
    if not reconstruction_ok: reasons.append("full FFN FP32→FP16→FP32 reconstruction evidence is not PASS")
    return {"ok": not reasons, "reasons": reasons, "rows": rows_by_case}


def _table(values, unit, left_label="X0 FP32 storage", right_label="X1 FP16 storage"):
    if values is None: return "unavailable"
    lines = [f"| Pair | Order | {left_label} | {right_label} | X1-X0 | Benefit | Parity |", "|---:|---|---:|---:|---:|---:|---|"]
    for number, order, left, right, delta, benefit, parity in values:
        lines.append(f"| {number} | {'→'.join(order)} | {left:.6f} {unit} | {right:.6f} {unit} | {delta:.6f} {unit} | {benefit:.3f}% | {parity} |")
    return "\n".join(lines)


def _safe(rows, name):
    try:
        values = _values(rows, name)
        return values if len(values) == PAIR_COUNT else None
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def render(summary, evaluation, report_path: Path):
    rows = evaluation["rows"]
    ok = evaluation["ok"]
    lines = [f"# CPU-R6 results — exact FP16-storage FFN", "", f"Status: **{'accepted' if ok else 'rejected'}**.", "",
             "## Scope", "- X0 is CPU-R5 W1: CPU-R1/R2/R4/R5 enabled, one participant pinned to CPU 0, CPU-R3 row4 disabled.",
             "- X1 adds only exact FP16 storage for all dense FFN gate/up/down matrices. Every weight is validated by FP32→FP16→FP32 bit equality before activation; the kernel converts eight halves to FP32 registers and keeps the existing FMA/reduction order.",
             "- Attention, KV cache, weights on disk, quantization, workers, affinities, statistics ABI and defaults are unchanged. FP32 vectors remain allocated, so the compact payload is additional runtime memory, not a total-RAM reduction.", "",
             "## Benchmark command", "```powershell", "python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r6 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r6/results", "```", "",
             "Speed receipts have profiling disabled; diagnostic receipts are separate. Raw receipts: `native_cpu/benchmarks/cpu-r6/results/*.json`; build identity: `native_cpu/benchmarks/cpu-r6/build-info.json`.", "",
             "## Summary", "| Case | Metric | X0 median | X1 median | Median X1-X0 | Δ stdev | Median benefit | Favorable |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for length in (256, 1792):
        speed = rows.get(f"cpu-r6-speed-X0-X1-{length}", [])
        diagnostic = rows.get(f"cpu-r6-diagnostics-X0-X1-{length}", [])
        for name, unit in (("native_generation_seconds", "s"), ("decode_tokens_per_second", "tok/s")):
            values = _safe(speed, name) if len(speed) == PAIR_COUNT else None
            if values is None: lines.append(f"| speed {length} | {name} | unavailable | unavailable | unavailable | unavailable | unavailable | 0/6 |")
            else:
                stats = _stats(values, lower_is_better=(name == "native_generation_seconds"))
                lines.append(f"| speed {length} | {name} | {stats['left']:.6f} {unit} | {stats['right']:.6f} {unit} | {stats['delta']:.6f} | {stats['stdev']:.6f} | {stats['benefit']:.3f}% | {stats['favorable']}/6 |")
        for name, label in (("ffn_prefill_ns", "FFN prefill"), ("ffn_decode_ns", "FFN decode")):
            values = _safe(diagnostic, name) if len(diagnostic) == PAIR_COUNT else None
            if values is None: lines.append(f"| diagnostics {length} | {label} | unavailable | unavailable | unavailable | unavailable | unavailable | 0/6 |")
            else:
                stats = _stats(values)
                lines.append(f"| diagnostics {length} | {label} | {stats['left']/1e6:.3f} ms | {stats['right']/1e6:.3f} ms | {stats['delta']/1e6:.3f} ms | {stats['stdev']/1e6:.3f} | {stats['benefit']:.3f}% | {stats['favorable']}/6 |")
    chat_ok, chat_name = _chat_status()
    reconstruction_ok, reconstruction_name, reconstruction = _reconstruction_status()
    lines += ["", "## Native contract tests", "- `mm_test_kernels` checks signed zeros, FP16-small values, exact round trips, a non-representable value and a final incomplete column block.", "- `mm_test_runtime` checks exact logits with odd FFN tails, activation and disable/rejection paths, quantized fallback and row4 mutual exclusion.", "", "## Weight reconstruction", f"- Artifact: `{reconstruction_name}`; {reconstruction.get('matrix_count', 0)} FFN matrices and {reconstruction.get('elements', 0):,} elements: **{'PASS' if reconstruction_ok else 'REJECT'}** (zero bit mismatches required).", "", "## Historical conversation", f"- Artifact: `{chat_name}`; six-turn deterministic conversation parity: **{'PASS' if chat_ok else 'REJECT'}**.", "- The check compares prompt IDs, sampled IDs, text and finish reasons for X0 and X1.", "", "## Paired details"]
    for length in (256, 1792):
        speed = rows.get(f"cpu-r6-speed-X0-X1-{length}", [])
        diagnostic = rows.get(f"cpu-r6-diagnostics-X0-X1-{length}", [])
        lines += [f"### Prefix {length}: decode generation seconds", _table(_safe(speed, "native_generation_seconds"), "s")]
        for name, label in (("ffn_prefill_ns", "FFN prefill"), ("ffn_decode_ns", "FFN decode")):
            lines += ["", f"### Prefix {length}: {label}", _table(_safe(diagnostic, name), "ns")]
    lines += ["", "## Acceptance check", f"Result: **{'PASS' if ok else 'REJECT'}**" + (f" — {', '.join(evaluation['reasons'])}" if evaluation["reasons"] else "."), "", "Acceptance requires direct raw-entry parity, six alternating pairs, all-pair FFN diagnostic reductions, measurable decode benefit in long context, and no reproducible short-context regression. F16C availability and exact reconstruction are activation prerequisites; no benchmark is run with the diagnostic profiler enabled for final speed figures."]
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf8")


def main() -> int:
    try:
        summary = load("summary.json")
        evaluation = evaluate(summary)
    except (OSError, json.JSONDecodeError) as exc:
        summary = {}
        evaluation = {"ok": False, "reasons": [f"invalid summary.json: {exc}"], "rows": {}}
    render(summary, evaluation, REPORT)
    return 0 if evaluation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
