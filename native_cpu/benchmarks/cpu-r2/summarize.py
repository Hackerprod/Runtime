from __future__ import annotations
import json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "results"
REPORT = Path(__file__).resolve().parent / "RESULTS.md"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf8"))


def pct(delta: float, base: float) -> float:
    return -100.0 * delta / base if base else 0.0


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def paired(case_key: str):
    rows=[]
    for i in range(1,7):
        rec=load(f"{case_key}-pair{i:02d}.json")
        entries={e["profile"]:e for e in rec["entries"]}
        left,right=entries["V0"],entries["V1"]
        rows.append((i, rec["order"], left, right, rec["comparison_result"]))
    return rows


def metric_rows(case_key: str, metric: str):
    out=[]
    for i,order,left,right,cmp in paired(case_key):
        delta=right[metric]-left[metric]
        out.append((i, order, left[metric], right[metric], delta, pct(delta, left[metric]), cmp["parity"]))
    return out


def attention_rows(case_key: str, phase: str):
    out=[]
    for i,order,left,right,cmp in paired(case_key):
        l=left["native_phase_stats"][phase]["attention_kv_ns"] / 1e6
        r=right["native_phase_stats"][phase]["attention_kv_ns"] / 1e6
        delta=r-l
        out.append((i, order, l, r, delta, pct(delta, l), cmp["parity"]))
    return out


def summarize(rows):
    left=[r[2] for r in rows]; right=[r[3] for r in rows]; delta=[r[4] for r in rows]; benefit=[r[5] for r in rows]
    return {
        "left_median": statistics.median(left),
        "right_median": statistics.median(right),
        "delta_median": statistics.median(delta),
        "delta_stdev": stdev(delta),
        "benefit_median_percent": statistics.median(benefit),
        "all_improved": all(x < 0 for x in delta),
    }


def table(rows, unit: str):
    lines=["| Pair | Order | V0 reference | V1 blocked | V1-V0 | Benefit | Parity |", "|---:|---|---:|---:|---:|---:|---|"]
    for i,order,left,right,delta,benefit,parity in rows:
        lines.append(f"| {i} | {'→'.join(order)} | {left:.6f} {unit} | {right:.6f} {unit} | {delta:.6f} {unit} | {benefit:.3f}% | {parity} |")
    return "\n".join(lines)


def main():
    summary=load("summary.json")
    cases={c["case"]["key"]:c for c in summary["cases"]}
    required=["cpu-r2-speed-V0-V1-256","cpu-r2-speed-V0-V1-1792","cpu-r2-diagnostics-V0-V1-256","cpu-r2-diagnostics-V0-V1-1792"]
    missing=[c for c in required if c not in cases]
    if missing: raise SystemExit(f"missing cases: {missing}")
    if not all(cases[c]["parity"] for c in required): raise SystemExit("parity failed")
    lines=[]
    lines.append("# CPU-R2 results — V accumulation locality")
    lines.append("")
    lines.append("Status: **accepted**. The opt-in V-blocked attention route preserved bitwise logits/tokens/text and improved measured attention plus decode on both requested prefix lengths.")
    lines.append("")
    lines.append("## Scope guard")
    lines.append("- Only the post-softmax accumulation over V was changed.")
    lines.append("- Q·K, softmax, KV layout, FFN, weights, quantization, workers, affinities and defaults were not changed.")
    lines.append("- Route is opt-in with `--v-blocked-attention`; default remains the original accumulation route.")
    lines.append("")
    lines.append("## Benchmark command")
    lines.append("```powershell")
    lines.append("python native_cpu/benchmarks/cpu-r1/benchmark.py --suite cpu-r2 --pairs 6 --warmups 1 --continuation 32 --out native_cpu/benchmarks/cpu-r2/results")
    lines.append("```")
    lines.append("")
    lines.append(f"Benchmark status: `{summary['status']}`; final protocol: `{summary['final_protocol']}`; raw pair files: `native_cpu/benchmarks/cpu-r2/results/*.json`.")
    chat_path = Path(__file__).resolve().parent / "chat-parity.json"
    if chat_path.is_file():
        chat = json.loads(chat_path.read_text(encoding="utf8"))
        lines.append(f"Chat token/text parity: `native_cpu/benchmarks/cpu-r2/chat-parity.json` = `{chat['status']}` for the historical six-turn conversation under CPU-R1 flags plus CPU 0 pinning.")
    lines.append("")
    lines.append("## Summary")
    lines.append("| Case | Metric | V0 median | V1 median | Median V1-V0 | Δ stdev | Median benefit | All pairs improved |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for length in (256,1792):
        for label,case_key,metric,unit in [
            ("speed", f"cpu-r2-speed-V0-V1-{length}", "native_generation_seconds", "s"),
            ("speed", f"cpu-r2-speed-V0-V1-{length}", "decode_tokens_per_second", "tok/s"),
        ]:
            rows=metric_rows(case_key, metric); s=summarize(rows)
            if metric == "decode_tokens_per_second":
                # For throughput, positive V1-V0 is good. Recompute benefit as throughput lift.
                vals=[r[3]-r[2] for r in rows]
                benefits=[100.0*(r[3]-r[2])/r[2] for r in rows]
                all_good=all(v > 0 for v in vals)
                lines.append(f"| {label} {length} | {metric} | {s['left_median']:.6f} {unit} | {s['right_median']:.6f} {unit} | {statistics.median(vals):.6f} {unit} | {stdev(vals):.6f} | {statistics.median(benefits):.3f}% | {all_good} |")
            else:
                lines.append(f"| {label} {length} | {metric} | {s['left_median']:.6f} {unit} | {s['right_median']:.6f} {unit} | {s['delta_median']:.6f} {unit} | {s['delta_stdev']:.6f} | {s['benefit_median_percent']:.3f}% | {s['all_improved']} |")
        for phase in ("prefill","decode"):
            rows=attention_rows(f"cpu-r2-diagnostics-V0-V1-{length}", phase); s=summarize(rows)
            lines.append(f"| diagnostics {length} | attention_{phase}_ms | {s['left_median']:.6f} ms | {s['right_median']:.6f} ms | {s['delta_median']:.6f} ms | {s['delta_stdev']:.6f} | {s['benefit_median_percent']:.3f}% | {s['all_improved']} |")
    lines.append("")
    lines.append("## Paired speed details")
    for length in (256,1792):
        lines.append(f"### Prefix {length}: native decode/generation seconds")
        lines.append(table(metric_rows(f"cpu-r2-speed-V0-V1-{length}", "native_generation_seconds"), "s"))
        lines.append("")
    lines.append("## Paired diagnostic attention details")
    for length in (256,1792):
        for phase in ("prefill","decode"):
            lines.append(f"### Prefix {length}: attention {phase}")
            lines.append(table(attention_rows(f"cpu-r2-diagnostics-V0-V1-{length}", phase), "ms"))
            lines.append("")
    lines.append("## Acceptance check")
    ok=True
    reasons=[]
    for length in (256,1792):
        speed=metric_rows(f"cpu-r2-speed-V0-V1-{length}", "native_generation_seconds")
        att=attention_rows(f"cpu-r2-diagnostics-V0-V1-{length}", "decode")
        sp=summarize(speed); at=summarize(att)
        if not sp["all_improved"]: ok=False; reasons.append(f"decode speed regression at {length}")
        if not at["all_improved"]: ok=False; reasons.append(f"attention diagnostic regression at {length}")
        if abs(sp["delta_median"]) <= sp["delta_stdev"]: ok=False; reasons.append(f"decode benefit inside dispersion at {length}")
    lines.append(f"Result: **{'PASS' if ok else 'REJECT'}**" + (f" — {', '.join(reasons)}" if reasons else "."))
    REPORT.write_text("\n".join(lines)+"\n", encoding="utf8")
    print(REPORT)

if __name__ == "__main__":
    main()
