"""Render an offline Markdown report from benchmark.py JSON evidence."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def med(values):
    values = [float(x) for x in values if x is not None]
    return statistics.median(values) if values else None


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def load_results(root):
    summary = root / "summary.json"
    if summary.exists():
        data = json.loads(summary.read_text(encoding="utf8"))
        records = {p.stem: json.loads(p.read_text(encoding="utf8"))
                   for p in root.glob("*-pair*.json")}
        return data, records
    raise FileNotFoundError(f"missing benchmark summary: {summary}")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(out)


def validate(data, records):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cpu_r1_benchmark_report", Path(__file__).with_name("benchmark.py"))
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    cases = data["cases"]
    if data["status"] != "PASS" or not data["final_protocol"]:
        raise ValueError("not a final PASS run")
    options = engine.build_parser().parse_args(["--out", "."])
    options.prefix_lengths = [64, 256, 1024, 1792]
    expected = {c["key"]: c for c in engine.make_cases(options)}
    if len(cases) != 27 or {c["case"]["key"] for c in cases} != set(expected):
        raise ValueError("incorrect final case matrix")
    if len(records) != 162:
        raise ValueError("expected exactly 162 paired artifacts")
    for item in cases:
        case = item["case"]
        if case != expected[case["key"]]:
            raise ValueError("case configuration mismatch")
        rows = sorted((r for r in records.values() if r["case"] == case), key=lambda r:r["pair"])
        if [r["pair"] for r in rows] != list(range(1,7)):
            raise ValueError("missing or duplicate pairs")
        for r in rows:
            order = engine.ordered_pair(case["left"], case["right"], r["pair"])
            if r["identity"] != data["identity"] or r["status"] != "PASS" or r["order"] != order:
                raise ValueError("identity/status/order mismatch")
            if len(r["entries"]) != 2 or [e["profile"] for e in r["entries"]] != order:
                raise ValueError("entries do not match recorded order")
            parity = engine.compare_entries(*r["entries"], case["workload"])
            if not parity["parity"] or parity != r["comparison_result"]:
                raise ValueError("raw fidelity mismatch")
        if engine.aggregate(case, rows) != item:
            raise ValueError("summary differs from raw records")


def conversation(cases):
    rows = []
    for item in cases:
        c = item["case"]
        if c.get("family") not in ("attribution", "scaling") or c.get("workload") not in ("historical", "second"):
            continue
        p = item["profiles"]
        def v(profile, key):
            x = p[profile][key]; return fmt(x.get("median")) + " ± " + fmt(x.get("stdev"))
        for profile in (c["left"], c["right"]):
            rows.append([c["key"], c["family"], c["workload"], profile,
                         v(profile, "prefill_seconds"), v(profile, "native_generation_seconds"),
                         v(profile, "sampling_seconds"), v(profile, "total_seconds"), v(profile, "conversation_seconds")])
    return rows


def forced(cases):
    rows = []
    for item in cases:
        c = item["case"]
        if c.get("family") != "forced": continue
        p = item["profiles"]
        for profile in (c["left"], c["right"]):
            def measurement(key):
                v=p[profile][key]
                return fmt(v["median"])+" ± "+fmt(v["stdev"])
            rows.append([c["key"],c["length"],profile,measurement("prefill_seconds"),
                         measurement("native_generation_seconds"),measurement("prefill_tokens_per_second"),
                         measurement("decode_tokens_per_second")])
    return rows


def diagnostics(records):
    buckets = {}
    for record in records.values():
        c = record.get("case", {})
        if c.get("family") != "diagnostics": continue
        for entry in record.get("entries", []):
            key = entry["profile"]
            for phase, stats in entry.get("native_phase_stats", {}).items():
                names = {"qkv":"qkv_ns", "attention":"attention_kv_ns", "output":"output_projection_ns", "ffn":"ffn_ns", "vocab":"vocab_head_ns", "remaining":"remaining_ops_ns", "wait":"controller_wait_ns"}
                if any(n not in stats for n in names.values()) or "participant_compute_ns" not in stats: raise ValueError("incomplete diagnostic stats")
                b = buckets.setdefault((key, phase), {**{n: [] for n in names}, **{f"participant{i}": [] for i in range(len(stats["participant_compute_ns"]))}})
                for name, actual in names.items(): b[name].append(stats[actual] / 1e6)
                for i, value in enumerate(stats["participant_compute_ns"]): b[f"participant{i}"].append(value / 1e6)
    rows = []
    for (profile, phase), b in sorted(buckets.items()):
        participants = [k for k in b if k.startswith("participant")]
        active = int(profile[1]) if profile.startswith("P") else 1
        per_worker = ", ".join(f"{i}: {fmt(med(b['participant'+str(i)]))}" for i in range(active))
        rows.append([profile, phase, *[fmt(med(b[k])) for k in ("qkv", "attention", "output", "ffn", "vocab", "remaining")], per_worker, fmt(med(b["wait"]))])
    return rows


def profile_overhead(cases):
    rows = []
    for item in cases:
        c = item.get("case", {})
        if c.get("family") != "diagnostics":
            continue
        pairs = item.get("paired_differences", [])
        overhead = [p["right_minus_left"]["total_seconds"] * 1000 for p in pairs if "total_seconds" in p.get("right_minus_left", {})]
        reduction = [p["total_time_reduction"] * 100 for p in pairs if "total_time_reduction" in p]
        if overhead: rows.append([c.get("profile", c.get("left")), fmt(med(overhead)), fmt(statistics.stdev(overhead)), fmt(min(overhead)), fmt(max(overhead)), fmt(med(reduction), 2)])
    return rows


def historical(records):
    rows = []
    entries = {}
    for record in records.values():
        c = record.get("case", {})
        if c.get("family") == "attribution" and c.get("workload") == "historical" and record.get("pair") == 1:
            for e in record.get("entries", []): entries[e["profile"]] = e
    if set(entries) != {"R0", "R1", "R2", "R3"}: raise ValueError("historical pair 1 lacks R0/R1/R2/R3")
    turns = min((len(e.get("turns", [])) for e in entries.values()), default=0)
    for i in range(turns):
        values = []
        for profile in ("R0", "R1", "R2", "R3"):
            t = entries.get(profile, {}).get("turns", [])[i]
            values.append(f"{profile}: {t.get('prompt_tokens_total','—')}/{t.get('prefix_tokens_reused','—')}/{t.get('prefill_tokens_evaluated','—')}/{t.get('lm_head_calls','—')}")
        rows.append([i + 1, "<br>".join(values)])
    return rows


def render(data, records, source, output):
    cases = data["cases"]
    lines = ["# CPU R1 Benchmark Report", "", "Offline summary of recorded benchmark evidence. Timing is wall-clock data; worker times overlap and are not summed.", ""]
    lines += ["## Conversation", "", table(["Case", "Family", "Workload", "Profile", "Prefill median ± stdev s", "Native decode median ± stdev s", "Sampling median ± stdev s", "Turn sum median ± stdev s", "Conversation wall median ± stdev s"], conversation(cases)), ""]
    lines += ["## Forced contexts", "", "Each value is median ± sample standard deviation within its own six-pair comparison.", "", table(["Case","Context", "Profile", "Prefill s","Native decode s","Prefill tok/s", "Decode tok/s"], forced(cases)), ""]
    drows = diagnostics(records)
    participant_headers = ["Participant: compute ms"]
    lines += ["## Diagnostics (milliseconds)", "", "Participant compute and controller wait are separate. Participant work overlaps and MUST NOT be added to elapsed time.", "", table(["Profile", "Phase", "QKV", "Attention", "Output", "FFN", "Vocab", "Remaining", *participant_headers, "Controller wait"], drows), ""]
    lines += ["### Profiling overhead (paired on − off)", "", "Negative estimates and wide dispersion are measurement noise, not evidence that profiling improves performance. No pairs were discarded.", "", table(["Profile", "Median overhead ms", "Stdev ms", "Min ms", "Max ms", "Median wall reduction %"], profile_overhead(cases)), ""]
    reductions = []
    for item in cases:
        if item["case"].get("family") in ("attribution", "scaling"):
            values = [p["total_time_reduction"] * 100 for p in item.get("paired_differences", [])]
            reductions.append([item["case"]["key"], fmt(med(values)), fmt(statistics.stdev(values) if len(values) > 1 else 0), fmt(min(values) if values else None), fmt(max(values) if values else None)])
    lines += ["## Paired reduction distributions (per comparison; never pooled)", "", table(["Case", "Median %", "Stdev %", "Min %", "Max %"], reductions), ""]
    lines += ["## Historical canonical prompt (pair 1)", "", "Each cell is `prompt_tokens_total / prefix_tokens_reused / prefill_tokens_evaluated / lm_head_calls`.", "", table(["Turn", "R0–R3"], historical(records)), ""]
    links = sorted(source.glob("*.json"))
    lines += ["## Raw evidence", "", "\n".join(f"- [{p.name}]({p.relative_to(output.parent).as_posix()})" for p in links) or "No raw artifacts.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data, records = load_results(args.results)
    validate(data, records)
    args.output.write_text(render(data, records, args.results, args.output), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
