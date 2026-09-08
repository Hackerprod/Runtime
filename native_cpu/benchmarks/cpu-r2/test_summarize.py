"""Synthetic evidence tests for the CPU-R2 verdict/report boundary."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("cpu_r2_summarize", HERE / "summarize.py")
summarize = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(summarize)


def _case(key: str) -> dict:
    length = int(key.rsplit("-", 1)[1])
    return {
        "key": key,
        "family": "cpu-r2-diagnostics" if "diagnostics" in key else "cpu-r2-speed",
        "left": "V0",
        "right": "V1",
        "workload": "forced",
        "length": length,
        **({"diagnostics": True} if "diagnostics" in key else {}),
    }


def _entry(profile: str, case: dict, pair: int) -> dict:
    faster = profile == "V1"
    return {
        "profile": profile,
        "prefix_token_ids": [1, 2],
        "continuation_token_ids": [3],
        "prefill_logits_sha256": "same-prefill",
        "decode_logits_sha256": ["same-decode"],
        "position_after": 3,
        "native_generation_seconds": 8.0 + pair * 0.01 if faster else 10.0 + pair * 0.01,
        "decode_tokens_per_second": 1.25 if faster else 1.0,
        "native_phase_stats": {
            "prefill": {"attention_kv_ns": 800.0 if faster else 1000.0},
            "decode": {"attention_kv_ns": 800.0 if faster else 1000.0},
        },
    }


def _fixture(root: Path) -> Path:
    out = root / "results"
    out.mkdir()
    identity = {"fixture": 1}
    cases = []
    for key in summarize.REQUIRED_CASES:
        case = _case(key)
        cases.append({"case": case, "pairs": summarize.PAIR_COUNT, "parity": True})
        for pair in range(1, summarize.PAIR_COUNT + 1):
            order = ["V0", "V1"] if pair % 2 else ["V1", "V0"]
            entries = [_entry(profile, case, pair) for profile in order]
            record = {
                "identity": identity,
                "case": case,
                "pair": pair,
                "order": order,
                "entries": entries,
                "comparison_result": summarize.compare_entries(entries[0], entries[1], "forced"),
                "status": "PASS",
            }
            (out / f"{key}-pair{pair:02d}.json").write_text(json.dumps(record), encoding="utf8")
    summary = {"identity": identity, "cases": cases, "status": "PASS", "final_protocol": True}
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf8")
    return out


def _run(root: Path) -> tuple[int, str]:
    out = _fixture(root)
    report = root / "RESULTS.md"
    code = summarize.main(out=out, report_path=report)
    return code, report.read_text(encoding="utf8")


def test_acceptance_uses_one_verdict_for_report_and_exit_code(tmp_path):
    code, text = _run(tmp_path)
    assert code == 0
    assert "Status: **accepted**" in text
    assert "Result: **PASS**" in text


def test_speed_regression_rejects_without_accepted_header(tmp_path):
    out = _fixture(tmp_path)
    path = out / "cpu-r2-speed-V0-V1-256-pair01.json"
    record = json.loads(path.read_text(encoding="utf8"))
    v1 = next(entry for entry in record["entries"] if entry["profile"] == "V1")
    v1["native_generation_seconds"] = 11.0
    path.write_text(json.dumps(record), encoding="utf8")
    report = tmp_path / "RESULTS.md"
    assert summarize.main(out=out, report_path=report) != 0
    text = report.read_text(encoding="utf8")
    assert "Status: **accepted**" not in text
    assert "Result: **REJECT**" in text


def test_direct_logits_divergence_rejects_even_when_stored_parity_is_true(tmp_path):
    out = _fixture(tmp_path)
    path = out / "cpu-r2-speed-V0-V1-256-pair01.json"
    record = json.loads(path.read_text(encoding="utf8"))
    v1 = next(entry for entry in record["entries"] if entry["profile"] == "V1")
    v1["decode_logits_sha256"] = ["different-decode"]
    path.write_text(json.dumps(record), encoding="utf8")
    report = tmp_path / "RESULTS.md"
    assert summarize.main(out=out, report_path=report) != 0
    text = report.read_text(encoding="utf8")
    assert "Status: **accepted**" not in text
    assert "direct parity mismatch" in text


def test_missing_pair_rejects_and_returns_nonzero(tmp_path):
    out = _fixture(tmp_path)
    (out / "cpu-r2-diagnostics-V0-V1-1792-pair06.json").unlink()
    report = tmp_path / "RESULTS.md"
    assert summarize.main(out=out, report_path=report) != 0
    text = report.read_text(encoding="utf8")
    assert "Status: **accepted**" not in text
    assert "missing evidence" in text
