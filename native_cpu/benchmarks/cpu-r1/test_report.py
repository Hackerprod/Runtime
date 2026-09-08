"""Bounded synthetic tests for CPU-R1 report validation/rendering."""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import pytest
HERE=Path(__file__).parent

def load(name):
 s=importlib.util.spec_from_file_location(name,HERE/(name+".py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
report=load("report"); engine=load("benchmark")
class Opt: suite="all";subset=None;prefix_lengths=[64,256,1024,1792];diagnostic_context=256

def record(case,n,identity):
 order=engine.ordered_pair(case["left"],case["right"],n); entries=[]
 for profile in order:
  if case["workload"] in engine.CONVERSATIONS:
   turn={"prompt_token_ids":[1,2],"sampled_ids":[3],"text":"x","finish_reason":"length","generated_tokens":1,"prompt_tokens_total":2,"prefix_tokens_reused":0,"prefill_tokens_evaluated":2,"lm_head_calls":{"prefill":1,"decode":1},"prefill_seconds":1.,"native_generation_seconds":1.,"sampling_seconds":.1,"prompt_prepare_seconds":.1,"frontend_other_seconds":.1,"generation_seconds":1.,"native_seconds":2.,"total_seconds":2.3}
   entry={"profile":profile,"turns":[turn],"conversation_seconds":2.3,**{k:turn[k] for k in engine.TIMES+engine.COUNTS if k in turn}}
  elif case["family"]=="diagnostics":
   st={k:1 for k in ("qkv_ns","attention_kv_ns","output_projection_ns","ffn_ns","vocab_head_ns","remaining_ops_ns","controller_wait_ns")};st["participant_compute_ns"]=[1,1,1,1] if profile.startswith("P") else [1]
   entry={"profile":profile,"prefix_token_ids":[1],"continuation_token_ids":[2],"prefill_logits_sha256":"a","decode_logits_sha256":["b"],"position_after":2,"total_seconds":2.,"native_phase_stats":{"prefill":st,"decode":st}}
  else:
   entry={"profile":profile,"prefix_token_ids":[1],"continuation_token_ids":[2],"prefill_logits_sha256":"a","decode_logits_sha256":["b"],"position_after":2,"prefill_seconds":1.,"native_generation_seconds":1.,"total_seconds":2.,"prefill_tokens_per_second":1.,"decode_tokens_per_second":1.}
  entries.append(entry)
 cmp=engine.compare_entries(entries[0],entries[1],case["workload"])
 return {"identity":identity,"case":case,"pair":n,"order":order,"entries":entries,"comparison_result":cmp,"status":"PASS"}

def fixture():
 cases=engine.make_cases(Opt());identity={"id":1};records={};summaries=[]
 for c in cases:
  rs=[record(c,n,identity) for n in range(1,7)]
  records.update({f"{c['key']}-pair{r['pair']:02d}":r for r in rs});summaries.append(engine.aggregate(c,rs))
 data={"identity":identity,"cases":summaries,"status":"PASS","final_protocol":True}
 return data,records

def test_validate_accepts_complete_synthetic_matrix():
 data,records=fixture();report.validate(data,records)

def test_validate_rejects_unbalanced_order():
 data,records=fixture(); key=next(iter(records));records[key]["order"]=records[key]["order"][::-1]
 with pytest.raises(ValueError):report.validate(data,records)

def test_validate_rejects_identity_mismatch():
 data,records=fixture();next(iter(records.values()))["identity"]={"id":2}
 with pytest.raises(ValueError):report.validate(data,records)

def test_validate_rejects_altered_raw_token():
 data,records=fixture();r=next(iter(records.values()));r["entries"][0]["turns"][0]["sampled_ids"]=[9]
 with pytest.raises(ValueError):report.validate(data,records)

def test_diagnostics_requires_component_and_participant_keys():
 data,records=fixture();assert report.diagnostics(records)
 r=next(v for v in records.values() if v["case"]["family"]=="diagnostics");del r["entries"][0]["native_phase_stats"]["prefill"]["qkv_ns"]
 with pytest.raises(ValueError):report.diagnostics(records)

def test_table_header_and_row_widths_are_consistent():
 text=report.table(["A","B"],[[1,2],[3,4]])
 assert all(line.count("|")==3 for line in text.splitlines())

def test_historical_collects_r0_to_r3_from_three_pair1_records():
 data,records=fixture();rows=report.historical(records)
 assert len(rows)==1 and all(p+":" in rows[0][1] for p in ("R0","R1","R2","R3"))
