"""Pure orchestration tests for CPU-R1 benchmark receipts (no model/DLL)."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location("cpu_r1_benchmark",HERE/"benchmark.py")
benchmark=importlib.util.module_from_spec(spec);spec.loader.exec_module(benchmark)
def test_case_matrix_has_requested_boundaries():
 class A: suite="all";subset=None;prefix_lengths=[64,256,1024,1792];diagnostic_context=256
 cases=benchmark.make_cases(A());keys={c["key"] for c in cases}
 assert "attribution-R0-R1-historical" in keys and "scaling-S1-P4-historical" in keys
 assert {c["key"] for c in cases if c["family"]=="forced"}=={f"forced-S1-{p}-{n}" for p in ("S0","P2","P4") for n in A.prefix_lengths}
 assert {c["key"] for c in cases if c["family"]=="diagnostics"}=={"diagnostics-S1","diagnostics-P2","diagnostics-P4"}
def test_balanced_order_and_signed_delta():
 assert [benchmark.ordered_pair("R0","R3",n) for n in range(1,7)]==[["R0","R3"],["R3","R0"]]*3
 case={"left":"R0","right":"R3"};e=lambda p,t:{"profile":p,"total_seconds":t,"conversation_seconds":t}
 out=benchmark.aggregate(case,[{"pair":1,"order":["R0","R3"],"entries":[e("R0",10),e("R3",8)],"comparison_result":{"parity":True}}])
 assert out["paired_differences"][0]["right_minus_left"]["total_seconds"]==-2
def test_forced_continuation_is_identical_across_context_lengths():
 a,b=benchmark.forced_ids(64,32,6400),benchmark.forced_ids(256,32,6400)
 assert a[1].tolist()==b[1].tolist() and len(a[0])==64 and len(b[0])==256
def test_compare_fails_closed_for_missing_or_empty_evidence():
 assert benchmark.compare_entries({}, {}, "historical")["parity"] is False
 assert benchmark.compare_entries({"turns":[]},{"turns":[]},"historical")["parity"] is False
 assert benchmark.compare_entries({"prefix_token_ids":[]},{"prefix_token_ids":[]},"forced")["parity"] is False
def test_compare_requires_all_fidelity_fields():
 turn={"prompt_token_ids":[1],"sampled_ids":[2],"text":"x","finish_reason":"eos","generated_tokens":0}
 assert benchmark.compare_entries({"turns":[turn]},{"turns":[dict(turn,text="y")]},"historical")["parity"] is False
 assert benchmark.compare_entries({"turns":[turn]},{"turns":[turn]},"historical")["parity"] is True
def test_summary_and_aggregate_include_dispersion():
 assert benchmark.summarize([1,2,3])["stdev"]>0
 case={"left":"S1","right":"P2"};e=lambda p,t:{"profile":p,"total_seconds":t,"conversation_seconds":t}
 recs=[{"pair":n,"order":benchmark.ordered_pair("S1","P2",n),"entries":[e("S1",10+n),e("P2",8+n)],"comparison_result":{"parity":True}} for n in range(1,7)]
 out=benchmark.aggregate(case,recs);assert out["pairs"]==6 and out["profiles"]["S1"]["total_seconds"]["count"]==6
 assert len(out["paired_differences"])==6
def test_effective_configuration_rejects_mismatch():
 class Runtime: threads=1;cpus=[0];row_weights=[1];selective_logits=False
 with pytest.raises(AssertionError): benchmark.effective(Runtime(),{"threads":1,"cpus":[0],"row_weights":[1],"selective_logits":True,"reuse_kv":False},False)

def test_full_default_case_matrix_count_and_workloads():
 class A: suite="all";subset=None;prefix_lengths=[64,256,1024,1792];diagnostic_context=256
 cases=benchmark.make_cases(A())
 assert len(cases)==27
 assert sum(c["family"]=="attribution" for c in cases)==6
 assert sum(c["family"]=="scaling" for c in cases)==6
 assert sum(c["family"]=="forced" for c in cases)==12
 assert sum(c["family"]=="diagnostics" for c in cases)==3

def test_audited_prompts_match_fidelity_validator():
 import importlib.util
 p=HERE/"validate.py"; s=importlib.util.spec_from_file_location("cpu_r1_validate",p); v=importlib.util.module_from_spec(s); s.loader.exec_module(v)
 assert benchmark.HISTORICAL_PROMPTS==v.HISTORICAL_PROMPTS
 assert benchmark.SECOND_PROMPTS==v.SECOND_PROMPTS

def test_resume_pair_accepts_identity_case_pair_and_both_orders(tmp_path):
 case={"key":"attribution-R0-R1-historical","left":"R0","right":"R1","workload":"historical"}; meta={"identity":1}
 for number in (1,2):
  order=benchmark.ordered_pair(case["left"],case["right"],number)
  path=tmp_path/f"p{number}.json"; path.write_text(__import__("json").dumps({"identity":meta,"case":case,"pair":number,"status":"PASS","order":order}))
  assert benchmark.resume_pair(path,meta,case,number)["pair"]==number

def test_resume_pair_rejects_identity_case_pair_order_and_fail(tmp_path):
 case={"key":"x","left":"R0","right":"R1","workload":"historical"}; meta={"identity":1}; path=tmp_path/"x.json"
 base={"identity":meta,"case":case,"pair":1,"status":"PASS","order":["R0","R1"]}
 for changed in ({"identity":{"identity":2}}, {"case":dict(case,key="y")}, {"pair":2}, {"order":["R1","R0"]}, {"status":"FAIL"}):
  value=dict(base); value.update(changed); path.write_text(__import__("json").dumps(value))
  with pytest.raises(ValueError): benchmark.resume_pair(path,meta,case,1)

def test_aggregate_ba_order_still_orients_deltas_by_profile():
 case={"left":"S1","right":"P2"}; e=lambda p,t:{"profile":p,"total_seconds":t,"conversation_seconds":t}
 rec={"pair":2,"order":["P2","S1"],"entries":[e("P2",8),e("S1",10)],"comparison_result":{"parity":True}}
 out=benchmark.aggregate(case,[rec])
 assert out["paired_differences"][0]["right_minus_left"]["total_seconds"]==-2
