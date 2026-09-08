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


def test_cpu_r2_case_matrix_is_bounded_to_requested_prefixes():
 class A: suite="cpu-r2";subset=None;prefix_lengths=[64,256,1024,1792];diagnostic_context=256
 cases=benchmark.make_cases(A())
 assert [c["key"] for c in cases]==[
  "cpu-r2-speed-V0-V1-256",
  "cpu-r2-diagnostics-V0-V1-256",
  "cpu-r2-speed-V0-V1-1792",
  "cpu-r2-diagnostics-V0-V1-1792",
 ]
 assert all(c["left"]=="V0" and c["right"]=="V1" and c["workload"]=="forced" for c in cases)
 assert [bool(c.get("diagnostics")) for c in cases]==[False, True, False, True]
 assert benchmark.PROFILES["V0"]["selective_logits"] and benchmark.PROFILES["V0"]["reuse_kv"]
 assert benchmark.PROFILES["V0"]["cpus"]==[0] and benchmark.PROFILES["V0"]["threads"]==1
 assert benchmark.PROFILES["V1"]["v_blocked_attention"] is True

def test_effective_configuration_checks_v_blocked_attention():
 class Runtime:
  threads=1;cpus=[0];row_weights=[1];selective_logits=True;v_blocked_attention=False
 cfg={"threads":1,"cpus":[0],"row_weights":[1],"selective_logits":True,"reuse_kv":True,"v_blocked_attention":True}
 with pytest.raises(AssertionError): benchmark.effective(Runtime(),cfg,False)

def test_cpu_r3_case_matrix_is_bounded_to_requested_prefixes():
 class A: suite="cpu-r3";subset=None;prefix_lengths=[64,256,1024,1792];diagnostic_context=256
 cases=benchmark.make_cases(A())
 assert [c["key"] for c in cases]==[
  "cpu-r3-speed-F0-F1-256",
  "cpu-r3-diagnostics-F0-F1-256",
  "cpu-r3-speed-F0-F1-1792",
  "cpu-r3-diagnostics-F0-F1-1792",
 ]
 assert all(c["left"]=="F0" and c["right"]=="F1" and c["workload"]=="forced" for c in cases)
 assert [bool(c.get("diagnostics")) for c in cases]==[False,True,False,True]
 assert benchmark.PROFILES["F0"]["v_blocked_attention"] and not benchmark.PROFILES["F0"]["ffn_row4"]
 assert benchmark.PROFILES["F1"]["v_blocked_attention"] and benchmark.PROFILES["F1"]["ffn_row4"]

def test_effective_configuration_checks_ffn_row4():
 class Runtime:
  threads=1;cpus=[0];row_weights=[1];selective_logits=True;v_blocked_attention=True;ffn_row4=False
 cfg={"threads":1,"cpus":[0],"row_weights":[1],"selective_logits":True,"reuse_kv":True,"v_blocked_attention":True,"ffn_row4":True}
 with pytest.raises(AssertionError): benchmark.effective(Runtime(),cfg,False)

def test_cpu_r4_case_matrix_is_bounded_to_requested_prefixes():
 class A: suite="cpu-r4";subset=None;prefix_lengths=[64,256,1024,1792];diagnostic_context=256
 cases=benchmark.make_cases(A())
 assert [c["key"] for c in cases]==[
  "cpu-r4-speed-K0-K1-256",
  "cpu-r4-diagnostics-K0-K1-256",
  "cpu-r4-speed-K0-K1-1792",
  "cpu-r4-diagnostics-K0-K1-1792",
 ]
 assert all(c["left"]=="K0" and c["right"]=="K1" and c["workload"]=="forced" for c in cases)
 assert [bool(c.get("diagnostics")) for c in cases]==[False,True,False,True]
 assert benchmark.PROFILES["K0"]["v_blocked_attention"] and not benchmark.PROFILES["K0"]["gqa_k_shared"]
 assert benchmark.PROFILES["K1"]["v_blocked_attention"] and benchmark.PROFILES["K1"]["gqa_k_shared"]

def test_effective_configuration_checks_gqa_k_shared():
 class Runtime:
  threads=1;cpus=[0];row_weights=[1];selective_logits=True;v_blocked_attention=True;ffn_row4=False;gqa_k_shared=False
 cfg={"threads":1,"cpus":[0],"row_weights":[1],"selective_logits":True,"reuse_kv":True,"v_blocked_attention":True,"ffn_row4":False,"gqa_k_shared":True}
 with pytest.raises(AssertionError): benchmark.effective(Runtime(),cfg,False)
