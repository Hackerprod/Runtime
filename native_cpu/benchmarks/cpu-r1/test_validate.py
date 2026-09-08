"""Unit coverage for CPU-R1 fidelity comparison (no model/DLL execution)."""
import importlib.util, json
from pathlib import Path
import numpy as np
PATH = Path(__file__).with_name("validate.py")
spec = importlib.util.spec_from_file_location("cpu_r1_validate", PATH); v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

def test_first_vector_mismatch_is_bitwise_and_json_safe():
    x = np.array([1.0, -2.0], dtype=np.float32)
    assert v.first_vector_mismatch(x, x.copy(), turn="t", position=3, token_id=4) is None
    y = x.copy(); y[1] = np.nextafter(y[1], np.float32(0)); d = v.first_vector_mismatch(x, y, turn="t", position=3, token_id=4)
    assert d["first_index"] == 1; json.dumps(d, allow_nan=False)

def _turn(**changes):
    row = {"sampled_ids":[1,2], "text":"ok", "finish_reason":"length", "positions":[4,5], "_vectors":[np.ones(2,np.float32),np.ones(2,np.float32)], "metrics":{"prompt_token_ids":[8,9]}}
    row.update(changes); return row
def _conv(*turns): return {"turns":list(turns)}

def test_compare_conversations_exact_match_passes(): assert v.compare_conversations(_conv(_turn()), _conv(_turn()), "exact") is None

def test_compare_conversations_tiny_bit_difference_fails():
    r = _turn(_vectors=[np.ones(2,np.float32), np.array([1.0,np.nextafter(np.float32(1),np.float32(2))],np.float32)])
    d=v.compare_conversations(_conv(_turn()),_conv(r),"bits"); assert d and d["first_index"]==1

def test_compare_conversations_reports_generation_fields():
    for field,value in (("sampled_ids",[1,3]),("text","different"),("finish_reason","eos"),("positions",[4,6])):
        d=v.compare_conversations(_conv(_turn()),_conv(_turn(**{field:value})),"field"); assert d["kind"]=="generation" and d["field"]==field

def test_compare_conversations_reports_prompt_and_vector_count():
    d=v.compare_conversations(_conv(_turn()),_conv(_turn(metrics={"prompt_token_ids":[1]})),"prompt"); assert d["kind"]=="prompt_ids"
    d=v.compare_conversations(_conv(_turn()),_conv(_turn(_vectors=[np.ones(2,np.float32)])),"count"); assert d["kind"]=="vector_count"

def test_compare_conversations_reports_turn_count_and_vector_shape():
    assert v.compare_conversations(_conv(_turn()),_conv(),"turns")["kind"]=="turn_count"
    r=_turn(_vectors=[np.ones(3,np.float32),np.ones(3,np.float32)]); d=v.compare_conversations(_conv(_turn()),_conv(r),"shape"); assert d and d["first_index"]==0

def test_mismatch_preserves_actual_position_and_token():
    r=_turn(_vectors=[np.array([0.0,1.0],np.float32),np.ones(2,np.float32)]); d=v.compare_conversations(_conv(_turn()),_conv(r),"trace")
    assert d["position"]==4 and d["token_id"]==1 and d["decision"]==0

def test_historical_prompts_match_audited_input():
    assert v.HISTORICAL_PROMPTS == ["Can you help me with programming?", "Can you help me with programming?", "talk me about how study programming", "c#", "I like c# languaje", "talk me more about languajes please"]
