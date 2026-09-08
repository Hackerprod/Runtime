"""Bounded, real-weight CPU-R1 fidelity validation.

This is intentionally separate from :mod:`benchmark`: it proves that the
selective-logits and persistent-KV routes are equivalent, rather than timing
them.  Every run uses the native FP32 runtime and records compact vector
hashes; complete vectors are emitted only for the first mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from native_cpu.tools.native import NativeRuntime  # noqa: E402
from native_cpu.tools.chat import ChatSession, load_local_tokenizer, DEFAULT_CHAT_SYSTEM  # noqa: E402

HISTORICAL_PROMPTS = [
    "Can you help me with programming?",
    "Can you help me with programming?",
    "talk me about how study programming",
    "c#",
    "I like c# languaje",
    "talk me more about languajes please",
]
SECOND_PROMPTS = ["Name one benefit of tests.", "Define latency briefly."]

FROZEN = ROOT / "native_cpu" / "validation" / "cpu-r1" / "frozen" / "minimind_cpu.dll"
DEFAULT_MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
DEFAULT_TOKENIZER = ROOT / "checkpoints" / "minimind-3-hf"
DEFAULT_LIBRARY = ROOT / "native_cpu" / "build" / "Release" / "minimind_cpu.dll"
DEFAULT_OUTPUT = ROOT / "native_cpu" / "validation" / "cpu-r1" / "real-fidelity.json"


def _json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return str(value)

def _sha256(path: Path) -> str | None:
    if not path.is_file(): return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def vector_receipt(vector: Any, position: int, token_id: int) -> dict[str, Any]:
    """Return a stable FP32 hash, retaining no large data in normal output."""
    raw = np.asarray(vector, dtype="<f4").reshape(-1)
    return {"position": int(position), "token_id": int(token_id),
            "vocab_size": int(raw.size),
            "sha256": hashlib.sha256(raw.tobytes()).hexdigest()}


def first_vector_mismatch(left: Any, right: Any, *, turn: str, position: int,
                          token_id: int) -> dict[str, Any] | None:
    """Return diagnostic evidence for the first *bitwise* mismatch."""
    a = np.asarray(left, dtype="<f4").reshape(-1)
    b = np.asarray(right, dtype="<f4").reshape(-1)
    if a.shape != b.shape:
        index = 0
    else:
        different = np.flatnonzero(a.view(np.uint32) != b.view(np.uint32))
        if not different.size:
            return None
        index = int(different[0])
    return {"turn": turn, "position": int(position), "token_id": int(token_id),
            "first_index": index,
            "argmax_left": int(np.argmax(a)) if a.size else None,
            "argmax_right": int(np.argmax(b)) if b.size else None,
            "left": a.tolist(), "right": b.tolist()}


def _configure(runtime: Any, *, selective: bool, profiling: bool = False) -> None:
    """Use the R1 names, with an explicit compatibility path for old DLLs."""
    method = getattr(runtime, "configure_selective_logits", None)
    if method is None:
        method = getattr(runtime, "set_selective_logits", None)
    if method is None and selective:
        raise RuntimeError("runtime lacks configure_selective_logits; R1 cannot be claimed")
    if method is not None:
        method(bool(selective))
    profile = getattr(runtime, "configure_profile", None)
    if profile is not None:
        try:
            params = inspect.signature(profile).parameters
            if "profiling" in params:
                profile(profiling=bool(profiling))
            else:
                profile(bool(profiling))
        except (TypeError, ValueError):
            profile(bool(profiling))


def _reset(runtime: Any) -> None:
    runtime.reset()
    reset_stats = getattr(runtime, "reset_stats", None)
    if reset_stats is not None:
        reset_stats()


def _eval_partitions(runtime: Any, ids: list[int], partitions: list[int], *, label: str) -> dict[str, Any]:
    _reset(runtime)
    vectors = []
    cursor = 0
    for size in partitions:
        if size <= 0 or cursor + size > len(ids):
            raise ValueError("invalid partition")
        chunk = ids[cursor:cursor + size]
        logits = runtime.eval(np.asarray(chunk, dtype=np.int32))
        vectors.append(vector_receipt(logits, int(runtime.position) - 1, chunk[-1]))
        cursor += size
    if cursor != len(ids):
        raise ValueError("partitions do not cover sequence")
    return {"label": label, "partitions": partitions, "position": int(runtime.position), "vectors": vectors}


def _compare_eval(runtime: Any, ids: list[int], *, prefix: list[int] | None = None) -> dict[str, Any]:
    """Compare whole, token-by-token and bounded 1/2/4-token partitions."""
    out: dict[str, Any] = {"length": len(ids), "mismatches": []}
    captures: dict[str, list[np.ndarray]] = {}; capture_positions: dict[str, list[int]] = {}
    for selective in (False, True):
        _configure(runtime, selective=selective)
        for name, parts in (("whole", [len(ids)]), ("token", [1] * len(ids)),
                            ("p2", [2] * (len(ids) // 2) + ([1] if len(ids) % 2 else [])),
                            ("p4", [4] * (len(ids) // 4) + ([len(ids) % 4] if len(ids) % 4 else []))):
            _reset(runtime)
            if prefix:
                runtime.eval(np.asarray(prefix, dtype=np.int32))
            cursor = 0; values = []; positions = []
            for n in parts:
                chunk = ids[cursor:cursor + n]; values.append(np.array(runtime.eval(np.asarray(chunk, dtype=np.int32)), copy=True)); positions.append(int(runtime.position) - 1); cursor += n
            captures[f"{'on' if selective else 'off'}:{name}"] = values
            capture_positions[f"{'on' if selective else 'off'}:{name}"] = positions
            out[f"{'on' if selective else 'off'}_{name}"] = [vector_receipt(v, pos, ids[min(pos - (len(prefix or []) if prefix else 0), len(ids)-1)]) for v, pos in zip(values, positions)]
    for name in ("whole", "token", "p2", "p4"):
        off, on = captures[f"off:{name}"], captures[f"on:{name}"]
        # Every route returns logits only at the end of each eval call.  Thus
        # compare same-call boundaries, plus the final decision of each route.
        for i, (a, b) in enumerate(zip(off, on)):
            pos = capture_positions[f"off:{name}"][i]
            local = pos - (len(prefix or []) if prefix else 0)
            mismatch = first_vector_mismatch(a, b, turn=f"{name}:off-vs-on", position=pos,
                                              token_id=ids[min(len(ids) - 1, max(0, local))])
            if mismatch:
                out["mismatches"].append(mismatch); return out
    # Partition endpoints must also equal the token-by-token reference from
    # the same selective mode (including an already-evaluated prefix offset).
    offset = len(prefix or [])
    for mode in ("off", "on"):
        tokens = captures[f"{mode}:token"]
        for name in ("whole", "p2", "p4"):
            for i, value in enumerate(captures[f"{mode}:{name}"]):
                pos = capture_positions[f"{mode}:{name}"][i]
                ref = tokens[pos - offset]
                mismatch = first_vector_mismatch(ref, value, turn=f"{mode}:{name}-vs-token", position=pos,
                                                  token_id=ids[pos - offset])
                if mismatch:
                    out["mismatches"].append(mismatch); return out
    return out


def _conversation(runtime, tokenizer, *, selective, reuse_kv, temperature, seed, max_new_tokens, prompts):
    _configure(runtime, selective=selective)
    session = ChatSession(runtime, tokenizer, context_limit=runtime.max_context,
        max_new_tokens=max_new_tokens, temperature=temperature, top_k=50, top_p=.85,
        seed=seed, system=DEFAULT_CHAT_SYSTEM, reuse_kv=reuse_kv)
    turns = []
    for prompt in prompts:
        captured = []; positions = []; original_sample = session._sample
        def capture(logits):
            captured.append(np.array(logits, dtype="<f4", copy=True))
            positions.append(int(runtime.position)-1)
            return original_sample(logits)
        session._sample = capture
        result = session.turn(prompt)
        session._sample = original_sample
        assert runtime.position == len(session.cached_token_ids)
        assert len(result.sampled_ids) == len(captured)
        turns.append({"user":prompt,"text":result.text,"sampled_ids":result.sampled_ids,
            "finish_reason":"eos" if result.sampled_ids and result.sampled_ids[-1]==tokenizer.eos_token_id else "length",
            "logits":[vector_receipt(v,p,t) for v,p,t in zip(captured,positions,result.sampled_ids)],
            "_vectors":captured,"positions":positions,"metrics":result.metrics,
            "cached_token_ids":list(session.cached_token_ids),"position":int(runtime.position)})
    return {"reuse_kv":reuse_kv,"selective_logits":selective,"temperature":temperature,"turns":turns}


def compare_conversations(left, right, label):
    a, b = left["turns"], right["turns"]
    if len(a) != len(b): return {"kind":"turn_count","profile":label,"left":len(a),"right":len(b)}
    for index, (x,y) in enumerate(zip(a,b)):
        vx,vy=x["_vectors"],y["_vectors"]
        first = None
        for decision,(v,w) in enumerate(zip(vx,vy)):
            first = first_vector_mismatch(v,w,turn=f"{label}:{index}",
                position=x["positions"][decision],token_id=x["sampled_ids"][decision])
            if first:
                first["decision"]=decision
                break
        if first: return first
        for field in ("sampled_ids","text","finish_reason","positions"):
            if x[field]!=y[field]: return {"kind":"generation","profile":label,"turn":index,
                "field":field,"left":x[field],"right":y[field],
                "first_logits_left":vx[0].tolist() if vx else None,"first_logits_right":vy[0].tolist() if vy else None}
        if len(vx)!=len(vy): return {"kind":"vector_count","profile":label,"turn":index,"left":len(vx),"right":len(vy)}
        if x["metrics"]["prompt_token_ids"]!=y["metrics"]["prompt_token_ids"]:
            return {"kind":"prompt_ids","profile":label,"turn":index}
    return None

def _truncation(runtime, ids):
    prefix=list(ids[:8]); rows=[]; errors=[]
    if not runtime.supports_truncate: raise RuntimeError("truncation capability required")
    for target in (0,3,7,8):
        _reset(runtime); runtime.eval(np.asarray(prefix,dtype=np.int32))
        epoch=runtime.cache_identity; runtime.truncate(target)
        assert runtime.position==target and not runtime.logits_valid and runtime.cache_identity!=epoch
        suffix=[(prefix[-1]+3)%runtime.vocab_size, (prefix[-1]+5)%runtime.vocab_size]
        got=runtime.eval(np.asarray(suffix,dtype=np.int32))
        _reset(runtime); clean=runtime.eval(np.asarray(prefix[:target]+suffix,dtype=np.int32))
        mismatch=first_vector_mismatch(clean,got,turn="truncate",position=target+1,token_id=suffix[-1])
        if mismatch: errors.append(mismatch)
        before=(runtime.position,runtime.cache_identity,runtime.logits_valid)
        for invalid in (-1,runtime.position+1,2**32,1.5):
            try: runtime.truncate(invalid)
            except (ValueError,RuntimeError): pass
            else: raise AssertionError("invalid truncate accepted")
            assert before==(runtime.position,runtime.cache_identity,runtime.logits_valid)
        rows.append({"target":target,"logits_sha256":vector_receipt(got,target+1,suffix[-1]),"position":runtime.position})
    return {"mismatches":errors,"cases":rows}

def validate(args):
    model,library=Path(args.model),Path(args.library)
    tokenizer=load_local_tokenizer(args.tokenizer)
    workloads={"historical":HISTORICAL_PROMPTS,"second":SECOND_PROMPTS}
    if args.prompts is not None:
        if not args.prompts: raise ValueError("custom prompts must not be empty")
        workloads={"custom":args.prompts}
    elif args.workload!="all": workloads={args.workload:workloads[args.workload]}
    records={"contract":"CPU-R1-real-fidelity-bitwise","context_limit":args.context_limit,
        "config":{"seed":args.seed,"maximum":args.max_new_tokens,"second_maximum":args.second_max_new_tokens,
                  "system":DEFAULT_CHAT_SYSTEM,"top_k":50,"top_p":.85,"workloads":workloads},
        "sha256":{"model":_sha256(model),"library":_sha256(library),
                  "frozen_library":_sha256(args.frozen),"tokenizer":_sha256(Path(args.tokenizer)/"tokenizer.json"),
                  "chat_template":_sha256(Path(args.tokenizer)/"chat_template.jinja"),
                  "validation_script":_sha256(Path(__file__))},"eval":{},"conversations":{},"errors":[]}
    layouts={"S1":{"threads":1,"cpus":[0]},"P2":{"threads":2,"cpus":[0,6],"row_weights":[1332,992]},
             "P4":{"threads":4,"cpus":[0,2,4,6],"row_weights":[1,1,1,1]}}
    ids=[1,4,3,5,6,9,7,8]
    for name,options in layouts.items():
        with NativeRuntime(model,library,args.context_limit,**options) as runtime:
            checks={"empty":_compare_eval(runtime,ids),"prefix":_compare_eval(runtime,ids,prefix=[1,5]),
                    "truncate":_truncation(runtime,ids)}
            records["eval"][name]=checks
            for check in checks.values(): records["errors"].extend(check.get("mismatches",[]))
        print("EVAL",name,"completed",flush=True)
    for workload,prompts in workloads.items():
        maximum=args.second_max_new_tokens if workload=="second" else args.max_new_tokens
        conv={}
        # R0-R3 attribution uses one identical pinned participant in the same DLL.
        routes=[("R0",False,False,layouts["S1"],library),("R1",True,False,layouts["S1"],library),
                ("R2",False,True,layouts["S1"],library),("R3",True,True,layouts["S1"],library),
                ("P2",True,True,layouts["P2"],library),("P4",True,True,layouts["P4"],library),
                ("frozen",False,False,layouts["S1"],Path(args.frozen))]
        for name,selective,reuse,options,dll in routes:
            if not dll.is_file(): raise FileNotFoundError(dll)
            with NativeRuntime(model,dll,args.context_limit,**options) as runtime:
                for mode,temperature in (("greedy",0.),("sampled",.9)):
                    key=f"{name}_{mode}"
                    conv[key]=_conversation(runtime,tokenizer,selective=selective,reuse_kv=reuse,
                        temperature=temperature,seed=args.seed,max_new_tokens=maximum,prompts=prompts)
                    print("CONVERSATION",workload,key,"completed",flush=True)
        for key,value in conv.items():
            if key.startswith("R0_"): continue
            mode=key.rsplit("_",1)[1]
            mismatch=compare_conversations(conv["R0_"+mode],value,workload+":"+key)
            if mismatch: records["errors"].append(mismatch)
        for value in conv.values():
            for turn in value["turns"]: turn.pop("_vectors")
        records["conversations"][workload]=conv
    records["status"]="FAIL" if records["errors"] else "PASS"
    return records

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    p.add_argument("--frozen", type=Path, default=FROZEN)
    p.add_argument("--checkpoint", type=Path, help="Alias for --model, retained for runner manifests")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--context-limit", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--second-max-new-tokens", type=int, default=64)
    p.add_argument("--workload", choices=("historical", "second", "all"), default="all")
    p.add_argument("--prompts", nargs="*", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint is not None:
        args.model = args.checkpoint
    if args.context_limit < 2 or args.max_new_tokens < 0:
        raise SystemExit("context-limit must be >=2 and max-new-tokens must be >=0")
    result = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing historical evidence.
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
