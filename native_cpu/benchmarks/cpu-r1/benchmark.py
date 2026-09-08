"""Sequential paired CPU-R1 benchmarks. Profiling is separate; no hardware-traffic claims."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, platform, re, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from native_cpu.tools.chat import ChatSession, LocalTokenizer, DEFAULT_CHAT_SYSTEM
from native_cpu.tools.native import NativeRuntime
HISTORICAL_PROMPTS=["Can you help me with programming?","Can you help me with programming?","talk me about how study programming","c#","I like c# languaje","talk me more about languajes please"]
SECOND_PROMPTS=["Name one benefit of tests.","Define latency briefly."]
CONVERSATIONS={"historical":HISTORICAL_PROMPTS,"second":SECOND_PROMPTS}
PROFILES={n:dict(selective_logits=s,reuse_kv=k,threads=1,cpus=[0],row_weights=[1],v_blocked_attention=False) for n,s,k in [("R0",False,False),("R1",True,False),("R2",False,True),("R3",True,True)]}
PROFILES.update(S0=dict(PROFILES["R3"],cpus=None),S1=dict(PROFILES["R3"]),P2=dict(PROFILES["R3"],threads=2,cpus=[0,6],row_weights=[1332,992]),P4=dict(PROFILES["R3"],threads=4,cpus=[0,2,4,6],row_weights=[1]*4),V0=dict(PROFILES["R3"]),V1=dict(PROFILES["R3"],v_blocked_attention=True))
ATTRIBUTION_PAIRS=[("R0","R1"),("R0","R2"),("R0","R3")]
SCALING_PAIRS=[("S1","S0"),("S1","P2"),("S1","P4")]
TIMES=("prefill_seconds","native_generation_seconds","sampling_seconds","prompt_prepare_seconds","frontend_other_seconds","generation_seconds","native_seconds","total_seconds")
COUNTS=("prompt_tokens_total","prefix_tokens_reused","prefill_tokens_evaluated","decode_tokens_evaluated","generated_tokens")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def vector_sha(logits): return hashlib.sha256(np.asarray(logits,dtype="<f4").tobytes()).hexdigest()

def write_json(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    os.replace(tmp,path)

def environment():
    data={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    try:
        import psutil
        data["system_cpu_percent"]=psutil.cpu_percent(interval=.2)
    except ImportError: data["system_cpu_percent"]=None
    if os.name=="nt":
        class Power(ctypes.Structure):
            _fields_=[("ac",ctypes.c_ubyte),("flags",ctypes.c_ubyte),("battery",ctypes.c_ubyte),("status",ctypes.c_ubyte),("remaining",ctypes.c_ulong),("full",ctypes.c_ulong)]
        p=Power()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p)): data["power"]={"ac_line":p.ac,"battery_percent":p.battery}
        plan=subprocess.check_output(["powercfg","/getactivescheme"],text=True,encoding="utf8",errors="replace")
        data["power_scheme_guid"]=re.search(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",plan).group(0)
    return data

def summarize(samples):
    v=list(samples)
    return dict(count=len(v),median=statistics.median(v) if v else None,min=min(v) if v else None,max=max(v) if v else None,stdev=statistics.stdev(v) if len(v)>1 else 0.)

def ordered_pair(a,b,pair): return [a,b] if pair%2 else [b,a]

def forced_ids(length,continuation,vocab):
    return np.asarray([(i*17+11)%vocab for i in range(length)],dtype=np.int32),np.asarray([(i*31+7)%vocab for i in range(continuation)],dtype=np.int32)

def warmup(rt,reps):
    prefix,suffix=forced_ids(64,8,rt.vocab_size)
    for _ in range(reps):
        rt.reset();rt.eval(prefix)
        for t in suffix: rt.eval(np.asarray([t],dtype=np.int32))
    rt.reset();rt.reset_stats()

def effective(rt,cfg,diagnostics,session=None):
    actual=dict(threads=rt.threads,cpus=rt.cpus,row_weights=rt.row_weights,selective_logits=rt.selective_logits,reuse_kv=session.reuse_kv if session else False,v_blocked_attention=getattr(rt,"v_blocked_attention",False),diagnostics=diagnostics)
    assert actual["threads"]==cfg["threads"] and actual["row_weights"]==cfg["row_weights"]
    assert actual["cpus"]==(cfg["cpus"] if cfg["cpus"] is not None else [-1]*cfg["threads"])
    assert actual["selective_logits"]==cfg["selective_logits"]
    assert actual["v_blocked_attention"]==cfg.get("v_blocked_attention",False)
    if session: assert actual["reuse_kv"]==cfg["reuse_kv"]
    return actual

def run_forced(rt,length,continuation,diagnostics):
    prefix,suffix=forced_ids(length,continuation,rt.vocab_size)
    rt.reset();rt.reset_stats();before=rt.stats
    start=time.perf_counter();logits=rt.eval(prefix);prefill=time.perf_counter()-start
    prefill_hash=vector_sha(logits);middle=rt.stats;hashes=[];durations=[]
    start_loop=time.perf_counter()
    for token in suffix:
        ids=np.asarray([token],dtype=np.int32)
        start=time.perf_counter();logits=rt.eval(ids);elapsed=time.perf_counter()-start
        durations.append(elapsed);hashes.append(vector_sha(logits))
    loop_seconds=time.perf_counter()-start_loop;after=rt.stats
    assert rt.position==length+continuation
    if not diagnostics: assert after["qkv_ns"]==0 and after["controller_wait_ns"]==0
    return dict(prefix_token_ids=prefix.tolist(),continuation_token_ids=suffix.tolist(),prefill_logits_sha256=prefill_hash,decode_logits_sha256=hashes,position_after=rt.position,prefill_tokens_evaluated=length,decode_tokens_evaluated=continuation,prefill_seconds=prefill,native_generation_seconds=sum(durations),decode_call_seconds=durations,decode_loop_with_hashing_seconds=loop_seconds,native_seconds=prefill+sum(durations),total_seconds=prefill+sum(durations),prefill_tokens_per_second=length/prefill,decode_tokens_per_second=continuation/sum(durations),lm_head_calls=dict(prefill=middle["lm_head_calls"]-before["lm_head_calls"],decode=after["lm_head_calls"]-middle["lm_head_calls"]),native_phase_stats=dict(prefill=ChatSession._stats_delta(middle,before),decode=ChatSession._stats_delta(after,middle)))

def run_conversation(rt,tokenizer,cfg,prompts,maximum):
    session=ChatSession(rt,tokenizer,context_limit=2048,max_new_tokens=maximum,temperature=.9,top_k=50,top_p=.85,seed=0,system=DEFAULT_CHAT_SYSTEM,reuse_kv=cfg["reuse_kv"])
    turns=[];started=time.perf_counter()
    for prompt in prompts:
        result=session.turn(prompt);row=dict(result.metrics)
        if "prompt_token_ids" not in row: raise RuntimeError("actual canonical prompt IDs required")
        row.update(user=prompt,text=result.text,sampled_ids=result.sampled_ids,finish_reason="eos" if result.sampled_ids and result.sampled_ids[-1]==tokenizer.eos_token_id else "length",generation_seconds=result.decode_seconds)
        assert rt.position==len(session.cached_token_ids)
        assert row["prefix_tokens_reused"]+row["prefill_tokens_evaluated"]==row["prompt_tokens_total"]==len(row["prompt_token_ids"])
        row["native_seconds"]=row["prefill_seconds"]+row["native_generation_seconds"]
        row["frontend_other_seconds"]=row["total_seconds"]-row["prompt_prepare_seconds"]-row["prefill_seconds"]-row["native_generation_seconds"]-row["sampling_seconds"]
        turns.append(row)
    totals={k:sum(t[k] for t in turns) for k in TIMES+COUNTS}
    totals["conversation_seconds"]=time.perf_counter()-started
    totals["lm_head_calls"]={p:sum(t["lm_head_calls"][p] for t in turns) for p in ("prefill","decode")}
    totals["decode_tokens_per_second"]=totals["decode_tokens_evaluated"]/totals["native_generation_seconds"] if totals["decode_tokens_evaluated"] else None
    totals["turns"]=turns
    return totals,session

def compare_entries(left,right,workload):
    if workload in CONVERSATIONS:
        a,b=left.get("turns",[]),right.get("turns",[])
        if len(a)!=len(b) or not a: return dict(parity=False,reason="turn count")
        for i,(x,y) in enumerate(zip(a,b)):
            for k in ("prompt_token_ids","sampled_ids","text","finish_reason","generated_tokens"):
                if k not in x or k not in y or x[k]!=y[k]: return dict(parity=False,turn=i,field=k,left=x.get(k),right=y.get(k))
    else:
        for k in ("prefix_token_ids","continuation_token_ids","prefill_logits_sha256","decode_logits_sha256","position_after"):
            if k not in left or k not in right or left[k]!=right[k]: return dict(parity=False,field=k,left=left.get(k),right=right.get(k))
    return {"parity":True}

def make_cases(args):
    cases=[]
    if args.suite in ("conversation","all"):
        for family,pairs in (("attribution",ATTRIBUTION_PAIRS),("scaling",SCALING_PAIRS)):
            for a,b in pairs:
                for workload in CONVERSATIONS: cases.append(dict(key=f"{family}-{a}-{b}-{workload}",family=family,left=a,right=b,workload=workload))
    if args.suite in ("forced","all"):
        for a,b in SCALING_PAIRS:
            for n in args.prefix_lengths: cases.append(dict(key=f"forced-{a}-{b}-{n}",family="forced",left=a,right=b,workload="forced",length=n))
    if args.suite in ("diagnostics","all"):
        for p in ("S1","P2","P4"): cases.append(dict(key=f"diagnostics-{p}",family="diagnostics",left=p+"-off",right=p+"-on",profile=p,workload="diagnostics",length=args.diagnostic_context))
    if args.suite == "cpu-r2":
        for n in (256,1792):
            cases.append(dict(key=f"cpu-r2-speed-V0-V1-{n}",family="cpu-r2-speed",left="V0",right="V1",workload="forced",length=n))
            cases.append(dict(key=f"cpu-r2-diagnostics-V0-V1-{n}",family="cpu-r2-diagnostics",left="V0",right="V1",workload="forced",length=n,diagnostics=True))
    if args.subset:
        requested=args.subset.split(",")
        cases=[c for c in cases if any(x in (c["key"],c["family"],c["workload"]) for x in requested)]
    if not cases: raise ValueError("selection has no cases")
    return cases

def run_entry(args,case,active,tokenizer):
    diagnostics=bool(case.get("diagnostics")) or (case["workload"]=="diagnostics" and active.endswith("-on"))
    cfg=PROFILES[case.get("profile",active)];before=environment()
    with NativeRuntime(args.model,args.library,2048,threads=cfg["threads"],cpus=cfg["cpus"],row_weights=cfg["row_weights"]) as rt:
        rt.configure_selective_logits(cfg["selective_logits"]);rt.configure_v_blocked_attention(cfg.get("v_blocked_attention",False));rt.configure_profile(diagnostics);warmup(rt,args.warmups)
        if case["workload"] in CONVERSATIONS:
            maximum=args.max_new_tokens if case["workload"]=="historical" else args.second_max_new_tokens
            data,session=run_conversation(rt,tokenizer,cfg,CONVERSATIONS[case["workload"]],maximum)
            actual=effective(rt,cfg,diagnostics,session)
            assert data["turns"][-1]["native"]["qkv_ns"]==0
        else:
            data=run_forced(rt,case["length"],args.continuation,diagnostics);actual=effective(rt,cfg,diagnostics)
    data.update(profile=active,effective=actual,environment_before=before,environment_after=environment())
    return data

def aggregate(case,pairs):
    a,b=case["left"],case["right"];keys=list(TIMES)+list(COUNTS)+["conversation_seconds","decode_tokens_per_second","prefill_tokens_per_second"]
    result=dict(case=case,pairs=len(pairs),parity=all(p["comparison_result"]["parity"] for p in pairs),profiles={},paired_differences=[],order=[p["order"] for p in pairs])
    for name in (a,b):
        rows=[next(e for e in p["entries"] if e["profile"]==name) for p in pairs]
        result["profiles"][name]={k:summarize(r[k] for r in rows) for k in keys if all(isinstance(r.get(k),(int,float)) for r in rows)}
    for p in pairs:
        left=next(e for e in p["entries"] if e["profile"]==a);right=next(e for e in p["entries"] if e["profile"]==b)
        result["paired_differences"].append(dict(pair=p["pair"],right_minus_left={k:right[k]-left[k] for k in keys if isinstance(left.get(k),(int,float)) and isinstance(right.get(k),(int,float))},total_time_reduction=1-right["total_seconds"]/left["total_seconds"]))
    return result

def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",type=Path,default=ROOT/"native_cpu/artifacts/minimind-fp32.bin");p.add_argument("--library",type=Path,default=ROOT/"native_cpu/build/Release/minimind_cpu.dll");p.add_argument("--tokenizer",type=Path,default=ROOT/"checkpoints/minimind-3-hf")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--suite",choices=("conversation","forced","diagnostics","all","cpu-r2"),default="all")
    p.add_argument("--subset",help="comma-separated exact case, family or workload names")
    p.add_argument("--pairs",type=int,default=6);p.add_argument("--warmups",type=int,default=1);p.add_argument("--max-new-tokens",type=int,default=256);p.add_argument("--second-max-new-tokens",type=int,default=64)
    p.add_argument("--prefix-lengths",default="64,256,1024,1792");p.add_argument("--diagnostic-context",type=int,default=256);p.add_argument("--continuation",type=int,default=32);p.add_argument("--resume",action="store_true")
    return p

def identity(args):
    import subprocess
    return dict(schema="cpu-r1/v3",source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),model_sha256=sha(args.model),library_sha256=sha(args.library),tokenizer_sha256={n:sha(args.tokenizer/n) for n in ("tokenizer.json","tokenizer_config.json","chat_template.jinja")},sources_sha256={n:sha(ROOT/n) for n in ("native_cpu/tools/chat.py","native_cpu/tools/native.py","native_cpu/tools/compare_chat.py","native_cpu/benchmarks/cpu-r1/benchmark.py")},python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),config=dict(profiles=PROFILES,conversations=CONVERSATIONS,context=2048,pairs=args.pairs,warmup_prefix=64,warmup_decode=8,warmups=args.warmups,maximum=args.max_new_tokens,second_maximum=args.second_max_new_tokens,prefix_lengths=args.prefix_lengths,diagnostic_context=args.diagnostic_context,continuation=args.continuation,temperature=.9,top_k=50,top_p=.85,seed=0,system=DEFAULT_CHAT_SYSTEM))

def resume_pair(path,meta,case,pair):
    prior=json.loads(path.read_text(encoding="utf8"))
    if prior["identity"]!=meta or prior["case"]!=case or prior["pair"]!=pair: raise ValueError(f"resume identity mismatch: {path}")
    if prior["status"]!="PASS": raise ValueError(f"cannot resume an invalid pair: {path}")
    if prior["order"]!=ordered_pair(case["left"],case["right"],pair): raise ValueError("resume order mismatch")
    return prior

def main(argv=None):
    if not __debug__: raise RuntimeError("run without -O: benchmark invariants must remain enabled")
    args=build_parser().parse_args(argv);args.prefix_lengths=[int(x) for x in args.prefix_lengths.split(",")]
    if args.pairs<1 or args.warmups<1 or args.continuation<1 or min(args.max_new_tokens,args.second_max_new_tokens)<0: raise ValueError("positive pairs/warmups/continuation required")
    if any(n<1 or n+args.continuation>2048 for n in args.prefix_lengths+[args.diagnostic_context]): raise ValueError("prefix+continuation must fit 2048")
    args.out.mkdir(parents=True,exist_ok=True);meta=identity(args);tokenizer=LocalTokenizer(args.tokenizer);summaries=[]
    for case in make_cases(args):
        pairs=[]
        for number in range(1,args.pairs+1):
            path=args.out/f"{case['key']}-pair{number:02d}.json"
            if path.exists():
                if not args.resume: raise FileExistsError(path)
                pairs.append(resume_pair(path,meta,case,number));continue
            order=ordered_pair(case["left"],case["right"],number)
            print(f"PAIR {case['key']} {number}/{args.pairs} {order}",flush=True)
            entries=[run_entry(args,case,name,tokenizer) for name in order]
            comparison=compare_entries(entries[0],entries[1],case["workload"])
            record=dict(identity=meta,case=case,pair=number,order=order,entries=entries,comparison_result=comparison,status="PASS" if comparison["parity"] else "FAIL")
            write_json(path,record);pairs.append(record)
            if not comparison["parity"]: raise RuntimeError(f"parity failed; preserved {path}")
        summary=aggregate(case,pairs);summaries.append(summary);write_json(args.out/f"{case['key']}-summary.json",summary)
    write_json(args.out/"summary.json",dict(identity=meta,cases=summaries,status="PASS",final_protocol=args.pairs>=6 and args.pairs%2==0,limits=["Native-call timing includes Python/ctypes boundary.","Worker times overlap; do not sum as elapsed time.","No DRAM traffic, temperature or energy counters; unrelated processes not terminated.","Free generation has fixed policy; forced decode has identical IDs per profile.","Profiling-on runs are separate diagnostics, not final speed measurements."]))
    return 0
if __name__=="__main__": raise SystemExit(main())
