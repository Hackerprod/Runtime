"""Bounded CPU-R1 frontend smoke checks; execution is explicit and opt-in."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
LAUNCHER = ROOT / "compare_cpu" / "Native_CPP.cmd"

def run(command, *, stdin=None):
    env = os.environ.copy(); env.update(MINIMIND_NO_PAUSE="1", MINIMIND_PYTHON=sys.executable, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    p = subprocess.run(command, cwd=ROOT, input=stdin, text=True, capture_output=True, encoding="utf-8", errors="replace", env=env)
    return {"command": [str(x) for x in command], "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def json_lines(stdout):
    return [json.loads(line) for line in stdout.splitlines() if line.lstrip().startswith("{")]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=8); args = p.parse_args(argv)
    records = {"checks": [], "errors": []}
    from native_cpu.tools import compare_chat
    records["parser_defaults"] = {"selective_logits": compare_chat.make_parser().parse_args([]).selective_logits,
                                   "reuse_kv": compare_chat.make_parser().parse_args([]).reuse_kv}
    if records["parser_defaults"] != {"selective_logits": False, "reuse_kv": False}: records["errors"].append("flags not off by default")
    reject = run([sys.executable,"-m","native_cpu.tools.compare_chat","--backend","original","--reuse-kv"])
    records["original_reject"] = reject
    if reject["returncode"] != 1 or "--reuse-kv is supported only by the native backend" not in reject["stderr"]: records["errors"].append("original backend did not explicitly reject --reuse-kv")
    if not LAUNCHER.is_file(): records["errors"].append(f"missing launcher: {LAUNCHER}")
    else:
        base = ["cmd", "/c", str(LAUNCHER.relative_to(ROOT)), "--prompt", "Can you help me with programming?", "--max-new-tokens", str(args.max_new_tokens), "--metrics-json"]
        for label, extra in (("default", []), ("optin", ["--selective-logits", "--reuse-kv"])):
            result = run(base + extra); result["metrics"] = json_lines(result["stdout"]); records["checks"].append({"label": label, **result})
            if result["returncode"] != 0 or not result["metrics"]: records["errors"].append(f"{label} smoke failed")
        if all(len(c["metrics"]) == 1 for c in records["checks"]):
            a, b = records["checks"][0]["metrics"][0], records["checks"][1]["metrics"][0]
            if (a.get("text"), a.get("sampled_ids")) != (b.get("text"), b.get("sampled_ids")): records["errors"].append("default/opt-in generation differs")
        # Interactive clear must restore the seeded generation policy.
        result = run(["cmd", "/c", str(LAUNCHER.relative_to(ROOT)), "--max-new-tokens", str(args.max_new_tokens), "--metrics-json", "--selective-logits", "--reuse-kv"], stdin="hello\n/clear\nhello\n/quit\n")
        result["metrics"] = json_lines(result["stdout"]); records["checks"].append({"label":"clear-repeat", **result})
        if result["returncode"] != 0 or len(result["metrics"]) != 2: records["errors"].append("/clear smoke did not produce exactly two responses")
        elif result["metrics"][0].get("sampled_ids") != result["metrics"][1].get("sampled_ids"): records["errors"].append("/clear did not restore seeded generation")
    guard_code = """import builtins, sys
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch imported')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
from native_cpu.tools.compare_chat import main
rc = main(['--backend','native','--prompt','hello','--max-new-tokens','1','--metrics-json','--selective-logits','--reuse-kv'])
assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
raise SystemExit(rc)
"""
    guard = run([sys.executable, "-c", guard_code])
    records["checks"].append({"label":"no-torch-import", **guard})
    if guard["returncode"] != 0: records["errors"].append("compare frontend imported torch")
    # Legacy frozen DLLs must either expose truncate or safely fall back to a
    # full reset; they must never claim a reused prefix.
    legacy = {"status": "FAIL"}
    try:
        from native_cpu.tools.native import NativeRuntime
        from native_cpu.tools.chat import ChatSession, LocalTokenizer
        model = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
        frozen = ROOT / "native_cpu" / "validation" / "cpu-r1" / "frozen" / "minimind_cpu.dll"
        tokroot = ROOT / "checkpoints" / "minimind-3-hf"
        if not model.is_file() or not frozen.is_file(): raise FileNotFoundError("legacy validation requires frozen DLL and model")
        traces = {}
        for reuse in (False,True):
            with NativeRuntime(model, frozen, 2048) as runtime:
                assert not runtime.supports_truncate
                session = ChatSession(runtime, LocalTokenizer(tokroot), max_new_tokens=1, context_limit=2048, reuse_kv=reuse, temperature=0)
                turns = [session.turn(prompt) for prompt in ("hello","world")]
                traces[str(reuse)] = [{"text":r.text,"sampled_ids":r.sampled_ids,"metrics":r.metrics} for r in turns]
                assert turns[1].prefix_tokens_reused == 0
                if reuse: assert turns[1].cache_action["reason"] == "truncate_unsupported"
        assert [(r["text"],r["sampled_ids"]) for r in traces["False"]] == [(r["text"],r["sampled_ids"]) for r in traces["True"]]
        legacy = {"status":"PASS", "traces":traces}
    except Exception as exc:
        legacy = {"status":"ERROR", "error":str(exc)}
        records["errors"].append("legacy fallback validation failed: " + str(exc))
    records["checks"].append({"label":"legacy-dll-fallback", **legacy})
    records["status"] = "FAIL" if records["errors"] else "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream: json.dump(records, stream, ensure_ascii=False, indent=2); stream.write("\n")
    return 1 if records["errors"] else 0

if __name__ == "__main__": raise SystemExit(main())
