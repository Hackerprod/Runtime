"""CPU-E4.1: long-context no-match lookup guard."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL = ROOT / "native_cpu" / "artifacts" / "minimind-fp32.bin"
TOKENIZER = ROOT / "checkpoints" / "minimind-3-hf"
BASELINE = ROOT / "native_cpu" / "benchmarks" / "cpu-e2" / "baseline" / "minimind_cpu.dll"
CANDIDATE = ROOT / "native_cpu" / "benchmarks" / "cpu-e4" / "baseline" / "minimind_cpu.dll"
EXPECTED_BASELINE_SHA256 = "3ff9418181d29ad9f7761e2a48866f06759027b4599ef1d024f24ac4cd3acfd5"
EXPECTED_CANDIDATE_SHA256 = "b410e642e2249a8982a9a83b134f3d7f5e1f37c677f11d10900ed0e213cb97ab"
PREFIX_LENGTH = 1792
MAX_NEW_TOKENS = 32
PAIR_COUNT = 3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_sha(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def no_match_prefix(tokenizer) -> np.ndarray:
    # Use a natural long context so the target produces a full 32-token sample,
    # then finish with a unique paragraph.  The listed follower substitutions
    # poison the deterministic copied continuations: lookup may find a 12-gram,
    # but no proposed block is useful and therefore no verification is called.
    body = " ".join(
        f"The system receives sequence number {index}; this sentence provides context for measuring CPU execution and token lookup marker {index}."
        for index in range(500)
    )
    suffix = "Now consider this fresh subject: explain how unit tests improve software reliability in practice."
    body_ids = tokenizer._tokenizer.encode(body, add_special_tokens=False).ids
    suffix_ids = tokenizer._tokenizer.encode(suffix, add_special_tokens=False).ids
    values = body_ids[:PREFIX_LENGTH - len(suffix_ids)] + suffix_ids
    if len(values) != PREFIX_LENGTH:
        raise RuntimeError(f"unexpected no-match prefix length: {len(values)}")
    key = values[-12:]
    if any(values[index:index + 12] == key for index in range(PREFIX_LENGTH - 12)):
        raise RuntimeError("no-match prefix unexpectedly contains the final 12-gram")
    poisoned_followers = {
        1365: 4312, 1369: 391, 1509: 3475, 1540: 2766, 1570: 4744,
        1600: 331, 1603: 3475, 1631: 115, 1634: 2765, 1662: 370,
        1665: 4742, 1691: 6082, 1725: 383, 1728: 328, 1756: 2653,
        1759: 111, 1763: 3475,
    }
    for index, value in poisoned_followers.items():
        values[index] = value
    return np.asarray(values, dtype=np.int32)


def run_entry(library: Path, label: str):
    from native_cpu.tools.chat import ChatSession, LocalTokenizer
    from native_cpu.tools.native import NativeRuntime

    tokenizer = LocalTokenizer(TOKENIZER)
    prefix = no_match_prefix(tokenizer)
    with NativeRuntime(MODEL, library, 2048, threads=1, cpus=[0], row_weights=[1]) as runtime:
        runtime.configure_profile(False)
        runtime.reset()
        warmup = np.asarray([(index * 31 + 7) % runtime.vocab_size for index in range(64)], dtype=np.int32)
        runtime.eval(warmup)
        runtime.reset()
        prepare_started = time.perf_counter()
        initial_logits = runtime.eval(prefix)
        prepare_seconds = time.perf_counter() - prepare_started

        session = ChatSession(runtime, tokenizer, context_limit=2048,
                              max_new_tokens=MAX_NEW_TOKENS, temperature=0.9,
                              top_k=50, top_p=0.85, seed=0, reuse_kv=True)
        session.cached_token_ids = prefix.tolist()
        session._cache_identity = session._identity()
        session._cache_trusted = True

        decode_started = time.perf_counter()
        generated, sampled_ids, decode_evaluated, native_seconds, sampling_seconds, speculative = session._decode_tokens(initial_logits, tokenizer.eos_token_id)
        decode_seconds = time.perf_counter() - decode_started
        position_after = runtime.position

        # Probe one continuation token after the timed generation. This gives
        # an exact logits/state comparison without consuming the sampler RNG.
        probe_id = np.asarray([1234], dtype=np.int32)
        continuation_logits = session._eval_and_commit(probe_id)
        continuation_position = runtime.position

        if len(sampled_ids) != MAX_NEW_TOKENS or position_after != PREFIX_LENGTH + MAX_NEW_TOKENS - 1:
            raise RuntimeError(f"unexpected generation boundary for {label}: tokens={len(sampled_ids)} position={position_after}")
        return {
            "profile": label,
            "library_sha256": sha(library),
            "prefix_length": PREFIX_LENGTH,
            "prefix_token_ids_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
            "initial_logits_sha256": vector_sha(initial_logits),
            "prepare_seconds": prepare_seconds,
            "decode_seconds": decode_seconds,
            "native_generation_seconds": native_seconds,
            "sampling_seconds": sampling_seconds,
            "decode_tokens_evaluated": decode_evaluated,
            "sampled_ids": sampled_ids,
            "generated_ids": generated,
            "position_after": position_after,
            "continuation_probe_id": int(probe_id[0]),
            "continuation_logits_sha256": vector_sha(continuation_logits),
            "continuation_position": continuation_position,
            "speculative_decode": speculative,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (MODEL, TOKENIZER / "tokenizer.json", BASELINE, CANDIDATE):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline_sha = sha(BASELINE)
    candidate_sha = sha(CANDIDATE)
    if baseline_sha != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(f"CPU-E4.1 refuses CPU-E2 baseline: expected {EXPECTED_BASELINE_SHA256}, found {baseline_sha}")
    if candidate_sha != EXPECTED_CANDIDATE_SHA256:
        raise RuntimeError(f"CPU-E4.1 refuses CPU-E4 candidate: expected {EXPECTED_CANDIDATE_SHA256}, found {candidate_sha}")
    identity = {
        "schema": "cpu-e4.1/no-match-v1",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_sha256": sha(MODEL),
        "baseline_library_sha256": baseline_sha,
        "candidate_library_sha256": candidate_sha,
        "prefix_length": PREFIX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.9, "top_k": 50, "top_p": 0.85, "seed": 0,
        "threads": 1, "cpus": [0], "row_weights": [1], "profiler": False,
        "workload": "1792-token prefix with no useful 12-to-4 speculative hit",
        "timed_operation": "decode including lookup and normal steps; context preparation excluded",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for number in range(1, PAIR_COUNT + 1):
        order = ("A", "B") if number % 2 else ("B", "A")
        print(f"PAIR no-match-1792 {number}/{PAIR_COUNT} {order}", flush=True)
        entries = {}
        for label in order:
            entries[label] = run_entry(BASELINE if label == "A" else CANDIDATE, label)
        left, right = entries["A"], entries["B"]
        comparison = {
            "sampled_ids_parity": left["sampled_ids"] == right["sampled_ids"],
            "generated_ids_parity": left["generated_ids"] == right["generated_ids"],
            "initial_logits_parity": left["initial_logits_sha256"] == right["initial_logits_sha256"],
            "continuation_logits_parity": left["continuation_logits_sha256"] == right["continuation_logits_sha256"],
            "position_parity": left["position_after"] == right["position_after"],
            "continuation_position_parity": left["continuation_position"] == right["continuation_position"],
            "candidate_verify_calls_zero": right["speculative_decode"]["verify_calls"] == 0,
            "candidate_accepted_tokens_zero": right["speculative_decode"]["accepted_tokens"] == 0,
        }
        comparison["parity"] = all(comparison.values())
        if not comparison["parity"]:
            raise RuntimeError(f"CPU-E4.1 parity failed on pair {number}: {comparison}")
        paired = {
            "pair": number,
            "decode_time_reduction": 1.0 - right["decode_seconds"] / left["decode_seconds"],
            "decode_tok_s_change": (1.0 / right["decode_seconds"]) / (1.0 / left["decode_seconds"]) - 1.0,
            "lookup_seconds": right["speculative_decode"]["lookup_seconds"],
        }
        record = {"identity": identity, "pair": number, "order": list(order),
                  "entries": entries, "comparison_result": comparison, "paired": paired,
                  "status": "PASS"}
        write_json(args.out / f"cpu-e4.1-no-match-pair{number:02d}.json", record)
        pairs.append(record)
    reductions = [p["paired"]["decode_time_reduction"] for p in pairs]
    summary = {
        "identity": identity,
        "pairs": PAIR_COUNT,
        "parity": True,
        "profiles": {
            label: {
                "decode_seconds_median": statistics.median(p["entries"][label]["decode_seconds"] for p in pairs),
                "prepare_seconds_median": statistics.median(p["entries"][label]["prepare_seconds"] for p in pairs),
                "native_generation_seconds_median": statistics.median(p["entries"][label]["native_generation_seconds"] for p in pairs),
            } for label in ("A", "B")
        },
        "paired_decode_time_reductions": reductions,
        "median_decode_time_reduction": statistics.median(reductions),
        "favorable_pairs": sum(value > 0 for value in reductions),
        "candidate_verify_calls": [p["entries"]["B"]["speculative_decode"]["verify_calls"] for p in pairs],
        "candidate_lookup_seconds": [p["entries"]["B"]["speculative_decode"]["lookup_seconds"] for p in pairs],
        "status": "PASS",
    }
    write_json(args.out / "cpu-e4.1-no-match-summary.json", summary)
    write_json(args.out / "summary.json", {"identity": identity, "summary": summary, "status": "PASS"})


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
