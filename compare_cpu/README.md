# CPU backend comparison

`Native_CPP.cmd` and `Original_PyTorch.cmd` launch the same local comparison
frontend with different inference backends. They are safe to double-click from
any working directory, including paths containing spaces.

Open this folder in File Explorer and double-click the desired launcher. No
installation, conversion or download is needed on this laptop. Wait for the `>`
prompt after the loading message; PyTorch initialization can take several seconds.

The serial launchers for both backends use the same prompt, local
tokenizer/template, and `chat` profile by default: the official demo's MiniMind
system message, temperature 0.9, top-k 50, top-p 0.85, seed 0, context 2048,
256 new tokens, and one CPU thread.
Use `/clear` between interactive cases to reset both conversation and random
generator before repeating the same prompt on the other backend.
Run only one generation at a time when comparing speed, to avoid CPU contention.

Examples from PowerShell opened in this folder:

```powershell
.\Native_CPP.cmd --prompt "Explain a CPU cache." --metrics-json
.\Original_PyTorch.cmd --prompt "Explain a CPU cache." --metrics-json
.\Native_CPP_T0.cmd --prompt "Explain a CPU cache." --metrics-json
```

`Native_CPP.cmd` runs the consolidated CPU-U1 production route with one native
participant by default. `Native_CPP_T0.cmd` delegates to it with the selected laptop profile: two total native participants (caller plus
one persistent worker), logical CPUs `0,6` in Windows group 0, and contiguous
row-shard weights `1332,992`. This is specific to the Ryzen AI 5 330 test laptop,
where logical CPU 0 is the fast physical-core anchor and logical CPU 6 is the
compact-core participant. FP32 kernels and weights are identical between
launchers. The caller affinity is scoped/restored per `mm_eval`; the worker
spins during evaluation and parks between requests. Energy behavior was not
measured, so this is not a universal faster-setting claim.

The equivalent explicit options are:

```powershell
python -m native_cpu.tools.compare_chat --backend native --threads 2 --cpus 0,6 --row-weights 1332,992
```

`--threads` counts the caller. `--cpus` names Windows group-0 logical CPU IDs
in participant order, and `--row-weights` controls contiguous row ownership.
If overriding the T0 launcher's thread count, provide matching CPU and weight
lists; use `Native_CPP.cmd --threads N` for an unbound, equally weighted team. The original
PyTorch launcher rejects non-default native-only options. The historical
optimization switches are no longer exposed by the current executable.

For the original numerical-comparison policy, explicitly add `--profile parity`.
It uses no system message and greedy decoding (temperature 0, top-k 0, top-p 1).
That is a diagnostic profile, **not the default chat configuration**. You can
override `--system`, `--temperature`, `--top-k`, `--top-p`, and `--seed` explicitly.

Without `--prompt`, each launcher opens the interactive chat. Commands are
`/clear`, `/exit`, and `/quit`. `MINIMIND_NO_PAUSE=1` suppresses the double-click pause for automation. Set
`MINIMIND_PYTHON` to choose another Python executable; otherwise `C:\Python314\python.exe` is
preferred, then `python` on PATH.

The native launcher uses the verified MMCPU binary and library defaults from
`native_cpu.tools.compare_chat`. The original launcher uses the unchanged
PyTorch MiniMind reference checkpoint. No network fallback is enabled. This is
a matched FP32 comparison: the original backend calls the unchanged MiniMind
model, but shares our tokenizer/session/sampler. It is **not** the actual
Transformers `generate()` or Streamlit execution path. Use the separate official
demo below to test that path directly.

Responses are printed complete, not streamed. The summary reports model prefill,
decode time and decode tokens/second separately from loading. `finish=length`
means the configured output limit was reached; increase `--max-new-tokens` if
needed while keeping the same value for both backends. `finish=eos` means the
model ended its answer. The original backend uses `flash_attn=False` to match
the validated reference settings; it is not a claim about the fastest possible
PyTorch configuration.

## Canonical output-throughput guard

`Canonical_TokS.cmd` runs three fresh CPU-E6 processes with the fixed prompt
`Could you tell me about programming languages?` and the default chat sampling
policy. It verifies the published CPU-E6 DLL SHA-256 before running and stores
raw JSON and stderr receipts in `canonical-results/cpu-e6/`. The displayed `output
tok/s` counts generated output tokens; `legacy target tok/s` is the secondary
native-evaluation rate retained for diagnosing older reports.

## Run the unmodified official web demo

**This local launcher is not a replica of the public deployment.** On September
7, 2026, a fresh session of the [public ModelScope demo](https://www.modelscope.cn/studios/gongjy/MiniMind)
selected `minimind-3-moe` by default (198M total, about 64M active parameters).
Our local checkpoint is the **dense** `minimind-3` (about 64M total). Select the
same model before comparing responses. The hosted dense weight SHA and all six
config/tokenizer files match our local download; the model selection, not a
different dense download, is the verified difference. The public app also omits
the `.half()` call present in the GitHub demo. See
`native_cpu/validation/hosted-demo-audit.json` for the observed deployment details.

Double-click `Official_Web_Demo.cmd`, wait for startup, then open
[MiniMind on localhost](http://127.0.0.1:8501). Keep its terminal open; press Ctrl+C there to stop it.
The launcher uses the isolated `.venv-web-demo` environment and runs the original
`scripts/web_demo.py` from its required `scripts` working directory. The local
`scripts/minimind-3` junction points to the existing verified checkpoint; it does
not duplicate or download weights. The server binds only to localhost.
Automatic source watching is disabled: Streamlit's watcher otherwise probes
unrelated Transformers modules and produces optional-vision dependency errors.
This does not change model loading or generation; restart the app after edits.

Select `minimind-3`. Leave Thinking and Tools disabled for the text-only
comparison. The upstream app defaults to **zero history rounds**; increase the
History rounds slider if you want it to remember previous turns. For a bounded
CPU test, lower Max Length from the upstream 8192 to 256.

This is an independent upstream baseline, not a speed-matched third backend:
it uses Hugging Face Qwen3, FP16/SDPA, Torch sampling and a new random seed per
turn. Our two launchers use FP32, the same NumPy sampler and fixed seed 0.
Even with the same seed, different random generators need not produce the same
text. The sampler also uses FP64 probabilities rather than HF's FP32: exact
nucleus cutoffs or ties can select different boundary candidates. Matching the
sampling policy does not mean bit-identical HF generation.
Our CLI also retains bounded history and prints complete responses rather
than streaming them.

## Why lowercase `hola` used to repeat

The previous launchers omitted the official system message and defaulted to
greedy decoding. Both our FP32 backends **and the actual official HF model**
repeated `Hola` with that bare prompt. Adding only the official system message
while keeping greedy decoding ended that loop in the controlled test. The
default chat profile now includes that message and the demo's sampling policy.
No weights, C++ math or user input spelling were changed to hide the symptom.

Evidence is in `native_cpu/validation/hola-baseline.json`,
`hola-policy-ablation.json`, and `hola-official-hf.json`. Some sampled answers
are still in English or Chinese, and some are incoherent. Matching the demo's
policy fixes our setup mismatch; it does not establish reliable Spanish quality.

The real-launcher regression in `chat-launchers-after-fix.json` checks lowercase
`hola`, case variants, `/clear`, an English follow-up and a Spanish greeting with
full transcripts. Both backends produced identical text in all seven cases;
the default lowercase greeting ended after 36 tokens. The English name question
received an irrelevant greeting and the longer Spanish greeting reached the
256-token cap. Those are remaining conversation failures, not passing quality
tests. `chat-default-no-torch.json` separately verifies the direct native CLI
works with Torch/Transformers/TensorFlow imports actively prohibited.

The independent `web-demo-apptest.json` runs the actual unmodified Streamlit
script with the real checkpoint, a harness-fixed Torch seed 0 and a 256-token
limit. Its matched English name question also received an irrelevant greeting;
the separate Spanish prompt was answered with `¿Cómo estás?`. The UI completed
without exceptions, but that is not a conversation-quality pass. Special tokens
are hidden by the upstream streamer, so this test does not claim observed EOS.

## T0 affinity comparison evidence

The final paired comparison is between a locally archived baseline C++ DLL and the new
C++ DLL with the same FP32 kernels, weights, and model—not original PyTorch
versus native. It uses the same six-turn programming transcript, complete
retained history, and chat sampling policy. Across four AB/BA pairs (eight
complete conversations), all 48 turn texts, prompt IDs, and sampled IDs
matched; each conversation performed 733 decode evaluations.

For the selected `1332,992` row weights, median prefill changed from 18.025723 s
to 16.918335 s (-6.14%), decode throughput from 84.642327 to 87.757669
tokens/s (+3.6806%), and total time from 26.694330 s to 25.299208 s (-5.2263%).
All four paired trials improved both reported measures, but the predeclared
promotion gate (at least 5% decode **and** at least 5% total-latency gain) was
not met. Keep `Native_CPP.cmd` as the default; `Native_CPP_T0.cmd` is explicit
opt-in. Earlier pre-cleanup measurements were invalidated because four old
Python test processes consumed the CPU concurrently. The final report is
[`native_cpu/benchmarks/t0-affinity-v1/summary.json`](../native_cpu/benchmarks/t0-affinity-v1/summary.json).

To reproduce against an archived baseline DLL when that local archive is available:

```powershell
.\compare_cpu\Native_CPP.cmd --library native_cpu/validation/t0-affinity-v1/baseline/minimind_cpu.dll
```
