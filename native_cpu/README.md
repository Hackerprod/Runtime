# MiniMind on CPU, with native C++ inference

This directory adds a CPU runtime while preserving the **official MiniMind model
and training code**.
The transformer, attention, KV cache, normalization, projections and logits run
in C++20. Python supplies the original Hugging Face BPE/chat template through
`tokenizers` and sandboxed Jinja2, and sampling through a small `ctypes` boundary.
The chat frontend imports neither Transformers nor PyTorch. It is not a standalone C++ tokenizer
or a C++ training framework.

The supported checkpoint is the dense **MiniMind-3, approximately 64M parameters**.
No custom MoE, model training, GPU service, or architecture change is involved.

## Run the downloaded model

For a ready-to-use interactive comparison, open
[`compare_cpu`](../compare_cpu/README.md): `Native_CPP.cmd` runs our C++ backend
and `Original_PyTorch.cmd` runs the original MiniMind model in CPU FP32. Both
use matching generation settings and show separate prefill/decode timings.
Their default `chat` profile includes the official demo's system message and
sampling (temperature 0.9, top-k 50, top-p 0.85). `--profile parity` restores the
bare greedy diagnostic policy. The separate `Official_Web_Demo.cmd` runs the
untouched upstream Streamlit/Hugging Face implementation for an independent
baseline; see the comparison guide for its different dtype, RNG and history.

Run commands from the repository root. On this laptop Python is
`C:\Python314\python.exe`; substitute that executable for `python` if necessary.

```powershell
python -m native_cpu.tools.chat --model native_cpu/artifacts/minimind-fp32.bin --tokenizer checkpoints/minimind-3-hf
```

For the laptop-specific T0-style affinity/row-sharding profile, use
`compare_cpu/Native_CPP_T0.cmd`. It delegates to the existing native launcher
with two total compute participants (the caller plus one persistent worker),
Windows group-0 logical CPUs `0,6`, and contiguous row weights `1332:992`:

```powershell
.\compare_cpu\Native_CPP_T0.cmd --prompt "Explain a CPU cache." --metrics-json
```

CPU-R3's four-row FFN experiment is retained only as historical evidence in
[`benchmarks/cpu-r3/RESULTS.md`](benchmarks/cpu-r3/RESULTS.md); it was rejected
and is not part of the production runtime.

CPU-R1, R2, R4, R5 and R6 are now consolidated as the single production route
for the supported MiniMind-3 checkpoint. The runtime selects contiguous row
sharding, blocked V accumulation, shared-K/V GQA and exact FP16-storage/FP32-
compute FFN automatically after validating the model shape, CPU instructions and
bit-for-bit weight reconstruction. There are no optimization switches in the
chat launcher; an incompatibility fails clearly instead of silently selecting a
different precision path. CPU-E1 remains an isolated kernel experiment and is
not connected to inference. The consolidation evidence is in
[`benchmarks/cpu-u1/RESULTS.md`](benchmarks/cpu-u1/RESULTS.md).

This selection is for the Ryzen AI 5 330 laptop: logical CPU 0 is the fast
physical-core anchor and logical CPU 6 is the compact-core participant. The
FP32 kernels, weights, and math are unchanged. The caller's CPU affinity is
scoped to each `mm_eval` and restored afterward; the background worker spins
during an active evaluation and parks between requests. CPU-energy tradeoffs
were not measured, and this profile is not a universal speed claim. The
one-participant consolidated CPU-U1 route remains the default comparison
launcher; T0 affinity is still explicit.

CPU-R4.1 restores the pre-CPU-R4 binary layout of `MmRuntimeStats` (1,136
bytes on x64). Its Q·K diagnostic is queried separately through the optional
`mm_get_attention_qk_ns` symbol, so old compatible DLLs report that metric as
unavailable rather than zero. The Python binding rejects the published CPU-R4
DLL with the shifted statistics layout before calling `mm_get_stats`; see
[`benchmarks/cpu-r4.1/RESULTS.md`](benchmarks/cpu-r4.1/RESULTS.md).

The historical CPU-R5 and CPU-R6 experiments remain documented for lineage. R5
shared-V reads each V block once for the two GQA heads while preserving separate
softmax distributions; see [`benchmarks/cpu-r5/RESULTS.md`](benchmarks/cpu-r5/RESULTS.md).
R6 validates every dense
`gate_proj`, `up_proj` and `down_proj` weight by an FP32→FP16→FP32 bit-for-bit
round trip, prepares the half storage once, and converts eight weights directly
to FP32 registers in the existing FMA GEMV order. Runtime F16C/AVX2/FMA support
is required; a non-reconstructible tensor, quantized FFN tensor, or missing
instruction support rejects the production load. The
FP32 vectors remain allocated for reference, so compact storage is reported as
additional runtime memory rather than a total-RAM reduction.

The bounded CPU-R6 campaign accepted this route: FFN time fell by a median
22.540%/20.860% in prefill (prefixes 256/1792) and 21.944%/18.777% in decode;
decode throughput improved 14.987% and 15.724%, with exact logits and all six
historical turns matching. These are paired measurements for X0 (CPU-R5 W1)
versus X1 (FP16 storage), not a cumulative claim over earlier experiments. Full
raw receipts, reconstruction evidence and the fail-closed report are in
[`benchmarks/cpu-r6/RESULTS.md`](benchmarks/cpu-r6/RESULTS.md).

Use `/clear` to reset the conversation and random generator, and `/exit` to quit.
A one-shot invocation:

```powershell
python -m native_cpu.tools.chat --model native_cpu/artifacts/minimind-fp32.bin --tokenizer checkpoints/minimind-3-hf --prompt "Hello! Who are you?" --max-new-tokens 64 --metrics-json
```

The consolidated route keeps FP32 arithmetic (with exact FP16 FFN storage). The experimental `minimind-q4.bin` is smaller,
but was slower than native FP32 and materially changed responses in this test.
Neither variant improves the language abilities of the pretrained model.

## Measured baseline: September 7, 2026

On this laptop (Ryzen AI 5 330, Windows, MSVC Release), using one compute thread,
three prompts of 26–35 tokens and three timed repetitions after warm-up:

| Backend | Decode, tokens/s | Prompt prefill, seconds | Weight file |
|---|---:|---:|---:|
| Original PyTorch FP32 | 39.5–44.4 | 0.110–0.123 | — |
| Native FP32 AVX2 | **112.2–114.1** | 0.210–0.285 | 255.66 MB |
| Native FP32 scalar | 26.9–28.9 | 0.915–1.208 | 255.66 MB |
| Native Q4 AVX2, experimental | 55.5–69.6 | 0.365–0.478 | 56.59 MB |

Ranges are per-prompt medians, not a universal throughput guarantee. Decode uses
the same 15 teacher-forced token evaluations for each backend; loading is excluded.
**Native prefill is still slower** because it is sequential rather than batched.

FP32 scalar and AVX2 passed all compared logits at `atol=rtol=1e-4` and exactly
matched all three 16-token greedy streams. Worst absolute error: `8.32e-5`.
Q4 decode top-1 agreement was only 75–93.75%, with KL divergence 0.134–0.292;
all three greedy streams changed. Q4 is not accepted as a faithful replacement.
Spanish FP32 reproduced the original model's weak response, rather than improving
it. These short numerical cases are not a general chatbot-quality evaluation.

Full evidence: `validation/parity-initial.json` (SHA-256
`f33e78219ab0e93c52552ca2e99695a717cc9d40cf0a3e50ab65da36804177d7`).

The tokenizer-only frontend also matched the original AutoTokenizer on 36 exact
render/token-ID/decode cases and 20 random decode cases. Actual 64-token native
chat, multi-turn history, reset and invalid-input handling were exercised with
Torch, Transformers and TensorFlow imports actively prohibited. See
`validation/tokenizer-parity.json` and `validation/chat-native-pure.json`.

## Reproduce the build and conversion

Requirements: x86-64 C++20 compiler, CMake 3.24+ and Python. Install the tokenizer,
NumPy and exporter dependencies with
`python -m pip install -r native_cpu/requirements-runtime.txt`.
The numerical validator additionally requires Transformers, PyTorch and pytest;
downloads use the Hugging Face CLI. The tested compiler is MSVC 19.50 on
Windows; non-Windows builds are not yet validated.

```powershell
python -m native_cpu.tools.build
```

The helper builds Release, runs CTest with assertions enabled, and writes compiler,
flags, source and library hashes to `native_cpu/build/build-info.json`. On this
Windows environment it normalizes environment-variable casing and uses serial
MSBuild with node reuse disabled. Override `--cmake` or `--generator` when needed.

The downloaded inputs come from
[jingyaogong/minimind-3](https://huggingface.co/jingyaogong/minimind-3), pinned to
`f92512d4cd6142fa9acc0d6022375049a8974bf6`. Re-download only if absent:

```powershell
hf download jingyaogong/minimind-3 model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja --revision f92512d4cd6142fa9acc0d6022375049a8974bf6 --local-dir checkpoints/minimind-3-hf
hf cache verify jingyaogong/minimind-3 --revision f92512d4cd6142fa9acc0d6022375049a8974bf6 --local-dir checkpoints/minimind-3-hf
```

Export to **new** paths; the exporter refuses to overwrite existing artifacts:

```powershell
python -m native_cpu.tools.export_weights --source checkpoints/minimind-3-hf/model.safetensors --config checkpoints/minimind-3-hf/config.json --output native_cpu/artifacts/minimind-fp32.bin --quantization fp32
python -m native_cpu.tools.export_weights --source checkpoints/minimind-3-hf/model.safetensors --config checkpoints/minimind-3-hf/config.json --output native_cpu/artifacts/minimind-q4.bin --quantization q4
```

Each binary has a neighboring SHA-256 manifest. Checkpoint weights, binaries and
generated build/validation files are ignored by Git.

## Verify numerical behavior

```powershell
python -m pytest native_cpu/tests -q -p no:cacheprovider
python -m native_cpu.tools.validate_parity --library native_cpu/build/Release/minimind_cpu.dll --fp32-model native_cpu/artifacts/minimind-fp32.bin --fp32-manifest native_cpu/artifacts/minimind-fp32.bin.manifest.json --q4-model native_cpu/artifacts/minimind-q4.bin --q4-manifest native_cpu/artifacts/minimind-q4.bin.manifest.json --build-info native_cpu/build/build-info.json --decode-tokens 16 --bench-repeats 3 --output native_cpu/validation/parity.json
```

The validator independently loads the pinned original Python model, verifies
checkpoint/source hashes and strictly checks all tensor keys and shapes. It tests
English, Spanish and Chinese prompts, token-by-token caching, full-prefill final
logits, reset, and greedy decoding. FP32 scalar and AVX2 must meet the declared
numerical tolerance and generate identical token streams. Q4 reports divergence
separately instead of pretending quantization is exact. Prefill and decode timings
are measured separately with identical tokens and one compute thread per backend.

## Runtime boundaries

| Component | Implementation |
|---|---|
| Dense graph | 8 layers, width 768, SwiGLU width 2432, tied vocabulary 6400 |
| Attention | 8 query / 4 KV heads, per-head Q/K RMSNorm, half-split RoPE |
| Cache | Preallocated, unrepeated K/V; checked capacity and reset |
| Kernels | FP32 and signed Q4/group-32 GEMV; runtime AVX2/FMA detection, scalar fallback |
| Quantization | Linear weights only; embeddings and norms remain FP32 |
| File format | Strict little-endian `MMCPU001`, exact tensor set, positive scales, CRC32 |
| Conversation | Original non-thinking template and demo system/sampling by default; explicit greedy parity profile; complete-pair history eviction |

The kernel approach adapts the row-wise FP32/Q4 and AVX2/FMA work in the supplied
`cpu-native-arch` directory. That source remains untouched. Old custom expert
routing and weights are not part of this runtime.

The frontend defaults to 2048 context tokens and rejects an oversized current
turn. The original model limit is 32768; loader allocations also have a 1 GiB
combined cache/table/scratch cap. Runtime handles are single-session objects:
do not call one handle concurrently. Prefill currently uses sequential GEMV,
not batched GEMM, so long-prompt performance needs separate optimization.

## T0-style affinity and row sharding

`compare_chat.py` accepts these native-only options:

| Option | Meaning |
|---|---|
| `--threads N` | Total native compute participants, including the caller (1–64) |
| `--cpus A,B,...` | Windows group-0 logical CPU indices in participant order |
| `--row-weights A,B,...` | Positive contiguous row-shard weights, one per participant |

The selected laptop profile is `--threads 2 --cpus 0,6 --row-weights 1332,992`.
When overriding this T0 launcher's thread count, supply matching CPU and weight
lists. For an unbound team with equal row weights, use `Native_CPP.cmd --threads N`.
The
original PyTorch launcher rejects non-default native-only options rather than
silently ignoring them. The old serial DLL remains compatible with default
options; explicit parallel options require the parallel-capable DLL.

The T0 row candidates came from `cpu-native-arch/archive/t0r/t0r_int8_sharded.csv`:
rows `1332,807,965,992` mapped to CPUs `0,2,4,6`. The selected `0,6` pair is a
starting profile validated on this MiniMind workload, not a universal optimal
FP32 ratio. Its final paired comparison is between the frozen baseline C++ DLL
and the new C++ DLL with the same FP32 kernels, weights, and model; the archived DLL is local-only and is not a
PyTorch-versus-C++ comparison.

Across four AB/BA pairs (eight complete six-turn conversations), all 48 turn
texts, prompt IDs, and sampled IDs matched. The four trials were faster on both
reported measures, but the conservative promotion gate (at least 5% decode and
at least 5% total-latency improvement) was not met: median prefill changed from
18.025723 s to 16.918335 s (-6.14%), decode throughput from 84.642327 to
87.757669 tokens/s (+3.6806%), and total time from 26.694330 s to 25.299208 s
(-5.2263%). Keep `Native_CPP.cmd` as the default; `Native_CPP_T0.cmd` remains
explicit opt-in. Earlier pre-cleanup measurements were invalidated because
four old Python test processes consumed the CPU concurrently. Full evidence:
[`benchmarks/t0-affinity-v1/summary.json`](benchmarks/t0-affinity-v1/summary.json).

## Roadmap after the faithful baseline

1. Preserve pinned-weight numerical parity as the regression gate.
2. Profile and optimize batched prefill, then decode kernels and thread scheduling;
   retain changes only after real hardware benchmarks and parity checks.
3. Evaluate quantization against a broader conversation corpus before changing the
   default from FP32. Small prompt samples do not establish language quality.
4. Add a native tokenizer/standalone executable if removing Python becomes useful.
5. Only then experiment with model architecture or training. C++ inference does
   not require replacing the proven upstream training pipeline.
