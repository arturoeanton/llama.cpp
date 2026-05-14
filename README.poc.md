# poc.c — slotted-real chat demo launcher

A 100-line C program that launches the modified `llama-cli` with every flag
needed to run **Meta-Llama-3.1-70B-Instruct Q3_K_XL (~35 GB on disk)** on a
machine where the full model does not fit comfortably in RAM. With
`--slots-resident 2` the peak memory footprint stays around **11 GB**.

`poc.c` is intentionally a launcher only. The actual slotted-real
infrastructure (loader filter, per-slot graph builder, hot-swap runtime)
lives inside `llama.cpp` itself. See [`README.slot.md`](README.slot.md) for
the architecture and design notes.

## Prerequisites

- macOS or Linux
- `clang` (or any C compiler)
- A built `llama-cli` (the one in this fork, with slotted-real support)
- A local copy of Llama 3.1 70B Q3_K_XL at
  `~/models/llama31-70b/Meta-Llama-3.1-70B-Instruct-Q3_K_XL.gguf`

Build `llama-cli` first:

```bash
cmake -B build -DGGML_METAL=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -j --target llama-cli
```

## Build the launcher

From the repository root:

```bash
clang poc.c -o poc
```

The launcher is a single source file, no dependencies beyond `libc`.

## Run

```bash
./poc           # 16 tokens per turn (default)
./poc 32        # 32 tokens per turn
./poc 8         # 8 tokens per turn
```

The launcher checks that `./build/bin/llama-cli` and the model file exist,
then `execv`s `llama-cli` with the following hardcoded flags:

```
--n-gpu-layers 0
--no-mmap
--no-warmup
--no-repack
--ctx-size 128
--batch-size 8
--ubatch-size 8
--slot-layers 10
--slotted-real
--slots-resident 2
--slotted-chat-poc
-n <N>
```

`-n <N>` is taken from `argv[1]` if you pass it, otherwise defaults to 16.

## What you see

```
=========================================================================
  llama.cpp slotted-real demo
=========================================================================
  Model:     Meta-Llama-3.1-70B-Instruct Q3_K_XL (~35 GB on disk)
  Mode:      CPU only, --no-mmap, --no-repack
  Slot plan: 8 slots x 10 layers,  --slots-resident 2

  This is intentionally slow. It proves a 35GB 70B model can run
  with a reduced memory footprint (~11 GB peak instead of ~35 GB).
=========================================================================

Loading model... |/-\
=======================================================================
  Type a prompt and press Enter. Empty line / Ctrl-D exits.
=======================================================================

> Hi
, I'm looking

> The capital of France is
 a city of love

> Hello, my name is
 John and I am a 30-year
```

Each turn:
1. Resets the KV cache (no conversation memory between turns).
2. Tokenises the prompt with BOS.
3. Feeds it through the slotted decode path (one token per `llama_decode_slotted_real_test` call).
4. Greedy-samples the requested number of tokens.
5. Prints each token as it is produced.

Press Enter on an empty line or Ctrl-D to exit.

## Why the hardcoded flags

| Flag | Value | Reason |
|---|---|---|
| `--slot-layers 10` | 10 | 80 layers / 10 = 8 uniform slots. Avoids the heterogeneous-slot mismatch that breaks size-based grouping. |
| `--slots-resident 2` | 2 | Minimum viable. Peak footprint ~11 GB. Higher values use more RAM and reduce hot-swap frequency. |
| `--no-mmap` | on | Required so the resident set size reported by the OS reflects the real memory cost. |
| `--no-repack` | on | The hot-swap path writes raw GGUF bytes directly into the pool buffers; the SIMD-friendly repack format would need on-the-fly re-packing, which is FASE 5 territory. |
| `--ctx-size 128` | 128 | KV cache stays small. KV cache size is independent of slot residency. |
| `--batch-size 8` / `--ubatch-size 8` | 8 | The slotted path is batch=1; these are just safe defaults. |
| `--no-warmup` | on | The default warmup path tries to build a full cgraph through the filtered model, which would dereference NULL tensors for non-resident layers. |

The hot-swap pool policy defaults to **pin-and-scratch** (introduced in the
4A-2b benchmark round). This reduced runtime by 19% on MacBook Air M4 24GB
with Llama 3.1 70B Q3_K_XL versus the legacy round-robin policy, with
identical output and no additional peak memory footprint. To force the
legacy behaviour for A/B comparison, run `llama-cli` directly and add
`--slotted-round-robin`; the launcher itself has no flag for it.

If you need to point the launcher at a different model path, edit the
`MODEL_REL` macro at the top of `poc.c`.

## Performance expectations

Generation time scales roughly linearly with `n_predict`. On an Apple M4 with
the SSD's page cache mostly warm, you can expect:

- ~5 minutes for `./poc 4` (4 tokens, including model load and prompt processing)
- ~7 minutes for `./poc 16`
- ~15 minutes for `./poc 32`

The hot-swap path re-reads roughly 30 GB per forward pass (8 slots minus the
two residents, ~4.4 GB each). For long generations the OS page cache usually
absorbs most of this; reported `block input operations` should stay low.

## Cross-platform validation

Both `poc` (70B Q3_K_XL) and `poc_tulu` (405B Q3_K_M) have also been
smoke-tested on a **ThinkPad-class Intel Core i7 Evo laptop with 32 GB
RAM** running Linux / x86_64. Both ran to completion with coherent
output, no source changes required. Wall-clock was roughly **~2× the
M4 24 GB target for 70B** and **~3× for 405B**, attributed to slower
NVMe sustained throughput plus the AVX2 dispatch path being less
optimised than the NEON `sdot` path for Q-quant `matmul` on Apple
Silicon.

The MacBook Air M4 24 GB stays the primary optimisation target. The
x86 result is included here because it demonstrates that the
slotted-real path itself is not tied to any one ISA or to Apple
Silicon's unified-memory architecture — same binary, same flags,
different machine, model runs.

## Limitations

1. **Llama 3.1 family only.** Models with heterogeneous layer structure
   (Gemma 4 with MoE, Mixtral, etc.) currently fail in the hot-swap path with
   tensor-size mismatches. See `README.slot.md` for details.
2. **CPU only.** Metal/CUDA paths have not been adapted.
3. **No conversation memory.** Each turn is independent.
4. **Slow.** This is a memory-efficiency demo, not a performance demo.
5. **`--no-repack` cost.** Without the SIMD weight-packing, CPU matmul is
   roughly 30% slower per layer than the default path.
6. **Slot 0 weights stay valid until the first hot-swap into pool 0.** After
   that, model.layers[0..9] still point at pool 0's tensors but the *data*
   has been overwritten. Anything that touches those layer indices outside
   the slotted decode loop will read whichever logical slot was last loaded.

## Files this PoC touches

- `poc.c` (this directory) — the launcher
- `tools/cli/cli.cpp` — the `--slotted-chat-poc` chat loop
- `src/llama-slotted-runtime.{h,cpp}` — the hot-swap pool runtime
- `src/llama-context.cpp` / `src/llama-context.h` — `decode_slotted_real_test`
- `src/models/llama.cpp` — slot-aware graph builder
- `include/llama.h` — public C API extensions

For a deeper walkthrough of what each piece does, read
[`README.slot.md`](README.slot.md).

## Companion launcher: `poc_tulu.c` (Llama 3.1 405B Q3_K_M)

`poc.c` targets the 70B Q3_K_XL. A sibling launcher, `poc_tulu.c`, points
the same `llama-cli` binary at **Llama 3.1 Tulu-3 405B Q3_K_M** (~200 GB
across 5 shards), with a peak memory footprint of **~12.6 GB** on a 24 GB
MacBook Air M4. This was enabled by FASE 4A-3 (multi-shard + mixed-quant
support in `src/llama-slotted-runtime.{h,cpp}`).

Build and run:

```bash
clang poc_tulu.c -o poc_tulu
./poc_tulu          # 4 tokens per turn (default, ~9 min on 24 GB M4)
./poc_tulu 8        # 8 tokens per turn (~20 min)
./poc_tulu 1        # 1 token (smoke test)
```

`poc_tulu` expects shard 1 of the model at
`$HOME/models/llama31-405b-tulu-q3km/Llama-3.1-Tulu-3-405B-Q3_K_M/Llama-3.1-Tulu-3-405B-Q3_K_M-00001-of-00005.gguf`,
with the other 4 shards alongside it. The runtime auto-discovers them from
the `split.count` metadata key, so only the path to shard 1 is given on
the `llama-cli` command line. Edit `MODEL_REL` at the top of `poc_tulu.c`
to point elsewhere.

The hardcoded flags differ from `poc.c` only in the slot plan:

| Flag | `poc` (70B) | `poc_tulu` (405B) | Reason |
|---|---|---|---|
| `--slot-layers` | 10 | 3 | 80 / 10 = 8 slots vs 126 / 3 = 42 slots. Both uniform. |
| Default `-n` | 16 | 4 | The 405B is ~138 s/token; defaults aim for a similar wall-clock budget per demo turn. |

Trade-off vs 70B: ~138 s/token instead of ~20-30 s/token, because each
forward pass re-reads tens of GB of weights and the page cache hit rate
is lower with a 200 GB working set on 24 GB of RAM. This is intentional;
the goal is to demonstrate that slot-based execution scales to 405B-class
models on consumer hardware, not to be fast.

See [`README.slot.md`](README.slot.md), section "Llama 3.1 Tulu-3 405B
Q3_K_M (FASE 4A-3)" for the full measurement table.

## Disclosure

This is experimental research code in a private fork. It is not intended for
upstream submission to `ggml-org/llama.cpp` as-is. The goal is to validate
the slotted-execution idea on a real GGUF model before deciding whether to
invest in a production implementation. See `CONTRIBUTING.md` and `AGENTS.md`
for the upstream project's contribution policy.
