# slotted-real — running LLMs larger than RAM on commodity hardware

This document describes the slotted-real experiment built on top of
`llama.cpp` in this fork. The goal is to validate, on a real GGUF model,
whether a transformer can be executed in **slots** of contiguous layers,
keeping only a small number of slots resident in memory at any moment and
streaming the rest from disk on demand.

If you want to *run* the demo, read [`README.poc.md`](README.poc.md). This
document covers design, phases, and known limitations.

## TL;DR

- **Goal:** run models that don't fit in RAM with a bounded peak memory
  footprint, even at the cost of wall-clock time.
- **Approach:** load only `slots_resident` slots of transformer layers; build
  a separate cgraph per slot; pass hidden state through a CPU staging buffer
  between slots; hot-swap weight bytes from the GGUF on demand.
- **Validated on Llama 3.1 70B Q3_K_XL** with `slots_resident=2`,
  `--slot-layers 10` (8 uniform slots), generating coherent multi-token output
  at ~11 GB peak.
- **Also validated on Llama 3.1 Tulu-3 405B Q3_K_M** (~200 GB across 5 shards)
  with `slots_resident=2`, `--slot-layers 3` (42 slots), generating coherent
  output at ~12.6 GB peak on a 24 GB MacBook Air M4 (~138 s/token).
- **Not validated on Gemma 4 31B** — the model's heterogeneous layer *shapes*
  (MoE intermixed with dense, variable head counts) still break the simple
  pool-based hot-swap. Treated as a known limitation.
- **Performance cost:** ~20× slower than a fully resident baseline for 70B;
  ~100-200× for 405B (page-cache-bound). The technique trades wall-clock time
  for peak memory.

## Motivation

Consumer hardware is converging on 16-32 GB of unified or system RAM. Open
weights are shipping in increasingly larger quants:
- Llama 3.1 70B Q3_K_XL ≈ 35 GB
- Llama 3.1 70B Q4_K_M ≈ 42 GB
- Llama 3.3 70B Q4_K_M ≈ 42 GB
- Gemma 4 31B Q4_K_M ≈ 18 GB
- Future Mixtral / OLMoE / DeepSeek MoE variants well over 50 GB

The choices today are:
1. Buy a workstation. Not always possible.
2. Use a smaller model. Quality regression.
3. Let the OS swap. Pathological tail latency, occasional OOM kills.

Slotted inference is a fourth option: keep only `K` slots of layers physically
resident, stream the rest from the GGUF on demand. It is intrinsically slow
but it lets a model that does not fit in RAM still produce correct output.

## Design

### Slot plan

A *slot* is a contiguous range of transformer layers `[il_start, il_end]`.
The full transformer is partitioned into N slots. The slot plan is built
once from the GGUF metadata (no weights loaded), using one of:

- `--slot-layers N` — fixed number of layers per slot. `N` should divide
  the total layer count to keep all slots uniform.
- `--slot-size-mb N` — approximate weight bytes per slot (greedy grouping).
  Currently this produces variable-sized slots, which breaks the homogeneous-
  pool hot-swap. Use `--slot-layers` instead for hot-swap.

### Loader filter (FASE 4A-1)

`include/llama.h` exposes a per-layer load filter:

```c
typedef bool (*llama_layer_filter_t)(int32_t il, void * user_data);

struct llama_model_params {
    ...
    llama_layer_filter_t layer_filter;
    void *               layer_filter_user_data;
};
```

When the filter returns `false` for layer `il`, the loader skips that layer's
tensors entirely:
- `create_tensor` returns `nullptr` (counted as skipped, no buffer alloc)
- `load_all_data` never iterates them
- `model.layers[il].wq` etc. stay `nullptr`

This drops the peak memory footprint roughly proportionally to the fraction
of layers loaded. Measured on Gemma 4 31B Q4_K_M:

| slots_resident | resident_weight_MB | peak_memory_footprint_MB |
|---:|---:|---:|
| 1 | 2,872 | 5,203 |
| 2 | 4,796 | 7,129 |
| 5 | 10,514 | 12,849 |
| 9 (baseline) | 17,806 | 20,133 |

The filter must skip `TENSOR_DUPLICATED` paths (e.g. Gemma's `rope_freqs`
shared across non-SWA layers) to keep the `n_created` count consistent with
the GGUF tensor count.

### Per-slot cgraph (FASE 4A-2a)

The model architecture's graph builder is parameterised to emit nodes for a
subrange of layers and to expose the hidden state at the slot boundary as an
output / input.

`llm_graph_params` (in `src/llama-graph.h`) gets three new fields, all with
defaults that preserve the original behaviour:

```cpp
int           slot_il_start    = 0;
int           slot_il_end      = -1;    // -1 -> full range
ggml_tensor * slot_inpL_carry  = nullptr;
```

In `src/models/llama.cpp` (and `gemma4.cpp`) the constructor uses these:

- `slot_il_start == 0` and no carry → build inpL from token embeddings (the
  canonical path).
- `slot_il_start > 0` → create a new input tensor `slot_inp_carry` in the
  cgraph's `ctx_compute` and use it as `inpL`. The driver writes data into
  it via `ggml_backend_tensor_set` between slot computes.
- Loop body unchanged: `for (int il = slot_il_start; il <= slot_il_end; ++il)`.
- `slot_il_end == n_layer - 1` → run `output_norm + lm_head` and set
  `res->t_logits` (final slot).
- Otherwise → `ggml_set_output(cur)` and `res->t_embd = cur` so the driver
  can read the per-slot hidden state.

Validation: with `slots_resident == total_slots` (no eviction), the slotted
path produces the **same top-1 token and logit, bit-for-bit, across 16 tokens
of greedy generation**, compared against the unmodified `llama_decode` path on
Gemma 4 31B Q4_K_M. The slotted graph builder is therefore mathematically
equivalent to the monolithic builder.

### Hot-swap runtime (FASE 4A-2b)

`src/llama-slotted-runtime.{h,cpp}` implements the runtime that streams
non-resident slots from the GGUF into the resident slot buffers on demand.

Key state in `slotted_hot_swap_state`:

- A separate read-only `fd` to the GGUF file.
- A `gguf_tensors` map (built from `gguf_init_from_file` metadata) of
  `name -> (offset, size)`.
- `slot_ranges`: `[(il_start, il_end), ...]` for every logical slot.
- `pool_initial_layers[p]`: a snapshot of `model.layers[il]` for the layers
  in pool `p`, captured at setup time.
- `pool_current_slot[p]`: which logical slot is currently materialised in pool
  `p` (initially `p`).

To make logical slot `K` resident in pool `P`:

1. `ggml_backend_sched_synchronize(sched)` — wait for any prior compute that
   referenced this pool's tensors to finish.
2. For each tensor `blk.<phy>.X` in `model.tensors_by_name` with
   `phy_il ∈ [P_start, P_end]`:
   - Compute the equivalent logical name: replace `<phy_il>` with
     `K_start + (phy_il - P_start)`.
   - Look up its `(offset, size)` in the GGUF map.
   - **Refuse the swap if the sizes do not match** (this is the Gemma blocker).
   - `pread()` the bytes from the GGUF fd.
   - `ggml_backend_tensor_set(dst_tensor, bytes, ...)` writes them into the
     pool's tensor buffer (no repack — `--no-repack` is forced).
3. Rebind: `model.layers[K_start..K_end] = pool_initial_layers[P]`. Now the
   graph builder's references to `model.layers[K_start + i].wq` etc. resolve
   to the pool tensors that physically hold slot K's bytes.
4. `pool_current_slot[P] = K`.

Pool assignment is round-robin: `pool_idx = K % slots_resident`. Each
forward pass re-swaps every non-resident slot since the pool contents from
the previous pass are stale (and the residents 0..R-1 themselves need to
be swapped back in on subsequent passes after the pool was reused).

### Multi-shard GGUFs (FASE 4A-3)

GGUFs larger than ~30 GB are split into multiple `-NNNNN-of-NNNNN.gguf`
files (the standard llama.cpp convention). The runtime detects the split via
the `split.count` metadata key on the main shard and uses the public helpers
`llama_split_prefix` / `llama_split_path` to enumerate the rest. It opens an
`O_RDONLY` fd per shard and builds a unified tensor map:

```
name  ->  { shard_idx, off_in_shard_data, size, type }
```

At swap time, the source `pread()` targets `gguf_data_bases[shard_idx] +
off_in_shard_data` on `gguf_fds[shard_idx]`. The rest of the swap path is
shard-agnostic. Single-file GGUFs fall through with a 1-element shard
vector.

### Mixed-quant variants (FASE 4A-3)

K-quant variants like Q3_K_M assign *different quant types* to different
layers within the same model (e.g. Q5_K for `ffn_down` in a fraction of
layers, Q3_K for the rest). The pool tensors are sized by the loader for
the resident slots' types; when a non-resident slot's source has a different
type at the same name:

1. The runtime mutates `dst->type` and recomputes `dst->nb[]` from the
   source's `ggml_blck_size` / `ggml_type_size`. The shape (`ne[]`) is
   assumed unchanged.
2. If `src_size > initial_alloc_size[dst]` (snapshotted at setup), the
   runtime allocates a per-tensor override buffer on the heap and points
   `dst->data` at it. Otherwise it just memcpys into the loader-allocated
   storage (which has room to spare).
3. The CPU compute backend reads `dst->type` / `dst->nb[]` / `dst->data`
   at every graph build, so the next slot's cgraph sees the right kernel
   for the right bytes.

`set_ms` in the swap log no longer goes through `ggml_backend_tensor_set`
in this path; we issue a direct `memcpy` since the override buffer is not
known to the backend buffer allocator. With `--n-gpu-layers 0` and CPU
buffers this is equivalent to the prior code for non-mutated tensors.

This works for variants where the per-layer **shape** is uniform but the
**type** varies (Q3_K_M, Q4_K_M, Q5_K_M, etc.). It does **not** work for
models with heterogeneous shapes (MoE with intermixed dense/sparse layers,
variable head counts) — those still fail because `recompute_nb_for_type`
keeps `ne[]` from the resident slot.

Measured on Llama 3.1 Tulu-3 405B Q3_K_M (5 shards, 1138 tensors total),
`--slot-layers 3` → 42 logical slots, `--slots-resident 2`:

- Slot 2 first swap: `retyped=2 resized=0 bytes=4.68 GB`.
- Slot 3 first swap: `retyped=1 resized=0 bytes=4.57 GB`.
- Slot 4..41 first swaps: `retyped=0 resized=0 bytes=4.57 GB` each.

`resized=0` everywhere means the loader-allocated buffers from the
resident slots happened to be large enough (the resident slot had the
higher-quant variant for those tensors). No heap overrides were needed
for the run.

### Decode driver

`llama_context::decode_slotted_real_test` (in `src/llama-context.cpp`)
mirrors the parts of `decode()` that handle batch initialisation, KV cache
preparation, and output ID mapping, with the inner ubatch loop replaced by:

```cpp
for each ubatch:
    mctx->apply()
    hidden = zero_vec(n_embd * n_tokens)
    for each slot K in plan:
        if pool[K % R] != K:
            sched_synchronize()
            hot_swap_swap_in(K, K % R)
        build slot graph (slot_il_start, slot_il_end set in params)
        sched_alloc_graph
        set_inputs
        if not first slot: tensor_set(carry, hidden)
        graph_compute
        if not final slot: tensor_get(t_embd, hidden)
        else: tensor_get_async(t_logits -> output buffer)
sched_synchronize
finalize output_ids[]
```

### Bypasses needed for the integration

A few existing `llama.cpp` code paths assume the model is fully materialised.
We disable them when `cparams.slotted_real_skip_sched_reserve` is true:

- `llama_context::sched_reserve()` — initialises `gf_res_prev`, `gf_res_reserve`,
  and `sched`, but skips the full-graph reservation pass that would try to
  build a cgraph touching NULL tensors.
- `common_context_can_seq_rm()` — short-circuits the 2-token probe decode
  that server-context.cpp does on init.
- The repack buffer (`--no-repack` is forced when slotted-real + decode test
  are both active).

## CLI flag reference

These flags live in `common/arg.cpp` and are CLI-only (`LLAMA_EXAMPLE_CLI`).

### Instrumentation (FASE 1-3)

| Flag | Effect |
|---|---|
| `--slotted-test` | Build slot plan from the loaded model and log it. No behaviour change. |
| `--slot-size-mb N` | Group layers up to ~N MiB per slot. |
| `--slot-layers N` | Group layers into fixed N-layer slots. Takes priority over `--slot-size-mb`. |
| `--slotted-log` | Verbose per-slot logging during decode. |
| `--slotted-json PATH` | Write slot plan + summary to PATH as JSON. |
| `--slotted-simulate-prefetch` | Run a `std::thread` per slot transition that simulates prefetching the next slot. |
| `--slotted-bandwidth-mb-s N` | Bandwidth used by `mode=sleep` to convert bytes to ms. |
| `--slotted-prefetch-mode MODE` | `sleep` (default) or `dummy-read` (pread from a separate fd). |

### Loader filter (FASE 4A-1)

| Flag | Effect |
|---|---|
| `--slotted-real` | Install the layer filter so only the first `--slots-resident` slots load. |
| `--slots-resident N` | Number of slots to keep resident in RAM. Default 2. |

### Slotted decode (FASE 4A-2a, b)

| Flag | Effect |
|---|---|
| `--slotted-decode-test` | Run `llama_decode_slotted_test` (or `_real_test` if combined with `--slotted-real`) on the `-p` prompt, sample `-n` tokens greedily, print. |
| `--slotted-decode-baseline` | Run normal `llama_decode` on the same prompt and sampling for comparison. |
| `--slotted-chat-poc` | Interactive chat loop using the slotted-real runtime. Used by `poc.c`. |
| `--slotted-round-robin` | Opt out of the default pin-and-scratch hot-swap policy and use the legacy round-robin one (pool_idx = slot_idx % R). See "Hot-swap pool policy" below. |

## Hot-swap pool policy

The runtime supports two replacement policies for the slot buffer pool:

- **`pin-and-scratch` (default, when `slots_resident >= 2`)**: pins slots
  `0..R-2` in pools `0..R-2` for the lifetime of the run; pool `R-1` is the
  single scratch slot that rotates through `slots R-1..N-1`. Slots `0..R-2`
  always hit. Steady-state swap count per forward pass: `N - R + 1`.
- **`round-robin` (legacy, opt-in via `--slotted-round-robin`)**: assigns
  `pool_idx = slot_idx % R`. Every steady-state pass re-swaps all `N` slots
  because the pool ends each pass holding the last `R` accessed slots.
  Steady-state swap count per forward pass: `N`.

For our reference workload (Llama 3.1 70B Q3_K_XL, `N=8`, `R=2`, prompt
"The capital of France is", `-n 8`):

| Metric                | round-robin | pin-and-scratch | delta |
|---|---:|---:|---:|
| Total swaps           | 102 | 90 | -12 (-11.8%) |
| Bytes read            | 425 GB | 375 GB | -50 GB (-11.8%) |
| Sum read_ms           | 264.2 s | 195.0 s | -26.2% |
| Sum set_ms            | 157.1 s | 68.7 s | -56.3% |
| **Real time**         | **311.2 s** | **251.9 s** | **-59.3 s (-19.0%)** |
| Peak memory footprint | 11,357 MB | 11,362 MB | identical |
| OS swaps              | 0 | 0 | identical |
| Generated text        | `a city of love, art, fashion` | `a city of love, art, fashion` | identical |

> pin-and-scratch is the default policy because it reduced runtime by 19%
> on MacBook Air M4 24GB with Llama 3.1 70B Q3_K_XL, with identical output
> and no additional peak memory footprint.

The unexpectedly large drop in `set_ms` (56% vs the swap-count drop of
12%) is because round-robin writes to **both** pool buffers in alternation,
keeping both sets of pool pages hot in the CPU cache. Pin-and-scratch only
ever writes to pool `R-1` after init, so the pinned pool's pages can stay
cold and there is less cache competition on the write side.

## Files

```
include/llama.h                              # public API additions
common/common.h, common/arg.cpp              # common_params + CLI flags
common/common.cpp                            # forward to cparams
common/slotted-inference.h, .cpp             # plan builder, FASE 3 callback runtime
src/llama-graph.h                            # slot fields in llm_graph_params
src/llama-cparams.h                          # cparams.slotted_real_skip_sched_reserve
src/llama-model-loader.h, .cpp               # layer filter integration in create_tensor
src/llama-model.cpp                          # exported size accounting per layer
src/llama-context.h, .cpp                    # decode_slotted_test + decode_slotted_real_test
src/llama-slotted-runtime.h, .cpp            # hot-swap pool + C API
src/llama.cpp                                # forward filter from model params to loader
src/models/llama.cpp                         # slot-aware Llama graph builder
src/models/gemma4.cpp                        # slot-aware Gemma 4 graph builder
tools/cli/cli.cpp                            # test harness + chat loop
poc.c                                        # demo launcher (root of repo)
```

## Phase log

| Phase | Goal | Result |
|---|---|---|
| FASE 0 | Map the code base (loader, graph, decode, CLI). | File:line references collected. |
| FASE 1 | Add CLI flags + slot planner, log only. | Plan printed for Gemma 4 31B (60 layers, 9 slots @ 2 GB). |
| FASE 2 | Slot planner data structures. | Folded into FASE 1. |
| FASE 3 | Async load simulation + per-slot timing via cb_eval. | `io_hidden_percent` measured on CPU (100%) and Metal partial (40-86%). |
| FASE 4A-1 | Loader filter, no inference. | Peak footprint scales linearly with `slots_resident`. |
| FASE 4A-2a | Per-slot graph execution, all slots loaded. | Top-1 token and logit match the monolithic baseline bit-for-bit, validated over 16 tokens. |
| FASE 4A-2b | Hot-swap + 1 token with `slots_resident=2`. | Works on Llama 3.1 70B Q3_K_XL (35 GB → 11 GB peak); blocked on Gemma 4 31B by layer-structure heterogeneity. |
| FASE 4A-3 | Multi-shard GGUFs + mixed-quant per-layer types. | Works on Llama 3.1 Tulu-3 405B Q3_K_M (~200 GB across 5 shards → 12.6 GB peak on 24 GB M4). |
| FASE 4B | M4 kernel lab (microbenchmark). | Designed, not implemented yet. |
| FASE 5 | True eviction with per-slot tensor recreation. | Out of scope for this round. |

## Validated configurations

### Llama 3.1 70B Q3_K_XL (FASE 4A-2b)

```
Model:       Meta-Llama-3.1-70B-Instruct Q3_K_XL  (35 GB, single file)
Backend:     CPU only (Apple M4 / 24 GB)
Build:       cmake -B build -DGGML_METAL=OFF -DLLAMA_BUILD_SERVER=ON
Flags:       --n-gpu-layers 0 --no-mmap --no-warmup --no-repack
             --ctx-size 128 --batch-size 8 --ubatch-size 8
             --slot-layers 10 --slotted-real --slots-resident 2
             --slotted-chat-poc -n {4|8|16|32}
Result:      Coherent generation. Peak memory footprint ~11 GB across runs.
             Zero OS-level swaps. Hard page faults < 700.
             ~20-30 s per generated token at n=4..8.
```

Examples of `Hello, my name is` → ` John and I am a 30-year-old` and
`The capital of France is` → ` a city of love, art, fashion`.

The numbers in the "Phase log" / FASE 4A-2b row above were measured with
the legacy round-robin policy. With the default pin-and-scratch policy
the same run completes in 19% less wall-clock time on the same hardware
(see "Hot-swap pool policy" below).

### Llama 3.1 Tulu-3 405B Q3_K_M (FASE 4A-3)

```
Model:       Llama-3.1-Tulu-3-405B-Q3_K_M  (~200 GB across 5 shards)
Backend:     CPU only (Apple M4 / 24 GB)
Flags:       --n-gpu-layers 0 --no-mmap --no-warmup
             --ctx-size 128 --batch-size 8 --ubatch-size 8
             --slot-layers 3 --slotted-real --slots-resident 2
             --slotted-chat-poc -n 4 --verbosity 4
Result:      Coherent generation.
             Peak memory footprint: 12.6 GB.
             5 shards detected, 1138 tensors mapped, 0 'not found' warnings.
             368 total hot-swaps over the run.
             1.68 TB of bytes streamed (the OS page cache absorbed almost all
             of it: block_input_operations = 0).
             4 tokens in 552 s (~138 s/token).
             3 retypes total (all in the first swaps of slots 2 and 3,
             corresponding to Q3_K_M's per-layer Q5_K upgrades).
             0 heap-override allocations (loader buffers were large enough).
```

Example: `The capital of France is` → ` a city that needs`.

The 405B run uses the same `llama-cli` binary; the only differences from
the 70B configuration are the model path, `--slot-layers 3` (126 layers /
3 = 42 slots), and dropping `--no-repack` (repack stays off implicitly via
the slotted-real path). `poc.c` is not used (it hardcodes the 70B path);
the run is launched via direct `llama-cli` invocation. See the snippet at
the bottom of [`README.poc.md`](README.poc.md) for the exact command.

## Known limitations

1. **Heterogeneous *shapes* still break hot-swap.** As of FASE 4A-3, the
   runtime tolerates per-layer **type** variation (Q3_K_M, Q4_K_M, Q5_K_M:
   different quant types for the same tensor name in different layers) by
   mutating `dst->type` / `dst->nb[]` at swap time. It still does **not**
   tolerate per-layer **shape** variation: Gemma 4 with intermixed dense/MoE
   FFN, Mixtral, mixed-head counts, alternative attention with optional
   v_proj, etc. all keep `ne[]` non-uniform across layers and would require
   reshaping at swap time, which we don't do.

2. **`--no-repack` is required.** The CPU_REPACK buffer type formats weights
   at allocation time for SIMD-friendly layouts. Replicating that
   transformation on-the-fly during a hot-swap is FASE 5 work. Without
   repack, CPU matmul is ~30% slower.

3. **`--no-mmap` is required for honest measurement.** With mmap the OS
   pages weights in lazily; reported RSS understates true memory pressure.
   The hot-swap path itself does not depend on mmap being off, but
   the demo uses `--no-mmap` so the `/usr/bin/time -l` numbers are
   meaningful.

4. **One model architecture has been adapted at a time.** The slot-aware
   graph builder modifications live in `src/models/llama.cpp` and
   `src/models/gemma4.cpp`. Other architectures need the same pattern
   replicated: add `slot_il_start/end` reading, branch on `slot_is_first`
   for the embedding input, branch on `slot_is_final` for the output norm /
   lm_head, and replace the layer loop bounds.

5. **`sched_reserve` is skipped, not replaced.** The slotted decode path
   reserves its own per-slot graphs as it goes. The first forward pass pays
   a larger allocation cost than subsequent ones because the scheduler has
   not yet seen any slot's graph topology.

6. **KV cache is allocated for all layers.** It is small (~200 MiB at
   `--ctx-size 128`) and not the bottleneck, but the slotted-real run still
   allocates KV space for the layers whose weights were skipped by the load
   filter. This is intentional: KV state must persist across slot rotations
   within a forward pass.

7. **Pool replacement policy.** The default is `pin-and-scratch` (pin R-1
   slots, use 1 scratch); legacy round-robin remains available via
   `--slotted-round-robin`. See "Hot-swap pool policy" for the measured
   trade-off. Both are still suboptimal vs Belady (optimal for the cyclic
   access pattern), but Belady-style lookahead is not implemented here.

## Disclosure

This is experimental research code in a private fork. It is not intended for
upstream submission to `ggml-org/llama.cpp` as-is. See `AGENTS.md` and
`CONTRIBUTING.md` for the upstream contribution policy. The slotted-real
work was developed with substantial AI assistance and should be reviewed
end-to-end by a human contributor before any portion of it is proposed
upstream.
