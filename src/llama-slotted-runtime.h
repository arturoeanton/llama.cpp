#pragma once

// FASE 4A-2b: slotted hot-swap runtime.
//
// At load time we use the layer filter (FASE 4A-1) to only materialize the
// tensors for the first `slots_resident` slots. Those slots act as a buffer
// "pool". When the decode driver needs a non-resident slot K, we read its raw
// bytes from the GGUF file into the pool slot's tensor buffers and rebind the
// per-layer pointers in `model.layers[]` so the gemma4 graph builder picks up
// the new data for layers K_start..K_end.
//
// The runtime opens its own fd on the GGUF so its reads don't interact with
// the loader's mmap or fread state.
//
// FASE 4A-3 extensions (to support Llama 3.1 405B Q3_K_M):
//   - multi-shard GGUFs (-NNNNN-of-NNNNN.gguf): open every shard, build a
//     unified tensor map name -> (shard_idx, offset, size, type).
//   - heterogeneous quant types across logical slots (mixed-quant variants
//     like Q3_K_M): on swap, if the source tensor's type/size differs from
//     the pool-resident dst tensor's, mutate dst->type and dst->nb[] to match
//     the source, and (if the source bytes don't fit the loader-allocated
//     buffer) point dst->data at a per-tensor override buffer we own.
//
// Constraints still in force:
//   - text-only Llama / Gemma 4
//   - --no-repack (raw quant bytes go straight to the buffer, no SIMD pack)
//   - batch_size 1
//   - all logical slots must share the same per-layer *shape* (ne[]). Only
//     the quant type may vary between layers.

#include "llama.h"
#include "llama-model.h"
#include "ggml.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Hot-swap pool replacement policy.
//
// ROUND_ROBIN: pool_idx = slot_idx % R. Every pass after the first re-swaps
//   every slot because the pool ends each pass holding the last R slots.
//   Steady state = N swaps/pass.
//
// PIN_SCRATCH: pins slots 0..R-2 in pools 0..R-2 permanently; pool R-1 is the
//   single scratch that rotates through slots R-1..N-1. Slot 0..R-2 always
//   hit. Steady state = N - (R - 1) = N - R + 1 swaps/pass.
//   For R=2, N=8 this saves 1 swap per steady-state pass.
//   Requires R >= 2 (with R == 1, there is nothing to pin).
enum slotted_hot_swap_policy_t {
    SLOTTED_POOL_ROUND_ROBIN = 0,
    SLOTTED_POOL_PIN_SCRATCH = 1,
};

// Per-tensor location in a (possibly multi-shard) GGUF.
//
// `shard_idx`         index into slotted_hot_swap_state::gguf_fds.
// `off_in_shard_data` offset relative to that shard's data section start
//                     (i.e. add gguf_data_bases[shard_idx] to get a file offset).
// `size`              raw tensor size in bytes (matches gguf_get_tensor_size).
// `type`              ggml_type as stored in the GGUF (Q3_K, Q5_K, etc.).
struct gguf_tensor_loc {
    int       shard_idx        = 0;
    uint64_t  off_in_shard_data = 0;
    uint64_t  size             = 0;
    int       type             = 0; // really enum ggml_type
};

struct slotted_hot_swap_state {
    // The model whose layer pointers will be mutated.
    llama_model * model = nullptr;

    // Multi-shard GGUF: one entry per shard, parallel arrays. Single-file
    // GGUFs use vectors of length 1.
    std::vector<std::string> gguf_paths;
    std::vector<int>         gguf_fds;
    std::vector<uint64_t>    gguf_data_bases;

    // Unified tensor map across all shards.
    std::unordered_map<std::string, gguf_tensor_loc> gguf_tensors;

    // Slot plan: each entry is [layer_start, layer_end] (inclusive).
    std::vector<std::pair<int,int>> slot_ranges;
    int slots_resident = 0;
    enum slotted_hot_swap_policy_t policy = SLOTTED_POOL_ROUND_ROBIN;

    // For each pool slot (0..slots_resident-1), snapshot of `model.layers[il]`
    // for il in the pool's physical layer range. We restore from here when
    // rebinding model.layers[K_start..K_end] for a hot-swapped logical slot.
    // pool_initial_layers[p][i] = model.layers[ slot_ranges[p].first + i ] as
    // captured at setup time, before any mutation.
    std::vector<std::vector<llama_layer>> pool_initial_layers;

    // pool_current_slot[p] = which logical slot is currently materialized in
    // pool p. Initially p (= the resident slot). Updated by swap_in.
    std::vector<int> pool_current_slot;

    // Mixed-quant support (FASE 4A-3).
    //
    // initial_alloc_size[t] = bytes the loader allocated for pool tensor t,
    // captured before any swap mutates dst->type / dst->nb. If a hot-swap
    // source needs more bytes than this, we fall back to override_bufs.
    std::unordered_map<ggml_tensor *, size_t> initial_alloc_size;

    // override_bufs[t] = malloc'd buffer that has replaced t's loader-backed
    // storage because a hot-swap source exceeded initial_alloc_size[t].
    // Sized to the largest source size ever observed for t. Lives until
    // slotted_hot_swap_destroy(). We use a raw vector<uint8_t> so the
    // destructor frees automatically.
    std::unordered_map<ggml_tensor *, std::vector<uint8_t>> override_bufs;

    // For restoring dst tensor metadata at destroy time.
    std::unordered_map<ggml_tensor *, int>     initial_type; // ggml_type
    std::unordered_map<ggml_tensor *, std::array<size_t, GGML_MAX_DIMS>> initial_nb;
    std::unordered_map<ggml_tensor *, void *>  initial_data;

    // Per-swap stats reported back to the caller.
    uint64_t total_bytes_read = 0;
    double   total_read_ms    = 0.0;
    double   total_set_ms     = 0.0;
    int      total_swaps      = 0;
};

// Open GGUF, build the offset map, snapshot pool layers. Returns false on error.
bool slotted_hot_swap_setup(slotted_hot_swap_state & st,
                            llama_model * model,
                            const std::string & gguf_path,
                            const std::vector<std::pair<int,int>> & slot_ranges,
                            int slots_resident,
                            enum slotted_hot_swap_policy_t policy);

// Map slot_idx -> pool_idx according to the current policy. The decode driver
// uses this to decide where each logical slot lives.
int slotted_hot_swap_pool_idx(const slotted_hot_swap_state & st, int slot_idx);

// Make logical slot `slot_idx` resident in pool slot `pool_idx`.
// If pool_idx already holds slot_idx, this is a no-op (returns true quickly).
// Otherwise reads each tensor of slot_idx's layers from the GGUF, writes into
// the pool's tensor buffers, and rebinds `model.layers[slot.range] =
// pool_initial_layers[pool_idx]`.
// Returns false on error (with a logged reason).
bool slotted_hot_swap_swap_in(slotted_hot_swap_state & st, int slot_idx, int pool_idx);

// Restore the pool slots' layer bindings to their initial state, close fd.
void slotted_hot_swap_destroy(slotted_hot_swap_state & st);

// Public wrapper type referenced by the C API (definition kept in the same
// header so the context implementation can reach in to drive hot-swaps).
struct llama_slotted_hot_swap {
    slotted_hot_swap_state st;
};
