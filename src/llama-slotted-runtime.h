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
// Constraints (FASE 4A-2b):
//   - text-only Gemma 4
//   - --no-repack (raw Q4_K_M bytes go straight to the buffer, no SIMD pack)
//   - batch_size 1
//   - homogeneous slot layouts (all slots' per-layer tensor shapes match the
//     pool slots' shapes -- true for Gemma 4 31B; we assert at swap time)

#include "llama.h"
#include "llama-model.h"

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

struct slotted_hot_swap_state {
    // The model whose layer pointers will be mutated.
    llama_model * model = nullptr;

    // GGUF path + fd opened with O_RDONLY for hot-swap reads.
    std::string gguf_path;
    int         gguf_fd        = -1;
    uint64_t    gguf_file_size = 0;
    uint64_t    gguf_data_base = 0; // offset where tensor data starts in GGUF

    // GGUF tensor map: tensor name -> (data offset relative to gguf_data_base, size in bytes).
    // Built once at setup. Used to read raw bytes for non-resident slots.
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> gguf_tensors;

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
