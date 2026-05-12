#pragma once

// Experimental: slotted-inference instrumentation.
//
// FASE 1+2: PLAN and LOG (common_slot_plan, common_slotted_build_plan, ...).
// FASE 3:   per-slot compute timing via cb_eval + async load simulation.
// No weights are moved, no math is changed in either phase.
//
// Activation is opt-in via --slotted-test. With the flag off this header
// is unreachable from any hot path and the helpers are no-ops.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

struct llama_model;
struct llama_context;
struct common_params;
struct ggml_tensor;

struct common_slot_desc {
    int      slot_id      = 0;
    int      layer_start  = 0;     // inclusive
    int      layer_end    = 0;     // inclusive
    uint64_t weight_bytes = 0;     // sum of per-layer weight bytes
    uint64_t attn_bytes   = 0;
    uint64_t ffn_bytes    = 0;
    uint64_t other_bytes  = 0;     // norms / ssm / biases / etc.
};

struct common_slot_plan {
    bool                          enabled            = false;
    int                           n_layer            = 0;
    int                           slot_layers_used   = 0; // 0 if planning by size
    int                           slot_size_mb_used  = 0; // 0 if planning by layers
    uint64_t                      non_layer_bytes    = 0; // embd / lm_head / final norm
    std::vector<common_slot_desc> slots;
};

// Build a slot plan from CLI params. Inputs:
//   - params.slotted_test: master switch. If false, returns an empty disabled plan.
//   - params.slot_layers:  if > 0, group layers in fixed-size groups.
//   - params.slot_size_mb: else, approximate target size per slot in MiB.
// One of slot_layers / slot_size_mb must be > 0 when slotted_test is on, otherwise
// we default to a single slot covering all layers (caller should warn).
common_slot_plan common_slotted_build_plan(const llama_model * model, const common_params & params);

// Print the plan to the standard log channels (LOG_INF). Idempotent.
void common_slotted_print_plan(const common_slot_plan & plan);

// Write the plan to PATH as JSON. Returns true on success. The schema is the same
// the FASE 3 summary will append into.
bool common_slotted_write_plan_json(const common_slot_plan & plan, const std::string & path);

// =========================================================================
// FASE 4A: build slot plan from GGUF metadata (before model load).
// =========================================================================

// Parse the GGUF at `model_path` and build a slot plan using the same
// grouping rules as common_slotted_build_plan(). Does NOT load weights.
// Returns a plan with enabled=false on parse error or empty file.
common_slot_plan common_slotted_build_plan_from_gguf(
        const std::string & model_path,
        const common_params & params);

// Boolean test: is layer `il` covered by the first `slots_resident` slots of `plan`?
// Used as the body of the C-API layer_filter callback.
bool common_slotted_layer_is_resident(const common_slot_plan & plan, int32_t il, int slots_resident);

// C-compatible adapter that fits the llama_model_params.layer_filter signature.
// Pass `&filter_state` as user_data, where filter_state is the struct below.
struct common_slot_filter_state {
    const common_slot_plan * plan;
    int                      slots_resident;
};
bool common_slotted_layer_filter_cb(int32_t il, void * user_data);

// =========================================================================
// FASE 3: per-slot timing via cb_eval + async load simulation.
// =========================================================================

// One record emitted at the end of each forward pass, for each slot that fired.
struct common_slot_run_record {
    int      token_index   = 0;   // 0 = prefill pass; 1, 2, ... = decode tokens
    bool     is_prefill    = false;
    int      slot_id       = 0;
    int      layer_start   = 0;
    int      layer_end     = 0;
    uint64_t weight_bytes  = 0;
    double   compute_ms    = 0.0;
    double   async_load_ms = 0.0; // simulated load time of the NEXT slot, kicked off here
    double   wait_next_ms  = 0.0; // time spent blocked waiting for the prefetch to finish
};

// Prefetch simulation modes.
enum class common_slot_prefetch_mode {
    sleep      = 0, // pure cpu-side sleep modelled by bandwidth_mb_s
    dummy_read = 1, // real pread() from a separate fd on the model file
};

// Mutable state shared between the cb_eval thread and the prefetch worker.
// Held by value in cli main(); pointer is passed as cb_eval_user_data.
struct common_slot_runtime {
    common_slot_plan          plan;
    std::vector<int>          layer_to_slot;   // size n_layer

    // configuration (copied at init)
    bool                      simulate_prefetch = false;
    bool                      verbose_log       = false;
    double                    bandwidth_mb_s    = 0.0;
    common_slot_prefetch_mode prefetch_mode     = common_slot_prefetch_mode::sleep;

    // dummy-read state (only used in dummy_read mode)
    int                       dummy_fd          = -1;
    uint64_t                  dummy_file_size   = 0;

    // becomes true after init; cb_eval is a no-op until then.
    bool                      ready             = false;

    // per-forward-pass live state.
    int                       current_slot      = -1;
    int                       last_seen_layer   = -1;
    int                       token_index       = 0;
    bool                      first_node_in_pass = true;
    std::chrono::steady_clock::time_point slot_enter_time;

    // async prefetch state.
    std::thread                           prefetch_thread;
    bool                                  prefetch_active      = false;
    int                                   prefetch_target_slot = -1;
    std::chrono::steady_clock::time_point prefetch_start_time;
    // actual time the prefetch worker ran for, written by the worker BEFORE it
    // returns. Read by the compute thread only after .join(), so no atomicity
    // is needed -- join() is an acquire fence.
    double                                prefetch_actual_ms   = 0.0;

    // scratch: timings being accumulated for each slot within the current pass.
    struct pass_acc {
        double compute_ms    = 0.0;
        double async_load_ms = 0.0;
        double wait_next_ms  = 0.0;
    };
    std::vector<pass_acc>          per_slot_pass;
    std::vector<common_slot_run_record> records;
};

// Initialize the runtime AFTER the model is loaded and the plan is built.
// `model_path` is opened (read-only) for dummy-read mode; pass empty string to skip.
// Returns false on error; the runtime stays disabled in that case.
bool common_slotted_runtime_init(common_slot_runtime & rt,
                                 common_slot_plan && plan,
                                 const common_params & params,
                                 const std::string & model_path);

// Commit any in-flight forward-pass timings to `rt.records`. Must be called
// AFTER the last llama_decode() / llama_encode() but BEFORE printing the
// summary, otherwise the records for the final forward pass are lost.
void common_slotted_runtime_flush(common_slot_runtime & rt);

// Drop file descriptors / join lingering threads. Safe to call multiple times.
void common_slotted_runtime_shutdown(common_slot_runtime & rt);

// ggml_backend_sched_eval_callback adapter. Pass as `params.cb_eval`,
// with `&runtime` as `cb_eval_user_data`. Safe to set before the runtime is
// ready -- it will simply early-return until `rt.ready == true`.
bool common_slotted_cb_eval(ggml_tensor * t, bool ask, void * user_data);

// Print per-record [slot-run] lines (if requested) and the aggregated summary.
// Also computes io_hidden_percent, avg compute/load/wait. Idempotent.
void common_slotted_print_summary(const common_slot_runtime & rt);

// Write the plan + per-slot records + summary as JSON to PATH. Returns true on success.
bool common_slotted_write_summary_json(const common_slot_runtime & rt, const std::string & path);
