#include "slotted-inference.h"

#include "common.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <numeric>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;

// Parse a GGUF tensor name of the form "blk.<N>.<rest>".
// Mirrors the logic in src/llama-model.cpp::slotted_parse_blk_prefix so we
// can categorize tensors without having a loaded llama_model handy.
bool parse_blk_prefix_name(const std::string & name, int & out_il, std::string & out_rest) {
    static const std::string prefix = "blk.";
    if (name.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    size_t i = prefix.size();
    int il = 0;
    bool has_digit = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        il = il * 10 + (name[i] - '0');
        ++i;
        has_digit = true;
    }
    if (!has_digit || i >= name.size() || name[i] != '.') {
        return false;
    }
    out_il   = il;
    out_rest = name.substr(i + 1);
    return true;
}

// Append one slot description with its accumulated size info.
void finalize_slot(common_slot_plan & plan, common_slot_desc & cur) {
    cur.other_bytes = (cur.weight_bytes > cur.attn_bytes + cur.ffn_bytes)
        ? cur.weight_bytes - cur.attn_bytes - cur.ffn_bytes
        : 0;
    plan.slots.push_back(cur);
    cur = common_slot_desc{};
    cur.slot_id     = (int) plan.slots.size();
    cur.layer_start = plan.slots.back().layer_end + 1;
    cur.layer_end   = cur.layer_start;
}

} // namespace

common_slot_plan common_slotted_build_plan(const llama_model * model, const common_params & params) {
    common_slot_plan plan;
    if (!params.slotted_test) {
        return plan;
    }
    if (model == nullptr) {
        return plan;
    }

    plan.enabled           = true;
    plan.n_layer           = llama_model_n_layer(model);
    plan.slot_layers_used  = params.slot_layers;
    plan.slot_size_mb_used = (params.slot_layers > 0) ? 0 : params.slot_size_mb;
    plan.non_layer_bytes   = llama_model_non_layer_weight_bytes(model);

    if (plan.n_layer <= 0) {
        return plan;
    }

    // Plan by fixed number of layers.
    if (params.slot_layers > 0) {
        common_slot_desc cur;
        cur.slot_id     = 0;
        cur.layer_start = 0;
        cur.layer_end   = 0;
        int layers_in_cur = 0;
        for (int il = 0; il < plan.n_layer; ++il) {
            uint64_t attn = 0, ffn = 0;
            const uint64_t tot = llama_model_layer_weight_bytes(model, il, &attn, &ffn);
            cur.weight_bytes += tot;
            cur.attn_bytes   += attn;
            cur.ffn_bytes    += ffn;
            cur.layer_end     = il;
            ++layers_in_cur;
            if (layers_in_cur == params.slot_layers && il + 1 < plan.n_layer) {
                finalize_slot(plan, cur);
                layers_in_cur = 0;
            }
        }
        // emit the last (possibly short) slot
        cur.other_bytes = (cur.weight_bytes > cur.attn_bytes + cur.ffn_bytes)
            ? cur.weight_bytes - cur.attn_bytes - cur.ffn_bytes
            : 0;
        plan.slots.push_back(cur);
        return plan;
    }

    // Plan by approximate slot size in MiB. Greedy: keep adding layers until
    // the next layer would exceed the target, then start a new slot.
    if (params.slot_size_mb > 0) {
        const uint64_t target_bytes = (uint64_t) params.slot_size_mb * MiB;
        common_slot_desc cur;
        cur.slot_id     = 0;
        cur.layer_start = 0;
        cur.layer_end   = 0;
        for (int il = 0; il < plan.n_layer; ++il) {
            uint64_t attn = 0, ffn = 0;
            const uint64_t tot = llama_model_layer_weight_bytes(model, il, &attn, &ffn);

            const bool slot_has_content = cur.weight_bytes > 0;
            const bool would_overflow   = cur.weight_bytes + tot > target_bytes;
            if (slot_has_content && would_overflow) {
                finalize_slot(plan, cur);
            }
            cur.weight_bytes += tot;
            cur.attn_bytes   += attn;
            cur.ffn_bytes    += ffn;
            cur.layer_end     = il;
        }
        cur.other_bytes = (cur.weight_bytes > cur.attn_bytes + cur.ffn_bytes)
            ? cur.weight_bytes - cur.attn_bytes - cur.ffn_bytes
            : 0;
        plan.slots.push_back(cur);
        return plan;
    }

    // Fallback: single slot covering everything (caller already warned).
    common_slot_desc cur;
    cur.slot_id     = 0;
    cur.layer_start = 0;
    cur.layer_end   = plan.n_layer - 1;
    for (int il = 0; il < plan.n_layer; ++il) {
        uint64_t attn = 0, ffn = 0;
        const uint64_t tot = llama_model_layer_weight_bytes(model, il, &attn, &ffn);
        cur.weight_bytes += tot;
        cur.attn_bytes   += attn;
        cur.ffn_bytes    += ffn;
    }
    cur.other_bytes = (cur.weight_bytes > cur.attn_bytes + cur.ffn_bytes)
        ? cur.weight_bytes - cur.attn_bytes - cur.ffn_bytes
        : 0;
    plan.slots.push_back(cur);
    return plan;
}

void common_slotted_print_plan(const common_slot_plan & plan) {
    if (!plan.enabled) {
        return;
    }

    // experimental: print at LOG_LEVEL_OUTPUT so the plan is visible regardless
    // of the host tool's verbosity (llama-cli forces ERROR by default).
    LOG("[slot-plan]\n");
    LOG("  n_layer              = %d\n", plan.n_layer);
    if (plan.slot_layers_used > 0) {
        LOG("  grouping             = slot_layers=%d\n", plan.slot_layers_used);
    } else if (plan.slot_size_mb_used > 0) {
        LOG("  grouping             = slot_size_mb=%d\n", plan.slot_size_mb_used);
    } else {
        LOG("  grouping             = none (single slot)\n");
    }
    LOG("  non_layer_weight_mb  = %.2f (embeddings + final norm + lm_head)\n",
            (double) plan.non_layer_bytes / (double) MiB);
    LOG("  total_slots          = %zu\n", plan.slots.size());

    uint64_t max_slot_bytes = 0;
    for (const auto & s : plan.slots) {
        LOG("[slot %d]\n", s.slot_id);
        LOG("  layers               = %d..%d (%d)\n",
                s.layer_start, s.layer_end, s.layer_end - s.layer_start + 1);
        LOG("  weight_mb            = %.2f\n", (double) s.weight_bytes / (double) MiB);
        LOG("  attn_weight_mb       = %.2f\n", (double) s.attn_bytes   / (double) MiB);
        LOG("  ffn_weight_mb        = %.2f\n", (double) s.ffn_bytes    / (double) MiB);
        LOG("  other_weight_mb      = %.2f\n", (double) s.other_bytes  / (double) MiB);
        if (s.weight_bytes > max_slot_bytes) {
            max_slot_bytes = s.weight_bytes;
        }
    }
    LOG("[slot-plan-summary]\n");
    LOG("  estimated_slot_ram_mb = %.2f (largest slot)\n",
            (double) max_slot_bytes / (double) MiB);
    LOG("  estimated_double_buffer_ram_mb = %.2f (2x largest + non-layer)\n",
            (double)(2 * max_slot_bytes + plan.non_layer_bytes) / (double) MiB);
}

bool common_slotted_write_plan_json(const common_slot_plan & plan, const std::string & path) {
    if (!plan.enabled || path.empty()) {
        return false;
    }
    std::ofstream out(path);
    if (!out.is_open()) {
        LOG_ERR("slotted: failed to open '%s' for JSON output\n", path.c_str());
        return false;
    }
    out << "{\n";
    out << "  \"enabled\": true,\n";
    out << "  \"n_layer\": " << plan.n_layer << ",\n";
    out << "  \"slot_layers_used\": " << plan.slot_layers_used << ",\n";
    out << "  \"slot_size_mb_used\": " << plan.slot_size_mb_used << ",\n";
    out << "  \"non_layer_bytes\": " << plan.non_layer_bytes << ",\n";
    out << "  \"slots\": [\n";
    for (size_t i = 0; i < plan.slots.size(); ++i) {
        const auto & s = plan.slots[i];
        out << "    {";
        out << " \"slot_id\": " << s.slot_id << ",";
        out << " \"layer_start\": " << s.layer_start << ",";
        out << " \"layer_end\": " << s.layer_end << ",";
        out << " \"weight_bytes\": " << s.weight_bytes << ",";
        out << " \"attn_bytes\": " << s.attn_bytes << ",";
        out << " \"ffn_bytes\": " << s.ffn_bytes << ",";
        out << " \"other_bytes\": " << s.other_bytes;
        out << " }";
        if (i + 1 < plan.slots.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return out.good();
}

// =========================================================================
// FASE 3 — runtime
// =========================================================================

namespace {

// Parse the trailing "-<digits>" of a tensor name. Returns the layer index,
// or -1 if the name has no such suffix. Examples:
//   "attn_norm-12"      -> 12
//   "Qcur-0"            -> 0
//   "attn_inp_kq_mask"  -> -1   (no '-')
//   "logits_seq_5"      -> -1   (uses '_' not '-')
int parse_layer_suffix(const char * name) {
    if (name == nullptr) return -1;
    const size_t n = std::strlen(name);
    if (n < 2) return -1;
    size_t i = n;
    int    il = 0;
    int    mult = 1;
    bool   saw_digit = false;
    while (i > 0 && name[i - 1] >= '0' && name[i - 1] <= '9') {
        il += (name[i - 1] - '0') * mult;
        mult *= 10;
        --i;
        saw_digit = true;
    }
    if (!saw_digit || i == 0 || name[i - 1] != '-') return -1;
    return il;
}

void start_prefetch_if_needed(common_slot_runtime & rt, int slot_idx) {
    if (!rt.simulate_prefetch)                                     return;
    if (slot_idx < 0 || slot_idx >= (int) rt.plan.slots.size())    return;
    if (rt.prefetch_active && rt.prefetch_thread.joinable()) {
        // shouldn't normally happen; defensively join the previous worker
        rt.prefetch_thread.join();
        rt.prefetch_active = false;
    }

    const uint64_t bytes  = rt.plan.slots[slot_idx].weight_bytes;
    const double   bw_bps = rt.bandwidth_mb_s * 1024.0 * 1024.0;
    const auto     mode   = rt.prefetch_mode;
    const int      fd     = rt.dummy_fd;
    const uint64_t fsize  = rt.dummy_file_size;

    rt.prefetch_target_slot = slot_idx;
    rt.prefetch_start_time  = std::chrono::steady_clock::now();
    rt.prefetch_active      = true;
    rt.prefetch_actual_ms   = 0.0;
    double * out_actual_ms  = &rt.prefetch_actual_ms;

    rt.prefetch_thread = std::thread([bytes, bw_bps, mode, fd, fsize, out_actual_ms]() {
        const auto t0 = std::chrono::steady_clock::now();
        if (mode == common_slot_prefetch_mode::dummy_read && fd >= 0 && fsize > 0) {
            constexpr size_t CHUNK = 1u << 20; // 1 MiB
            std::vector<char> buf(CHUNK);
            uint64_t read_so_far = 0;
            uint64_t off = 0;
            while (read_so_far < bytes) {
                const size_t to_read = (size_t) std::min<uint64_t>(CHUNK, bytes - read_so_far);
                const off_t  this_off = (off_t)(off % fsize);
                const ssize_t n = pread(fd, buf.data(), to_read, this_off);
                if (n <= 0) break;
                off         += (uint64_t) n;
                read_so_far += (uint64_t) n;
            }
        } else if (bw_bps > 0.0 && bytes > 0) {
            const double load_s = (double) bytes / bw_bps;
            std::this_thread::sleep_for(std::chrono::duration<double>(load_s));
        }
        const auto t1 = std::chrono::steady_clock::now();
        *out_actual_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    });
}

void commit_pass_records(common_slot_runtime & rt) {
    for (size_t s = 0; s < rt.per_slot_pass.size(); ++s) {
        const auto & p = rt.per_slot_pass[s];
        if (p.compute_ms == 0.0 && p.async_load_ms == 0.0 && p.wait_next_ms == 0.0) continue;
        common_slot_run_record rec;
        rec.token_index   = rt.token_index;
        rec.is_prefill    = (rt.token_index == 0);
        rec.slot_id       = (int) s;
        rec.layer_start   = rt.plan.slots[s].layer_start;
        rec.layer_end     = rt.plan.slots[s].layer_end;
        rec.weight_bytes  = rt.plan.slots[s].weight_bytes;
        rec.compute_ms    = p.compute_ms;
        rec.async_load_ms = p.async_load_ms;
        rec.wait_next_ms  = p.wait_next_ms;
        rt.records.push_back(rec);
        if (rt.verbose_log) {
            const double MiB = 1024.0 * 1024.0;
            LOG("[slot-run] phase=%s token=%d slot=%d layers=%d..%d "
                "weight_mb=%.2f compute_ms=%.2f async_load_ms=%.2f wait_next_ms=%.2f\n",
                rec.is_prefill ? "prefill" : "decode",
                rec.token_index, rec.slot_id, rec.layer_start, rec.layer_end,
                (double) rec.weight_bytes / MiB,
                rec.compute_ms, rec.async_load_ms, rec.wait_next_ms);
        }
    }
}

void finalize_pass(common_slot_runtime & rt) {
    if (rt.current_slot < 0) return;
    const auto now = std::chrono::steady_clock::now();
    const double slot_ms = std::chrono::duration<double, std::milli>(now - rt.slot_enter_time).count();
    if (rt.current_slot < (int) rt.per_slot_pass.size()) {
        rt.per_slot_pass[rt.current_slot].compute_ms += slot_ms;
    }
    if (rt.prefetch_active && rt.prefetch_thread.joinable()) {
        rt.prefetch_thread.join();
        rt.prefetch_active = false;
    }
    commit_pass_records(rt);
}

} // namespace

bool common_slotted_runtime_init(common_slot_runtime & rt,
                                 common_slot_plan && plan,
                                 const common_params & params,
                                 const std::string & model_path) {
    rt.plan              = std::move(plan);
    rt.simulate_prefetch = params.slotted_simulate_prefetch;
    rt.verbose_log       = params.slotted_log;
    rt.bandwidth_mb_s    = (double) params.slotted_bandwidth_mb_s;
    rt.prefetch_mode     = (params.slotted_prefetch_mode == "dummy-read")
                              ? common_slot_prefetch_mode::dummy_read
                              : common_slot_prefetch_mode::sleep;

    if (!rt.plan.enabled) {
        return false;
    }

    // build layer_to_slot index
    rt.layer_to_slot.assign(rt.plan.n_layer, -1);
    for (const auto & s : rt.plan.slots) {
        for (int il = s.layer_start; il <= s.layer_end && il < rt.plan.n_layer; ++il) {
            rt.layer_to_slot[il] = s.slot_id;
        }
    }
    rt.per_slot_pass.assign(rt.plan.slots.size(), {});

    // open the model file in a SEPARATE fd for dummy-read mode. This is a
    // distinct kernel file object so reads don't interfere with the mmap.
    if (rt.simulate_prefetch
        && rt.prefetch_mode == common_slot_prefetch_mode::dummy_read
        && !model_path.empty()) {
        rt.dummy_fd = ::open(model_path.c_str(), O_RDONLY);
        if (rt.dummy_fd >= 0) {
            struct stat st{};
            if (::fstat(rt.dummy_fd, &st) == 0 && st.st_size > 0) {
                rt.dummy_file_size = (uint64_t) st.st_size;
            } else {
                ::close(rt.dummy_fd);
                rt.dummy_fd = -1;
            }
        }
        if (rt.dummy_fd < 0) {
            LOG("slotted: dummy-read mode requested but failed to open '%s'; "
                "falling back to sleep mode\n", model_path.c_str());
            rt.prefetch_mode = common_slot_prefetch_mode::sleep;
        }
    }

    rt.ready = true;
    return true;
}

void common_slotted_runtime_flush(common_slot_runtime & rt) {
    if (!rt.ready || !rt.plan.enabled) return;
    finalize_pass(rt);
    // reset live-pass state so a subsequent run starts clean
    rt.current_slot       = -1;
    rt.last_seen_layer    = -1;
    rt.first_node_in_pass = true;
    rt.per_slot_pass.assign(rt.plan.slots.size(), {});
}

void common_slotted_runtime_shutdown(common_slot_runtime & rt) {
    if (rt.prefetch_active && rt.prefetch_thread.joinable()) {
        rt.prefetch_thread.join();
        rt.prefetch_active = false;
    }
    if (rt.dummy_fd >= 0) {
        ::close(rt.dummy_fd);
        rt.dummy_fd = -1;
    }
    rt.ready = false;
}

bool common_slotted_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;

    auto * rt = static_cast<common_slot_runtime *>(user_data);
    if (rt == nullptr || !rt->ready || !rt->plan.enabled) return true;

    const auto now = std::chrono::steady_clock::now();
    const int  layer = parse_layer_suffix(t->name);
    if (layer < 0 || layer >= (int) rt->layer_to_slot.size()) {
        // non-layer node: keep going, attribute nothing
        return true;
    }
    const int new_slot = rt->layer_to_slot[layer];
    if (new_slot < 0) return true;

    // detect new forward pass via layer regression.
    if (layer < rt->last_seen_layer) {
        finalize_pass(*rt);
        rt->token_index++;
        rt->first_node_in_pass = true;
        rt->current_slot       = -1;
        rt->per_slot_pass.assign(rt->plan.slots.size(), {});
    }
    rt->last_seen_layer = layer;

    if (rt->first_node_in_pass) {
        rt->first_node_in_pass = false;
        rt->current_slot       = new_slot;
        rt->slot_enter_time    = now;
        start_prefetch_if_needed(*rt, new_slot + 1);
        return true;
    }

    if (new_slot != rt->current_slot) {
        // slot transition: charge elapsed time to the slot we are LEAVING.
        const double slot_ms = std::chrono::duration<double, std::milli>(now - rt->slot_enter_time).count();
        if (rt->current_slot < (int) rt->per_slot_pass.size()) {
            rt->per_slot_pass[rt->current_slot].compute_ms += slot_ms;
        }

        // join the prefetch for the slot we are ENTERING.
        if (rt->prefetch_active && rt->prefetch_target_slot == new_slot && rt->prefetch_thread.joinable()) {
            const auto wait_start = std::chrono::steady_clock::now();
            rt->prefetch_thread.join();
            const auto wait_end = std::chrono::steady_clock::now();
            const double wait_ms = std::chrono::duration<double, std::milli>(wait_end - wait_start).count();
            // load_ms is the actual time the worker spent doing I/O / sleeping
            // (recorded inside the worker right before it returned). The
            // wall-clock from prefetch_start to wait_end is NOT the same thing:
            // most of that wall-clock is the compute hiding the load behind it.
            const double load_ms = rt->prefetch_actual_ms;
            if (rt->current_slot < (int) rt->per_slot_pass.size()) {
                rt->per_slot_pass[rt->current_slot].async_load_ms = load_ms;
                rt->per_slot_pass[rt->current_slot].wait_next_ms  = wait_ms;
            }
            rt->prefetch_active = false;
        }

        // start prefetch for the slot AFTER the one we are entering.
        start_prefetch_if_needed(*rt, new_slot + 1);

        rt->current_slot    = new_slot;
        rt->slot_enter_time = now;
    }

    return true;
}

void common_slotted_print_summary(const common_slot_runtime & rt) {
    if (!rt.plan.enabled) return;

    // finalize the very last pass if cb_eval never saw a layer regression.
    // (in practice we call this AFTER inference is fully done, so any in-flight
    // pass should already be committed; but if not, we have nothing actionable
    // to do here because this is a `const` view.)

    if (rt.records.empty()) {
        LOG("[slotted-summary]\n");
        LOG("  no per-slot records captured -- did the model run at all?\n");
        return;
    }

    const double MiB = 1024.0 * 1024.0;

    int    n_records         = 0;
    double sum_compute_ms    = 0.0;
    double sum_async_load_ms = 0.0;
    double sum_wait_next_ms  = 0.0;
    int    n_prefill         = 0;
    int    n_decode_records  = 0;
    double prefill_total_ms  = 0.0;
    double decode_total_ms   = 0.0;
    uint64_t max_slot_bytes  = 0;
    int    transitions_with_prefetch = 0;

    for (const auto & r : rt.records) {
        ++n_records;
        sum_compute_ms    += r.compute_ms;
        sum_async_load_ms += r.async_load_ms;
        sum_wait_next_ms  += r.wait_next_ms;
        if (r.async_load_ms > 0.0) ++transitions_with_prefetch;
        if (r.is_prefill) {
            ++n_prefill;
            prefill_total_ms += r.compute_ms;
        } else {
            ++n_decode_records;
            decode_total_ms += r.compute_ms;
        }
        if (r.weight_bytes > max_slot_bytes) max_slot_bytes = r.weight_bytes;
    }

    const double avg_compute = n_records > 0 ? sum_compute_ms / n_records : 0.0;
    const double avg_load    = transitions_with_prefetch > 0 ? sum_async_load_ms / transitions_with_prefetch : 0.0;
    const double avg_wait    = transitions_with_prefetch > 0 ? sum_wait_next_ms  / transitions_with_prefetch : 0.0;
    const double io_hidden_percent = (sum_async_load_ms > 0.0)
        ? (1.0 - (sum_wait_next_ms / sum_async_load_ms)) * 100.0
        : 0.0;

    LOG("[slotted-summary]\n");
    LOG("  total_records              = %d (= n_passes x n_slots, 1 record per slot per pass)\n", n_records);
    LOG("  prefill_records            = %d (sum_compute_ms = %.2f)\n", n_prefill, prefill_total_ms);
    LOG("  decode_records             = %d (sum_compute_ms = %.2f)\n", n_decode_records, decode_total_ms);
    LOG("  avg_compute_ms_per_slot    = %.2f\n", avg_compute);
    if (rt.simulate_prefetch) {
        LOG("  avg_async_load_ms          = %.2f (mode=%s, bandwidth_mb_s=%.0f)\n",
            avg_load,
            rt.prefetch_mode == common_slot_prefetch_mode::dummy_read ? "dummy-read" : "sleep",
            rt.bandwidth_mb_s);
        LOG("  avg_wait_next_ms           = %.2f\n", avg_wait);
        LOG("  io_hidden_percent          = %.2f%%\n", io_hidden_percent);
    } else {
        LOG("  prefetch simulation        = disabled (pass --slotted-simulate-prefetch)\n");
    }
    LOG("  estimated_weight_ram_mb    = %.2f (largest slot)\n", (double) max_slot_bytes / MiB);
    LOG("  estimated_double_buffer_ram_mb = %.2f (2x largest + non-layer)\n",
        (double)(2 * max_slot_bytes + rt.plan.non_layer_bytes) / MiB);
}

bool common_slotted_write_summary_json(const common_slot_runtime & rt, const std::string & path) {
    if (!rt.plan.enabled || path.empty()) return false;
    std::ofstream out(path);
    if (!out.is_open()) {
        LOG_ERR("slotted: failed to open '%s' for JSON output\n", path.c_str());
        return false;
    }

    double sum_compute = 0.0, sum_load = 0.0, sum_wait = 0.0;
    for (const auto & r : rt.records) {
        sum_compute += r.compute_ms;
        sum_load    += r.async_load_ms;
        sum_wait    += r.wait_next_ms;
    }
    const double io_hidden_percent = (sum_load > 0.0) ? (1.0 - sum_wait / sum_load) * 100.0 : 0.0;

    out << "{\n";
    out << "  \"enabled\": true,\n";
    out << "  \"n_layer\": " << rt.plan.n_layer << ",\n";
    out << "  \"slot_layers_used\": "  << rt.plan.slot_layers_used  << ",\n";
    out << "  \"slot_size_mb_used\": " << rt.plan.slot_size_mb_used << ",\n";
    out << "  \"non_layer_bytes\": "   << rt.plan.non_layer_bytes   << ",\n";
    out << "  \"simulate_prefetch\": " << (rt.simulate_prefetch ? "true" : "false") << ",\n";
    out << "  \"bandwidth_mb_s\": "    << rt.bandwidth_mb_s         << ",\n";
    out << "  \"prefetch_mode\": \""
        << (rt.prefetch_mode == common_slot_prefetch_mode::dummy_read ? "dummy-read" : "sleep")
        << "\",\n";
    out << "  \"slots\": [\n";
    for (size_t i = 0; i < rt.plan.slots.size(); ++i) {
        const auto & s = rt.plan.slots[i];
        out << "    {";
        out << " \"slot_id\": " << s.slot_id << ",";
        out << " \"layer_start\": " << s.layer_start << ",";
        out << " \"layer_end\": " << s.layer_end << ",";
        out << " \"weight_bytes\": " << s.weight_bytes << ",";
        out << " \"attn_bytes\": " << s.attn_bytes << ",";
        out << " \"ffn_bytes\": " << s.ffn_bytes << ",";
        out << " \"other_bytes\": " << s.other_bytes;
        out << " }";
        if (i + 1 < rt.plan.slots.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"records\": [\n";
    for (size_t i = 0; i < rt.records.size(); ++i) {
        const auto & r = rt.records[i];
        out << "    {";
        out << " \"token_index\": " << r.token_index << ",";
        out << " \"is_prefill\": "  << (r.is_prefill ? "true" : "false") << ",";
        out << " \"slot_id\": "     << r.slot_id << ",";
        out << " \"layer_start\": " << r.layer_start << ",";
        out << " \"layer_end\": "   << r.layer_end   << ",";
        out << " \"weight_bytes\": "<< r.weight_bytes<< ",";
        out << " \"compute_ms\": "  << r.compute_ms  << ",";
        out << " \"async_load_ms\": "<< r.async_load_ms << ",";
        out << " \"wait_next_ms\": "<< r.wait_next_ms;
        out << " }";
        if (i + 1 < rt.records.size()) out << ",";
        out << "\n";
    }
    out << "  ],\n";
    out << "  \"summary\": {\n";
    out << "    \"total_records\": "      << rt.records.size() << ",\n";
    out << "    \"sum_compute_ms\": "     << sum_compute       << ",\n";
    out << "    \"sum_async_load_ms\": "  << sum_load          << ",\n";
    out << "    \"sum_wait_next_ms\": "   << sum_wait          << ",\n";
    out << "    \"io_hidden_percent\": "  << io_hidden_percent << "\n";
    out << "  }\n";
    out << "}\n";
    return out.good();
}

// =========================================================================
// FASE 4A: build slot plan from GGUF metadata (no model load needed).
// =========================================================================

namespace {

// Group `per_layer_total` into slots according to `params.slot_layers` /
// `params.slot_size_mb`. Mirrors the logic of common_slotted_build_plan() but
// works on raw byte arrays rather than a llama_model. Also propagates
// attn / ffn sub-totals so the breakdown matches.
void group_layers_into_slots(common_slot_plan & plan,
                             const std::vector<uint64_t> & per_layer_total,
                             const std::vector<uint64_t> & per_layer_attn,
                             const std::vector<uint64_t> & per_layer_ffn,
                             const common_params & params) {
    const int n_layer = (int) per_layer_total.size();
    if (n_layer <= 0) return;

    auto close_slot = [](common_slot_desc & cur) {
        cur.other_bytes = (cur.weight_bytes > cur.attn_bytes + cur.ffn_bytes)
            ? cur.weight_bytes - cur.attn_bytes - cur.ffn_bytes : 0;
    };

    if (params.slot_layers > 0) {
        common_slot_desc cur;
        cur.slot_id = 0; cur.layer_start = 0; cur.layer_end = 0;
        int in_cur = 0;
        for (int il = 0; il < n_layer; ++il) {
            cur.weight_bytes += per_layer_total[il];
            cur.attn_bytes   += per_layer_attn[il];
            cur.ffn_bytes    += per_layer_ffn[il];
            cur.layer_end     = il;
            ++in_cur;
            if (in_cur == params.slot_layers && il + 1 < n_layer) {
                close_slot(cur);
                plan.slots.push_back(cur);
                cur = common_slot_desc{};
                cur.slot_id     = (int) plan.slots.size();
                cur.layer_start = il + 1;
                cur.layer_end   = il + 1;
                in_cur = 0;
            }
        }
        close_slot(cur);
        plan.slots.push_back(cur);
        return;
    }

    if (params.slot_size_mb > 0) {
        const uint64_t target = (uint64_t) params.slot_size_mb * MiB;
        common_slot_desc cur;
        cur.slot_id = 0; cur.layer_start = 0; cur.layer_end = 0;
        for (int il = 0; il < n_layer; ++il) {
            const bool slot_has_content = cur.weight_bytes > 0;
            const bool would_overflow   = cur.weight_bytes + per_layer_total[il] > target;
            if (slot_has_content && would_overflow) {
                close_slot(cur);
                plan.slots.push_back(cur);
                cur = common_slot_desc{};
                cur.slot_id     = (int) plan.slots.size();
                cur.layer_start = il;
                cur.layer_end   = il;
            }
            cur.weight_bytes += per_layer_total[il];
            cur.attn_bytes   += per_layer_attn[il];
            cur.ffn_bytes    += per_layer_ffn[il];
            cur.layer_end     = il;
        }
        close_slot(cur);
        plan.slots.push_back(cur);
        return;
    }

    // fallback: one big slot
    common_slot_desc cur;
    cur.slot_id = 0; cur.layer_start = 0; cur.layer_end = n_layer - 1;
    for (int il = 0; il < n_layer; ++il) {
        cur.weight_bytes += per_layer_total[il];
        cur.attn_bytes   += per_layer_attn[il];
        cur.ffn_bytes    += per_layer_ffn[il];
    }
    close_slot(cur);
    plan.slots.push_back(cur);
}

} // namespace

common_slot_plan common_slotted_build_plan_from_gguf(const std::string & model_path,
                                                     const common_params & params) {
    common_slot_plan plan;
    if (model_path.empty()) {
        return plan;
    }

    struct gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(model_path.c_str(), gp);
    if (gctx == nullptr) {
        LOG_ERR("slotted: failed to parse GGUF '%s' for slot planning\n", model_path.c_str());
        return plan;
    }

    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    // First pass: discover n_layer.
    int max_il = -1;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        int il = -1;
        std::string rest;
        if (parse_blk_prefix_name(name, il, rest) && il > max_il) {
            max_il = il;
        }
    }
    const int n_layer = max_il + 1;
    if (n_layer <= 0) {
        gguf_free(gctx);
        LOG_ERR("slotted: no blk.<N>.* tensors found in GGUF; cannot plan slots\n");
        return plan;
    }

    // Second pass: accumulate sizes per layer.
    std::vector<uint64_t> per_layer_total(n_layer, 0);
    std::vector<uint64_t> per_layer_attn (n_layer, 0);
    std::vector<uint64_t> per_layer_ffn  (n_layer, 0);
    uint64_t non_layer_bytes = 0;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        const size_t sz   = gguf_get_tensor_size(gctx, i);
        int il = -1;
        std::string rest;
        if (parse_blk_prefix_name(name, il, rest)) {
            if (il < n_layer) {
                per_layer_total[il] += sz;
                if (rest.find("norm") == std::string::npos) {
                    if (rest.compare(0, 4, "attn") == 0) {
                        per_layer_attn[il] += sz;
                    } else if (rest.compare(0, 3, "ffn") == 0) {
                        per_layer_ffn[il] += sz;
                    }
                }
            }
        } else {
            non_layer_bytes += sz;
        }
    }

    gguf_free(gctx);

    plan.enabled           = true;
    plan.n_layer           = n_layer;
    plan.slot_layers_used  = params.slot_layers;
    plan.slot_size_mb_used = (params.slot_layers > 0) ? 0 : params.slot_size_mb;
    plan.non_layer_bytes   = non_layer_bytes;

    group_layers_into_slots(plan, per_layer_total, per_layer_attn, per_layer_ffn, params);

    return plan;
}

bool common_slotted_layer_is_resident(const common_slot_plan & plan, int32_t il, int slots_resident) {
    if (!plan.enabled || il < 0 || slots_resident <= 0) return false;
    const int n = std::min<int>(slots_resident, (int) plan.slots.size());
    for (int i = 0; i < n; ++i) {
        if (il >= plan.slots[i].layer_start && il <= plan.slots[i].layer_end) {
            return true;
        }
    }
    return false;
}

bool common_slotted_layer_filter_cb(int32_t il, void * user_data) {
    if (user_data == nullptr) return true;
    const auto * st = static_cast<const common_slot_filter_state *>(user_data);
    if (st->plan == nullptr) return true;
    return common_slotted_layer_is_resident(*st->plan, il, st->slots_resident);
}
