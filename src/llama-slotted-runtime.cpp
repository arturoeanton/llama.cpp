#include "llama-slotted-runtime.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

// Parse "blk.<N>.<rest>" out of a tensor name. Mirrors the helper used in
// src/llama-model.cpp and common/slotted-inference.cpp.
bool parse_blk(const std::string & name, int & out_il, std::string & out_rest) {
    static const std::string prefix = "blk.";
    if (name.compare(0, prefix.size(), prefix) != 0) return false;
    size_t i = prefix.size();
    int il = 0;
    bool has_digit = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        il = il * 10 + (name[i] - '0');
        ++i;
        has_digit = true;
    }
    if (!has_digit || i >= name.size() || name[i] != '.') return false;
    out_il   = il;
    out_rest = name.substr(i + 1);
    return true;
}

inline double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

int slotted_hot_swap_pool_idx(const slotted_hot_swap_state & st, int slot_idx) {
    // Pin-and-scratch needs at least R=2 to make sense (1 pin + 1 scratch).
    if (st.policy == SLOTTED_POOL_PIN_SCRATCH && st.slots_resident >= 2) {
        // Slots 0 .. R-2 stay pinned to their initial pool index.
        if (slot_idx < st.slots_resident - 1) {
            return slot_idx;
        }
        // Everything else uses the single scratch pool slot at index R-1.
        return st.slots_resident - 1;
    }
    // Default / fallback: round-robin.
    return slot_idx % st.slots_resident;
}

bool slotted_hot_swap_setup(slotted_hot_swap_state & st,
                            llama_model * model,
                            const std::string & gguf_path,
                            const std::vector<std::pair<int,int>> & slot_ranges,
                            int slots_resident,
                            enum slotted_hot_swap_policy_t policy) {
    if (model == nullptr || gguf_path.empty() || slot_ranges.empty() || slots_resident <= 0) {
        LLAMA_LOG_ERROR("%s: invalid arguments\n", __func__);
        return false;
    }
    if (slots_resident > (int) slot_ranges.size()) {
        LLAMA_LOG_ERROR("%s: slots_resident (%d) > total slots (%zu)\n",
                __func__, slots_resident, slot_ranges.size());
        return false;
    }

    st.model           = model;
    st.gguf_path       = gguf_path;
    st.slot_ranges     = slot_ranges;
    st.slots_resident  = slots_resident;
    st.policy          = policy;

    // Open GGUF read-only (separate fd from the loader's mmap/fread).
    st.gguf_fd = ::open(gguf_path.c_str(), O_RDONLY);
    if (st.gguf_fd < 0) {
        LLAMA_LOG_ERROR("%s: open('%s') failed\n", __func__, gguf_path.c_str());
        return false;
    }
    struct stat sb{};
    if (::fstat(st.gguf_fd, &sb) < 0 || sb.st_size <= 0) {
        LLAMA_LOG_ERROR("%s: fstat failed\n", __func__);
        ::close(st.gguf_fd);
        st.gguf_fd = -1;
        return false;
    }
    st.gguf_file_size = (uint64_t) sb.st_size;

    // Read tensor metadata + the data-base offset via gguf_init_from_file.
    // We use no_alloc=true: we just want offsets, not buffers.
    {
        struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
        gguf_context * gctx = gguf_init_from_file(gguf_path.c_str(), gp);
        if (gctx == nullptr) {
            LLAMA_LOG_ERROR("%s: gguf_init_from_file failed\n", __func__);
            ::close(st.gguf_fd);
            st.gguf_fd = -1;
            return false;
        }
        st.gguf_data_base = gguf_get_data_offset(gctx);
        const int64_t n_tensors = gguf_get_n_tensors(gctx);
        st.gguf_tensors.reserve((size_t) n_tensors);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gctx, i);
            const size_t off  = gguf_get_tensor_offset(gctx, i);
            const size_t sz   = gguf_get_tensor_size(gctx, i);
            st.gguf_tensors.emplace(std::string(name), std::make_pair((uint64_t) off, (uint64_t) sz));
        }
        gguf_free(gctx);
    }

    // Snapshot the initial llama_layer state for each pool slot.
    st.pool_initial_layers.resize(slots_resident);
    st.pool_current_slot.assign(slots_resident, 0);
    for (int p = 0; p < slots_resident; ++p) {
        const int il_start = slot_ranges[p].first;
        const int il_end   = slot_ranges[p].second;
        st.pool_current_slot[p] = p;
        auto & dst = st.pool_initial_layers[p];
        dst.reserve(il_end - il_start + 1);
        for (int il = il_start; il <= il_end; ++il) {
            dst.push_back(model->layers[il]);
        }
    }

    const char * policy_name = (policy == SLOTTED_POOL_PIN_SCRATCH) ? "pin-scratch" : "round-robin";
    LLAMA_LOG_INFO("%s: hot-swap state ready: gguf='%s' fd=%d data_off=%llu tensors=%zu pools=%d policy=%s\n",
            __func__, gguf_path.c_str(), st.gguf_fd,
            (unsigned long long) st.gguf_data_base, st.gguf_tensors.size(), slots_resident, policy_name);

    return true;
}

bool slotted_hot_swap_swap_in(slotted_hot_swap_state & st, int slot_idx, int pool_idx) {
    if (st.model == nullptr || st.gguf_fd < 0) {
        LLAMA_LOG_ERROR("%s: state not initialized\n", __func__);
        return false;
    }
    if (slot_idx < 0 || slot_idx >= (int) st.slot_ranges.size()) {
        LLAMA_LOG_ERROR("%s: slot_idx %d out of range\n", __func__, slot_idx);
        return false;
    }
    if (pool_idx < 0 || pool_idx >= st.slots_resident) {
        LLAMA_LOG_ERROR("%s: pool_idx %d out of range [0,%d)\n", __func__, pool_idx, st.slots_resident);
        return false;
    }
    if (st.pool_current_slot[pool_idx] == slot_idx) {
        // Already resident in this pool. No-op.
        LLAMA_LOG_INFO("%s: slot=%d already in pool=%d (no-op)\n", __func__, slot_idx, pool_idx);
        return true;
    }

    const int log_start = st.slot_ranges[slot_idx].first;
    const int log_end   = st.slot_ranges[slot_idx].second;
    const int phy_start = st.slot_ranges[pool_idx].first;
    const int phy_end   = st.slot_ranges[pool_idx].second;
    const int log_size  = log_end - log_start + 1;
    const int phy_size  = phy_end - phy_start + 1;
    if (log_size != phy_size) {
        LLAMA_LOG_ERROR("%s: slot %d has %d layers but pool %d has %d (size mismatch)\n",
                __func__, slot_idx, log_size, pool_idx, phy_size);
        return false;
    }

    LLAMA_LOG_INFO("%s: swapping logical slot %d (layers %d..%d) into pool %d (phys %d..%d)\n",
            __func__, slot_idx, log_start, log_end, pool_idx, phy_start, phy_end);

    // For every tensor in tensors_by_name belonging to a physical layer in this
    // pool, find its logical-slot counterpart in the GGUF and stream its bytes
    // into the pool's tensor buffer.
    const auto & pool_initial = st.pool_initial_layers[pool_idx];

    uint64_t bytes_this_swap = 0;
    int      tensors_touched = 0;
    double   read_ms_this    = 0.0;
    double   set_ms_this     = 0.0;
    constexpr size_t CHUNK   = 1u << 20; // 1 MiB scratch buffer
    std::vector<uint8_t> scratch(CHUNK);

    // For fast lookup of the pool's tensors by physical layer index + suffix,
    // we'll iterate tensors_by_name and pick out the ones with the expected
    // "blk.<phy>.*" prefix. For each, derive the logical name and stream bytes.
    const auto & tensors = st.model->tensors_by_name;

    for (const auto & kv : tensors) {
        const std::string & name = kv.first;
        ggml_tensor * dst_tensor = kv.second;
        int         phy_il = -1;
        std::string suffix;
        if (!parse_blk(name, phy_il, suffix)) continue;
        if (phy_il < phy_start || phy_il > phy_end) continue;

        const int offset_in_pool = phy_il - phy_start;
        const int log_il = log_start + offset_in_pool;

        // Build the logical tensor name: replace "blk.<phy_il>." with "blk.<log_il>."
        char log_name[GGML_MAX_NAME];
        std::snprintf(log_name, sizeof(log_name), "blk.%d.%s", log_il, suffix.c_str());

        auto it = st.gguf_tensors.find(log_name);
        if (it == st.gguf_tensors.end()) {
            // Some tensors may be optional / present only for certain layers (e.g. rope_freqs
            // is non-SWA-only, expert tensors, etc.). If the destination has data but the
            // source doesn't, that's a real problem. If both are absent, fine. For FASE 4A-2b
            // we conservatively warn and skip.
            LLAMA_LOG_WARN("%s: source tensor '%s' not found in GGUF (skipping)\n", __func__, log_name);
            continue;
        }
        const uint64_t src_off  = st.gguf_data_base + it->second.first;
        const uint64_t src_size = it->second.second;

        const size_t dst_size = ggml_nbytes(dst_tensor);
        if (src_size != dst_size) {
            // FASE 4A-2b discovery: Gemma 4 31B has heterogeneous layers (MoE
            // pattern varies). Pool-based hot-swap requires same-shape tensors
            // across slots. For now we skip mismatched tensors with a warning
            // and continue -- the resulting cgraph will compute on partial
            // (stale) data and produce wrong logits. Stage 1 only verifies no
            // crash; correctness requires a different slotting strategy.
            LLAMA_LOG_WARN("%s: size mismatch for '%s' -> '%s': src=%llu dst=%zu (skipping)\n",
                    __func__, log_name, name.c_str(),
                    (unsigned long long) src_size, dst_size);
            continue;
        }

        // Read bytes from GGUF and write into the pool's tensor.
        uint64_t   bytes_done = 0;
        const auto t_read_start = std::chrono::steady_clock::now();
        while (bytes_done < src_size) {
            const size_t to_read = (size_t) std::min<uint64_t>(CHUNK, src_size - bytes_done);
            const ssize_t n = ::pread(st.gguf_fd, scratch.data(), to_read,
                                       (off_t)(src_off + bytes_done));
            if (n <= 0) {
                LLAMA_LOG_ERROR("%s: pread failed for '%s' at off=%llu (%zd)\n",
                        __func__, log_name,
                        (unsigned long long)(src_off + bytes_done), n);
                return false;
            }
            const auto t_set_start = std::chrono::steady_clock::now();
            ggml_backend_tensor_set(dst_tensor, scratch.data(), bytes_done, (size_t) n);
            set_ms_this += ms_since(t_set_start);
            bytes_done += (uint64_t) n;
        }
        read_ms_this += ms_since(t_read_start);

        bytes_this_swap += src_size;
        ++tensors_touched;
    }

    // Rebind: model.layers[log_start..log_end] = pool_initial[pool_idx][0..]
    // After this, the graph builder dereferences model.layers[log_il].wq and gets the
    // tensor pointer that physically holds slot_idx's bytes.
    for (int i = 0; i < log_size; ++i) {
        st.model->layers[log_start + i] = pool_initial[i];
    }

    // Mark the previously-resident slot as no longer in this pool (its layers'
    // model.layers[...] still reference the same tensors, whose data we just
    // overwrote -- those slots must be hot-swapped back before being computed).
    st.pool_current_slot[pool_idx] = slot_idx;

    // Stats.
    st.total_bytes_read += bytes_this_swap;
    st.total_read_ms    += read_ms_this;
    st.total_set_ms     += set_ms_this;
    st.total_swaps      += 1;

    LLAMA_LOG_INFO("%s: slot=%d pool=%d tensors=%d bytes=%llu read_ms=%.2f set_ms=%.2f\n",
            __func__, slot_idx, pool_idx, tensors_touched,
            (unsigned long long) bytes_this_swap, read_ms_this, set_ms_this);

    return true;
}

// ---------------------------------------------------------------------------
// Public C API wrappers.
// ---------------------------------------------------------------------------

extern "C" {

struct llama_slotted_hot_swap * llama_slotted_hot_swap_init(
        struct llama_model * model,
                const char * gguf_path,
                    int32_t  slot_layers,
                    int32_t  slot_size_mb,
                    int32_t  slots_resident,
                    int32_t  policy) {
    if (model == nullptr || gguf_path == nullptr) {
        LLAMA_LOG_ERROR("%s: null args\n", __func__);
        return nullptr;
    }
    const int n_layer = llama_model_n_layer(model);
    if (n_layer <= 0) {
        LLAMA_LOG_ERROR("%s: n_layer <= 0\n", __func__);
        return nullptr;
    }

    // Build per-layer weight totals from GGUF metadata directly. We cannot
    // use llama_model_layer_weight_bytes() here because the model was loaded
    // with the slotted filter active and reports 0 bytes for non-resident
    // layers. The GGUF has all the metadata for every layer.
    std::vector<uint64_t> per_layer(n_layer, 0);
    {
        struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
        gguf_context * gctx = gguf_init_from_file(gguf_path, gp);
        if (gctx == nullptr) {
            LLAMA_LOG_ERROR("%s: gguf_init_from_file('%s') failed for plan\n", __func__, gguf_path);
            return nullptr;
        }
        const int64_t n_tensors = gguf_get_n_tensors(gctx);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(gctx, i);
            int         il = -1;
            std::string suffix;
            if (parse_blk(name, il, suffix) && il < n_layer) {
                per_layer[il] += gguf_get_tensor_size(gctx, i);
            }
        }
        gguf_free(gctx);
    }

    std::vector<std::pair<int,int>> ranges;
    if (slot_layers > 0) {
        for (int il = 0; il < n_layer; il += slot_layers) {
            const int end = std::min<int>(il + slot_layers, n_layer) - 1;
            ranges.emplace_back(il, end);
        }
    } else if (slot_size_mb > 0) {
        const uint64_t target = (uint64_t) slot_size_mb * 1024ull * 1024ull;
        uint64_t cur = 0;
        int      st_start = 0;
        for (int il = 0; il < n_layer; ++il) {
            const uint64_t lb = per_layer[il];
            if (cur > 0 && cur + lb > target) {
                ranges.emplace_back(st_start, il - 1);
                st_start = il;
                cur = 0;
            }
            cur += lb;
        }
        ranges.emplace_back(st_start, n_layer - 1);
    } else {
        ranges.emplace_back(0, n_layer - 1);
    }

    LLAMA_LOG_INFO("%s: built %zu slot ranges from GGUF metadata\n", __func__, ranges.size());
    for (size_t i = 0; i < ranges.size(); ++i) {
        LLAMA_LOG_INFO("%s:   slot[%zu] = layers %d..%d\n", __func__, i, ranges[i].first, ranges[i].second);
    }

    enum slotted_hot_swap_policy_t pol =
        (policy == 1) ? SLOTTED_POOL_PIN_SCRATCH : SLOTTED_POOL_ROUND_ROBIN;

    auto * hs = new llama_slotted_hot_swap;
    if (!slotted_hot_swap_setup(hs->st, model, gguf_path, ranges, slots_resident, pol)) {
        delete hs;
        return nullptr;
    }
    return hs;
}

int32_t llama_slotted_hot_swap_swap_in(
        struct llama_slotted_hot_swap * hs,
                              int32_t   slot_idx,
                              int32_t   pool_idx) {
    if (hs == nullptr) return -1;
    return slotted_hot_swap_swap_in(hs->st, slot_idx, pool_idx) ? 0 : -1;
}

void llama_slotted_hot_swap_free(struct llama_slotted_hot_swap * hs) {
    if (hs == nullptr) return;
    slotted_hot_swap_destroy(hs->st);
    delete hs;
}

} // extern "C"

void slotted_hot_swap_destroy(slotted_hot_swap_state & st) {
    if (st.gguf_fd >= 0) {
        ::close(st.gguf_fd);
        st.gguf_fd = -1;
    }
    // Best-effort restore: put the initial layer bindings back so that any
    // subsequent traversal of model.layers[] sees the original (potentially
    // stale-data) pool layers rather than dangling logical-slot rebinds.
    for (int p = 0; p < st.slots_resident; ++p) {
        const int il_start = st.slot_ranges[p].first;
        const int il_end   = st.slot_ranges[p].second;
        const auto & src = st.pool_initial_layers[p];
        for (int i = 0; i <= il_end - il_start; ++i) {
            st.model->layers[il_start + i] = src[i];
        }
    }
    st.gguf_tensors.clear();
    st.pool_initial_layers.clear();
    st.pool_current_slot.clear();
    st.model = nullptr;
}
