#include "llama-slotted-runtime.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
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

// Recompute nb[] for a row-major dense tensor whose `type` and `ne[]` are set.
// This mirrors the convention ggml uses for tensors with quant blocks along
// dim 0: nb[0] = type_size, nb[1] = nelems_row / block_size * type_size, etc.
void recompute_nb_for_type(ggml_tensor * t) {
    const enum ggml_type ty = (enum ggml_type) t->type;
    const int64_t blck = ggml_blck_size(ty);
    const size_t  tsz  = ggml_type_size(ty);
    if (blck <= 0 || tsz == 0) return;
    t->nb[0] = tsz;
    t->nb[1] = (size_t)((t->ne[0] / blck) * tsz);
    for (int d = 2; d < GGML_MAX_DIMS; ++d) {
        t->nb[d] = t->nb[d - 1] * (size_t) t->ne[d - 1];
    }
}

// Discover all shards for a possibly-split GGUF path. Returns a list with at
// least the input path (if not a split file) or every shard path (if split).
// On any parse / open / metadata error we return a single-element vector
// containing just the original path and let the caller proceed -- if it's not
// actually a split it'll just look like a 1-shard model.
std::vector<std::string> discover_shards(const std::string & path) {
    std::vector<std::string> out;

    // Open the user-supplied path, read split metadata.
    struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gp);
    if (gctx == nullptr) {
        out.push_back(path);
        return out;
    }
    uint16_t n_split = 0;
    {
        const int kid = gguf_find_key(gctx, "split.count");
        if (kid >= 0) {
            n_split = gguf_get_val_u16(gctx, kid);
        }
    }
    uint16_t this_idx = 0;
    {
        const int kid = gguf_find_key(gctx, "split.no");
        if (kid >= 0) {
            this_idx = gguf_get_val_u16(gctx, kid);
        }
    }
    gguf_free(gctx);

    if (n_split <= 1) {
        out.push_back(path);
        return out;
    }

    // Derive the prefix from the input path using llama's helper, then
    // synthesise each shard path. We assume the caller passed shard 0.
    constexpr size_t BUF = 4096;
    std::vector<char> buf(BUF, 0);
    int pre_len = llama_split_prefix(buf.data(), buf.size(), path.c_str(),
                                     (int32_t) this_idx, (int32_t) n_split);
    if (pre_len <= 0) {
        LLAMA_LOG_WARN("slotted: split.count=%u but llama_split_prefix failed on '%s'; "
                       "falling back to single-shard\n", (unsigned) n_split, path.c_str());
        out.push_back(path);
        return out;
    }
    const std::string prefix(buf.data(), (size_t) pre_len);

    out.reserve((size_t) n_split);
    for (int32_t i = 0; i < (int32_t) n_split; ++i) {
        int n = llama_split_path(buf.data(), buf.size(), prefix.c_str(), i, (int32_t) n_split);
        if (n <= 0) {
            LLAMA_LOG_WARN("slotted: llama_split_path failed for idx=%d / %u\n",
                           i, (unsigned) n_split);
            // Best effort: fall back to single shard.
            out.clear();
            out.push_back(path);
            return out;
        }
        out.emplace_back(buf.data(), (size_t) n);
    }

    LLAMA_LOG_INFO("slotted: discovered %u shard(s) for '%s'\n", (unsigned) n_split, path.c_str());
    return out;
}

// Snapshot the original (loader-allocated) metadata for each pool tensor so
// destroy() can restore. Stores type, nb, data pointer keyed by tensor.
void snapshot_pool_tensor_metadata(slotted_hot_swap_state & st) {
    for (int p = 0; p < st.slots_resident; ++p) {
        const int il_start = st.slot_ranges[p].first;
        const int il_end   = st.slot_ranges[p].second;

        for (const auto & kv : st.model->tensors_by_name) {
            const std::string & name = kv.first;
            ggml_tensor * t = kv.second;
            int il = -1;
            std::string suffix;
            if (!parse_blk(name, il, suffix)) continue;
            if (il < il_start || il > il_end)  continue;

            st.initial_alloc_size[t] = ggml_nbytes(t);
            st.initial_type[t]       = (int) t->type;
            std::array<size_t, GGML_MAX_DIMS> nb{};
            for (int d = 0; d < GGML_MAX_DIMS; ++d) nb[d] = t->nb[d];
            st.initial_nb[t]   = nb;
            st.initial_data[t] = t->data;
        }
    }
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
    st.slot_ranges     = slot_ranges;
    st.slots_resident  = slots_resident;
    st.policy          = policy;

    // Discover all shards (single-file → 1 entry).
    st.gguf_paths = discover_shards(gguf_path);
    st.gguf_fds.assign(st.gguf_paths.size(), -1);
    st.gguf_data_bases.assign(st.gguf_paths.size(), 0);

    // Open each shard read-only and build the unified tensor map.
    for (size_t s = 0; s < st.gguf_paths.size(); ++s) {
        const std::string & p = st.gguf_paths[s];
        st.gguf_fds[s] = ::open(p.c_str(), O_RDONLY);
        if (st.gguf_fds[s] < 0) {
            LLAMA_LOG_ERROR("%s: open('%s') failed for shard %zu\n", __func__, p.c_str(), s);
            // Cleanup partial state.
            for (size_t k = 0; k < s; ++k) {
                if (st.gguf_fds[k] >= 0) ::close(st.gguf_fds[k]);
                st.gguf_fds[k] = -1;
            }
            return false;
        }
        struct stat sb{};
        if (::fstat(st.gguf_fds[s], &sb) < 0 || sb.st_size <= 0) {
            LLAMA_LOG_ERROR("%s: fstat failed on shard %zu ('%s')\n", __func__, s, p.c_str());
            ::close(st.gguf_fds[s]);
            st.gguf_fds[s] = -1;
            return false;
        }

        struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
        gguf_context * gctx = gguf_init_from_file(p.c_str(), gp);
        if (gctx == nullptr) {
            LLAMA_LOG_ERROR("%s: gguf_init_from_file failed on shard %zu ('%s')\n",
                            __func__, s, p.c_str());
            return false;
        }
        st.gguf_data_bases[s] = gguf_get_data_offset(gctx);

        const int64_t n_tensors = gguf_get_n_tensors(gctx);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char *      name = gguf_get_tensor_name(gctx, i);
            const size_t      off  = gguf_get_tensor_offset(gctx, i);
            const size_t      sz   = gguf_get_tensor_size(gctx, i);
            const enum ggml_type tt = gguf_get_tensor_type(gctx, i);

            gguf_tensor_loc loc;
            loc.shard_idx         = (int) s;
            loc.off_in_shard_data = (uint64_t) off;
            loc.size              = (uint64_t) sz;
            loc.type              = (int) tt;
            // First wins (shouldn't collide across shards since the loader
            // forbids duplicates; if it does, log and ignore the dup).
            auto ins = st.gguf_tensors.emplace(std::string(name), loc);
            if (!ins.second) {
                LLAMA_LOG_WARN("%s: tensor '%s' present in multiple shards; using shard %d\n",
                               __func__, name, ins.first->second.shard_idx);
            }
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

    // Snapshot per-tensor metadata so destroy() can restore.
    snapshot_pool_tensor_metadata(st);

    const char * policy_name = (policy == SLOTTED_POOL_PIN_SCRATCH) ? "pin-scratch" : "round-robin";
    LLAMA_LOG_INFO("%s: hot-swap state ready: shards=%zu tensors=%zu pools=%d policy=%s\n",
            __func__, st.gguf_paths.size(), st.gguf_tensors.size(), slots_resident, policy_name);
    for (size_t s = 0; s < st.gguf_paths.size(); ++s) {
        LLAMA_LOG_INFO("%s:   shard[%zu]: fd=%d data_off=%llu path='%s'\n",
                __func__, s, st.gguf_fds[s],
                (unsigned long long) st.gguf_data_bases[s], st.gguf_paths[s].c_str());
    }

    return true;
}

bool slotted_hot_swap_swap_in(slotted_hot_swap_state & st, int slot_idx, int pool_idx) {
    if (st.model == nullptr || st.gguf_fds.empty()) {
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

    const auto & pool_initial = st.pool_initial_layers[pool_idx];

    uint64_t bytes_this_swap = 0;
    int      tensors_touched = 0;
    int      tensors_retyped = 0;
    int      tensors_resized = 0;
    double   read_ms_this    = 0.0;
    double   set_ms_this     = 0.0;
    constexpr size_t CHUNK   = 1u << 20; // 1 MiB scratch buffer
    std::vector<uint8_t> scratch(CHUNK);

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
            // Genuinely absent from the GGUF (across all shards). For optional
            // tensors (rope_freqs etc.) this is fine; if the graph needs it,
            // we'll crash later anyway.
            LLAMA_LOG_WARN("%s: source tensor '%s' not found in GGUF (skipping)\n",
                           __func__, log_name);
            continue;
        }
        const gguf_tensor_loc & loc = it->second;
        if (loc.shard_idx < 0 || loc.shard_idx >= (int) st.gguf_fds.size()) {
            LLAMA_LOG_ERROR("%s: bogus shard_idx=%d for '%s'\n", __func__, loc.shard_idx, log_name);
            return false;
        }
        const int      fd       = st.gguf_fds[loc.shard_idx];
        const uint64_t src_off  = st.gguf_data_bases[loc.shard_idx] + loc.off_in_shard_data;
        const uint64_t src_size = loc.size;
        const enum ggml_type src_type = (enum ggml_type) loc.type;

        // Resolve dst's loader-allocated buffer size (snapshotted at setup).
        size_t init_alloc = 0;
        {
            auto it2 = st.initial_alloc_size.find(dst_tensor);
            init_alloc = (it2 != st.initial_alloc_size.end()) ? it2->second : ggml_nbytes(dst_tensor);
        }

        // Decide whether we need to mutate dst's type/nb (mixed-quant).
        const bool need_retype = (src_type != (enum ggml_type) dst_tensor->type) ||
                                 (src_size != (uint64_t) ggml_nbytes(dst_tensor));

        // Decide whether we need an override buffer (src doesn't fit dst's
        // loader allocation, or already on override and dst->data still points
        // at a now-too-small override).
        uint8_t * write_ptr = (uint8_t *) dst_tensor->data;
        if (src_size > (uint64_t) init_alloc) {
            // Use / grow the override buffer for this tensor.
            auto & buf = st.override_bufs[dst_tensor];
            if (buf.size() < src_size) {
                buf.resize((size_t) src_size);
            }
            dst_tensor->data = buf.data();
            write_ptr        = buf.data();
            ++tensors_resized;
        } else if (st.override_bufs.count(dst_tensor) != 0) {
            // We already overrode this tensor previously and the override is
            // big enough -- keep using it.
            write_ptr = (uint8_t *) dst_tensor->data;
        }

        if (need_retype) {
            // Same shape (ne[]) assumed; only type changes. Mutate type and nb.
            dst_tensor->type = src_type;
            recompute_nb_for_type(dst_tensor);
            ++tensors_retyped;
        }

        // Read bytes from the right shard's fd and copy into dst's storage.
        uint64_t   bytes_done = 0;
        const auto t_read_start = std::chrono::steady_clock::now();
        while (bytes_done < src_size) {
            const size_t to_read = (size_t) std::min<uint64_t>(CHUNK, src_size - bytes_done);
            const ssize_t n = ::pread(fd, scratch.data(), to_read,
                                       (off_t)(src_off + bytes_done));
            if (n <= 0) {
                LLAMA_LOG_ERROR("%s: pread failed for '%s' (shard=%d) at off=%llu (%zd)\n",
                        __func__, log_name, loc.shard_idx,
                        (unsigned long long)(src_off + bytes_done), n);
                return false;
            }
            const auto t_set_start = std::chrono::steady_clock::now();
            // Direct memcpy when dst->data is a host pointer (override buffer
            // or CPU backend buffer). ggml_backend_tensor_set is safer when
            // the buffer lives on a non-CPU backend, but here we are CPU-only
            // and we may have replaced data with a malloc'd buffer that the
            // backend doesn't know about.
            std::memcpy(write_ptr + bytes_done, scratch.data(), (size_t) n);
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

    st.pool_current_slot[pool_idx] = slot_idx;

    st.total_bytes_read += bytes_this_swap;
    st.total_read_ms    += read_ms_this;
    st.total_set_ms     += set_ms_this;
    st.total_swaps      += 1;

    LLAMA_LOG_INFO("%s: slot=%d pool=%d tensors=%d retyped=%d resized=%d bytes=%llu read_ms=%.2f set_ms=%.2f\n",
            __func__, slot_idx, pool_idx, tensors_touched, tensors_retyped, tensors_resized,
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

    // Build per-layer weight totals from GGUF metadata, summing across all
    // shards so the slot planner sees the full model.
    std::vector<uint64_t> per_layer(n_layer, 0);
    {
        const std::vector<std::string> shards = discover_shards(std::string(gguf_path));
        for (const auto & p : shards) {
            struct gguf_init_params gp = { /*no_alloc=*/ true, /*ctx=*/ nullptr };
            gguf_context * gctx = gguf_init_from_file(p.c_str(), gp);
            if (gctx == nullptr) {
                LLAMA_LOG_ERROR("%s: gguf_init_from_file('%s') failed for plan\n", __func__, p.c_str());
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
    // Close all shard fds.
    for (size_t s = 0; s < st.gguf_fds.size(); ++s) {
        if (st.gguf_fds[s] >= 0) {
            ::close(st.gguf_fds[s]);
            st.gguf_fds[s] = -1;
        }
    }

    // Restore original (loader-allocated) tensor metadata + data pointers so
    // that anything that traverses model.layers[] after destroy() sees the
    // tensors as the loader left them. Override buffers (vector<uint8_t>) get
    // freed when override_bufs goes out of scope below.
    if (st.model != nullptr) {
        for (const auto & kv : st.initial_type) {
            ggml_tensor * t = kv.first;
            t->type = (enum ggml_type) kv.second;
        }
        for (const auto & kv : st.initial_nb) {
            ggml_tensor * t = kv.first;
            for (int d = 0; d < GGML_MAX_DIMS; ++d) t->nb[d] = kv.second[d];
        }
        for (const auto & kv : st.initial_data) {
            ggml_tensor * t = kv.first;
            t->data = kv.second;
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
    }

    st.gguf_paths.clear();
    st.gguf_fds.clear();
    st.gguf_data_bases.clear();
    st.gguf_tensors.clear();
    st.pool_initial_layers.clear();
    st.pool_current_slot.clear();
    st.initial_alloc_size.clear();
    st.initial_type.clear();
    st.initial_nb.clear();
    st.initial_data.clear();
    st.override_bufs.clear();
    st.model = nullptr;
}
