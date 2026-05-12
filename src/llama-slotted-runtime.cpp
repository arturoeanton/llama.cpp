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

// Forward decl: the background worker entry point.
static void slotted_prefetch_worker(slotted_hot_swap_state * st);

bool slotted_hot_swap_setup(slotted_hot_swap_state & st,
                            llama_model * model,
                            const std::string & gguf_path,
                            const std::vector<std::pair<int,int>> & slot_ranges,
                            int slots_resident,
                            enum slotted_hot_swap_policy_t policy,
                            bool async_prefetch) {
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
    st.pools.clear();
    st.pools.reserve(slots_resident);
    for (int p = 0; p < slots_resident; ++p) {
        const int il_start = slot_ranges[p].first;
        const int il_end   = slot_ranges[p].second;
        st.pool_current_slot[p] = p;
        auto & dst = st.pool_initial_layers[p];
        dst.reserve(il_end - il_start + 1);
        for (int il = il_start; il <= il_end; ++il) {
            dst.push_back(model->layers[il]);
        }

        auto ps = std::make_unique<slotted_pool_state>();
        ps->current_slot = p;
        ps->in_use       = false;
        ps->loading      = false;
        st.pools.emplace_back(std::move(ps));
    }

    // Snapshot per-tensor metadata so destroy() can restore.
    snapshot_pool_tensor_metadata(st);

    // FASE 4C: async prefetch only makes sense with round-robin (R >= 2),
    // because pin-and-scratch has only one rotating pool which is exactly
    // the one main thread is reading. Refuse to enable it otherwise.
    st.async_prefetch_enabled = false;
    if (async_prefetch) {
        if (policy != SLOTTED_POOL_ROUND_ROBIN) {
            LLAMA_LOG_WARN("%s: async_prefetch requested but policy is not round-robin; ignoring\n",
                           __func__);
        } else if (slots_resident < 2) {
            LLAMA_LOG_WARN("%s: async_prefetch requested but slots_resident=%d < 2; ignoring\n",
                           __func__, slots_resident);
        } else {
            st.async_prefetch_enabled = true;
            st.shutdown.store(false);
            st.prefetch_thread = std::thread(slotted_prefetch_worker, &st);
        }
    }

    const char * policy_name = (policy == SLOTTED_POOL_PIN_SCRATCH) ? "pin-scratch" : "round-robin";
    LLAMA_LOG_INFO("%s: hot-swap state ready: shards=%zu tensors=%zu pools=%d policy=%s async_prefetch=%s\n",
            __func__, st.gguf_paths.size(), st.gguf_tensors.size(), slots_resident, policy_name,
            st.async_prefetch_enabled ? "on" : "off");
    for (size_t s = 0; s < st.gguf_paths.size(); ++s) {
        LLAMA_LOG_INFO("%s:   shard[%zu]: fd=%d data_off=%llu path='%s'\n",
                __func__, s, st.gguf_fds[s],
                (unsigned long long) st.gguf_data_bases[s], st.gguf_paths[s].c_str());
    }

    return true;
}

// Pure I/O for one slot into one pool. Mutates pool tensors' data/type/nb to
// reflect the source's quant (FASE 4A-3 mixed-quant logic). Does NOT rebind
// model.layers[] and does NOT update pool_current_slot[]; the caller does
// that after taking whatever locks are needed.
//
// `tag` is just for log lines so the worker thread can identify itself.
// `from_worker` selects whether the per-load stats line is logged as the
// foreground swap or the bg worker swap.
static bool do_load_into_pool(slotted_hot_swap_state & st,
                              int slot_idx, int pool_idx,
                              const char * tag) {
    if (st.model == nullptr || st.gguf_fds.empty()) {
        LLAMA_LOG_ERROR("%s: state not initialized\n", tag);
        return false;
    }
    if (slot_idx < 0 || slot_idx >= (int) st.slot_ranges.size()) {
        LLAMA_LOG_ERROR("%s: slot_idx %d out of range\n", tag, slot_idx);
        return false;
    }
    if (pool_idx < 0 || pool_idx >= st.slots_resident) {
        LLAMA_LOG_ERROR("%s: pool_idx %d out of range [0,%d)\n", tag, pool_idx, st.slots_resident);
        return false;
    }

    const int log_start = st.slot_ranges[slot_idx].first;
    const int log_end   = st.slot_ranges[slot_idx].second;
    const int phy_start = st.slot_ranges[pool_idx].first;
    const int phy_end   = st.slot_ranges[pool_idx].second;
    const int log_size  = log_end - log_start + 1;
    const int phy_size  = phy_end - phy_start + 1;
    if (log_size != phy_size) {
        LLAMA_LOG_ERROR("%s: slot %d has %d layers but pool %d has %d (size mismatch)\n",
                tag, slot_idx, log_size, pool_idx, phy_size);
        return false;
    }

    LLAMA_LOG_INFO("%s: loading logical slot %d (layers %d..%d) into pool %d (phys %d..%d)\n",
            tag, slot_idx, log_start, log_end, pool_idx, phy_start, phy_end);

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

        char log_name[GGML_MAX_NAME];
        std::snprintf(log_name, sizeof(log_name), "blk.%d.%s", log_il, suffix.c_str());

        auto it = st.gguf_tensors.find(log_name);
        if (it == st.gguf_tensors.end()) {
            LLAMA_LOG_WARN("%s: source tensor '%s' not found in GGUF (skipping)\n", tag, log_name);
            continue;
        }
        const gguf_tensor_loc & loc = it->second;
        if (loc.shard_idx < 0 || loc.shard_idx >= (int) st.gguf_fds.size()) {
            LLAMA_LOG_ERROR("%s: bogus shard_idx=%d for '%s'\n", tag, loc.shard_idx, log_name);
            return false;
        }
        const int      fd       = st.gguf_fds[loc.shard_idx];
        const uint64_t src_off  = st.gguf_data_bases[loc.shard_idx] + loc.off_in_shard_data;
        const uint64_t src_size = loc.size;
        const enum ggml_type src_type = (enum ggml_type) loc.type;

        size_t init_alloc = 0;
        {
            auto it2 = st.initial_alloc_size.find(dst_tensor);
            init_alloc = (it2 != st.initial_alloc_size.end()) ? it2->second : ggml_nbytes(dst_tensor);
        }

        const bool need_retype = (src_type != (enum ggml_type) dst_tensor->type) ||
                                 (src_size != (uint64_t) ggml_nbytes(dst_tensor));

        uint8_t * write_ptr = (uint8_t *) dst_tensor->data;
        if (src_size > (uint64_t) init_alloc) {
            // Map mutation must be serialized across main and worker threads.
            std::lock_guard<std::mutex> lk(st.runtime_mu);
            auto & buf = st.override_bufs[dst_tensor];
            if (buf.size() < src_size) {
                buf.resize((size_t) src_size);
            }
            dst_tensor->data = buf.data();
            write_ptr        = buf.data();
            ++tensors_resized;
        } else {
            std::lock_guard<std::mutex> lk(st.runtime_mu);
            if (st.override_bufs.count(dst_tensor) != 0) {
                write_ptr = (uint8_t *) dst_tensor->data;
            }
        }

        if (need_retype) {
            dst_tensor->type = src_type;
            recompute_nb_for_type(dst_tensor);
            ++tensors_retyped;
        }

        uint64_t   bytes_done = 0;
        const auto t_read_start = std::chrono::steady_clock::now();
        while (bytes_done < src_size) {
            const size_t to_read = (size_t) std::min<uint64_t>(CHUNK, src_size - bytes_done);
            const ssize_t n = ::pread(fd, scratch.data(), to_read,
                                       (off_t)(src_off + bytes_done));
            if (n <= 0) {
                LLAMA_LOG_ERROR("%s: pread failed for '%s' (shard=%d) at off=%llu (%zd)\n",
                        tag, log_name, loc.shard_idx,
                        (unsigned long long)(src_off + bytes_done), n);
                return false;
            }
            const auto t_set_start = std::chrono::steady_clock::now();
            std::memcpy(write_ptr + bytes_done, scratch.data(), (size_t) n);
            set_ms_this += ms_since(t_set_start);
            bytes_done += (uint64_t) n;
        }
        read_ms_this += ms_since(t_read_start);

        bytes_this_swap += src_size;
        ++tensors_touched;
    }

    // Stats: in async mode the same counters track both fg and bg work, so
    // guard the accumulation against concurrent updates from the worker.
    {
        std::lock_guard<std::mutex> lk(st.runtime_mu);
        st.total_bytes_read += bytes_this_swap;
        st.total_read_ms    += read_ms_this;
        st.total_set_ms     += set_ms_this;
        st.total_swaps      += 1;
    }

    LLAMA_LOG_INFO("%s: slot=%d pool=%d tensors=%d retyped=%d resized=%d bytes=%llu read_ms=%.2f set_ms=%.2f\n",
            tag, slot_idx, pool_idx, tensors_touched, tensors_retyped, tensors_resized,
            (unsigned long long) bytes_this_swap, read_ms_this, set_ms_this);

    return true;
}

// Rebind model.layers[log_start..log_end] to the pool's snapshotted layers.
// Always done on the main thread, with the pool lock NOT held (rebind only
// touches model.layers[], not pool tensors).
static void do_rebind_layers(slotted_hot_swap_state & st, int slot_idx, int pool_idx) {
    const int log_start = st.slot_ranges[slot_idx].first;
    const int log_end   = st.slot_ranges[slot_idx].second;
    const auto & pool_initial = st.pool_initial_layers[pool_idx];
    const int log_size = log_end - log_start + 1;
    for (int i = 0; i < log_size; ++i) {
        st.model->layers[log_start + i] = pool_initial[i];
    }
}

bool slotted_hot_swap_swap_in(slotted_hot_swap_state & st, int slot_idx, int pool_idx) {
    if (slot_idx < 0 || slot_idx >= (int) st.slot_ranges.size() ||
        pool_idx < 0 || pool_idx >= st.slots_resident) {
        LLAMA_LOG_ERROR("%s: bad indices slot=%d pool=%d\n", __func__, slot_idx, pool_idx);
        return false;
    }
    if (st.pool_current_slot[pool_idx] == slot_idx) {
        LLAMA_LOG_INFO("%s: slot=%d already in pool=%d (no-op)\n", __func__, slot_idx, pool_idx);
        return true;
    }
    if (!do_load_into_pool(st, slot_idx, pool_idx, __func__)) return false;
    do_rebind_layers(st, slot_idx, pool_idx);
    st.pool_current_slot[pool_idx] = slot_idx;
    if (pool_idx < (int) st.pools.size() && st.pools[pool_idx]) {
        st.pools[pool_idx]->current_slot = slot_idx;
    }
    return true;
}

// ---------------------------------------------------------------------------
// FASE 4C async-prefetch API.
// ---------------------------------------------------------------------------

void slotted_prefetch_enqueue(slotted_hot_swap_state & st, int slot_idx, int pool_idx) {
    if (!st.async_prefetch_enabled) return;
    if (slot_idx < 0 || slot_idx >= (int) st.slot_ranges.size()) return;
    if (pool_idx < 0 || pool_idx >= st.slots_resident)            return;

    // Cheap pre-check: don't bother queuing if the pool already holds the
    // wanted slot AND no one is loading it. Worker would skip anyway, but
    // this saves a queue cycle.
    {
        slotted_pool_state * ps = st.pools[pool_idx].get();
        std::lock_guard<std::mutex> lk(ps->mu);
        if (ps->current_slot == slot_idx && !ps->loading) {
            return;
        }
    }
    {
        std::lock_guard<std::mutex> lk(st.queue_mu);
        st.prefetch_queue.emplace(slot_idx, pool_idx);
    }
    st.queue_cv.notify_one();
}

bool slotted_pool_acquire(slotted_hot_swap_state & st, int slot_idx, int pool_idx) {
    if (slot_idx < 0 || slot_idx >= (int) st.slot_ranges.size() ||
        pool_idx < 0 || pool_idx >= st.slots_resident) {
        LLAMA_LOG_ERROR("%s: bad indices slot=%d pool=%d\n", __func__, slot_idx, pool_idx);
        return false;
    }

    slotted_pool_state * ps = st.pools[pool_idx].get();
    bool need_sync_load = false;
    bool was_hit        = false;
    const auto t_wait_start = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(ps->mu);
        // Wait for any bg load to finish (and for the in_use flag to clear,
        // though main shouldn't double-acquire).
        ps->cv.wait(lk, [&]{ return !ps->loading && !ps->in_use; });
        if (ps->current_slot == slot_idx) {
            was_hit = true;
        } else {
            need_sync_load = true;
        }
        ps->in_use = true;
    }
    st.prefetch_wait_ms += ms_since(t_wait_start);

    if (need_sync_load) {
        ++st.prefetch_misses;
        if (!do_load_into_pool(st, slot_idx, pool_idx, "slotted_pool_acquire(miss)")) {
            // Surface error and release the pool so the worker can recover.
            std::lock_guard<std::mutex> lk(ps->mu);
            ps->in_use = false;
            ps->cv.notify_all();
            return false;
        }
        std::lock_guard<std::mutex> lk(ps->mu);
        ps->current_slot = slot_idx;
        st.pool_current_slot[pool_idx] = slot_idx;
    } else {
        if (was_hit && st.async_prefetch_enabled) {
            ++st.prefetch_hits;
        }
    }

    do_rebind_layers(st, slot_idx, pool_idx);
    return true;
}

void slotted_pool_release(slotted_hot_swap_state & st, int pool_idx) {
    if (pool_idx < 0 || pool_idx >= st.slots_resident) return;
    slotted_pool_state * ps = st.pools[pool_idx].get();
    {
        std::lock_guard<std::mutex> lk(ps->mu);
        ps->in_use = false;
    }
    ps->cv.notify_all();
}

// Background worker. Pulls (slot, pool) jobs from the queue; for each, waits
// until the pool is idle, then runs a load. Skips a job if the pool already
// has the wanted slot when it is picked up.
static void slotted_prefetch_worker(slotted_hot_swap_state * st) {
    while (true) {
        std::pair<int,int> job;
        {
            std::unique_lock<std::mutex> lk(st->queue_mu);
            st->queue_cv.wait(lk, [st]{ return st->shutdown.load() || !st->prefetch_queue.empty(); });
            if (st->shutdown.load() && st->prefetch_queue.empty()) return;
            job = st->prefetch_queue.front();
            st->prefetch_queue.pop();
        }
        const int slot_idx = job.first;
        const int pool_idx = job.second;
        if (pool_idx < 0 || pool_idx >= (int) st->pools.size()) continue;

        slotted_pool_state * ps = st->pools[pool_idx].get();
        bool do_load = false;
        {
            std::unique_lock<std::mutex> lk(ps->mu);
            // Wait until main releases the pool.
            ps->cv.wait(lk, [&]{ return st->shutdown.load() || !ps->in_use; });
            if (st->shutdown.load()) return;
            if (ps->current_slot == slot_idx) {
                ++st->prefetch_skipped;
            } else {
                ps->loading = true;
                do_load = true;
            }
        }
        if (!do_load) continue;

        const auto t_start = std::chrono::steady_clock::now();
        const bool ok = do_load_into_pool(*st, slot_idx, pool_idx, "prefetch_worker");
        const double dt = ms_since(t_start);

        {
            std::lock_guard<std::mutex> lk(ps->mu);
            if (ok) {
                ps->current_slot = slot_idx;
                st->pool_current_slot[pool_idx] = slot_idx;
            }
            ps->loading = false;
            st->prefetch_bg_busy_ms += dt;
            ++st->prefetch_jobs_done;
        }
        ps->cv.notify_all();
    }
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
                    int32_t  policy,
                    int32_t  async_prefetch) {
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
    if (!slotted_hot_swap_setup(hs->st, model, gguf_path, ranges, slots_resident, pol, async_prefetch != 0)) {
        delete hs;
        return nullptr;
    }
    return hs;
}

int32_t llama_slotted_hot_swap_pool_acquire(
        struct llama_slotted_hot_swap * hs,
                              int32_t   slot_idx,
                              int32_t   pool_idx) {
    if (hs == nullptr) return -1;
    return slotted_pool_acquire(hs->st, slot_idx, pool_idx) ? 0 : -1;
}

void llama_slotted_hot_swap_pool_release(
        struct llama_slotted_hot_swap * hs,
                              int32_t   pool_idx) {
    if (hs == nullptr) return;
    slotted_pool_release(hs->st, pool_idx);
}

void llama_slotted_hot_swap_prefetch_enqueue(
        struct llama_slotted_hot_swap * hs,
                              int32_t   slot_idx,
                              int32_t   pool_idx) {
    if (hs == nullptr) return;
    slotted_prefetch_enqueue(hs->st, slot_idx, pool_idx);
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
    // Stop the prefetch worker first so it doesn't race with the fd close
    // or the metadata restoration below.
    if (st.async_prefetch_enabled) {
        st.shutdown.store(true);
        st.queue_cv.notify_all();
        // Notify all pool CVs in case the worker is waiting on an in_use pool.
        for (auto & ps : st.pools) {
            if (!ps) continue;
            std::lock_guard<std::mutex> lk(ps->mu);
            ps->cv.notify_all();
        }
        if (st.prefetch_thread.joinable()) {
            st.prefetch_thread.join();
        }
        st.async_prefetch_enabled = false;

        LLAMA_LOG_INFO("slotted_hot_swap_destroy: prefetch summary jobs=%d hits=%d misses=%d skipped=%d wait_ms=%.2f bg_busy_ms=%.2f\n",
                       st.prefetch_jobs_done, st.prefetch_hits, st.prefetch_misses,
                       st.prefetch_skipped, st.prefetch_wait_ms, st.prefetch_bg_busy_ms);
    }

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
    st.pools.clear();
    while (!st.prefetch_queue.empty()) st.prefetch_queue.pop();
    st.model = nullptr;
}
