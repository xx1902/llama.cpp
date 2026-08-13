#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml-cuda.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <cstdlib>

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
                        bool   physical_paged,
                    uint32_t   physical_page_size) :
    // const layer_filter_cb & filter,
    // const  layer_reuse_cb & reuse) :
    // model(model), hparams(model.hparams), v_trans(v_trans),
    // n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {
    model(model),
    hparams(model.hparams),
    type_k_cache(type_k),
    type_v_cache(type_v),
    offload_cache(offload),
    physical_paged(physical_paged),
    physical_page_size(physical_page_size == 0 ? 16 : physical_page_size),
    v_trans(v_trans),
    n_seq_max(n_seq_max),
    n_stream(unified ? 1 : n_seq_max),
    n_pad(n_pad),
    n_swa(n_swa),
    swa_type(swa_type) {

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer_kv = hparams.n_layer_kv();

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*n_layer_kv*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        // layers.push_back({ il, k, v, k_stream, v_stream, });
        layers.push_back({ il, k, v, k_stream, v_stream, buft });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < hparams.n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // 新增分页
    if (this->physical_paged) {
        init_physical_paged_storage();

        LLAMA_LOG_INFO(
                "%s: physical paged KV enabled, max cells = %u, page_size = %u, max_pages = %u\n",
                __func__,
                kv_size,
                this->physical_page_size,
                (kv_size + this->physical_page_size - 1) / this->physical_page_size);

        return;
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

llama_kv_cache::~llama_kv_cache() {
    for (auto & item : delta_async_builds) {
        auto & build = item.second;
        if (build.backend_job != nullptr && build.cancel != nullptr) {
            build.cancel(build.backend_job);
        }
    }
}

// 新增分页
void llama_kv_cache::init_physical_paged_storage() {
    page_table.resize(n_stream);
    physical_pages.resize(n_stream);

    for (uint32_t s = 0; s < n_stream; ++s) {
        page_table[s].resize(get_size());

        for (uint32_t i = 0; i < get_size(); ++i) {
            page_table[s][i].page_id = UINT32_MAX;
            page_table[s][i].offset = 0;
        }
    }

    for (auto & layer : layers) {
        layer.k = nullptr;
        layer.v = nullptr;
        layer.k_stream.clear();
        layer.v_stream.clear();
    }

    // graph_reserve 会在真实 decode 前提前构图，可能先调用 cpy_k/cpy_v。
    // 因此这里先为每个 stream 预分配第 0 页，避免构图阶段 page_table 为空。
    for (uint32_t s = 0; s < n_stream; ++s) {
        ensure_page_for_cell(s, 0);
    }

    LLAMA_LOG_INFO(
            "%s: initialized physical paged KV metadata, no full KV buffer allocated\n",
            __func__);
}
const llama_kv_cache::kv_page_ref & llama_kv_cache::get_page_ref(
        uint32_t stream_id,
        uint32_t cell_id) const {
    GGML_ASSERT(stream_id < page_table.size());
    GGML_ASSERT(cell_id < page_table[stream_id].size());

    return page_table[stream_id][cell_id];
}
void llama_kv_cache::ensure_page_for_cell(
        uint32_t stream_id,
        uint32_t cell_id) {
    GGML_ASSERT(stream_id < page_table.size());
    GGML_ASSERT(cell_id < page_table[stream_id].size());

    auto & ref = page_table[stream_id][cell_id];

    if (ref.page_id != UINT32_MAX) {
        return;
    }

    const uint32_t page_begin = (cell_id / physical_page_size) * physical_page_size;
    const uint32_t page_end = std::min(page_begin + physical_page_size, get_size());
    const uint32_t page_id = (uint32_t) physical_pages[stream_id].size();

    kv_page page;
    page.id = page_id;
    page.used = true;
    page.cell_begin = page_begin;
    page.cell_end = page_end;

    const size_t n_tensors = layers.size() * 2;
    const size_t ctx_size =
            ggml_tensor_overhead() * n_tensors
            + ggml_graph_overhead();

    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    page.ctx.reset(ggml_init(params));
    page.layers.resize(layers.size());

    for (uint32_t ikv = 0; ikv < layers.size(); ++ikv) {
        const uint32_t il = layers[ikv].il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        page.layers[ikv].k = ggml_new_tensor_2d(
                page.ctx.get(),
                type_k_cache,
                n_embd_k_gqa,
                physical_page_size);

        if (!v_trans) {
            page.layers[ikv].v = ggml_new_tensor_2d(
                    page.ctx.get(),
                    type_v_cache,
                    n_embd_v_gqa,
                    physical_page_size);
        } else {
            page.layers[ikv].v = ggml_new_tensor_2d(
                    page.ctx.get(),
                    type_v_cache,
                    physical_page_size,
                    n_embd_v_gqa);
        }
    }

    // 注意：这里先用原 KV 构造过程中第一块 buffer 的 backend 类型不容易直接拿到。
    GGML_ASSERT(!layers.empty());

    // 当前实验模型的 attention KV 都在 CUDA0，所以一个 page 使用同一个 buft。
    // 如果后面支持多 GPU / CPU+GPU 混合层，需要按 buft 分组创建多个 page buffer。
    ggml_backend_buffer_type_t buft = layers[0].buft;
    GGML_ASSERT(buft != nullptr);

    for (const auto & layer : layers) {
        GGML_ASSERT(layer.buft == buft && "physical paged KV currently expects all KV layers on the same backend");
    }

    page.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(page.ctx.get(), buft));

    if (!page.buf) {
        throw std::runtime_error("failed to allocate physical paged KV buffer");
    }

    ggml_backend_buffer_clear(page.buf.get(), 0);

    physical_pages[stream_id].push_back(std::move(page));

    for (uint32_t i = page_begin; i < page_end; ++i) {
        page_table[stream_id][i].page_id = page_id;
        page_table[stream_id][i].offset = i - page_begin;
    }

    LLAMA_LOG_INFO(
            "%s: allocated physical KV page stream=%u page=%u cells=[%u,%u)\n",
            __func__,
            stream_id,
            page_id,
            page_begin,
            page_end);
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        if (physical_paged) {
            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                const uint32_t stream_id = sinfo_new.strm[s];

                for (uint32_t idx : sinfo_new.idxs[s]) {
                    ensure_page_for_cell(stream_id, idx);
                }
            }
        }

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_physical_paged() const {
    return physical_paged;
}

uint32_t llama_kv_cache::get_physical_page_size() const {
    return physical_page_size;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

// 新增固定分页统计
llama_kv_cache::memory_usage_stats llama_kv_cache::get_memory_usage_stats(uint32_t page_size) const {
    memory_usage_stats stats;

    if (page_size == 0) {
        page_size = physical_paged ? physical_page_size : 16;
    }

    stats.page_size = page_size;

    const uint32_t cells_per_stream = get_size();

    stats.total_cells = cells_per_stream * n_stream;

    if (stats.total_cells == 0) {
        return stats;
    }

    const uint32_t pages_per_stream =
            (cells_per_stream + page_size - 1) / page_size;

    stats.total_pages = pages_per_stream * n_stream;

    // 物理分页模式下，layers[ikv].k/v 已经被置空，不能再用 size_k_bytes()/size_v_bytes()。
    // 这里分别计算：
    // - continuous_bytes：如果使用原始连续 KV，需要的理论完整 KV 大小。
    // - paged_bytes：当前已经真实分配的 page buffer 大小。
    if (physical_paged) {
        uint64_t bytes_per_cell_all_layers = 0;

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            bytes_per_cell_all_layers += ggml_row_size(type_k_cache, hparams.n_embd_k_gqa(il));

            if (!hparams.is_mla()) {
                bytes_per_cell_all_layers += ggml_row_size(type_v_cache, hparams.n_embd_v_gqa(il));
            }
        }

        stats.continuous_bytes =
                bytes_per_cell_all_layers * (uint64_t) cells_per_stream * (uint64_t) n_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            const auto & cells = v_cells[s];

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.is_empty(i)) {
                    stats.used_cells++;
                }
            }

            stats.used_pages += (uint32_t) physical_pages[s].size();

            for (const auto & page : physical_pages[s]) {
                if (page.buf) {
                    stats.paged_bytes += ggml_backend_buffer_get_size(page.buf.get());
                }
            }
        }

        stats.free_pages =
                stats.total_pages > stats.used_pages
                ? stats.total_pages - stats.used_pages
                : 0;

        stats.cell_used_rate =
                stats.total_cells > 0
                ? (double) stats.used_cells / (double) stats.total_cells
                : 0.0;

        stats.page_used_rate =
                stats.total_pages > 0
                ? (double) stats.used_pages / (double) stats.total_pages
                : 0.0;

        const uint32_t committed_cells = stats.used_pages * page_size;

        if (committed_cells > 0) {
            stats.page_waste_rate =
                    committed_cells > stats.used_cells
                    ? (double) (committed_cells - stats.used_cells) / (double) committed_cells
                    : 0.0;
        }

        return stats;
    }

    stats.continuous_bytes = (uint64_t) size_k_bytes() + (uint64_t) size_v_bytes();

    std::vector<uint8_t> page_used(stats.total_pages, 0);

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i)) {
                continue;
            }

            stats.used_cells++;

            const uint32_t page_in_stream = i / page_size;
            const uint32_t global_page_id = s * pages_per_stream + page_in_stream;

            if (global_page_id < page_used.size()) {
                page_used[global_page_id] = 1;
            }
        }
    }

    for (uint8_t used : page_used) {
        if (used) {
            stats.used_pages++;
        }
    }

    stats.free_pages =
            stats.total_pages > stats.used_pages
            ? stats.total_pages - stats.used_pages
            : 0;

    const double bytes_per_cell =
            (double) stats.continuous_bytes / (double) stats.total_cells;

    stats.paged_bytes =
            (uint64_t) (bytes_per_cell * (double) page_size * (double) stats.used_pages);

    stats.cell_used_rate =
            stats.total_cells > 0
            ? (double) stats.used_cells / (double) stats.total_cells
            : 0.0;

    stats.page_used_rate =
            stats.total_pages > 0
            ? (double) stats.used_pages / (double) stats.total_pages
            : 0.0;

    const uint32_t committed_cells = stats.used_pages * page_size;

    if (committed_cells > 0) {
        stats.page_waste_rate =
                (double) (committed_cells - stats.used_cells)
                / (double) committed_cells;
    }

    return stats;
}

// 新增 kv 差值
static float kv_probe_read_scalar(
        const std::vector<uint8_t> & buf,
        ggml_type type,
        uint32_t index) {
    switch (type) {
        case GGML_TYPE_F32:
            return reinterpret_cast<const float *>(buf.data())[index];

        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(
                    reinterpret_cast<const ggml_fp16_t *>(buf.data())[index]);

        case GGML_TYPE_BF16:
            return ggml_bf16_to_fp32(
                    reinterpret_cast<const ggml_bf16_t *>(buf.data())[index]);

        default:
            return 0.0f;
    }
}
static void kv_probe_write_scalar(
        std::vector<uint8_t> & raw,
        ggml_type type,
        uint32_t i,
        float value) {
    if (type == GGML_TYPE_F32) {
        float * ptr = reinterpret_cast<float *>(raw.data());
        ptr[i] = value;
        return;
    }

    if (type == GGML_TYPE_F16) {
        ggml_fp16_t * ptr = reinterpret_cast<ggml_fp16_t *>(raw.data());
        ptr[i] = ggml_fp32_to_fp16(value);
        return;
    }

    // 实验版先只支持 F16/F32 KV cache。
}

static void kv_probe_accumulate(
        const std::vector<float> & a,
        const std::vector<float> & b,
        double & l2_sum,
        double & cos_sum,
        int & count) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    double l2 = 0.0;

    const int n = std::min((int) a.size(), (int) b.size());

    for (int i = 0; i < n; i++) {
        const double da = a[i];
        const double db = b[i];
        const double diff = da - db;

        dot += da * db;
        na += da * da;
        nb += db * db;
        l2 += diff * diff;
    }

    const double eps = 1e-12;

    // 两边都是全 0，说明这个位置没有有效方向信息，不参与 cosine 平均。
    if (na < eps && nb < eps) {
        return;
    }

    const double cos =
            dot / (std::sqrt(na) * std::sqrt(nb) + eps);

    l2_sum += std::sqrt(l2 / std::max(1, n));
    cos_sum += cos;
    count++;
}

bool llama_kv_cache::seq_delta_probe(
        llama_seq_id seq_a,
        llama_seq_id seq_b,
        llama_pos p0,
        llama_pos p1,
        kv_delta_probe_stats & stats) const {
    if (seq_a < 0 || seq_b < 0) {
        return false;
    }

    if ((size_t) seq_a >= seq_to_stream.size() ||
            (size_t) seq_b >= seq_to_stream.size()) {
        return false;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 <= p0) {
        return false;
    }

    const uint32_t stream_a = seq_to_stream[seq_a];
    const uint32_t stream_b = seq_to_stream[seq_b];

    const auto & cells_a = v_cells[stream_a];
    const auto & cells_b = v_cells[stream_b];

    auto find_cell = [](
            const llama_kv_cells & cells,
            llama_seq_id seq_id,
            llama_pos pos,
            uint32_t & cell_id) -> bool {
        for (uint32_t i = 0; i < cells.size(); i++) {
            if (!cells.seq_has(i, seq_id)) {
                continue;
            }

            if (cells.pos_get(i) == pos) {
                cell_id = i;
                return true;
            }
        }

        return false;
    };

    stats = {};
    stats.n_layers = (int32_t) layers.size();
    stats.n_tokens = (int32_t) (p1 - p0);
    stats.layers.reserve(layers.size());

    double k_l2_all = 0.0;
    double v_l2_all = 0.0;
    double k_cos_all = 0.0;
    double v_cos_all = 0.0;
    int k_count_all = 0;
    int v_count_all = 0;

    for (const auto & layer : layers) {
        kv_delta_layer_stats layer_stats;

        layer_stats.layer_id = (int32_t) layer.il;

        double k_l2_layer = 0.0;
        double v_l2_layer = 0.0;
        double k_cos_layer = 0.0;
        double v_cos_layer = 0.0;
        int k_count_layer = 0;
        int v_count_layer = 0;

        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa =
                hparams.n_embd_k_gqa(il);

        const uint32_t n_embd_v_gqa =
                hparams.n_embd_v_gqa(il);

        ggml_tensor * k_a = layer.k_stream[stream_a];
        ggml_tensor * k_b = layer.k_stream[stream_b];

        ggml_tensor * v_a = layer.v_stream[stream_a];
        ggml_tensor * v_b = layer.v_stream[stream_b];

        const size_t k_size_row =
                ggml_row_size(k_a->type, n_embd_k_gqa);

        const size_t k_size_el =
                ggml_type_size(k_a->type);

        const size_t v_size_el =
                v_a ? ggml_type_size(v_a->type) : 0;

        int missing_cell_count = 0;
        int debug_print_count = 0;

        for (llama_pos pos = p0; pos < p1; pos++) {
            uint32_t cell_a = 0;
            uint32_t cell_b = 0;

            const bool has_a = find_cell(cells_a, seq_a, pos, cell_a);
            const bool has_b = find_cell(cells_b, seq_b, pos, cell_b);
            // debug日志取消
            const bool debug_kv_delta_cells = false;

            if (!has_a || !has_b) {
                missing_cell_count++;

                if (debug_kv_delta_cells && debug_print_count < 16) {
                    fprintf(stderr,
                            "kv_delta_probe missing cell: seq_a=%d seq_b=%d pos=%d has_a=%d has_b=%d\n",
                            seq_a,
                            seq_b,
                            (int) pos,
                            has_a ? 1 : 0,
                            has_b ? 1 : 0);
                    debug_print_count++;
                }

                continue;
            }

            if (debug_kv_delta_cells && debug_print_count < 16) {
                fprintf(stderr,
                        "kv_delta_probe cell map: seq_a=%d seq_b=%d pos=%d cell_a=%u cell_b=%u\n",
                        seq_a,
                        seq_b,
                        (int) pos,
                        cell_a,
                        cell_b);
                debug_print_count++;
            }

            {
                std::vector<uint8_t> raw_a(k_size_row);
                std::vector<uint8_t> raw_b(k_size_row);

                ggml_backend_tensor_get(
                        k_a,
                        raw_a.data(),
                        cell_a * k_size_row,
                        k_size_row);

                ggml_backend_tensor_get(
                        k_b,
                        raw_b.data(),
                        cell_b * k_size_row,
                        k_size_row);

                std::vector<float> fa(n_embd_k_gqa);
                std::vector<float> fb(n_embd_k_gqa);

                for (uint32_t i = 0; i < n_embd_k_gqa; i++) {
                    fa[i] = kv_probe_read_scalar(raw_a, k_a->type, i);
                    fb[i] = kv_probe_read_scalar(raw_b, k_b->type, i);
                }

                kv_probe_accumulate(
                        fa,
                        fb,
                        k_l2_layer,
                        k_cos_layer,
                        k_count_layer);
            }

            if (v_a != nullptr && v_b != nullptr) {
                std::vector<float> fa(n_embd_v_gqa);
                std::vector<float> fb(n_embd_v_gqa);

                if (!v_trans) {
                    const size_t v_size_row =
                            ggml_row_size(v_a->type, n_embd_v_gqa);

                    std::vector<uint8_t> raw_a(v_size_row);
                    std::vector<uint8_t> raw_b(v_size_row);

                    ggml_backend_tensor_get(
                            v_a,
                            raw_a.data(),
                            cell_a * v_size_row,
                            v_size_row);

                    ggml_backend_tensor_get(
                            v_b,
                            raw_b.data(),
                            cell_b * v_size_row,
                            v_size_row);

                    for (uint32_t i = 0; i < n_embd_v_gqa; i++) {
                        fa[i] = kv_probe_read_scalar(raw_a, v_a->type, i);
                        fb[i] = kv_probe_read_scalar(raw_b, v_b->type, i);
                    }
                } else {
                    for (uint32_t i = 0; i < n_embd_v_gqa; i++) {
                        std::vector<uint8_t> raw_a(v_size_el);
                        std::vector<uint8_t> raw_b(v_size_el);

                        const size_t offset_a =
                                (cell_a + i * cells_a.size()) * v_size_el;

                        const size_t offset_b =
                                (cell_b + i * cells_b.size()) * v_size_el;

                        ggml_backend_tensor_get(
                                v_a,
                                raw_a.data(),
                                offset_a,
                                v_size_el);

                        ggml_backend_tensor_get(
                                v_b,
                                raw_b.data(),
                                offset_b,
                                v_size_el);

                        fa[i] = kv_probe_read_scalar(raw_a, v_a->type, 0);
                        fb[i] = kv_probe_read_scalar(raw_b, v_b->type, 0);
                    }
                }

                kv_probe_accumulate(
                        fa,
                        fb,
                        v_l2_layer,
                        v_cos_layer,
                        v_count_layer);
            }
        }

        if (k_count_layer > 0) {
            layer_stats.k_l2_avg = k_l2_layer / k_count_layer;
            layer_stats.k_cos_avg = k_cos_layer / k_count_layer;

            k_l2_all += k_l2_layer;
            k_cos_all += k_cos_layer;
            k_count_all += k_count_layer;
        }

        if (v_count_layer > 0) {
            layer_stats.v_l2_avg = v_l2_layer / v_count_layer;
            layer_stats.v_cos_avg = v_cos_layer / v_count_layer;

            v_l2_all += v_l2_layer;
            v_cos_all += v_cos_layer;
            v_count_all += v_count_layer;
        }

        layer_stats.kv_l2_avg =
                (layer_stats.k_l2_avg + layer_stats.v_l2_avg) / 2.0;

        layer_stats.kv_cos_avg =
                (layer_stats.k_cos_avg + layer_stats.v_cos_avg) / 2.0;

        stats.layers.push_back(layer_stats);

        if (missing_cell_count > 0) {
            fprintf(stderr,
                    "kv_delta_probe warning: missing_cell_count=%d seq_a=%d seq_b=%d range=[%d,%d)\n",
                    missing_cell_count,
                    seq_a,
                    seq_b,
                    (int) p0,
                    (int) p1);
        }
    }

    if (k_count_all == 0 && v_count_all == 0) {
        fprintf(stderr,
                "kv_delta_probe failed: no comparable KV cells, seq_a=%d seq_b=%d range=[%d,%d)\n",
                seq_a,
                seq_b,
                (int) p0,
                (int) p1);

        stats.layers.clear();
        stats.can_reuse_as_delta = false;
        return false;
    }

    if (k_count_all > 0) {
        stats.k_l2_avg = k_l2_all / k_count_all;
        stats.k_cos_avg = k_cos_all / k_count_all;
    }

    if (v_count_all > 0) {
        stats.v_l2_avg = v_l2_all / v_count_all;
        stats.v_cos_avg = v_cos_all / v_count_all;
    }

    stats.kv_l2_avg =
            (stats.k_l2_avg + stats.v_l2_avg) / 2.0;

    stats.kv_cos_avg =
            (stats.k_cos_avg + stats.v_cos_avg) / 2.0;

    stats.can_reuse_as_delta =
            stats.kv_cos_avg > 0.98 || stats.kv_l2_avg < 0.05;

    return true;
}
// 量化相关
static int8_t kv_delta_quant_q8(float x, float scale) {
    if (scale <= 0.0f) {
        return 0;
    }

    int v = (int) std::round(x / scale);
    v = std::max(-127, std::min(127, v));

    return (int8_t) v;
}
static float kv_delta_dequant_q8(int8_t x, float scale) {
    return (float) x * scale;
}

// Debug-only dump for comparing the raw anchor/child KV values before Q8
// encoding. It is intentionally sampled because tensor_get can synchronize a
// GPU buffer and printing every element would dominate the experiment.
static bool kv_delta_debug_take_dump_slot() {
    const char * enabled = std::getenv("LLAMA_KV_DELTA_DEBUG");
    if (enabled == nullptr || std::atoi(enabled) == 0) {
        return false;
    }

    const char * limit_text = std::getenv("LLAMA_KV_DELTA_DEBUG_LIMIT");
    const int limit = limit_text == nullptr ? 1 : std::atoi(limit_text);
    if (limit <= 0) {
        return false;
    }

    static int dump_count = 0;
    if (dump_count >= limit) {
        return false;
    }
    ++dump_count;
    return true;
}

static void kv_delta_debug_dump_tensor(
        const char * kind,
        uint32_t layer_id,
        const ggml_tensor * anchor,
        const ggml_tensor * child,
        const llama_kv_cells & cells_anchor,
        const llama_kv_cells & cells_child,
        const std::vector<uint32_t> & anchor_cells,
        const std::vector<uint32_t> & child_cells,
        uint32_t n_embd,
        bool transposed,
        llama_pos p0,
        llama_pos p1) {
    if (anchor == nullptr || child == nullptr || n_embd == 0 || p1 <= p0) {
        return;
    }

    const uint32_t rows = std::min<uint32_t>(2, (uint32_t) (p1 - p0));
    const uint32_t cols = std::min<uint32_t>(8, n_embd);
    const size_t element_size = ggml_type_size(anchor->type);
    const size_t row_size = ggml_row_size(anchor->type, n_embd);

    fprintf(stderr,
            "[KV_DELTA_DEBUG] layer=%u kind=%s rows=%u cols=%u transposed=%d type=%d\n",
            layer_id, kind, rows, cols, transposed ? 1 : 0, (int) anchor->type);

    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t cell_a = anchor_cells[row];
        const uint32_t cell_b = child_cells[row];
        std::vector<uint8_t> raw_a(transposed ? element_size : row_size);
        std::vector<uint8_t> raw_b(transposed ? element_size : row_size);
        if (!transposed) {
            ggml_backend_tensor_get(anchor, raw_a.data(), cell_a * row_size, row_size);
            ggml_backend_tensor_get(child, raw_b.data(), cell_b * row_size, row_size);
        }

        std::vector<float> values_a(cols);
        std::vector<float> values_b(cols);
        std::vector<float> values_delta(cols);
        float max_abs = 0.0f;
        for (uint32_t col = 0; col < n_embd; ++col) {
            if (transposed) {
                ggml_backend_tensor_get(
                        anchor, raw_a.data(),
                        (cell_a + col * cells_anchor.size()) * element_size,
                        element_size);
                ggml_backend_tensor_get(
                        child, raw_b.data(),
                        (cell_b + col * cells_child.size()) * element_size,
                        element_size);
            }
            const float value_a = kv_probe_read_scalar(
                    raw_a, anchor->type, transposed ? 0 : col);
            const float value_b = kv_probe_read_scalar(
                    raw_b, child->type, transposed ? 0 : col);
            const float value_delta = value_b - value_a;
            max_abs = std::max(max_abs, std::fabs(value_delta));
            if (col < cols) {
                values_a[col] = value_a;
                values_b[col] = value_b;
                values_delta[col] = value_delta;
            }
        }

        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        fprintf(stderr, "  token=%lld anchor=[", (long long) (p0 + row));
        for (uint32_t col = 0; col < cols; ++col) {
            fprintf(stderr, "%s%.7g", col == 0 ? "" : ", ", values_a[col]);
        }
        fprintf(stderr, "] child=[");
        for (uint32_t col = 0; col < cols; ++col) {
            fprintf(stderr, "%s%.7g", col == 0 ? "" : ", ", values_b[col]);
        }
        fprintf(stderr, "] delta=[");
        for (uint32_t col = 0; col < cols; ++col) {
            fprintf(stderr, "%s%.7g", col == 0 ? "" : ", ", values_delta[col]);
        }
        fprintf(stderr, "] full_max_abs=%.7g scale=%.7g q8=[", max_abs, scale);
        for (uint32_t col = 0; col < cols; ++col) {
            fprintf(stderr, "%s%d", col == 0 ? "" : ", ",
                    (int) kv_delta_quant_q8(values_delta[col], scale));
        }
        fprintf(stderr, "]\n");
    }
}
bool llama_kv_cache::seq_delta_has_branch(llama_seq_id seq_id) const {
    return seq_delta_find_branch(seq_id) != nullptr;
}
const llama_kv_cache::kv_delta_branch * llama_kv_cache::seq_delta_find_branch(
        llama_seq_id seq_id) const {
    for (const auto & branch : delta_branches) {
        if (branch.child_seq_id == seq_id && branch.enabled) {
            return &branch;
        }
    }

    return nullptr;
}

bool llama_kv_cache::seq_delta_build_branch_async(
        llama_seq_id seq_anchor,
        llama_seq_id seq_child_full,
        llama_seq_id seq_child_delta,
        llama_pos p0,
        llama_pos p1,
        int32_t parent_node_id,
        int32_t child_node_id,
        uint64_t & job_id,
        int32_t layer_id) {
    job_id = 0;
    if (seq_anchor < 0 || seq_child_full < 0 || seq_child_delta < 0 || p1 <= p0 ||
            (size_t) seq_anchor >= seq_to_stream.size() ||
            (size_t) seq_child_full >= seq_to_stream.size() ||
            (size_t) seq_child_delta >= seq_to_stream.size()) {
        return false;
    }
    p0 = std::max<llama_pos>(0, p0);

    const uint32_t stream_anchor = seq_to_stream[seq_anchor];
    const uint32_t stream_child = seq_to_stream[seq_child_full];
    const auto & cells_anchor = v_cells[stream_anchor];
    const auto & cells_child = v_cells[stream_child];
    const int32_t n_tokens = (int32_t) (p1 - p0);

    auto find_cell = [](
            const llama_kv_cells & cells,
            llama_seq_id seq_id,
            llama_pos pos,
            uint32_t & cell_id) -> bool {
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.seq_has(i, seq_id) && cells.pos_get(i) == pos) {
                cell_id = i;
                return true;
            }
        }
        return false;
    };

    std::vector<uint32_t> anchor_cells(n_tokens);
    std::vector<uint32_t> child_cells(n_tokens);
    for (int32_t token = 0; token < n_tokens; ++token) {
        const llama_pos pos = p0 + token;
        if (!find_cell(cells_anchor, seq_anchor, pos, anchor_cells[token]) ||
                !find_cell(cells_child, seq_child_full, pos, child_cells[token])) {
            return false;
        }
    }

    kv_delta_async_build build;
    build.branch.anchor_seq_id = seq_anchor;
    build.branch.child_seq_id = seq_child_delta;
    build.branch.p0 = p0;
    build.branch.p1 = p1;
    build.branch.parent_node_id = parent_node_id;
    build.branch.child_node_id = child_node_id;
    build.branch.enabled = true;

    const bool dump_debug = kv_delta_debug_take_dump_slot();
    if (dump_debug) {
        std::vector<size_t> debug_layers = { 0, layers.size() / 2, layers.size() - 1 };
        std::sort(debug_layers.begin(), debug_layers.end());
        debug_layers.erase(
                std::unique(debug_layers.begin(), debug_layers.end()), debug_layers.end());
        for (const size_t layer_index : debug_layers) {
            const auto & layer = layers[layer_index];
            kv_delta_debug_dump_tensor(
                    "K", layer.il,
                    layer.k_stream[stream_anchor], layer.k_stream[stream_child],
                    cells_anchor, cells_child, anchor_cells, child_cells,
                    hparams.n_embd_k_gqa(layer.il), false, p0, p1);
            kv_delta_debug_dump_tensor(
                    "V", layer.il,
                    layer.v_stream[stream_anchor], layer.v_stream[stream_child],
                    cells_anchor, cells_child, anchor_cells, child_cells,
                    hparams.n_embd_v_gqa(layer.il), v_trans, p0, p1);
        }
    }

    std::vector<ggml_backend_cuda_kv_delta_input> inputs;
    inputs.reserve(layers.size() * 2);
    ggml_tensor * first_tensor = nullptr;

    auto add_input = [&](ggml_tensor * anchor, ggml_tensor * child,
                         uint32_t layer_id, bool is_k, uint32_t n_embd,
                         bool transposed) {
        if (anchor == nullptr || child == nullptr || anchor->buffer == nullptr ||
                child->buffer == nullptr) {
            return;
        }
        kv_delta_tensor delta;
        delta.layer_id = (int32_t) layer_id;
        delta.is_k = is_k;
        delta.p0 = p0;
        delta.p1 = p1;
        delta.n_embd = (int32_t) n_embd;
        delta.dtype = kv_delta_dtype::Q8;
        delta.q8.resize((size_t) n_tokens * n_embd);
        delta.scales.resize(n_tokens, 1.0f);
        build.branch.layer_deltas.push_back(std::move(delta));

        kv_delta_tensor & output = build.branch.layer_deltas.back();
        inputs.push_back({
                anchor,
                child,
                anchor_cells.data(),
                child_cells.data(),
                n_tokens,
                (int32_t) n_embd,
                transposed ? 1 : 0,
                (int32_t) cells_anchor.size(),
                (int32_t) cells_child.size(),
                output.q8.data(),
                output.scales.data()});
        build.branch.delta_q8_bytes += output.q8.size() * sizeof(int8_t);
        build.branch.delta_scale_bytes += output.scales.size() * sizeof(float);
        build.branch.full_kv_bytes_equivalent +=
                (uint64_t) n_tokens * n_embd * ggml_type_size(anchor->type);
        if (first_tensor == nullptr) {
            first_tensor = anchor;
        }
    };

    for (const auto & layer : layers) {
        if (layer_id >= 0 && (int32_t) layer.il != layer_id) {
            continue;
        }
        const uint32_t il = layer.il;
        add_input(
                layer.k_stream[stream_anchor], layer.k_stream[stream_child],
                il, true, hparams.n_embd_k_gqa(il), false);
        add_input(
                layer.v_stream[stream_anchor], layer.v_stream[stream_child],
                il, false, hparams.n_embd_v_gqa(il), v_trans);
    }
    if (inputs.empty() || first_tensor == nullptr) {
        return false;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(first_tensor->buffer);
    ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
    if (device == nullptr) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    auto submit = (ggml_backend_cuda_kv_delta_submit_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_delta_submit");
    build.finish = (ggml_backend_cuda_kv_delta_finish_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_delta_finish");
    build.cancel = (ggml_backend_cuda_kv_delta_cancel_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_kv_delta_cancel");
    if (submit == nullptr || build.finish == nullptr || build.cancel == nullptr) {
        return false;
    }

    build.backend_job = submit(inputs.data(), inputs.size());
    if (build.backend_job == nullptr) {
        return false;
    }

    job_id = next_delta_async_id++;
    if (job_id == 0) {
        job_id = next_delta_async_id++;
    }
    delta_async_builds.emplace(job_id, std::move(build));
    return true;
}

bool llama_kv_cache::seq_delta_build_branch_finish(uint64_t job_id) {
    auto it = delta_async_builds.find(job_id);
    if (it == delta_async_builds.end()) {
        return false;
    }
    kv_delta_async_build build = std::move(it->second);
    delta_async_builds.erase(it);
    const bool ok = build.backend_job != nullptr && build.finish != nullptr &&
            build.finish(build.backend_job);
    if (!ok) {
        return false;
    }
    auto existing = std::find_if(
            delta_branches.begin(), delta_branches.end(),
            [&](const kv_delta_branch & branch) {
                return branch.anchor_seq_id == build.branch.anchor_seq_id &&
                        branch.child_seq_id == build.branch.child_seq_id &&
                        branch.parent_node_id == build.branch.parent_node_id &&
                        branch.child_node_id == build.branch.child_node_id;
            });
    if (existing == delta_branches.end()) {
        delta_branches.push_back(std::move(build.branch));
        return true;
    }

    for (auto & source : build.branch.layer_deltas) {
        auto destination = std::find_if(
                existing->layer_deltas.begin(), existing->layer_deltas.end(),
                [&](const kv_delta_tensor & tensor) {
                    return tensor.layer_id == source.layer_id &&
                            tensor.is_k == source.is_k && tensor.n_embd == source.n_embd;
                });
        if (destination == existing->layer_deltas.end()) {
            existing->layer_deltas.push_back(std::move(source));
        } else if (destination->p1 == source.p0) {
            destination->q8.insert(
                    destination->q8.end(), source.q8.begin(), source.q8.end());
            destination->scales.insert(
                    destination->scales.end(), source.scales.begin(), source.scales.end());
            destination->p1 = source.p1;
        } else if (source.p1 == destination->p0) {
            source.q8.insert(source.q8.end(),
                    destination->q8.begin(), destination->q8.end());
            source.scales.insert(source.scales.end(),
                    destination->scales.begin(), destination->scales.end());
            source.p1 = destination->p1;
            *destination = std::move(source);
        } else {
            return false;
        }
    }
    existing->p0 = std::min(existing->p0, build.branch.p0);
    existing->p1 = std::max(existing->p1, build.branch.p1);
    existing->anchor_ref_bytes += build.branch.anchor_ref_bytes;
    existing->delta_q8_bytes += build.branch.delta_q8_bytes;
    existing->delta_scale_bytes += build.branch.delta_scale_bytes;
    existing->full_kv_bytes_equivalent += build.branch.full_kv_bytes_equivalent;
    return true;
}

bool llama_kv_cache::seq_delta_build_branch_cancel(uint64_t job_id) {
    auto it = delta_async_builds.find(job_id);
    if (it == delta_async_builds.end()) {
        return false;
    }
    kv_delta_async_build build = std::move(it->second);
    delta_async_builds.erase(it);
    if (build.backend_job != nullptr && build.cancel != nullptr) {
        build.cancel(build.backend_job);
    }
    return true;
}

bool llama_kv_cache::seq_delta_build_branch(
        llama_seq_id seq_anchor,
        llama_seq_id seq_child_full,
        llama_seq_id seq_child_delta,
        llama_pos p0,
        llama_pos p1,
        int32_t parent_node_id,
        int32_t child_node_id) {
    return seq_delta_build_branch_impl(
            seq_anchor, seq_child_full, seq_child_delta, p0, p1,
            parent_node_id, child_node_id, true);
}

bool llama_kv_cache::seq_delta_build_branch_cpu(
        llama_seq_id seq_anchor,
        llama_seq_id seq_child_full,
        llama_seq_id seq_child_delta,
        llama_pos p0,
        llama_pos p1,
        int32_t parent_node_id,
        int32_t child_node_id) {
    return seq_delta_build_branch_impl(
            seq_anchor, seq_child_full, seq_child_delta, p0, p1,
            parent_node_id, child_node_id, false);
}

bool llama_kv_cache::seq_delta_build_branch_impl(
        llama_seq_id seq_anchor,
        llama_seq_id seq_child_full,
        llama_seq_id seq_child_delta,
        llama_pos p0,
        llama_pos p1,
        int32_t parent_node_id,
        int32_t child_node_id,
        bool try_fused_cuda) {
    uint64_t async_job_id = 0;
    if (try_fused_cuda && seq_delta_build_branch_async(
                seq_anchor, seq_child_full, seq_child_delta, p0, p1,
                parent_node_id, child_node_id, async_job_id)) {
        if (seq_delta_build_branch_finish(async_job_id)) {
            return true;
        }
    }
    if (seq_anchor < 0 || seq_child_full < 0 || seq_child_delta < 0) {
        return false;
    }

    if ((size_t) seq_anchor >= seq_to_stream.size() ||
            (size_t) seq_child_full >= seq_to_stream.size() ||
            (size_t) seq_child_delta >= seq_to_stream.size()) {
        return false;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 <= p0) {
        return false;
    }

    const uint32_t stream_anchor = seq_to_stream[seq_anchor];
    const uint32_t stream_child  = seq_to_stream[seq_child_full];

    const auto & cells_anchor = v_cells[stream_anchor];
    const auto & cells_child  = v_cells[stream_child];

    auto find_cell = [](
            const llama_kv_cells & cells,
            llama_seq_id seq_id,
            llama_pos pos,
            uint32_t & cell_id) -> bool {
        for (uint32_t i = 0; i < cells.size(); i++) {
            if (!cells.seq_has(i, seq_id)) {
                continue;
            }

            if (cells.pos_get(i) == pos) {
                cell_id = i;
                return true;
            }
        }

        return false;
    };

    kv_delta_branch branch;

    branch.anchor_seq_id = seq_anchor;
    branch.child_seq_id = seq_child_delta;
    branch.p0 = p0;
    branch.p1 = p1;
    branch.parent_node_id = parent_node_id;
    branch.child_node_id = child_node_id;
    branch.enabled = true;

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa =
                hparams.n_embd_k_gqa(il);

        const uint32_t n_embd_v_gqa =
                hparams.n_embd_v_gqa(il);

        ggml_tensor * k_anchor = layer.k_stream[stream_anchor];
        ggml_tensor * k_child  = layer.k_stream[stream_child];

        ggml_tensor * v_anchor = layer.v_stream[stream_anchor];
        ggml_tensor * v_child  = layer.v_stream[stream_child];

        if (k_anchor != nullptr && k_child != nullptr) {
            kv_delta_tensor delta_k;

            delta_k.layer_id = (int32_t) il;
            delta_k.is_k = true;
            delta_k.p0 = p0;
            delta_k.p1 = p1;
            delta_k.n_embd = (int32_t) n_embd_k_gqa;
            delta_k.dtype = kv_delta_dtype::Q8;

            const int n_tokens = (int) (p1 - p0);
            delta_k.q8.resize((size_t) n_tokens * n_embd_k_gqa);
            delta_k.scales.resize(n_tokens, 1.0f);

            const size_t k_size_row =
                    ggml_row_size(k_anchor->type, n_embd_k_gqa);

            for (llama_pos pos = p0; pos < p1; pos++) {
                uint32_t cell_anchor = 0;
                uint32_t cell_child = 0;

                if (!find_cell(cells_anchor, seq_anchor, pos, cell_anchor) ||
                        !find_cell(cells_child, seq_child_full, pos, cell_child)) {
                    return false;
                }

                std::vector<uint8_t> raw_anchor(k_size_row);
                std::vector<uint8_t> raw_child(k_size_row);

                ggml_backend_tensor_get(
                        k_anchor,
                        raw_anchor.data(),
                        cell_anchor * k_size_row,
                        k_size_row);

                ggml_backend_tensor_get(
                        k_child,
                        raw_child.data(),
                        cell_child * k_size_row,
                        k_size_row);

                const int token_idx = (int) (pos - p0);

                float max_abs = 0.0f;

                for (uint32_t i = 0; i < n_embd_k_gqa; i++) {
                    const float a =
                            kv_probe_read_scalar(raw_anchor, k_anchor->type, i);

                    const float b =
                            kv_probe_read_scalar(raw_child, k_child->type, i);

                    const float d = b - a;

                    max_abs = std::max(max_abs, std::fabs(d));
                }

                const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
                delta_k.scales[token_idx] = scale;

                for (uint32_t i = 0; i < n_embd_k_gqa; i++) {
                    const float a =
                            kv_probe_read_scalar(raw_anchor, k_anchor->type, i);

                    const float b =
                            kv_probe_read_scalar(raw_child, k_child->type, i);

                    const float d = b - a;

                    delta_k.q8[(size_t) token_idx * n_embd_k_gqa + i] =
                            kv_delta_quant_q8(d, scale);
                }
            }

            branch.delta_q8_bytes +=
                    (uint64_t) delta_k.q8.size() * sizeof(int8_t);

            branch.delta_scale_bytes +=
                    (uint64_t) delta_k.scales.size() * sizeof(float);

            branch.full_kv_bytes_equivalent +=
                    (uint64_t) (p1 - p0) *
                    (uint64_t) n_embd_k_gqa *
                    (uint64_t) ggml_type_size(k_anchor->type);

            branch.layer_deltas.push_back(std::move(delta_k));
        }

        if (v_anchor != nullptr && v_child != nullptr) {
            kv_delta_tensor delta_v;

            delta_v.layer_id = (int32_t) il;
            delta_v.is_k = false;
            delta_v.p0 = p0;
            delta_v.p1 = p1;
            delta_v.n_embd = (int32_t) n_embd_v_gqa;
            delta_v.dtype = kv_delta_dtype::Q8;

            const int n_tokens = (int) (p1 - p0);
            delta_v.q8.resize((size_t) n_tokens * n_embd_v_gqa);
            delta_v.scales.resize(n_tokens, 1.0f);

            if (v_trans) {
                const size_t v_size_el =
                        ggml_type_size(v_anchor->type);

                for (llama_pos pos = p0; pos < p1; pos++) {
                    uint32_t cell_anchor = 0;
                    uint32_t cell_child = 0;

                    if (!find_cell(cells_anchor, seq_anchor, pos, cell_anchor) ||
                            !find_cell(cells_child, seq_child_full, pos, cell_child)) {
                        return false;
                    }

                    const int token_idx = (int) (pos - p0);
                    std::vector<float> diff(n_embd_v_gqa);

                    float max_abs = 0.0f;

                    for (uint32_t i = 0; i < n_embd_v_gqa; i++) {
                        std::vector<uint8_t> raw_anchor(v_size_el);
                        std::vector<uint8_t> raw_child(v_size_el);

                        const size_t offset_anchor =
                                (cell_anchor + i * cells_anchor.size()) * v_size_el;

                        const size_t offset_child =
                                (cell_child + i * cells_child.size()) * v_size_el;

                        ggml_backend_tensor_get(
                                v_anchor,
                                raw_anchor.data(),
                                offset_anchor,
                                v_size_el);

                        ggml_backend_tensor_get(
                                v_child,
                                raw_child.data(),
                                offset_child,
                                v_size_el);

                        const float a =
                                kv_probe_read_scalar(raw_anchor, v_anchor->type, 0);

                        const float b =
                                kv_probe_read_scalar(raw_child, v_child->type, 0);

                        diff[i] = b - a;
                        max_abs = std::max(max_abs, std::fabs(diff[i]));
                    }

                    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
                    delta_v.scales[token_idx] = scale;

                    for (uint32_t i = 0; i < n_embd_v_gqa; i++) {
                        delta_v.q8[(size_t) token_idx * n_embd_v_gqa + i] =
                                kv_delta_quant_q8(diff[i], scale);
                    }
                }
            } else {
                const size_t v_size_row =
                        ggml_row_size(v_anchor->type, n_embd_v_gqa);

                for (llama_pos pos = p0; pos < p1; pos++) {
                    uint32_t cell_anchor = 0;
                    uint32_t cell_child = 0;

                    if (!find_cell(cells_anchor, seq_anchor, pos, cell_anchor) ||
                            !find_cell(cells_child, seq_child_full, pos, cell_child)) {
                        return false;
                    }

                    std::vector<uint8_t> raw_anchor(v_size_row);
                    std::vector<uint8_t> raw_child(v_size_row);

                    ggml_backend_tensor_get(
                            v_anchor,
                            raw_anchor.data(),
                            cell_anchor * v_size_row,
                            v_size_row);

                    ggml_backend_tensor_get(
                            v_child,
                            raw_child.data(),
                            cell_child * v_size_row,
                            v_size_row);

                    const int token_idx = (int) (pos - p0);

                    float max_abs = 0.0f;

                    for (uint32_t i = 0; i < n_embd_v_gqa; i++) {
                        const float a =
                                kv_probe_read_scalar(raw_anchor, v_anchor->type, i);

                        const float b =
                                kv_probe_read_scalar(raw_child, v_child->type, i);

                        const float d = b - a;

                        max_abs = std::max(max_abs, std::fabs(d));
                    }

                    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
                    delta_v.scales[token_idx] = scale;

                    for (uint32_t i = 0; i < n_embd_v_gqa; i++) {
                        const float a =
                                kv_probe_read_scalar(raw_anchor, v_anchor->type, i);

                        const float b =
                                kv_probe_read_scalar(raw_child, v_child->type, i);

                        const float d = b - a;

                        delta_v.q8[(size_t) token_idx * n_embd_v_gqa + i] =
                                kv_delta_quant_q8(d, scale);
                    }
                }
            }

            branch.delta_q8_bytes +=
                    (uint64_t) delta_v.q8.size() * sizeof(int8_t);

            branch.delta_scale_bytes +=
                    (uint64_t) delta_v.scales.size() * sizeof(float);

            branch.full_kv_bytes_equivalent +=
                    (uint64_t) (p1 - p0) *
                    (uint64_t) n_embd_v_gqa *
                    (uint64_t) ggml_type_size(v_anchor->type);

            branch.layer_deltas.push_back(std::move(delta_v));
        }
    }

    delta_branches.push_back(std::move(branch));

    LLAMA_LOG_INFO(
            "%s: registered delta branch anchor_seq=%d child_delta_seq=%d range=[%d,%d), tensors=%zu\n",
            __func__,
            seq_anchor,
            seq_child_delta,
            (int) p0,
            (int) p1,
            delta_branches.back().layer_deltas.size());

    return true;
}

bool llama_kv_cache::seq_delta_materialize(
        llama_seq_id seq_anchor,
        llama_seq_id seq_child_full,
        llama_seq_id seq_dst,
        llama_pos p0,
        llama_pos p1,
        kv_delta_materialize_stats & stats) {
    stats = {};

    if (seq_anchor < 0 || seq_child_full < 0 || seq_dst < 0) {
        return false;
    }

    if ((size_t) seq_anchor >= seq_to_stream.size() ||
            (size_t) seq_child_full >= seq_to_stream.size() ||
            (size_t) seq_dst >= seq_to_stream.size()) {
        return false;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 <= p0) {
        return false;
    }

    const uint32_t stream_anchor = seq_to_stream[seq_anchor];
    const uint32_t stream_child  = seq_to_stream[seq_child_full];
    const uint32_t stream_dst    = seq_to_stream[seq_dst];

    auto & cells_anchor = v_cells[stream_anchor];
    auto & cells_child  = v_cells[stream_child];
    auto & cells_dst    = v_cells[stream_dst];

    auto find_cell = [](
            const llama_kv_cells & cells,
            llama_seq_id seq_id,
            llama_pos pos,
            uint32_t & cell_id) -> bool {
        for (uint32_t i = 0; i < cells.size(); i++) {
            if (!cells.seq_has(i, seq_id)) {
                continue;
            }

            if (cells.pos_get(i) == pos) {
                cell_id = i;
                return true;
            }
        }

        return false;
    };

    stats.n_layers = (int32_t) layers.size();
    stats.n_tokens = (int32_t) (p1 - p0);

    int materialized_token_count = 0;

    for (llama_pos pos = p0; pos < p1; pos++) {
        uint32_t cell_anchor = 0;
        uint32_t cell_child  = 0;
        uint32_t cell_dst    = 0;

        const bool has_anchor = find_cell(cells_anchor, seq_anchor, pos, cell_anchor);
        const bool has_child  = find_cell(cells_child,  seq_child_full, pos, cell_child);
        const bool has_dst    = find_cell(cells_dst,    seq_dst, pos, cell_dst);

        if (!has_anchor || !has_child || !has_dst) {
            stats.missing_cell_count++;
            continue;
        }

        materialized_token_count++;

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_k_gqa =
                    hparams.n_embd_k_gqa(il);

            const uint32_t n_embd_v_gqa =
                    hparams.n_embd_v_gqa(il);

            ggml_tensor * k_anchor = layer.k_stream[stream_anchor];
            ggml_tensor * k_child  = layer.k_stream[stream_child];
            ggml_tensor * k_dst    = layer.k_stream[stream_dst];

            ggml_tensor * v_anchor = layer.v_stream[stream_anchor];
            ggml_tensor * v_child  = layer.v_stream[stream_child];
            ggml_tensor * v_dst    = layer.v_stream[stream_dst];

            if (k_anchor != nullptr && k_child != nullptr && k_dst != nullptr) {
                const size_t k_size_row =
                        ggml_row_size(k_anchor->type, n_embd_k_gqa);

                std::vector<uint8_t> raw_anchor(k_size_row);
                std::vector<uint8_t> raw_child(k_size_row);
                std::vector<uint8_t> raw_dst(k_size_row);

                ggml_backend_tensor_get(
                        k_anchor,
                        raw_anchor.data(),
                        cell_anchor * k_size_row,
                        k_size_row);

                ggml_backend_tensor_get(
                        k_child,
                        raw_child.data(),
                        cell_child * k_size_row,
                        k_size_row);

                for (uint32_t i = 0; i < n_embd_k_gqa; i++) {
                    const float a =
                            kv_probe_read_scalar(raw_anchor, k_anchor->type, i);

                    const float b =
                            kv_probe_read_scalar(raw_child, k_child->type, i);

                    const float delta = b - a;
                    const float out = a + delta;

                    kv_probe_write_scalar(raw_dst, k_dst->type, i, out);
                }

                ggml_backend_tensor_set(
                        k_dst,
                        raw_dst.data(),
                        cell_dst * k_size_row,
                        k_size_row);

                stats.delta_fp32_bytes +=
                        (uint64_t) n_embd_k_gqa * sizeof(float);

                stats.materialized_kv_bytes +=
                        (uint64_t) k_size_row;
            }

            if (v_anchor != nullptr && v_child != nullptr && v_dst != nullptr) {
                const size_t v_size_el =
                        ggml_type_size(v_anchor->type);

                if (!v_trans) {
                    const size_t v_size_row =
                            ggml_row_size(v_anchor->type, n_embd_v_gqa);

                    std::vector<uint8_t> raw_anchor(v_size_row);
                    std::vector<uint8_t> raw_child(v_size_row);
                    std::vector<uint8_t> raw_dst(v_size_row);

                    ggml_backend_tensor_get(
                            v_anchor,
                            raw_anchor.data(),
                            cell_anchor * v_size_row,
                            v_size_row);

                    ggml_backend_tensor_get(
                            v_child,
                            raw_child.data(),
                            cell_child * v_size_row,
                            v_size_row);

                    for (uint32_t i = 0; i < n_embd_v_gqa; i++) {
                        const float a =
                                kv_probe_read_scalar(raw_anchor, v_anchor->type, i);

                        const float b =
                                kv_probe_read_scalar(raw_child, v_child->type, i);

                        const float delta = b - a;
                        const float out = a + delta;

                        kv_probe_write_scalar(raw_dst, v_dst->type, i, out);
                    }

                    ggml_backend_tensor_set(
                            v_dst,
                            raw_dst.data(),
                            cell_dst * v_size_row,
                            v_size_row);

                    stats.delta_fp32_bytes +=
                            (uint64_t) n_embd_v_gqa * sizeof(float);

                    stats.materialized_kv_bytes +=
                            (uint64_t) v_size_row;
                } else {
                    for (uint32_t i = 0; i < n_embd_v_gqa; i++) {
                        std::vector<uint8_t> raw_anchor(v_size_el);
                        std::vector<uint8_t> raw_child(v_size_el);
                        std::vector<uint8_t> raw_dst(v_size_el);

                        const size_t offset_anchor =
                                (cell_anchor + i * cells_anchor.size()) * v_size_el;

                        const size_t offset_child =
                                (cell_child + i * cells_child.size()) * v_size_el;

                        const size_t offset_dst =
                                (cell_dst + i * cells_dst.size()) * v_size_el;

                        ggml_backend_tensor_get(
                                v_anchor,
                                raw_anchor.data(),
                                offset_anchor,
                                v_size_el);

                        ggml_backend_tensor_get(
                                v_child,
                                raw_child.data(),
                                offset_child,
                                v_size_el);

                        const float a =
                                kv_probe_read_scalar(raw_anchor, v_anchor->type, 0);

                        const float b =
                                kv_probe_read_scalar(raw_child, v_child->type, 0);

                        const float delta = b - a;
                        const float out = a + delta;

                        kv_probe_write_scalar(raw_dst, v_dst->type, 0, out);

                        ggml_backend_tensor_set(
                                v_dst,
                                raw_dst.data(),
                                offset_dst,
                                v_size_el);
                    }

                    stats.delta_fp32_bytes +=
                            (uint64_t) n_embd_v_gqa * sizeof(float);

                    stats.materialized_kv_bytes +=
                            (uint64_t) n_embd_v_gqa * v_size_el;
                }
            }

            stats.materialized_layers++;
        }
    }

    stats.materialized_tokens = materialized_token_count;
    stats.ok =
            materialized_token_count > 0 &&
            stats.missing_cell_count == 0;

    return stats.ok;
}

bool llama_kv_cache::seq_delta_materialize_branch(
        llama_seq_id seq_anchor,
        llama_seq_id seq_child_delta,
        llama_seq_id seq_dst,
        llama_pos p0,
        llama_pos p1,
        kv_delta_materialize_stats & stats) {
    stats = {};

    const kv_delta_branch * branch = seq_delta_find_branch(seq_child_delta);
    if (branch == nullptr || branch->anchor_seq_id != seq_anchor) {
        return false;
    }
    if (seq_anchor < 0 || seq_dst < 0 ||
            (size_t) seq_anchor >= seq_to_stream.size() ||
            (size_t) seq_dst >= seq_to_stream.size()) {
        return false;
    }

    p0 = std::max(p0, branch->p0);
    p1 = std::min(p1, branch->p1);
    if (p1 <= p0) {
        return false;
    }

    const uint32_t stream_anchor = seq_to_stream[seq_anchor];
    const uint32_t stream_dst = seq_to_stream[seq_dst];
    auto & cells_anchor = v_cells[stream_anchor];
    auto & cells_dst = v_cells[stream_dst];

    auto find_cell = [](
            const llama_kv_cells & cells,
            llama_seq_id seq_id,
            llama_pos pos,
            uint32_t & cell_id) -> bool {
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.seq_has(i, seq_id) && cells.pos_get(i) == pos) {
                cell_id = i;
                return true;
            }
        }
        return false;
    };

    auto find_delta = [branch](int32_t layer_id, bool is_k) -> const kv_delta_tensor * {
        for (const auto & delta : branch->layer_deltas) {
            if (delta.layer_id == layer_id && delta.is_k == is_k) {
                return &delta;
            }
        }
        return nullptr;
    };

    stats.n_layers = (int32_t) layers.size();
    stats.n_tokens = (int32_t) (p1 - p0);

    for (llama_pos pos = p0; pos < p1; ++pos) {
        uint32_t cell_anchor = 0;
        uint32_t cell_dst = 0;
        if (!find_cell(cells_anchor, seq_anchor, pos, cell_anchor) ||
                !find_cell(cells_dst, seq_dst, pos, cell_dst)) {
            stats.missing_cell_count++;
            continue;
        }

        stats.materialized_tokens++;
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;
            const kv_delta_tensor * delta_k = find_delta((int32_t) il, true);
            const kv_delta_tensor * delta_v = find_delta((int32_t) il, false);
            bool wrote_layer = false;

            if (delta_k != nullptr && delta_k->dtype == kv_delta_dtype::Q8 &&
                    pos >= delta_k->p0 && pos < delta_k->p1) {
                const uint32_t n_embd = hparams.n_embd_k_gqa(il);
                const size_t token_index = (size_t) (pos - delta_k->p0);
                const size_t base = token_index * n_embd;
                if (token_index < delta_k->scales.size() &&
                        base + n_embd <= delta_k->q8.size()) {
                    ggml_tensor * anchor = layer.k_stream[stream_anchor];
                    ggml_tensor * dst = layer.k_stream[stream_dst];
                    if (anchor != nullptr && dst != nullptr) {
                        const size_t row_size = ggml_row_size(anchor->type, n_embd);
                        std::vector<uint8_t> raw_anchor(row_size);
                        std::vector<uint8_t> raw_dst(row_size);
                        ggml_backend_tensor_get(
                                anchor, raw_anchor.data(), cell_anchor * row_size, row_size);
                        const float scale = delta_k->scales[token_index];
                        for (uint32_t i = 0; i < n_embd; ++i) {
                            const float value =
                                    kv_probe_read_scalar(raw_anchor, anchor->type, i) +
                                    kv_delta_dequant_q8(delta_k->q8[base + i], scale);
                            kv_probe_write_scalar(raw_dst, dst->type, i, value);
                        }
                        ggml_backend_tensor_set(
                                dst, raw_dst.data(), cell_dst * row_size, row_size);
                        stats.delta_fp32_bytes += (uint64_t) n_embd * sizeof(float);
                        stats.materialized_kv_bytes += (uint64_t) row_size;
                        wrote_layer = true;
                    }
                }
            }

            if (delta_v != nullptr && delta_v->dtype == kv_delta_dtype::Q8 &&
                    pos >= delta_v->p0 && pos < delta_v->p1) {
                const uint32_t n_embd = hparams.n_embd_v_gqa(il);
                const size_t token_index = (size_t) (pos - delta_v->p0);
                const size_t base = token_index * n_embd;
                if (token_index < delta_v->scales.size() &&
                        base + n_embd <= delta_v->q8.size()) {
                    ggml_tensor * anchor = layer.v_stream[stream_anchor];
                    ggml_tensor * dst = layer.v_stream[stream_dst];
                    if (anchor != nullptr && dst != nullptr) {
                        const float scale = delta_v->scales[token_index];
                        if (!v_trans) {
                            const size_t row_size = ggml_row_size(anchor->type, n_embd);
                            std::vector<uint8_t> raw_anchor(row_size);
                            std::vector<uint8_t> raw_dst(row_size);
                            ggml_backend_tensor_get(
                                    anchor, raw_anchor.data(), cell_anchor * row_size, row_size);
                            for (uint32_t i = 0; i < n_embd; ++i) {
                                const float value =
                                        kv_probe_read_scalar(raw_anchor, anchor->type, i) +
                                        kv_delta_dequant_q8(delta_v->q8[base + i], scale);
                                kv_probe_write_scalar(raw_dst, dst->type, i, value);
                            }
                            ggml_backend_tensor_set(
                                    dst, raw_dst.data(), cell_dst * row_size, row_size);
                            stats.materialized_kv_bytes += (uint64_t) row_size;
                        } else {
                            const size_t value_size = ggml_type_size(anchor->type);
                            for (uint32_t i = 0; i < n_embd; ++i) {
                                std::vector<uint8_t> raw_anchor(value_size);
                                std::vector<uint8_t> raw_dst(value_size);
                                const size_t anchor_offset =
                                        (cell_anchor + i * cells_anchor.size()) * value_size;
                                const size_t dst_offset =
                                        (cell_dst + i * cells_dst.size()) * value_size;
                                ggml_backend_tensor_get(
                                        anchor, raw_anchor.data(), anchor_offset, value_size);
                                const float value =
                                        kv_probe_read_scalar(raw_anchor, anchor->type, 0) +
                                        kv_delta_dequant_q8(delta_v->q8[base + i], scale);
                                kv_probe_write_scalar(raw_dst, dst->type, 0, value);
                                ggml_backend_tensor_set(
                                        dst, raw_dst.data(), dst_offset, value_size);
                            }
                            stats.materialized_kv_bytes +=
                                    (uint64_t) n_embd * value_size;
                        }
                        stats.delta_fp32_bytes += (uint64_t) n_embd * sizeof(float);
                        wrote_layer = true;
                    }
                }
            }

            if (wrote_layer) {
                stats.materialized_layers++;
            }
        }
    }

    stats.ok = stats.materialized_tokens == stats.n_tokens &&
            stats.missing_cell_count == 0;
    return stats.ok;
}

bool llama_kv_cache::seq_delta_remove_branch(llama_seq_id seq_child_delta) {
    const size_t old_size = delta_branches.size();
    delta_branches.erase(
            std::remove_if(
                    delta_branches.begin(),
                    delta_branches.end(),
                    [seq_child_delta](const kv_delta_branch & branch) {
                        return branch.child_seq_id == seq_child_delta;
                    }),
            delta_branches.end());
    return delta_branches.size() != old_size;
}

bool llama_kv_cache::seq_delta_save_branch(
        llama_seq_id seq_child_delta,
        const char * path) const {
    if (path == nullptr) return false;
    const kv_delta_branch * branch = seq_delta_find_branch(seq_child_delta);
    if (branch == nullptr) return false;

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return false;

    const uint64_t magic = 0x4D4F42494C4F5241ULL; // "MOBILORA"
    const uint32_t version = 2;
    const uint32_t tensor_count = (uint32_t) branch->layer_deltas.size();
    output.write((const char *) &magic, sizeof(magic));
    output.write((const char *) &version, sizeof(version));
    output.write((const char *) &branch->p0, sizeof(branch->p0));
    output.write((const char *) &branch->p1, sizeof(branch->p1));
    output.write((const char *) &branch->parent_node_id, sizeof(branch->parent_node_id));
    output.write((const char *) &branch->child_node_id, sizeof(branch->child_node_id));
    output.write((const char *) &tensor_count, sizeof(tensor_count));
    output.write((const char *) &branch->full_kv_bytes_equivalent,
            sizeof(branch->full_kv_bytes_equivalent));

    for (const auto & delta : branch->layer_deltas) {
        const uint8_t is_k = delta.is_k ? 1 : 0;
        const uint8_t dtype = (uint8_t) delta.dtype;
        const uint64_t q8_size = delta.q8.size();
        const uint64_t scale_size = delta.scales.size();
        output.write((const char *) &delta.layer_id, sizeof(delta.layer_id));
        output.write((const char *) &is_k, sizeof(is_k));
        output.write((const char *) &delta.p0, sizeof(delta.p0));
        output.write((const char *) &delta.p1, sizeof(delta.p1));
        output.write((const char *) &delta.n_embd, sizeof(delta.n_embd));
        output.write((const char *) &dtype, sizeof(dtype));
        output.write((const char *) &q8_size, sizeof(q8_size));
        output.write((const char *) &scale_size, sizeof(scale_size));
        output.write((const char *) delta.q8.data(), (std::streamsize) q8_size);
        output.write((const char *) delta.scales.data(),
                (std::streamsize) (scale_size * sizeof(float)));
    }
    return (bool) output;
}

bool llama_kv_cache::seq_delta_load_branch(
        llama_seq_id seq_anchor,
        llama_seq_id seq_child_delta,
        const char * path,
        int32_t parent_node_id,
        int32_t child_node_id) {
    if (path == nullptr || seq_anchor < 0 || seq_child_delta < 0 ||
            (size_t) seq_anchor >= seq_to_stream.size() ||
            (size_t) seq_child_delta >= seq_to_stream.size()) return false;

    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    uint64_t magic = 0;
    uint32_t version = 0;
    llama_pos p0 = 0;
    llama_pos p1 = 0;
    int32_t stored_parent = -1;
    int32_t stored_child = -1;
    uint32_t tensor_count = 0;
    input.read((char *) &magic, sizeof(magic));
    input.read((char *) &version, sizeof(version));
    input.read((char *) &p0, sizeof(p0));
    input.read((char *) &p1, sizeof(p1));
    input.read((char *) &stored_parent, sizeof(stored_parent));
    input.read((char *) &stored_child, sizeof(stored_child));
    input.read((char *) &tensor_count, sizeof(tensor_count));
    if (!input || magic != 0x4D4F42494C4F5241ULL || (version != 1 && version != 2) ||
            p1 <= p0 || tensor_count > 2 * layers.size()) return false;

    seq_delta_remove_branch(seq_child_delta);
    kv_delta_branch branch;
    branch.anchor_seq_id = seq_anchor;
    branch.child_seq_id = seq_child_delta;
    branch.p0 = p0;
    branch.p1 = p1;
    branch.parent_node_id = parent_node_id >= 0 ? parent_node_id : stored_parent;
    branch.child_node_id = child_node_id >= 0 ? child_node_id : stored_child;
    branch.enabled = true;
    if (version >= 2) {
        input.read((char *) &branch.full_kv_bytes_equivalent,
                sizeof(branch.full_kv_bytes_equivalent));
        if (!input) return false;
    }

    for (uint32_t index = 0; index < tensor_count; ++index) {
        kv_delta_tensor delta;
        uint8_t is_k = 0;
        uint8_t dtype = 0;
        uint64_t q8_size = 0;
        uint64_t scale_size = 0;
        input.read((char *) &delta.layer_id, sizeof(delta.layer_id));
        input.read((char *) &is_k, sizeof(is_k));
        input.read((char *) &delta.p0, sizeof(delta.p0));
        input.read((char *) &delta.p1, sizeof(delta.p1));
        input.read((char *) &delta.n_embd, sizeof(delta.n_embd));
        input.read((char *) &dtype, sizeof(dtype));
        input.read((char *) &q8_size, sizeof(q8_size));
        input.read((char *) &scale_size, sizeof(scale_size));
        if (!input || q8_size > (1ULL << 32) || scale_size > (1ULL << 32)) return false;
        delta.is_k = is_k != 0;
        delta.dtype = (kv_delta_dtype) dtype;
        delta.q8.resize((size_t) q8_size);
        delta.scales.resize((size_t) scale_size);
        input.read((char *) delta.q8.data(), (std::streamsize) q8_size);
        input.read((char *) delta.scales.data(),
                (std::streamsize) (scale_size * sizeof(float)));
        if (!input || delta.dtype != kv_delta_dtype::Q8) return false;
        branch.delta_q8_bytes += q8_size;
        branch.delta_scale_bytes += scale_size * sizeof(float);
        branch.layer_deltas.push_back(std::move(delta));
    }
    delta_branches.push_back(std::move(branch));
    return true;
}

bool llama_kv_cache::seq_delta_get_branch_stats(
        llama_seq_id seq_child_delta,
        kv_delta_branch & stats) const {
    const kv_delta_branch * branch = seq_delta_find_branch(seq_child_delta);
    if (branch == nullptr) return false;
    stats = *branch;
    return true;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    if (physical_paged) {
        uint32_t result = 0;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const auto & cells = v_cells[sinfo.strm[s]];

            // 分页 KV 不要再固定 pad 到 256，否则 page_size=16 时会马上要求读取 256 个 KV。
            // 这里按 page_size 对齐，便于单页/多页管理。
            const uint32_t used = cells.used_max_p1();
            const uint32_t padded = used == 0 ? physical_page_size : GGML_PAD(used, physical_page_size);

            result = std::max(result, std::min(cells.size(), padded));
        }

        return result;
    }

    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    // 量化相关
    if (!sinfo.strm.empty()) {
        const llama_seq_id active_seq =
                (llama_seq_id) sinfo.strm[0];

        const kv_delta_branch * branch =
                seq_delta_find_branch(active_seq);

        if (branch != nullptr) {
            // 第一版先不在这里直接返回 compressed delta。
            // 因为 ggml 里需要构造 dense reconstructed tensor：
            // K_recon = K_anchor_view + dequant(delta)
            //
            // 这个函数当前只知道 il，不知道当前 attention mask 具体哪些 token 属于 delta。
            // 所以这里需要后续新增 build_delta_k_view(ctx, il, n_kv, sinfo, *branch)。
        }
    }
    // 分页相关
    if (physical_paged) {
        const int32_t ikv = map_layer_ids.at(il);

        GGML_ASSERT(sinfo.n_stream() == 1);

        const uint32_t stream_id = sinfo.strm[0];

        GGML_ASSERT(stream_id < physical_pages.size());
        GGML_ASSERT(!physical_pages[stream_id].empty());

        const uint32_t last_cell =
                sinfo.idxs[0].empty() ? 0 : sinfo.idxs[0].back();

        const auto & ref = get_page_ref(stream_id, last_cell);
        GGML_ASSERT(ref.page_id != UINT32_MAX);

        ggml_tensor * k = physical_pages[stream_id][ref.page_id].layers[ikv].k;
        GGML_ASSERT(k != nullptr);

        const uint32_t n_kv_page = std::min<uint32_t>(n_kv, physical_page_size);
        GGML_ASSERT(k != nullptr);

        const uint64_t n_embd_k_gqa = k->ne[0];

        GGML_ASSERT(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

        return ggml_view_4d(
                ctx,
                k,
                hparams.n_embd_head_k(il),
                hparams.n_head_kv(il),
                n_kv_page,
                1,
                ggml_row_size(k->type, hparams.n_embd_head_k(il)),
                ggml_row_size(k->type, n_embd_k_gqa),
                ggml_row_size(k->type, n_embd_k_gqa*physical_page_size),
                0);
    }

    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    if (physical_paged) {
        const int32_t ikv = map_layer_ids.at(il);

        GGML_ASSERT(sinfo.n_stream() == 1);

        const uint32_t stream_id = sinfo.strm[0];

        GGML_ASSERT(stream_id < physical_pages.size());
        GGML_ASSERT(!physical_pages[stream_id].empty());

        const uint32_t last_cell =
                sinfo.idxs[0].empty() ? 0 : sinfo.idxs[0].back();

        const auto & ref = get_page_ref(stream_id, last_cell);
        GGML_ASSERT(ref.page_id != UINT32_MAX);

        ggml_tensor * v = physical_pages[stream_id][ref.page_id].layers[ikv].v;
        GGML_ASSERT(v != nullptr);

        const uint32_t n_kv_page = std::min<uint32_t>(n_kv, physical_page_size);
        GGML_ASSERT(v != nullptr);

        if (!v_trans) {
            const uint64_t n_embd_v_gqa = v->ne[0];

            GGML_ASSERT(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

            return ggml_view_4d(
                    ctx,
                    v,
                    hparams.n_embd_head_v(il),
                    hparams.n_head_kv(il),
                    n_kv_page,
                    1,
                    ggml_row_size(v->type, hparams.n_embd_head_v(il)),
                    ggml_row_size(v->type, n_embd_v_gqa),
                    ggml_row_size(v->type, n_embd_v_gqa*physical_page_size),
                    0);
        }

        // v_trans=true 时，V 的布局是 [kv, embd]，和原始逻辑一致。
        const uint64_t n_embd_v_gqa = v->ne[1];

        GGML_ASSERT(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

        return ggml_view_4d(
                ctx,
                v,
                n_kv_page,
                hparams.n_head_kv(il),
                hparams.n_embd_head_v(il),
                1,
                ggml_row_size(v->type, physical_page_size*hparams.n_embd_head_v(il)),
                ggml_row_size(v->type, physical_page_size),
                ggml_row_size(v->type, physical_page_size*n_embd_v_gqa),
                0);
    }

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    // GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    if (physical_paged) {
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(!sinfo.idxs.empty());
        GGML_ASSERT(!sinfo.idxs[0].empty());

        const uint32_t stream_id = sinfo.strm[0];
        const auto & first_ref = get_page_ref(stream_id, sinfo.idxs[0][0]);
        GGML_ASSERT(first_ref.page_id != UINT32_MAX);

        // 第一版先要求一个 ubatch 写入同一个 page。
        // 如果跨 page，需要后续把 k_cur 按 page 拆成多次 ggml_set_rows。
        for (uint32_t idx : sinfo.idxs[0]) {
            const auto & ref = get_page_ref(stream_id, idx);
            GGML_ASSERT(ref.page_id == first_ref.page_id && "physical paged cpy_k currently requires one ubatch within one page");
        }

        ggml_tensor * page_k = physical_pages[stream_id][first_ref.page_id].layers[ikv].k;

        const int64_t n_embd_head = k_cur->ne[0];
        const int64_t n_head      = k_cur->ne[1];
        const int64_t n_tokens    = k_cur->ne[2];

        const int64_t n_embd_gqa = n_embd_head*n_head;

        GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

        k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

        return ggml_set_rows(ctx, page_k, k_cur, k_idxs);
    }

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    // GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    if (physical_paged) {
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(!sinfo.idxs.empty());
        GGML_ASSERT(!sinfo.idxs[0].empty());

        const uint32_t stream_id = sinfo.strm[0];
        const auto & first_ref = get_page_ref(stream_id, sinfo.idxs[0][0]);
        GGML_ASSERT(first_ref.page_id != UINT32_MAX);

        // 第一版先要求一个 ubatch 写入同一个 page。
        for (uint32_t idx : sinfo.idxs[0]) {
            const auto & ref = get_page_ref(stream_id, idx);
            GGML_ASSERT(ref.page_id == first_ref.page_id && "physical paged cpy_v currently requires one ubatch within one page");
        }

        ggml_tensor * page_v = physical_pages[stream_id][first_ref.page_id].layers[ikv].v;

        const int64_t n_embd_head = v_cur->ne[0];
        const int64_t n_head      = v_cur->ne[1];
        const int64_t n_tokens    = v_cur->ne[2];

        const int64_t n_embd_gqa = n_embd_head*n_head;

        GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

        if (!v_trans) {
            v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);
            return ggml_set_rows(ctx, page_v, v_cur, v_idxs);
        }

        v_cur = ggml_cont(ctx, ggml_transpose(ctx, v_cur));
        v_cur = ggml_reshape_2d(ctx, v_cur, n_tokens, n_embd_gqa);

        return ggml_set_rows(ctx, page_v, v_cur, v_idxs);
    }

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    // for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
    //     const int64_t offs = sinfo.strm[s]*get_size();

    //     for (uint32_t i = 0; i < sinfo.size(); ++i) {
    //         data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
    //     }
    // }
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            if (physical_paged) {
                const auto & ref = get_page_ref(sinfo.strm[s], sinfo.idxs[s][i]);
                GGML_ASSERT(ref.page_id != UINT32_MAX);
                data[s*sinfo.size() + i] = ref.offset;
            } else {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        // for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        //     const int64_t offs = sinfo.strm[s]*get_size();

        //     for (uint32_t i = 0; i < sinfo.size(); ++i) {
        //         data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        //     }
        // }
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                if (physical_paged) {
                    const auto & ref = get_page_ref(sinfo.strm[s], sinfo.idxs[s][i]);
                    GGML_ASSERT(ref.page_id != UINT32_MAX);
                    data[s*sinfo.size() + i] = ref.offset;
                } else {
                    data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
                }
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        // const int64_t kv_size = get_size();

        // const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        // for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        //     const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

        //     for (uint32_t i = 0; i < sinfo.size(); ++i) {
        //         for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
        //             data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
        //         }
        //     }
        // }

        const int64_t kv_size = physical_paged ? physical_page_size : get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = physical_paged ? 0 : sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                const uint32_t idx = sinfo.idxs[s][i];

                uint32_t page_offset = idx;
                if (physical_paged) {
                    const auto & ref = get_page_ref(sinfo.strm[s], idx);
                    GGML_ASSERT(ref.page_id != UINT32_MAX);
                    page_offset = ref.offset;
                }

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + page_offset;
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = -std::abs(p0 - p1);
                } else {
                    data[idst + j] = 0.0f;
                }

                continue;
skip:
                data[idst + j] = -INFINITY;
            }
        }
    }
}

template<bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<causal, swa, is_2d, false>(args, data);
    }
}

template<bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<causal, swa, false>(args, data);
    }
}

template<bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<causal, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    float * data = (float *) dst->data;

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (causal_attn) {
        set_input_kq_mask_impl<true> (args, data);
    } else {
        set_input_kq_mask_impl<false>(args, data);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;

    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && (seq_id == -1 || cells.seq_has(i, seq_id))) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read_to(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read_to(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        res = res && state_read_data(io, strm, cell_count, sinfo);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read_to(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read_to(&pos,      sizeof(pos));
            io.read_to(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read_to(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read_to(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read_to(&v_trans, sizeof(v_trans));
    io.read_to(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read_to(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read_to(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                ggml_backend_tensor_set(k, io.read(cell_count * k_size_row), sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                const void * src = io.read(cell_count * k_size_row);
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    ggml_backend_tensor_set(k, (const char*)src + i * k_size_row, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read_to(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    ggml_backend_tensor_set(v, io.read(cell_count * v_size_row), sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    const void * src = io.read(cell_count * v_size_row);
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        ggml_backend_tensor_set(v, (const char*)src + i * v_size_row, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read_to(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read_to(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        ggml_backend_tensor_set(v, io.read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const void * src = io.read(cell_count * v_size_el);
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            ggml_backend_tensor_set(v, (const char*)src + i * v_size_el, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

// llama_kv_cache_context::llama_kv_cache_context(
//         llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
//     n_kv = kv->get_size();

//     const uint32_t n_stream = kv->get_n_stream();

//     // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
//     sinfos.resize(1);
//     sinfos[0].s0 = 0;
//     sinfos[0].s1 = n_stream - 1;
//     sinfos[0].idxs.resize(n_stream);
//     for (uint32_t s = 0; s < n_stream; ++s) {
//         sinfos[0].strm.push_back(s);
//         sinfos[0].idxs[s].resize(1, 0);
//     }
// }

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);

    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }

    // 原始连续 KV 在 reserve 阶段仍然使用完整 KV size。
    // 物理分页 KV 在 reserve 阶段只构建一个 page 的图，否则 get_k/get_v 会要求读取完整 512 cells。
    if (kv->get_physical_paged()) {
        n_kv = kv->get_physical_page_size();
    } else {
        n_kv = kv->get_size();
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}
