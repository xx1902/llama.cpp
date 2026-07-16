#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // llama_kv_cache(
    //         const llama_model & model,
    //                 ggml_type   type_k,
    //                 ggml_type   type_v,
    //                      bool   v_trans,
    //                      bool   offload,
    //                      bool   unified,
    //                  uint32_t   kv_size,
    //                  uint32_t   n_seq_max,
    //                  uint32_t   n_pad,
    //                  uint32_t   n_swa,
    //            llama_swa_type   swa_type,
    //     const layer_filter_cb & filter,
    //     const  layer_reuse_cb & reuse);
    llama_kv_cache(
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
                        bool   physical_paged = false,
                    uint32_t   physical_page_size = 16);

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_physical_paged() const;
    uint32_t get_physical_page_size() const;

    // 新增固定分页
    bool get_has_shift() const;

        struct memory_usage_stats {
        uint32_t page_size = 0;

        uint32_t total_cells = 0;
        uint32_t used_cells = 0;

        uint32_t total_pages = 0;
        uint32_t used_pages = 0;
        uint32_t free_pages = 0;

        uint64_t continuous_bytes = 0;
        uint64_t paged_bytes = 0;

        double cell_used_rate = 0.0;
        double page_used_rate = 0.0;
        double page_waste_rate = 0.0;
    };

    memory_usage_stats get_memory_usage_stats(uint32_t page_size) const;

    // kv 差值相关
    struct kv_delta_layer_stats {
        int32_t layer_id = -1;

        double k_l2_avg = 0.0;
        double v_l2_avg = 0.0;
        double kv_l2_avg = 0.0;

        double k_cos_avg = 0.0;
        double v_cos_avg = 0.0;
        double kv_cos_avg = 0.0;
    };
    struct kv_delta_probe_stats {
        int32_t n_layers = 0;
        int32_t n_tokens = 0;

        double k_l2_avg = 0.0;
        double v_l2_avg = 0.0;
        double kv_l2_avg = 0.0;

        double k_cos_avg = 0.0;
        double v_cos_avg = 0.0;
        double kv_cos_avg = 0.0;

        bool can_reuse_as_delta = false;

        std::vector<kv_delta_layer_stats> layers;
    };
    struct kv_delta_materialize_stats {
        int32_t n_layers = 0;
        int32_t n_tokens = 0;

        int32_t materialized_layers = 0;
        int32_t materialized_tokens = 0;
        int32_t missing_cell_count = 0;

        uint64_t delta_fp32_bytes = 0;
        uint64_t materialized_kv_bytes = 0;

        bool ok = false;
    };
    // 量化相关
    enum class kv_delta_dtype {
        F16,
        Q8,
    };
    struct kv_delta_tensor {
        int32_t layer_id = -1;
        bool is_k = true;

        llama_pos p0 = 0;
        llama_pos p1 = 0;

        int32_t n_embd = 0;
        kv_delta_dtype dtype = kv_delta_dtype::Q8;

        // Q8: value ~= int8_value * scale
        std::vector<int8_t> q8;
        std::vector<float> scales;

        // F16 fallback/debug
        std::vector<ggml_fp16_t> f16;

        // 后续 GPU graph 版本再使用；当前先保留为空指针。
        ggml_tensor * q8_tensor = nullptr;
        ggml_tensor * scale_tensor = nullptr;
    };
    struct kv_delta_branch {
        llama_seq_id anchor_seq_id = -1;
        llama_seq_id child_seq_id = -1;

        llama_pos p0 = 0;
        llama_pos p1 = 0;

        int32_t parent_node_id = -1;
        int32_t child_node_id = -1;

        bool enabled = false;

        std::vector<kv_delta_tensor> layer_deltas;

        // 量化相关
        uint64_t anchor_ref_bytes = 0;
        uint64_t delta_q8_bytes = 0;
        uint64_t delta_scale_bytes = 0;
        uint64_t full_kv_bytes_equivalent = 0;
    };
    std::vector<kv_delta_branch> delta_branches;
    bool seq_delta_build_branch(
            llama_seq_id seq_anchor,
            llama_seq_id seq_child_full,
            llama_seq_id seq_child_delta,
            llama_pos p0,
            llama_pos p1,
            int32_t parent_node_id,
            int32_t child_node_id);

    bool seq_delta_has_branch(llama_seq_id seq_id) const;
    const kv_delta_branch * seq_delta_find_branch(llama_seq_id seq_id) const;

    bool seq_delta_materialize(
            llama_seq_id seq_anchor,
            llama_seq_id seq_child_full,
            llama_seq_id seq_dst,
            llama_pos p0,
            llama_pos p1,
            kv_delta_materialize_stats & stats);
    bool seq_delta_materialize_branch(
            llama_seq_id seq_anchor,
            llama_seq_id seq_child_delta,
            llama_seq_id seq_dst,
            llama_pos p0,
            llama_pos p1,
            kv_delta_materialize_stats & stats);
    bool seq_delta_remove_branch(llama_seq_id seq_child_delta);
    bool seq_delta_probe(
            llama_seq_id seq_a,
            llama_seq_id seq_b,
            llama_pos p0,
            llama_pos p1,
            kv_delta_probe_stats & stats) const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // 量化相关
    ggml_tensor * get_k_delta(
        ggml_context * ctx,
        int32_t il,
        uint32_t n_kv,
        const slot_info & sinfo,
        const kv_delta_branch & branch) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    // 新增固定分页
    ggml_type type_k_cache = GGML_TYPE_F16;
    ggml_type type_v_cache = GGML_TYPE_F16;
    bool offload_cache = true;

    bool physical_paged = false;
    uint32_t physical_page_size = 16;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        // 物理分页 KV 使用：记录这一层 KV 应该分配到哪个 backend。
        // 例如 CUDA0 / CPU。这样分页 page buffer 可以和原 KV buffer 放在同一个设备上。
        ggml_backend_buffer_type_t buft = nullptr;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    std::vector<llama_kv_cells> v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // 新增分页
    struct kv_page_layer {
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
    };

    struct kv_page {
        uint32_t id = 0;
        bool used = false;

        uint32_t cell_begin = 0;
        uint32_t cell_end = 0;

        std::vector<kv_page_layer> layers;

        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
    };

    struct kv_page_ref {
        uint32_t page_id = UINT32_MAX;
        uint32_t offset = 0;
    };

    std::vector<std::vector<kv_page_ref>> page_table;
    std::vector<std::vector<kv_page>> physical_pages;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    // 新增分页
    void init_physical_paged_storage();
    void ensure_page_for_cell(
            uint32_t stream_id,
            uint32_t cell_id);
    const kv_page_ref & get_page_ref(
            uint32_t stream_id,
            uint32_t cell_id) const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be layed out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
