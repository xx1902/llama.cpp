#pragma once

#include "llama-kv-cache.h"

#include <cstdint>
#include <memory>
#include <vector>

class llama_kv_cache_paged : public llama_memory_i {
public:
    llama_kv_cache_paged(
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
                     uint32_t   kv_page_size);

    ~llama_kv_cache_paged() override = default;
    // 新增固定分页
    llama_kv_cache::memory_usage_stats get_memory_usage_stats(uint32_t page_size) const;

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
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    struct page_stats {
        uint32_t page_size = 0;
        uint32_t total_pages = 0;
        uint32_t used_pages = 0;
        uint32_t free_pages = 0;
        double used_rate = 0.0;
    };

    page_stats get_page_stats() const;

private:
    uint32_t kv_page_size = 16;
    uint32_t kv_size = 0;

    std::unique_ptr<llama_kv_cache> base;
};