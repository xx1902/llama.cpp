#include "llama-kv-cache-paged.h"

#include <cstdio>

llama_kv_cache_paged::llama_kv_cache_paged(
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
                 uint32_t   kv_page_size) :
    kv_page_size(kv_page_size),
    kv_size(kv_size) {
    if (this->kv_page_size == 0) {
        this->kv_page_size = 16;
    }

    base.reset(new llama_kv_cache(
            model,
            type_k,
            type_v,
            v_trans,
            offload,
            unified,
            kv_size,
            n_seq_max,
            n_pad,
            n_swa,
            swa_type,
            filter,
            reuse));

    fprintf(stderr,
            "%s: experimental paged KV wrapper enabled, page_size = %u tokens, kv_size = %u tokens\n",
            __func__,
            this->kv_page_size,
            this->kv_size);
}

llama_memory_context_ptr llama_kv_cache_paged::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    return base->init_batch(balloc, n_ubatch, embd_all);
}

llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    return base->init_full();
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * lctx, bool optimize) {
    return base->init_update(lctx, optimize);
}
// 新增固定分页
llama_kv_cache::memory_usage_stats llama_kv_cache_paged::get_memory_usage_stats(uint32_t page_size) const {
    return base->get_memory_usage_stats(page_size);
}

bool llama_kv_cache_paged::get_can_shift() const {
    return base->get_can_shift();
}

void llama_kv_cache_paged::clear(bool data) {
    base->clear(data);
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return base->seq_rm(seq_id, p0, p1);
}

void llama_kv_cache_paged::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_paged::seq_keep(llama_seq_id seq_id) {
    base->seq_keep(seq_id);
}

void llama_kv_cache_paged::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    base->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_paged::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    base->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    return base->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    return base->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    return base->memory_breakdown();
}

void llama_kv_cache_paged::state_write(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    base->state_write(io, seq_id, flags);
}

void llama_kv_cache_paged::state_read(
        llama_io_read_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    base->state_read(io, seq_id, flags);
}

llama_kv_cache_paged::page_stats llama_kv_cache_paged::get_page_stats() const {
    page_stats s;

    s.page_size = kv_page_size;
    s.total_pages = (kv_size + kv_page_size - 1) / kv_page_size;

    // 当前版本只是 wrapper，还没有真正把 token 映射到 page。
    // 所以 used_pages 暂时只能作为接口连通性测试。
    const llama_pos max_pos = base->seq_pos_max(-1);

    if (max_pos >= 0) {
        s.used_pages = ((uint32_t) max_pos + kv_page_size) / kv_page_size;
        if (s.used_pages > s.total_pages) {
            s.used_pages = s.total_pages;
        }
    } else {
        s.used_pages = 0;
    }

    s.free_pages = s.total_pages > s.used_pages ? s.total_pages - s.used_pages : 0;

    if (s.total_pages > 0) {
        s.used_rate = (double) s.used_pages / (double) s.total_pages;
    }

    return s;
}