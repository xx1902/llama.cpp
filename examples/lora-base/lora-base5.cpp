// lora-base.cpp
//
// LoRA group tree + grouped KV prefix reuse prototype.
//
// This version uses the generated LoRA tree from build_lora_tree.py.
// Each group node keeps:
// - one anchor LoRA
// - several similar LoRA ids
// - several similar prompt templates
//
// Runtime flow:
// 1. Tokenize prompt.
// 2. Route prompt to the closest group by token-prefix similarity.
// 3. For each group, materialize the shared exact token prefix once.
// 4. For each request, copy the group prefix KV by llama_memory_seq_cp.
// 5. Bind LoRA path anchor -> child.
// 6. Evaluate suffix and decode.

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "llama.h"
#include "generated_lora_tree.hpp"

#include <windows.h>
#include <psapi.h>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#pragma comment(lib, "Psapi.lib")

struct lora_node_runtime {
    int id = 0;
    int parent_id = -1;
    std::string name;
    std::string short_name;
    std::string path;
    std::string group_name;
    bool is_anchor = false;
    llama_adapter_lora * adapter = nullptr;
};

struct prompt_pattern_runtime {
    std::string text;
    std::vector<llama_token> tokens;
};

struct lora_group_runtime {
    int group_id = 0;
    std::string group_name;
    int anchor_lora_id = 0;
    std::vector<int> lora_ids;
    std::vector<prompt_pattern_runtime> prompt_patterns;
};

struct request_item {
    int leaf_lora_id = 0;
    std::string prompt;
};

struct request_tokens {
    std::vector<llama_token> full;
    int routed_group_id = -1;
};

struct kv_group_cache_entry {
    int group_id = -1;
    llama_seq_id seq_id = -1;
    int prefix_len = 0;
    std::vector<llama_token> prefix_tokens;
};

struct sample_result {
    std::string mode;
    std::string group_name;
    std::string lora_name;

    int leaf_lora_id = 0;
    int n_prompt_tokens = 0;
    int n_prefix_tokens = 0;
    int n_suffix_tokens = 0;
    int n_predict = 0;

    double route_ms = 0.0;
    double prefix_ms = 0.0;
    double lora_bind_ms = 0.0;
    double suffix_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    double tps = 0.0;

    double gpu_start_mb = 0.0;
    double gpu_peak_mb = 0.0;
    double gpu_peak_delta_mb = 0.0;
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output";

static double now_ms() {
    return ggml_time_us() / 1000.0;
}

static double get_gpu_used_mb() {
    FILE * pipe = _popen(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            "r");

    if (pipe == nullptr) {
        return 0.0;
    }

    char buffer[128] = {};
    double used_mb = 0.0;

    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        used_mb = atof(buffer);
    }

    _pclose(pipe);

    return used_mb;
}

static bool tokenize_text(
        const llama_vocab * vocab,
        const std::string & text,
        std::vector<llama_token> & tokens) {
    const int n = -llama_tokenize(
            vocab,
            text.c_str(),
            (int) text.size(),
            nullptr,
            0,
            true,
            true);

    if (n <= 0) {
        return false;
    }

    tokens.resize(n);

    const int ret = llama_tokenize(
            vocab,
            text.c_str(),
            (int) text.size(),
            tokens.data(),
            (int) tokens.size(),
            true,
            true);

    return ret >= 0;
}

static void batch_add(
        llama_batch & batch,
        llama_token token,
        llama_pos pos,
        llama_seq_id seq_id,
        bool logits) {
    const int i = batch.n_tokens;

    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = seq_id;
    batch.logits[i] = logits ? 1 : 0;
    batch.n_tokens++;
}

static bool eval_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_seq_id seq_id,
        int start_pos,
        bool logits_last) {
    if (tokens.empty()) {
        return true;
    }

    const int max_chunk = 64;
    int offset = 0;

    while (offset < (int) tokens.size()) {
        const int chunk =
                std::min(max_chunk, (int) tokens.size() - offset);

        llama_batch batch = llama_batch_init(chunk, 0, 1);

        for (int i = 0; i < chunk; i++) {
            const int token_index = offset + i;

            const bool need_logits =
                    logits_last &&
                    token_index == (int) tokens.size() - 1;

            batch_add(
                    batch,
                    tokens[token_index],
                    start_pos + token_index,
                    seq_id,
                    need_logits);
        }

        const int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            fprintf(stderr,
                    "eval_tokens failed: seq=%d start=%d offset=%d chunk=%d\n",
                    seq_id,
                    start_pos,
                    offset,
                    chunk);
            return false;
        }

        offset += chunk;
    }

    return true;
}

static bool decode_one(
        llama_context * ctx,
        llama_token token,
        llama_seq_id seq_id,
        int pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);

    batch_add(
            batch,
            token,
            pos,
            seq_id,
            true);

    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    return ret == 0;
}

static void clear_lora(llama_context * ctx) {
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
}

static int common_prefix_len(
        const std::vector<llama_token> & a,
        const std::vector<llama_token> & b) {
    const int n = std::min((int) a.size(), (int) b.size());

    int i = 0;

    while (i < n && a[i] == b[i]) {
        i++;
    }

    return i;
}

static std::vector<lora_node_runtime> make_runtime_lora_nodes() {
    std::vector<lora_node_runtime> nodes;

    const std::vector<generated_lora_spec> specs =
            make_generated_lora_specs();

    for (const auto & spec : specs) {
        lora_node_runtime node;

        node.id = spec.id;
        node.parent_id = spec.parent_id;
        node.name = spec.name;
        node.short_name = spec.short_name;
        node.path = spec.path;
        node.group_name = spec.group_name;
        node.is_anchor = spec.is_anchor;

        nodes.push_back(node);
    }

    return nodes;
}

static std::vector<lora_group_runtime> make_runtime_groups(
        const llama_vocab * vocab) {
    std::vector<lora_group_runtime> groups;

    const std::vector<generated_lora_group> generated_groups =
            make_generated_lora_groups();

    for (const auto & src : generated_groups) {
        lora_group_runtime group;

        group.group_id = src.group_id;
        group.group_name = src.group_name;
        group.anchor_lora_id = src.anchor_lora_id;
        group.lora_ids = src.lora_ids;

        for (const auto & pattern : src.prompt_patterns) {
            prompt_pattern_runtime item;

            item.text = pattern.text;
            tokenize_text(vocab, item.text, item.tokens);

            group.prompt_patterns.push_back(item);
        }

        groups.push_back(group);
    }

    return groups;
}

static int route_prompt_to_group(
        const std::vector<lora_group_runtime> & groups,
        const std::vector<llama_token> & prompt_tokens) {
    int best_group = groups.empty() ? -1 : groups[0].group_id;
    int best_score = -1;

    for (const auto & group : groups) {
        for (const auto & pattern : group.prompt_patterns) {
            const int score =
                    common_prefix_len(prompt_tokens, pattern.tokens);

            if (score > best_score) {
                best_score = score;
                best_group = group.group_id;
            }
        }
    }

    return best_group;
}

static const lora_group_runtime * find_group(
        const std::vector<lora_group_runtime> & groups,
        int group_id) {
    for (const auto & group : groups) {
        if (group.group_id == group_id) {
            return &group;
        }
    }

    return nullptr;
}

static std::vector<int> build_lora_path(
        const std::vector<lora_node_runtime> & nodes,
        int leaf_id) {
    std::vector<int> path;
    int cur = leaf_id;

    while (cur >= 0 && cur < (int) nodes.size()) {
        path.push_back(cur);
        cur = nodes[cur].parent_id;
    }

    std::reverse(path.begin(), path.end());

    return path;
}

static double bind_lora_path(
        llama_context * ctx,
        const std::vector<lora_node_runtime> & nodes,
        int leaf_id) {
    const std::vector<int> path =
            build_lora_path(nodes, leaf_id);

    std::vector<llama_adapter_lora *> adapters;
    std::vector<float> scales;

    for (int id : path) {
        if (id >= 0 && id < (int) nodes.size() &&
                nodes[id].adapter != nullptr) {
            adapters.push_back(nodes[id].adapter);
            scales.push_back(1.0f);
        }
    }

    const double t0 = now_ms();

    const int ret = llama_set_adapters_lora(
            ctx,
            adapters.data(),
            adapters.size(),
            scales.data());

    const double t1 = now_ms();

    if (ret != 0) {
        fprintf(stderr, "failed to bind LoRA path for leaf %d\n", leaf_id);
    }

    return t1 - t0;
}

static std::vector<llama_token> group_common_prefix(
        const std::vector<request_tokens> & tokenized,
        const std::vector<int> & request_indices) {
    std::vector<llama_token> prefix;

    if (request_indices.empty()) {
        return prefix;
    }

    prefix = tokenized[request_indices[0]].full;

    for (int idx : request_indices) {
        const int len =
                common_prefix_len(prefix, tokenized[idx].full);

        prefix.resize(len);
    }

    return prefix;
}

static sample_result run_baseline_request(
        llama_context * ctx,
        const std::vector<lora_node_runtime> & lora_nodes,
        const std::vector<lora_group_runtime> & groups,
        const request_item & req,
        const request_tokens & toks,
        int request_index,
        int n_predict) {
    sample_result r;

    const lora_group_runtime * group =
            find_group(groups, toks.routed_group_id);

    r.mode = "baseline_no_group_kv_reuse";
    r.group_name = group ? group->group_name : "unknown";
    r.leaf_lora_id = req.leaf_lora_id;
    r.lora_name = lora_nodes[req.leaf_lora_id].short_name;
    r.n_prompt_tokens = (int) toks.full.size();
    r.n_prefix_tokens = 0;
    r.n_suffix_tokens = (int) toks.full.size();
    r.n_predict = n_predict;

    const llama_seq_id seq_id = request_index;
    const double gpu_start = get_gpu_used_mb();
    double gpu_peak = gpu_start;

    const double t0 = now_ms();

    r.lora_bind_ms =
            bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);

    const double suffix0 = now_ms();

    eval_tokens(
            ctx,
            toks.full,
            seq_id,
            0,
            true);

    const double suffix1 = now_ms();

    r.suffix_ms = suffix1 - suffix0;

    const llama_token repeated = toks.full.back();

    const double decode0 = now_ms();

    for (int i = 0; i < n_predict; i++) {
        const int pos = (int) toks.full.size() + i;

        decode_one(ctx, repeated, seq_id, pos);

        if (i == 0) {
            r.ttft_ms = now_ms() - t0;
        }

        gpu_peak = std::max(gpu_peak, get_gpu_used_mb());
    }

    const double decode1 = now_ms();

    r.decode_ms = decode1 - decode0;
    r.total_ms = decode1 - t0;
    r.tps = n_predict / std::max(0.001, r.decode_ms / 1000.0);
    r.gpu_start_mb = gpu_start;
    r.gpu_peak_mb = gpu_peak;
    r.gpu_peak_delta_mb = gpu_peak - gpu_start;

    return r;
}

static sample_result run_group_kv_reuse_request(
        llama_context * ctx,
        llama_memory_t mem,
        const std::vector<lora_node_runtime> & lora_nodes,
        const std::vector<lora_group_runtime> & groups,
        const std::vector<kv_group_cache_entry> & cache_entries,
        const request_item & req,
        const request_tokens & toks,
        int request_seq_id,
        int n_predict) {
    sample_result r;

    const lora_group_runtime * group =
            find_group(groups, toks.routed_group_id);

    r.mode = "group_node_kv_reuse_lora_tree";
    r.group_name = group ? group->group_name : "unknown";
    r.leaf_lora_id = req.leaf_lora_id;
    r.lora_name = lora_nodes[req.leaf_lora_id].short_name;
    r.n_prompt_tokens = (int) toks.full.size();
    r.n_predict = n_predict;

    const kv_group_cache_entry * hit = nullptr;

    for (const auto & entry : cache_entries) {
        if (entry.group_id == toks.routed_group_id) {
            hit = &entry;
            break;
        }
    }

    const double gpu_start = get_gpu_used_mb();
    double gpu_peak = gpu_start;

    const double t0 = now_ms();

    int prefix_len = 0;

    const double prefix0 = now_ms();

    if (hit != nullptr && hit->prefix_len > 0) {
        prefix_len = hit->prefix_len;

        llama_memory_seq_cp(
                mem,
                hit->seq_id,
                request_seq_id,
                0,
                hit->prefix_len);
    }

    const double prefix1 = now_ms();

    r.prefix_ms = prefix1 - prefix0;
    r.n_prefix_tokens = prefix_len;
    r.n_suffix_tokens = (int) toks.full.size() - prefix_len;

    r.lora_bind_ms =
            bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);

    std::vector<llama_token> suffix(
            toks.full.begin() + prefix_len,
            toks.full.end());

    const double suffix0 = now_ms();

    eval_tokens(
            ctx,
            suffix,
            request_seq_id,
            prefix_len,
            true);

    const double suffix1 = now_ms();

    r.suffix_ms = suffix1 - suffix0;

    const llama_token repeated =
            suffix.empty() ? toks.full.back() : suffix.back();

    const double decode0 = now_ms();

    for (int i = 0; i < n_predict; i++) {
        const int pos = (int) toks.full.size() + i;

        decode_one(ctx, repeated, request_seq_id, pos);

        if (i == 0) {
            r.ttft_ms = now_ms() - t0;
        }

        gpu_peak = std::max(gpu_peak, get_gpu_used_mb());
    }

    const double decode1 = now_ms();

    r.decode_ms = decode1 - decode0;
    r.total_ms = decode1 - t0;
    r.tps = n_predict / std::max(0.001, r.decode_ms / 1000.0);
    r.gpu_start_mb = gpu_start;
    r.gpu_peak_mb = gpu_peak;
    r.gpu_peak_delta_mb = gpu_peak - gpu_start;

    return r;
}

static void save_results(
        const std::vector<sample_result> & results) {
    std::filesystem::create_directories(output_dir);

    const std::string path =
            output_dir + "/group_node_kv_lora_tree_summary.csv";

    std::ofstream fout(path);

    fout << "mode,group_name,lora_name,leaf_lora_id,"
         << "n_prompt_tokens,n_prefix_tokens,n_suffix_tokens,n_predict,"
         << "route_ms,prefix_ms,lora_bind_ms,suffix_ms,"
         << "ttft_ms,decode_ms,total_ms,tps,"
         << "gpu_start_mb,gpu_peak_mb,gpu_peak_delta_mb\n";

    for (const auto & r : results) {
        fout << r.mode << ","
             << r.group_name << ","
             << r.lora_name << ","
             << r.leaf_lora_id << ","
             << r.n_prompt_tokens << ","
             << r.n_prefix_tokens << ","
             << r.n_suffix_tokens << ","
             << r.n_predict << ","
             << r.route_ms << ","
             << r.prefix_ms << ","
             << r.lora_bind_ms << ","
             << r.suffix_ms << ","
             << r.ttft_ms << ","
             << r.decode_ms << ","
             << r.total_ms << ","
             << r.tps << ","
             << r.gpu_start_mb << ","
             << r.gpu_peak_mb << ","
             << r.gpu_peak_delta_mb << "\n";
    }

    fprintf(stderr, "saved results to %s\n", path.c_str());
}

int main() {
    std::setlocale(LC_NUMERIC, "C");

    const std::string model_path =
            "D:/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf";

    const int ngl = 99;
    const int n_ctx = 4096;
    const int n_predict = 32;

    ggml_backend_load_all();

    llama_model_params model_params =
            llama_model_default_params();

    model_params.n_gpu_layers = ngl;

    llama_model * model =
            llama_model_load_from_file(
                    model_path.c_str(),
                    model_params);

    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab =
            llama_model_get_vocab(model);

    std::vector<lora_node_runtime> lora_nodes =
            make_runtime_lora_nodes();

    for (auto & node : lora_nodes) {
        node.adapter =
                llama_adapter_lora_init(
                        model,
                        node.path.c_str());

        if (node.adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", node.path.c_str());

            for (auto & old : lora_nodes) {
                if (old.adapter != nullptr) {
                    llama_adapter_lora_free(old.adapter);
                }
            }

            llama_model_free(model);
            return 1;
        }
    }

    std::vector<lora_group_runtime> groups =
            make_runtime_groups(vocab);

    std::vector<request_item> requests = {
        {
            0,
            "You are an animal introduction expert. Please introduce the characteristics of monkey.",
        },
        {
            1,
            "You are an animal introduction expert. Please introduce the characteristics of tiger.",
        },
        {
            2,
            "You are an animal introduction expert. Please introduce the characteristics of wolf.",
        },
        {
            3,
            "You are a marketing copywriting expert. Please write a short product slogan.",
        },
    };

    std::vector<request_tokens> tokenized;

    for (const auto & req : requests) {
        request_tokens toks;

        tokenize_text(vocab, req.prompt, toks.full);

        const double route0 = now_ms();
        toks.routed_group_id =
                route_prompt_to_group(groups, toks.full);
        const double route1 = now_ms();

        (void) route0;
        (void) route1;

        tokenized.push_back(toks);
    }

    fprintf(stderr, "request routing result:\n");

    for (int i = 0; i < (int) requests.size(); i++) {
        const lora_group_runtime * group =
                find_group(groups, tokenized[i].routed_group_id);

        fprintf(stderr,
                "  request %d -> group %s\n",
                i,
                group ? group->group_name.c_str() : "unknown");
    }

    std::vector<sample_result> results;

    {
        llama_context_params ctx_params =
                llama_context_default_params();

        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = 256;
        ctx_params.n_ubatch = 64;
        ctx_params.n_seq_max = (uint32_t) requests.size();
        ctx_params.no_perf = true;
        ctx_params.kv_unified = true;

        llama_context * ctx =
                llama_init_from_model(model, ctx_params);

        if (ctx == nullptr) {
            fprintf(stderr, "failed to create baseline context\n");
            return 1;
        }

        for (int i = 0; i < (int) requests.size(); i++) {
            sample_result r =
                    run_baseline_request(
                            ctx,
                            lora_nodes,
                            groups,
                            requests[i],
                            tokenized[i],
                            i,
                            n_predict);

            results.push_back(r);
        }

        clear_lora(ctx);
        llama_free(ctx);
    }

    {
        std::vector<kv_group_cache_entry> cache_entries;

        const int n_groups = (int) groups.size();
        const int request_seq_base = n_groups;

        llama_context_params ctx_params =
                llama_context_default_params();

        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = 256;
        ctx_params.n_ubatch = 64;
        ctx_params.n_seq_max =
                (uint32_t) (n_groups + requests.size() + 4);
        ctx_params.no_perf = true;
        ctx_params.kv_unified = true;

        llama_context * ctx =
                llama_init_from_model(model, ctx_params);

        if (ctx == nullptr) {
            fprintf(stderr, "failed to create group-kv context\n");
            return 1;
        }

        llama_memory_t mem =
                llama_get_memory(ctx);

        clear_lora(ctx);

        for (const auto & group : groups) {
            std::vector<int> indices;

            for (int i = 0; i < (int) tokenized.size(); i++) {
                if (tokenized[i].routed_group_id == group.group_id) {
                    indices.push_back(i);
                }
            }

            if (indices.empty()) {
                continue;
            }

            std::vector<llama_token> prefix =
                    group_common_prefix(tokenized, indices);

            if (prefix.empty()) {
                continue;
            }

            const llama_seq_id cache_seq_id =
                    (llama_seq_id) group.group_id;

            const double p0 = now_ms();

            eval_tokens(
                    ctx,
                    prefix,
                    cache_seq_id,
                    0,
                    false);

            const double p1 = now_ms();

            kv_group_cache_entry entry;

            entry.group_id = group.group_id;
            entry.seq_id = cache_seq_id;
            entry.prefix_len = (int) prefix.size();
            entry.prefix_tokens = prefix;

            cache_entries.push_back(entry);

            fprintf(stderr,
                    "group %s cached prefix_len=%d eval_ms=%.3f\n",
                    group.group_name.c_str(),
                    (int) prefix.size(),
                    p1 - p0);
        }

        for (int i = 0; i < (int) requests.size(); i++) {
            sample_result r =
                    run_group_kv_reuse_request(
                            ctx,
                            mem,
                            lora_nodes,
                            groups,
                            cache_entries,
                            requests[i],
                            tokenized[i],
                            request_seq_base + i,
                            n_predict);

            results.push_back(r);
        }

        clear_lora(ctx);
        llama_free(ctx);
    }

    save_results(results);

    for (auto & node : lora_nodes) {
        if (node.adapter != nullptr) {
            llama_adapter_lora_free(node.adapter);
        }
    }

    llama_model_free(model);

    fprintf(stderr, "group node KV reuse + LoRA tree experiment finished.\n");

    return 0;
}