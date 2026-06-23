// lora-base.cpp
//
// 多层 KV Prefix Tree + LoRA Tree 原型
//
// 当前做到：
// 1. 自动把 prompt token 插入 Trie，形成多层 KV prefix tree。
// 2. 对多个请求共享的 Trie 节点缓存 KV。
// 3. 请求执行时，从最深可复用 prefix 节点复制 KV。
// 4. LoRA 侧使用 anchor -> delta 的树路径绑定。
// 5. baseline 和 fused 分开 context，避免 seq_id 污染和异常耗时。
//
// 还没做到：
// 1. 真实 MobileLoRA delta = child - anchor 文件生成。
// 2. 根据 LoRA tensor 相似度在 C++ 内部直接构树。
// 3. 真正固定页 KV cache 替换，这里只做 prefix KV 复用。

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "llama.h"

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
#include <map>
#include <numeric>
#include <string>
#include <vector>

#pragma comment(lib, "Psapi.lib")

struct lora_node {
    int id = 0;
    int parent_id = -1;
    std::string name;
    std::string path;
    bool is_anchor = false;
    llama_adapter_lora * adapter = nullptr;
};

struct request_item {
    int leaf_lora_id = 0;
    std::string prompt;
};

struct request_tokens {
    std::vector<llama_token> full;
};

struct trie_node {
    int id = 0;
    int parent = -1;
    llama_token token = 0;
    int depth = 0;
    int pass_count = 0;
    bool cached = false;
    llama_seq_id seq_id = -1;
    std::map<llama_token, int> child;
};

struct sample_result {
    std::string mode;
    std::string lora_name;
    int leaf_lora_id = 0;

    int n_prefix_tokens = 0;
    int n_suffix_tokens = 0;
    int n_predict = 0;

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

static std::vector<int> build_lora_path(
        const std::vector<lora_node> & nodes,
        int leaf_id) {
    std::vector<int> path;
    int cur = leaf_id;

    while (cur >= 0) {
        path.push_back(cur);
        cur = nodes[cur].parent_id;
    }

    std::reverse(path.begin(), path.end());

    return path;
}

static double bind_lora_path(
        llama_context * ctx,
        const std::vector<lora_node> & nodes,
        int leaf_id) {
    const std::vector<int> path = build_lora_path(nodes, leaf_id);

    std::vector<llama_adapter_lora *> adapters;
    std::vector<float> scales;

    for (int id : path) {
        if (nodes[id].adapter != nullptr) {
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
        fprintf(stderr, "failed to bind LoRA path\n");
    }

    return t1 - t0;
}

static std::vector<trie_node> build_prompt_trie(
        const std::vector<request_tokens> & requests) {
    std::vector<trie_node> trie;

    trie_node root;
    root.id = 0;
    root.parent = -1;
    root.depth = 0;
    trie.push_back(root);

    for (const auto & req : requests) {
        int cur = 0;
        trie[cur].pass_count++;

        for (llama_token tok : req.full) {
            auto it = trie[cur].child.find(tok);

            if (it == trie[cur].child.end()) {
                trie_node node;
                node.id = (int) trie.size();
                node.parent = cur;
                node.token = tok;
                node.depth = trie[cur].depth + 1;

                trie[cur].child[tok] = node.id;
                trie.push_back(node);

                cur = node.id;
            } else {
                cur = it->second;
            }

            trie[cur].pass_count++;
        }
    }

    return trie;
}

static std::vector<llama_token> tokens_for_node(
        const std::vector<trie_node> & trie,
        int node_id) {
    std::vector<llama_token> reversed;

    int cur = node_id;

    while (cur > 0) {
        reversed.push_back(trie[cur].token);
        cur = trie[cur].parent;
    }

    std::reverse(reversed.begin(), reversed.end());

    return reversed;
}

static int find_deepest_cached_prefix(
        const std::vector<trie_node> & trie,
        const std::vector<llama_token> & tokens) {
    int cur = 0;
    int best = 0;

    for (llama_token tok : tokens) {
        auto it = trie[cur].child.find(tok);

        if (it == trie[cur].child.end()) {
            break;
        }

        cur = it->second;

        if (trie[cur].cached) {
            best = cur;
        }
    }

    return best;
}

static void materialize_cached_trie_nodes(
        llama_context * ctx,
        llama_memory_t mem,
        std::vector<trie_node> & trie) {
    llama_seq_id next_seq = 0;

    for (auto & node : trie) {
        if (node.id == 0) {
            continue;
        }

        if (node.pass_count < 2) {
            continue;
        }

        node.cached = true;
        node.seq_id = next_seq++;

        const int parent = node.parent;

        if (parent > 0 && trie[parent].cached) {
            llama_memory_seq_cp(
                    mem,
                    trie[parent].seq_id,
                    node.seq_id,
                    0,
                    trie[parent].depth);
        }

        std::vector<llama_token> one = { node.token };

        eval_tokens(
                ctx,
                one,
                node.seq_id,
                node.depth - 1,
                false);
    }

    fprintf(stderr,
            "materialized %d KV prefix tree seqs\n",
            (int) next_seq);
}

static int count_cached_nodes(const std::vector<trie_node> & trie) {
    int count = 0;

    for (const auto & node : trie) {
        if (node.id > 0 && node.pass_count >= 2) {
            count++;
        }
    }

    return count;
}

static sample_result run_baseline_request(
        llama_context * ctx,
        const std::vector<lora_node> & lora_nodes,
        const request_item & req,
        const request_tokens & toks,
        int request_index,
        int n_predict) {
    sample_result r;
    r.mode = "baseline_no_kv_tree";
    r.leaf_lora_id = req.leaf_lora_id;
    r.lora_name = lora_nodes[req.leaf_lora_id].name;
    r.n_prefix_tokens = 0;
    r.n_suffix_tokens = (int) toks.full.size();
    r.n_predict = n_predict;

    const llama_seq_id seq_id = request_index;
    const double gpu_start = get_gpu_used_mb();
    double gpu_peak = gpu_start;

    const double t0 = now_ms();

    r.lora_bind_ms = bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);

    const double suffix0 = now_ms();
    eval_tokens(ctx, toks.full, seq_id, 0, true);
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

static sample_result run_fused_request(
        llama_context * ctx,
        llama_memory_t mem,
        const std::vector<trie_node> & trie,
        const std::vector<lora_node> & lora_nodes,
        const request_item & req,
        const request_tokens & toks,
        int request_index,
        int n_predict) {
    sample_result r;
    r.mode = "multi_level_kv_tree_lora_tree";
    r.leaf_lora_id = req.leaf_lora_id;
    r.lora_name = lora_nodes[req.leaf_lora_id].name;
    r.n_predict = n_predict;

    const int cached_node =
            find_deepest_cached_prefix(trie, toks.full);

    const int prefix_len =
            cached_node > 0 ? trie[cached_node].depth : 0;

    r.n_prefix_tokens = prefix_len;
    r.n_suffix_tokens = (int) toks.full.size() - prefix_len;

    std::vector<llama_token> suffix(
            toks.full.begin() + prefix_len,
            toks.full.end());

    const llama_seq_id seq_id = request_index;
    const double gpu_start = get_gpu_used_mb();
    double gpu_peak = gpu_start;

    const double t0 = now_ms();

    const double cp0 = now_ms();

    if (cached_node > 0) {
        llama_memory_seq_cp(
                mem,
                trie[cached_node].seq_id,
                seq_id,
                0,
                prefix_len);
    }

    const double cp1 = now_ms();

    r.prefix_ms = cp1 - cp0;

    r.lora_bind_ms = bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);

    const double suffix0 = now_ms();
    eval_tokens(ctx, suffix, seq_id, prefix_len, true);
    const double suffix1 = now_ms();

    r.suffix_ms = suffix1 - suffix0;

    const llama_token repeated =
            suffix.empty() ? toks.full.back() : suffix.back();

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

static void save_results(const std::vector<sample_result> & results) {
    std::filesystem::create_directories(output_dir);

    const std::string path =
            output_dir + "/multi_level_kv_tree_lora_tree_summary.csv";

    std::ofstream fout(path);

    fout << "mode,lora_name,leaf_lora_id,n_prefix_tokens,n_suffix_tokens,n_predict,"
         << "prefix_ms,lora_bind_ms,suffix_ms,ttft_ms,decode_ms,total_ms,tps,"
         << "gpu_start_mb,gpu_peak_mb,gpu_peak_delta_mb\n";

    for (const auto & r : results) {
        fout << r.mode << ","
             << r.lora_name << ","
             << r.leaf_lora_id << ","
             << r.n_prefix_tokens << ","
             << r.n_suffix_tokens << ","
             << r.n_predict << ","
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
            llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab =
            llama_model_get_vocab(model);

    std::vector<lora_node> lora_nodes = {
        {
            0,
            -1,
            "anchor_marketing",
            "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
            true,
            nullptr,
        },
        {
            1,
            0,
            "delta_monkey",
            "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
            false,
            nullptr,
        },
        {
            2,
            0,
            "delta_tiger",
            "D:/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
            false,
            nullptr,
        },
        {
            3,
            0,
            "delta_wolf",
            "D:/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
            false,
            nullptr,
        },
    };

    for (auto & node : lora_nodes) {
        node.adapter =
                llama_adapter_lora_init(model, node.path.c_str());

        if (node.adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", node.path.c_str());
            llama_model_free(model);
            return 1;
        }
    }

    std::vector<request_item> requests = {
        {
            1,
            "You are a helpful assistant for product content. Please write a short marketing answer for a product launch.",
        },
        {
            2,
            "You are a helpful assistant for product content. Please write a short creative campaign idea for young users.",
        },
        {
            3,
            "You are a helpful assistant for product content. Please write a compact brand slogan for a premium product.",
        },
    };

    std::vector<request_tokens> tokenized;

    for (const auto & req : requests) {
        request_tokens item;
        tokenize_text(vocab, req.prompt, item.full);
        tokenized.push_back(item);
    }

    std::vector<trie_node> trie =
            build_prompt_trie(tokenized);

    const int cached_nodes = count_cached_nodes(trie);

    fprintf(stderr,
            "multi-level KV prefix tree cached nodes = %d\n",
            cached_nodes);

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
            results.push_back(run_baseline_request(
                    ctx,
                    lora_nodes,
                    requests[i],
                    tokenized[i],
                    i,
                    n_predict));
        }

        clear_lora(ctx);
        llama_free(ctx);
    }

    {
        llama_context_params ctx_params =
                llama_context_default_params();

        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = 256;
        ctx_params.n_ubatch = 64;
        ctx_params.n_seq_max =
                (uint32_t) (cached_nodes + requests.size() + 4);
        ctx_params.no_perf = true;
        ctx_params.kv_unified = true;

        llama_context * ctx =
                llama_init_from_model(model, ctx_params);

        if (ctx == nullptr) {
            fprintf(stderr, "failed to create fused context\n");
            return 1;
        }

        llama_memory_t mem = llama_get_memory(ctx);

        clear_lora(ctx);

        materialize_cached_trie_nodes(ctx, mem, trie);

        const int request_seq_base = cached_nodes + 1;

        for (int i = 0; i < (int) requests.size(); i++) {
            results.push_back(run_fused_request(
                    ctx,
                    mem,
                    trie,
                    lora_nodes,
                    requests[i],
                    tokenized[i],
                    request_seq_base + i,
                    n_predict));
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

    fprintf(stderr, "multi-level KV prefix tree + LoRA tree experiment finished.\n");

    return 0;
}