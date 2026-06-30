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
#include "generated_lora_tree_qwen2.5.hpp"

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

struct online_prefix_node {
    int node_id = -1;
    int group_id = -1;

    llama_seq_id cache_seq_id = -1;

    int prefix_len = 0;
    int hit_count = 0;

    std::string group_name;
    std::vector<llama_token> prefix_tokens;
};

struct online_route_result {
    int node_id = -1;
    int group_id = -1;

    int exact_prefix_len = 0;
    int suffix_len = 0;

    bool exact_prefix_hit = false;
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

    int online_node_id = -1;
    int exact_prefix_hit = 0;

    double estimated_saved_kv_mb = 0.0;

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

    double prefix_reuse_rate = 0.0;
    double suffix_delta_rate = 0.0;
};

// KV 差值相关
struct kv_delta_probe_result {
    std::string pair_name;
    std::string group_name;

    std::string memory_kind;
    std::string probe_status;

    int seq_a = -1;
    int seq_b = -1;

    int tokens_a = 0;
    int tokens_b = 0;
    int common_prefix_tokens = 0;
    int compared_tokens = 0;

    int probed_kv_modules = 0;
    int probed_layers = 0;
    int skipped_recurrent_layers = 0;

    double full_kv_l2_avg = 0.0;
    double full_kv_cos_avg = 0.0;

    double prefix_kv_l2_avg = 0.0;
    double prefix_kv_cos_avg = 0.0;

    double suffix_kv_l2_avg = 0.0;
    double suffix_kv_cos_avg = 0.0;

    int prefix_can_reuse = 0;
    int suffix_can_delta = 0;
    int should_open_new_branch = 0;
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

static void save_kv_delta_probe_results(
        const std::vector<kv_delta_probe_result> & results) {
    std::filesystem::create_directories(output_dir);

    const std::string path =
            output_dir + "/kv_delta_probe_summary.csv";

    std::ofstream fout(path);

    fout << "pair_name,group_name,memory_kind,probe_status,seq_a,seq_b,"
         << "tokens_a,tokens_b,common_prefix_tokens,compared_tokens,"
         << "probed_kv_modules,probed_layers,skipped_recurrent_layers,"
         << "full_kv_l2_avg,full_kv_cos_avg,"
         << "prefix_kv_l2_avg,prefix_kv_cos_avg,"
         << "suffix_kv_l2_avg,suffix_kv_cos_avg,"
         << "prefix_can_reuse,suffix_can_delta,should_open_new_branch\n";

    for (const auto & r : results) {
        fout << r.pair_name << ","
             << r.group_name << ","
             << r.memory_kind << ","
             << r.probe_status << ","
             << r.seq_a << ","
             << r.seq_b << ","
             << r.tokens_a << ","
             << r.tokens_b << ","
             << r.common_prefix_tokens << ","
             << r.compared_tokens << ","
             << r.probed_kv_modules << ","
             << r.probed_layers << ","
             << r.skipped_recurrent_layers << ","
             << r.full_kv_l2_avg << ","
             << r.full_kv_cos_avg << ","
             << r.prefix_kv_l2_avg << ","
             << r.prefix_kv_cos_avg << ","
             << r.suffix_kv_l2_avg << ","
             << r.suffix_kv_cos_avg << ","
             << r.prefix_can_reuse << ","
             << r.suffix_can_delta << ","
             << r.should_open_new_branch << "\n";
    }

    fprintf(stderr, "saved kv delta probe results to %s\n", path.c_str());
}

struct kv_delta_range_result {
    std::string memory_kind = "unknown";
    std::string probe_status = "failed";

    int probed_kv_modules = 0;
    int probed_layers = 0;
    int skipped_recurrent_layers = 0;

    double kv_l2_avg = 0.0;
    double kv_cos_avg = 0.0;

    int reusable = 0;
};

static kv_delta_range_result run_kv_delta_probe_range(
        llama_context * ctx,
        llama_seq_id seq_a,
        llama_seq_id seq_b,
        llama_pos p0,
        llama_pos p1) {
    kv_delta_range_result r;

    if (p1 <= p0) {
        r.probe_status = "empty_range";
        return r;
    }

    llama_kv_delta_probe_stats stats = {};

    const bool ok =
            llama_kv_seq_delta_probe(
                    ctx,
                    seq_a,
                    seq_b,
                    p0,
                    p1,
                    &stats);

    r.memory_kind = stats.memory_kind;
    r.probe_status = stats.probe_status;

    if (r.memory_kind.empty()) {
        r.memory_kind = "unknown";
    }

    if (r.probe_status.empty()) {
        r.probe_status = ok ? "ok" : "failed";
    }

    r.probed_kv_modules = stats.probed_kv_modules;
    r.probed_layers = stats.probed_layers;
    r.skipped_recurrent_layers = stats.skipped_recurrent_layers;

    r.kv_l2_avg = stats.kv_l2_avg;
    r.kv_cos_avg = stats.kv_cos_avg;
    r.reusable = stats.can_reuse_as_delta;

    return r;
}

static kv_delta_probe_result run_kv_delta_probe_pair(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::string & group_name,
        const std::string & pair_name,
        const std::string & prompt_a,
        const std::string & prompt_b,
        llama_seq_id seq_a,
        llama_seq_id seq_b) {
    kv_delta_probe_result r;

    r.group_name = group_name;
    r.pair_name = pair_name;
    r.seq_a = seq_a;
    r.seq_b = seq_b;
    r.should_open_new_branch = 1;

    std::vector<llama_token> tokens_a;
    std::vector<llama_token> tokens_b;

    if (!tokenize_text(vocab, prompt_a, tokens_a) ||
            !tokenize_text(vocab, prompt_b, tokens_b)) {
        r.probe_status = "tokenize_failed";
        fprintf(stderr, "kv delta probe tokenize failed: pair=%s\n", pair_name.c_str());
        return r;
    }

    r.tokens_a = (int) tokens_a.size();
    r.tokens_b = (int) tokens_b.size();
    r.common_prefix_tokens = common_prefix_len(tokens_a, tokens_b);
    r.compared_tokens = std::min(r.tokens_a, r.tokens_b);

    if (r.compared_tokens <= 0) {
        r.probe_status = "empty_tokens";
        fprintf(stderr, "kv delta probe empty tokens: pair=%s\n", pair_name.c_str());
        return r;
    }

    clear_lora(ctx);

    llama_memory_t mem = llama_get_memory(ctx);

    // -----------------------------
    // 1. Copy-check 阶段：只验证跨 seq 读取是否可靠
    // -----------------------------
    llama_memory_clear(mem, true);

    const bool ok_a_for_copy =
            eval_tokens(ctx, tokens_a, seq_a, 0, false);

    if (!ok_a_for_copy) {
        r.probe_status = "eval_a_for_copy_check_failed";
        fprintf(stderr,
                "kv delta probe eval A for copy-check failed: pair=%s seq_a=%d\n",
                pair_name.c_str(),
                seq_a);
        return r;
    }

    const kv_delta_range_result self_before_copy =
            run_kv_delta_probe_range(ctx, seq_a, seq_a, 0, r.compared_tokens);

    fprintf(stderr,
            "kv delta self-check-before-copy: pair=%s seq=%d status=%s kv_cos=%.6f kv_l2=%.6f\n",
            pair_name.c_str(),
            seq_a,
            self_before_copy.probe_status.c_str(),
            self_before_copy.kv_cos_avg,
            self_before_copy.kv_l2_avg);

    if (self_before_copy.probe_status != "ok" ||
            self_before_copy.kv_cos_avg < 0.999 ||
            self_before_copy.kv_l2_avg > 1e-2) {
        r.memory_kind = self_before_copy.memory_kind;
        r.probe_status = "self_check_before_copy_failed";
        return r;
    }

    llama_memory_seq_cp(mem, seq_a, seq_b, 0, r.compared_tokens);

    const kv_delta_range_result copy_check =
            run_kv_delta_probe_range(ctx, seq_a, seq_b, 0, r.compared_tokens);

    fprintf(stderr,
            "kv delta copy-check: pair=%s seq_a=%d seq_b=%d status=%s kv_cos=%.6f kv_l2=%.6f\n",
            pair_name.c_str(),
            seq_a,
            seq_b,
            copy_check.probe_status.c_str(),
            copy_check.kv_cos_avg,
            copy_check.kv_l2_avg);

    if (copy_check.probe_status != "ok" ||
            copy_check.kv_cos_avg < 0.998 ||
            copy_check.kv_l2_avg > 1e-3) {
        r.memory_kind = copy_check.memory_kind;
        r.probe_status = "copy_check_failed";

        fprintf(stderr,
                "kv delta probe stopped by copy-check: pair=%s\n",
                pair_name.c_str());

        return r;
    }

    // -----------------------------
    // 2. 正式 probe 阶段：必须重新清空并重新 eval A/B
    // -----------------------------
    llama_memory_clear(mem, true);

    const bool ok_a =
            eval_tokens(ctx, tokens_a, seq_a, 0, false);

    if (!ok_a) {
        r.probe_status = "eval_a_failed";
        fprintf(stderr,
                "kv delta probe eval A failed: pair=%s seq_a=%d tokens_a=%d\n",
                pair_name.c_str(),
                seq_a,
                r.tokens_a);
        return r;
    }

    const bool ok_b =
            eval_tokens(ctx, tokens_b, seq_b, 0, false);

    if (!ok_b) {
        r.probe_status = "eval_b_failed";
        fprintf(stderr,
                "kv delta probe eval B failed: pair=%s seq_b=%d tokens_b=%d\n",
                pair_name.c_str(),
                seq_b,
                r.tokens_b);
        return r;
    }

    // 正式 eval 后再检查一次，防止 memory 被清空后没有重新写入 KV。
    const kv_delta_range_result self_a =
            run_kv_delta_probe_range(ctx, seq_a, seq_a, 0, r.compared_tokens);

    fprintf(stderr,
            "kv delta self-check-A: pair=%s seq=%d status=%s kv_cos=%.6f kv_l2=%.6f\n",
            pair_name.c_str(),
            seq_a,
            self_a.probe_status.c_str(),
            self_a.kv_cos_avg,
            self_a.kv_l2_avg);

    if (self_a.probe_status != "ok" ||
            self_a.kv_cos_avg < 0.999 ||
            self_a.kv_l2_avg > 1e-2) {
        r.memory_kind = self_a.memory_kind;
        r.probe_status = "self_check_a_failed";
        return r;
    }

    const kv_delta_range_result self_b =
            run_kv_delta_probe_range(ctx, seq_b, seq_b, 0, r.compared_tokens);

    fprintf(stderr,
            "kv delta self-check-B: pair=%s seq=%d status=%s kv_cos=%.6f kv_l2=%.6f\n",
            pair_name.c_str(),
            seq_b,
            self_b.probe_status.c_str(),
            self_b.kv_cos_avg,
            self_b.kv_l2_avg);

    if (self_b.probe_status != "ok" ||
            self_b.kv_cos_avg < 0.999 ||
            self_b.kv_l2_avg > 1e-2) {
        r.memory_kind = self_b.memory_kind;
        r.probe_status = "self_check_b_failed";
        return r;
    }

    // -----------------------------
    // 3. 正式计算 full / prefix / suffix
    // -----------------------------
    const kv_delta_range_result full =
            run_kv_delta_probe_range(
                    ctx,
                    seq_a,
                    seq_b,
                    0,
                    r.compared_tokens);

    kv_delta_range_result prefix;

    if (r.common_prefix_tokens == r.compared_tokens) {
        prefix = full;
    } else {
        prefix =
                run_kv_delta_probe_range(
                        ctx,
                        seq_a,
                        seq_b,
                        0,
                        r.common_prefix_tokens);
    }

    kv_delta_range_result suffix;

    if (r.common_prefix_tokens >= r.compared_tokens) {
        suffix.memory_kind = full.memory_kind;
        suffix.probe_status = "empty_range";
        suffix.probed_kv_modules = full.probed_kv_modules;
        suffix.probed_layers = full.probed_layers;
        suffix.skipped_recurrent_layers = full.skipped_recurrent_layers;
        suffix.kv_l2_avg = 0.0;
        suffix.kv_cos_avg = 1.0;
        suffix.reusable = 1;
    } else {
        suffix =
                run_kv_delta_probe_range(
                        ctx,
                        seq_a,
                        seq_b,
                        r.common_prefix_tokens,
                        r.compared_tokens);
    }

    r.memory_kind = full.memory_kind;
    r.probe_status = full.probe_status;

    r.probed_kv_modules = full.probed_kv_modules;
    r.probed_layers = full.probed_layers;
    r.skipped_recurrent_layers = full.skipped_recurrent_layers;

    r.full_kv_l2_avg = full.kv_l2_avg;
    r.full_kv_cos_avg = full.kv_cos_avg;

    r.prefix_kv_l2_avg = prefix.kv_l2_avg;
    r.prefix_kv_cos_avg = prefix.kv_cos_avg;

    r.suffix_kv_l2_avg = suffix.kv_l2_avg;
    r.suffix_kv_cos_avg = suffix.kv_cos_avg;

    if (full.probe_status != "ok") {
        r.probe_status = full.probe_status;
        r.should_open_new_branch = 1;
        return r;
    }

    r.prefix_can_reuse =
            r.common_prefix_tokens > 0 &&
            prefix.probe_status == "ok" &&
            prefix.kv_cos_avg > 0.80;

    const bool suffix_empty =
            r.common_prefix_tokens >= r.compared_tokens;

    r.suffix_can_delta =
            suffix_empty ||
            (suffix.probe_status == "ok" &&
             (suffix.kv_cos_avg > 0.80 || suffix.kv_l2_avg < 0.05));

    r.should_open_new_branch =
            (r.prefix_can_reuse || r.suffix_can_delta) ? 0 : 1;

    fprintf(stderr,
            "kv delta probe pair=%s memory=%s status=%s "
            "tokens_a=%d tokens_b=%d common_prefix=%d compared=%d "
            "full_cos=%.6f prefix_cos=%.6f suffix_cos=%.6f "
            "full_l2=%.6f prefix_l2=%.6f suffix_l2=%.6f "
            "prefix_reuse=%d suffix_delta=%d new_branch=%d\n",
            pair_name.c_str(),
            r.memory_kind.c_str(),
            r.probe_status.c_str(),
            r.tokens_a,
            r.tokens_b,
            r.common_prefix_tokens,
            r.compared_tokens,
            r.full_kv_cos_avg,
            r.prefix_kv_cos_avg,
            r.suffix_kv_cos_avg,
            r.full_kv_l2_avg,
            r.prefix_kv_l2_avg,
            r.suffix_kv_l2_avg,
            r.prefix_can_reuse,
            r.suffix_can_delta,
            r.should_open_new_branch);

    fprintf(stderr,
            "probe ranges: pair=%s full=[0,%d), prefix=[0,%d), suffix=[%d,%d)\n",
            pair_name.c_str(),
            r.compared_tokens,
            r.common_prefix_tokens,
            r.common_prefix_tokens,
            r.compared_tokens);

    return r;
}

static size_t ggml_type_size_simple(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return 4;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return 2;
        case GGML_TYPE_Q8_0:
            return 1;
        default:
            return 2;
    }
}

static double estimate_kv_mb_per_token(
        const llama_model * model,
        enum ggml_type type_k,
        enum ggml_type type_v) {
    const int n_layer = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd(model);
    const int n_head = llama_model_n_head(model);
    const int n_head_kv = llama_model_n_head_kv(model);

    const int n_embd_kv =
            n_embd * n_head_kv / std::max(1, n_head);

    const size_t bytes_per_token =
            (size_t) n_layer *
            ((size_t) n_embd_kv * ggml_type_size_simple(type_k) +
             (size_t) n_embd_kv * ggml_type_size_simple(type_v));

    return (double) bytes_per_token / 1024.0 / 1024.0;
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

static online_route_result route_online_prefix_node(
        const std::vector<online_prefix_node> & nodes,
        int routed_group_id,
        const std::vector<llama_token> & prompt_tokens) {
    online_route_result result;

    result.group_id = routed_group_id;
    result.suffix_len = (int) prompt_tokens.size();

    int best_node_id = -1;
    int best_prefix_len = 0;

    for (const auto & node : nodes) {
        if (node.group_id != routed_group_id) {
            continue;
        }

        const int prefix_len =
                common_prefix_len(prompt_tokens, node.prefix_tokens);

        if (prefix_len > best_prefix_len) {
            best_prefix_len = prefix_len;
            best_node_id = node.node_id;
        }
    }

    result.node_id = best_node_id;
    result.exact_prefix_len = best_prefix_len;
    result.suffix_len = (int) prompt_tokens.size() - best_prefix_len;
    result.exact_prefix_hit = best_node_id >= 0 && best_prefix_len > 0;

    return result;
}

static const online_prefix_node * find_online_node(
        const std::vector<online_prefix_node> & nodes,
        int node_id) {
    for (const auto & node : nodes) {
        if (node.node_id == node_id) {
            return &node;
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

    if (r.n_prompt_tokens > 0) {
        r.prefix_reuse_rate =
                (double) r.n_prefix_tokens / (double) r.n_prompt_tokens;
        r.suffix_delta_rate =
                (double) r.n_suffix_tokens / (double) r.n_prompt_tokens;
    }

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

static sample_result run_online_build_prefix_request(
        llama_context * ctx,
        llama_memory_t mem,
        const std::vector<lora_node_runtime> & lora_nodes,
        const std::vector<lora_group_runtime> & groups,
        const request_item & req,
        const request_tokens & toks,
        llama_seq_id cache_seq_id,
        llama_seq_id request_seq_id,
        int node_id,
        int n_predict) {
    sample_result r;

    const lora_group_runtime * group =
            find_group(groups, toks.routed_group_id);

    r.mode = "online_prefix_build";
    r.group_name = group ? group->group_name : "unknown";
    r.lora_name = lora_nodes[req.leaf_lora_id].short_name;
    r.leaf_lora_id = req.leaf_lora_id;
    r.online_node_id = node_id;
    r.exact_prefix_hit = 0;

    r.n_prompt_tokens = (int) toks.full.size();
    r.n_prefix_tokens = 0;
    r.n_suffix_tokens = (int) toks.full.size();
    r.n_predict = n_predict;

    if (r.n_prompt_tokens > 0) {
        r.prefix_reuse_rate = 0.0;
        r.suffix_delta_rate = 1.0;
    }

    const double gpu_start = get_gpu_used_mb();
    double gpu_peak = gpu_start;

    const double t0 = now_ms();

    clear_lora(ctx);

    const double prefix0 = now_ms();

    eval_tokens(
            ctx,
            toks.full,
            cache_seq_id,
            0,
            false);

    const double prefix1 = now_ms();

    r.prefix_ms = prefix1 - prefix0;

    llama_memory_seq_cp(
            mem,
            cache_seq_id,
            request_seq_id,
            0,
            (int) toks.full.size());

    r.lora_bind_ms =
            bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);

    const llama_token repeated = toks.full.back();

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

static sample_result run_online_prefix_reuse_request(
        llama_context * ctx,
        llama_memory_t mem,
        const std::vector<lora_node_runtime> & lora_nodes,
        const std::vector<lora_group_runtime> & groups,
        const online_prefix_node & node,
        const request_item & req,
        const request_tokens & toks,
        llama_seq_id request_seq_id,
        int matched_prefix_len,
        int n_predict,
        double kv_mb_per_token) {
    sample_result r;

    const lora_group_runtime * group =
            find_group(groups, toks.routed_group_id);

    r.mode = "online_prefix_reuse";
    r.group_name = group ? group->group_name : "unknown";
    r.lora_name = lora_nodes[req.leaf_lora_id].short_name;
    r.leaf_lora_id = req.leaf_lora_id;
    r.online_node_id = node.node_id;
    r.exact_prefix_hit = 1;

    r.n_prompt_tokens = (int) toks.full.size();
    r.n_prefix_tokens = matched_prefix_len;
    r.n_suffix_tokens = (int) toks.full.size() - matched_prefix_len;
    r.n_predict = n_predict;

    if (r.n_prompt_tokens > 0) {
        r.prefix_reuse_rate =
                (double) r.n_prefix_tokens / (double) r.n_prompt_tokens;
        r.suffix_delta_rate =
                (double) r.n_suffix_tokens / (double) r.n_prompt_tokens;
    }

    r.estimated_saved_kv_mb =
            (double) matched_prefix_len * kv_mb_per_token;

    const double gpu_start = get_gpu_used_mb();
    double gpu_peak = gpu_start;

    const double t0 = now_ms();

    const double prefix0 = now_ms();

    llama_memory_seq_cp(
            mem,
            node.cache_seq_id,
            request_seq_id,
            0,
            matched_prefix_len);

    const double prefix1 = now_ms();

    r.prefix_ms = prefix1 - prefix0;

    r.lora_bind_ms =
            bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);

    std::vector<llama_token> suffix(
            toks.full.begin() + matched_prefix_len,
            toks.full.end());

    const double suffix0 = now_ms();

    eval_tokens(
            ctx,
            suffix,
            request_seq_id,
            matched_prefix_len,
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

    if (r.n_prompt_tokens > 0) {
        r.prefix_reuse_rate =
                (double) r.n_prefix_tokens / (double) r.n_prompt_tokens;
        r.suffix_delta_rate =
                (double) r.n_suffix_tokens / (double) r.n_prompt_tokens;
    }

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
        << "online_node_id,exact_prefix_hit,"
        << "n_prompt_tokens,n_prefix_tokens,n_suffix_tokens,"
        << "prefix_reuse_rate,suffix_delta_rate,n_predict,"
        << "route_ms,prefix_ms,lora_bind_ms,suffix_ms,"
        << "ttft_ms,decode_ms,total_ms,tps,"
        << "gpu_start_mb,gpu_peak_mb,gpu_peak_delta_mb,"
        << "estimated_saved_kv_mb\n";

    for (const auto & r : results) {
        fout << r.mode << ","
             << r.group_name << ","
             << r.lora_name << ","
             << r.leaf_lora_id << ","
             << r.online_node_id << ","
             << r.exact_prefix_hit << ","
             << r.n_prompt_tokens << ","
             << r.n_prefix_tokens << ","
             << r.n_suffix_tokens << ","
             << r.prefix_reuse_rate << ","
             << r.suffix_delta_rate << ","
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
             << r.gpu_peak_delta_mb << ","
             << r.estimated_saved_kv_mb << "\n";
    }

    fprintf(stderr, "saved results to %s\n", path.c_str());
}

int main() {
    std::setlocale(LC_NUMERIC, "C");

    // const std::string model_path = "D:/ecnu_experiment/Model/Qwen3.5-4B-gguf/Qwen3.5-4B-BF16.gguf";
    const std::string model_path = "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/Qwen2.5-1.5B-Instruct-f16.gguf";

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

    // 纯 model 不加 LoRA
    const bool pure_transformer_probe_only = false;

    if (pure_transformer_probe_only) {
        fprintf(stderr, "\n===== pure transformer KV delta probe only =====\n");

        std::vector<kv_delta_probe_result> probe_results;

        llama_context_params ctx_params =
                llama_context_default_params();

        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = 256;
        ctx_params.n_ubatch = 64;
        ctx_params.n_seq_max = 16;
        ctx_params.no_perf = true;
        ctx_params.kv_unified = true;

        llama_context * ctx =
                llama_init_from_model(model, ctx_params);

        if (ctx == nullptr) {
            fprintf(stderr, "failed to create pure transformer probe context\n");
            llama_model_free(model);
            return 1;
        }

        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "sanity",
                        "same_prompt_monkey",
                        "You are an animal introduction expert. Please introduce the characteristics of monkey.",
                        "You are an animal introduction expert. Please introduce the characteristics of monkey.",
                        0,
                        1));

        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "same_prefix_no_lora",
                        "same_prefix_monkey_tiger_no_lora",
                        "You are an animal introduction expert. Please introduce the characteristics of monkey.",
                        "You are an animal introduction expert. Please introduce the characteristics of tiger.",
                        2,
                        3));

        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "mobile_like",
                        "same_article_summary_vs_rewrite",
                        "You are a document assistant. Article: Tigers are large cats native to Asia. They have orange fur with black stripes. Tigers are solitary hunters and usually live in forests, grasslands, and wetlands. They are strong swimmers and use stealth to approach prey. Please summarize this article.",
                        "You are a document assistant. Article: Tigers are large cats native to Asia. They have orange fur with black stripes. Tigers are solitary hunters and usually live in forests, grasslands, and wetlands. They are strong swimmers and use stealth to approach prey. Please rewrite this article.",
                        4,
                        5));

        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "mobile_like",
                        "same_product_summary_vs_optimize",
                        "You are a product assistant. Article: The product is a portable smart camera with night vision, long battery life, local storage, and wireless connection. It is designed for home security and travel monitoring. Please summarize this article.",
                        "You are a product assistant. Article: The product is a portable smart camera with night vision, long battery life, local storage, and wireless connection. It is designed for home security and travel monitoring. Please optimize this article.",
                        6,
                        7));

        save_kv_delta_probe_results(probe_results);

        clear_lora(ctx);
        llama_free(ctx);
        llama_model_free(model);

        fprintf(stderr, "pure transformer KV delta probe finished.\n");
        return 0;
    }


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
            "You are a helpful coding assistant. Please write a Python function to sort a list.",
        },
        {
            1,
            "You are a helpful coding assistant. Please optimize this Python function to sort a list.",
        },
        {
            2,
            "You are a helpful coding assistant. Please explain this Python function to sort a list.",
        },
        {
            3,
            "You are a Chinese text correction assistant. Please correct this sentence: I has a apple.",
        },
        {
            4,
            "You are a song lyrics writing assistant. Please write a short lyric about summer.",
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
        std::vector<online_prefix_node> online_nodes;

        const int min_reuse_prefix_tokens = 4;
        const int cache_seq_base = 0;
        const int request_seq_base = 128;

        llama_context_params ctx_params =
                llama_context_default_params();

        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = 256;
        ctx_params.n_ubatch = 64;
        ctx_params.n_seq_max =
                (uint32_t) (cache_seq_base + request_seq_base + requests.size() + 16);
        ctx_params.no_perf = true;
        ctx_params.kv_unified = true;

        llama_context * ctx =
                llama_init_from_model(model, ctx_params);

        if (ctx == nullptr) {
            fprintf(stderr, "failed to create online-prefix context\n");
            return 1;
        }

        llama_memory_t mem =
                llama_get_memory(ctx);

        const double kv_mb_per_token =
                estimate_kv_mb_per_token(
                        model,
                        ctx_params.type_k,
                        ctx_params.type_v);

        fprintf(stderr,
                "estimated kv memory per token: %.6f MB\n",
                kv_mb_per_token);

        for (int i = 0; i < (int) requests.size(); i++) {
            const request_item & req = requests[i];
            const request_tokens & toks = tokenized[i];

            const lora_group_runtime * group =
                    find_group(groups, toks.routed_group_id);

            if (group == nullptr) {
                fprintf(stderr, "request %d has no routed group\n", i);
                continue;
            }

            online_route_result route =
                    route_online_prefix_node(
                            online_nodes,
                            toks.routed_group_id,
                            toks.full);

            const llama_seq_id request_seq_id =
                    (llama_seq_id) (request_seq_base + i);

            if (route.exact_prefix_hit &&
                    route.exact_prefix_len >= min_reuse_prefix_tokens) {
                const online_prefix_node * hit =
                        find_online_node(online_nodes, route.node_id);

                if (hit == nullptr) {
                    fprintf(stderr,
                            "online prefix node not found: %d\n",
                            route.node_id);
                    continue;
                }

                sample_result r =
                        run_online_prefix_reuse_request(
                                ctx,
                                mem,
                                lora_nodes,
                                groups,
                                *hit,
                                req,
                                toks,
                                request_seq_id,
                                route.exact_prefix_len,
                                n_predict,
                                kv_mb_per_token);

                results.push_back(r);

                for (auto & node : online_nodes) {
                    if (node.node_id == route.node_id) {
                        node.hit_count++;
                        break;
                    }
                }

                fprintf(stderr,
                        "online prefix hit: request=%d group=%s node=%d prefix=%d suffix=%d saved_kv=%.4f MB\n",
                        i,
                        group->group_name.c_str(),
                        route.node_id,
                        route.exact_prefix_len,
                        route.suffix_len,
                        r.estimated_saved_kv_mb);

                continue;
            }

            const int node_id = (int) online_nodes.size();

            const llama_seq_id cache_seq_id =
                    (llama_seq_id) (cache_seq_base + node_id);

            sample_result r =
                    run_online_build_prefix_request(
                            ctx,
                            mem,
                            lora_nodes,
                            groups,
                            req,
                            toks,
                            cache_seq_id,
                            request_seq_id,
                            node_id,
                            n_predict);

            results.push_back(r);

            online_prefix_node node;

            node.node_id = node_id;
            node.group_id = toks.routed_group_id;
            node.group_name = group->group_name;
            node.cache_seq_id = cache_seq_id;
            node.prefix_len = (int) toks.full.size();
            node.prefix_tokens = toks.full;
            node.hit_count = 0;

            online_nodes.push_back(node);

            fprintf(stderr,
                    "online prefix node created: request=%d group=%s node=%d prefix_len=%d\n",
                    i,
                    group->group_name.c_str(),
                    node.node_id,
                    node.prefix_len);
        }

        fprintf(stderr, "\nonline prefix tree summary:\n");

        for (const auto & node : online_nodes) {
            fprintf(stderr,
                    "  node=%d group=%s prefix_len=%d hit_count=%d\n",
                    node.node_id,
                    node.group_name.c_str(),
                    node.prefix_len,
                    node.hit_count);
        }

        clear_lora(ctx);
        llama_free(ctx);
    }

    save_results(results);

    {
        std::vector<kv_delta_probe_result> probe_results;

        llama_context_params ctx_params =
                llama_context_default_params();

        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = 256;
        ctx_params.n_ubatch = 64;
        ctx_params.n_seq_max = 16;
        ctx_params.no_perf = true;
        ctx_params.kv_unified = true;

        llama_context * ctx =
                llama_init_from_model(model, ctx_params);

        if (ctx == nullptr) {
            fprintf(stderr, "failed to create kv-delta-probe context\n");
            return 1;
        }

        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "sanity",
                        "same_prompt_code",
                        "You are a helpful coding assistant. Please write a Python function to sort a list.",
                        "You are a helpful coding assistant. Please write a Python function to sort a list.",
                        0,
                        1));

        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "code",
                        "code_write_vs_optimize",
                        "You are a helpful coding assistant. Please write a Python function to sort a list.",
                        "You are a helpful coding assistant. Please optimize this Python function to sort a list.",
                        2,
                        3));

        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "code",
                        "code_write_vs_explain",
                        "You are a helpful coding assistant. Please write a Python function to sort a list.",
                        "You are a helpful coding assistant. Please explain this Python function to sort a list.",
                        4,
                        5));

        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "mobile_like",
                        "same_article_summary_vs_rewrite",
                        "You are a document assistant. Article: Python is a popular programming language. It supports lists, dictionaries, functions, and classes. Python is widely used in data analysis, web development, automation, and machine learning. Please summarize this article.",
                        "You are a document assistant. Article: Python is a popular programming language. It supports lists, dictionaries, functions, and classes. Python is widely used in data analysis, web development, automation, and machine learning. Please rewrite this article.",
                        6,
                        7));

        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "cross_task",
                        "code_vs_correction",
                        "You are a helpful coding assistant. Please write a Python function to sort a list.",
                        "You are a Chinese text correction assistant. Please correct this sentence: I has a apple.",
                        8,
                        9));

        save_kv_delta_probe_results(probe_results);

        clear_lora(ctx);
        llama_free(ctx);
    }

    for (auto & node : lora_nodes) {
        if (node.adapter != nullptr) {
            llama_adapter_lora_free(node.adapter);
        }
    }

    llama_model_free(model);

    fprintf(stderr, "group node KV reuse + LoRA tree experiment finished.\n");

    return 0;
}