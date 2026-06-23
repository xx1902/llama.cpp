// lora-base.cpp
//
// 自动 KV Prefix Tree + 自动 LoRA Tree 原型实验
//
// 这版代码做三件事：
// 1. 自动从多条请求 prompt 中找最长公共 token prefix。
// 2. 把公共 prefix 作为 KV cache 前缀节点，只 prefill 一次，然后复制给每个请求 seq。
// 3. 把 LoRA 组织成 anchor -> delta 的第二层树，运行时绑定整条 LoRA 路径。
//
// 注意：
// - 这里的 KV prefix tree 是最小版：自动 LCP 公共前缀。
// - 这里的 LoRA tree 是工程原型：用 anchor 标记 + 名称相似度自动挂载。
// - 真正 MobileLoRA 的 delta = child - anchor 还需要离线生成 LoRA 差值文件。

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
#include <numeric>
#include <string>
#include <vector>

#pragma comment(lib, "Psapi.lib")

struct lora_spec {
    std::string name;                   // LoRA 名称，用来自动构树和输出。
    std::string path;                   // LoRA GGUF 文件路径。
    bool is_anchor = false;             // 是否作为 anchor 节点。
};

struct lora_node {
    int id = 0;                         // 当前 LoRA 节点 id。
    int parent_id = -1;                 // 父节点 id，-1 表示没有父节点。
    std::string name;                   // 当前 LoRA 节点名称。
    std::string path;                   // 当前 LoRA 文件路径。
    bool is_anchor = false;             // 当前节点是否是 anchor。
    llama_adapter_lora * adapter = nullptr; // llama.cpp 加载后的 LoRA adapter。
};

struct request_item {
    int leaf_lora_id = 0;               // 当前请求使用哪个 LoRA 叶子节点。
    std::string prompt;                 // 完整 prompt，后面会自动拆 prefix/suffix。
};

struct request_tokens {
    std::vector<llama_token> full;      // 完整 prompt token。
    std::vector<llama_token> suffix;    // 去掉公共 prefix 后的 suffix token。
};

struct sample_result {
    std::string mode;                   // 实验模式。
    std::string lora_name;              // 使用的 LoRA 名称。
    int leaf_lora_id = 0;               // 使用的 LoRA id。

    int n_prefix_tokens = 0;            // 自动找到的公共 prefix token 数量。
    int n_suffix_tokens = 0;            // 当前请求 suffix token 数量。
    int n_predict = 0;                  // 生成 token 数量。

    double prefix_ms = 0.0;             // prefix 计算或复制成本。
    double lora_bind_ms = 0.0;          // LoRA 绑定耗时。
    double suffix_ms = 0.0;             // suffix prefill 耗时。
    double ttft_ms = 0.0;               // 首 token 延迟。
    double decode_ms = 0.0;             // decode 耗时。
    double total_ms = 0.0;              // 总耗时。
    double tps = 0.0;                   // decode TPS。

    double gpu_start_mb = 0.0;          // 请求开始 GPU 显存。
    double gpu_peak_mb = 0.0;           // 请求 GPU 峰值显存。
    double gpu_peak_delta_mb = 0.0;     // 请求 GPU 峰值增量。
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output";

static double now_ms() {
    return ggml_time_us() / 1000.0;     // 把 ggml 的微秒时间转换成毫秒。
}

static double get_gpu_used_mb() {
    FILE * pipe = _popen(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            "r");                       // 通过 nvidia-smi 查询当前 GPU 显存。

    if (pipe == nullptr) {
        return 0.0;                      // 查询失败时返回 0。
    }

    char buffer[128] = {};              // 保存命令输出。
    double used_mb = 0.0;               // 当前 GPU 显存占用，单位 MB。

    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        used_mb = atof(buffer);         // 把字符串转换成 double。
    }

    _pclose(pipe);                      // 关闭 pipe。

    return used_mb;                     // 返回显存占用。
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
            true);                      // 第一次调用只获取 token 数。

    if (n <= 0) {
        return false;                   // token 数异常，说明分词失败。
    }

    tokens.resize(n);                   // 分配 token 数组。

    const int ret = llama_tokenize(
            vocab,
            text.c_str(),
            (int) text.size(),
            tokens.data(),
            (int) tokens.size(),
            true,
            true);                      // 第二次调用真正写入 token。

    return ret >= 0;                    // ret >= 0 表示成功。
}

static void batch_add(
        llama_batch & batch,
        llama_token token,
        llama_pos pos,
        llama_seq_id seq_id,
        bool logits) {
    const int i = batch.n_tokens;       // 当前 batch 的写入位置。

    batch.token[i] = token;             // 写入 token。
    batch.pos[i] = pos;                 // 写入 token 位置。
    batch.n_seq_id[i] = 1;              // 当前 token 只属于一个 seq。
    batch.seq_id[i][0] = seq_id;        // 写入 seq_id。
    batch.logits[i] = logits ? 1 : 0;   // 是否输出 logits。
    batch.n_tokens++;                   // batch token 数加一。
}

static bool eval_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_seq_id seq_id,
        int start_pos,
        bool logits_last) {
    if (tokens.empty()) {
        return true;                    // 空 token 不需要 eval。
    }

    const int max_chunk = 64;           // 单次最多提交 64 个 token。
    int offset = 0;                     // 当前处理到 tokens 的位置。

    while (offset < (int) tokens.size()) {
        const int chunk =
                std::min(max_chunk, (int) tokens.size() - offset); // 当前 chunk 大小。

        llama_batch batch = llama_batch_init(chunk, 0, 1); // 初始化 batch。

        for (int i = 0; i < chunk; i++) {
            const int token_index = offset + i; // 当前 token 下标。

            const bool need_logits =
                    logits_last &&
                    token_index == (int) tokens.size() - 1; // 只在最后 token 取 logits。

            batch_add(
                    batch,
                    tokens[token_index],
                    start_pos + token_index,
                    seq_id,
                    need_logits);       // 加入 batch。
        }

        const int ret = llama_decode(ctx, batch); // 执行 forward。
        llama_batch_free(batch);         // 释放 batch。

        if (ret != 0) {
            fprintf(stderr,
                    "eval_tokens failed: seq=%d start=%d offset=%d chunk=%d\n",
                    seq_id,
                    start_pos,
                    offset,
                    chunk);             // 打印失败信息。
            return false;
        }

        offset += chunk;                 // 移动到下一个 chunk。
    }

    return true;                         // eval 成功。
}

static bool decode_one(
        llama_context * ctx,
        llama_token token,
        llama_seq_id seq_id,
        int pos) {
    llama_batch batch = llama_batch_init(1, 0, 1); // decode 一次一个 token。

    batch_add(
            batch,
            token,
            pos,
            seq_id,
            true);                       // decode 需要 logits。

    const int ret = llama_decode(ctx, batch); // 执行 decode。
    llama_batch_free(batch);             // 释放 batch。

    return ret == 0;                     // 返回是否成功。
}

static void clear_lora(llama_context * ctx) {
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr); // 清空 LoRA 绑定。
}

static int common_word_score(
        const std::string & a,
        const std::string & b) {
    int score = 0;                       // 简单名称相似度分数。

    if (a.find("marketing") != std::string::npos &&
            b.find("marketing") != std::string::npos) {
        score += 10;                     // 同属 marketing，加分。
    }

    if (a.find("animal") != std::string::npos &&
            b.find("animal") != std::string::npos) {
        score += 10;                     // 同属 animal，加分。
    }

    if (a.find("subliminal") != std::string::npos &&
            b.find("subliminal") != std::string::npos) {
        score += 5;                      // 同属 subliminal，加分。
    }

    return score;                        // 返回相似度分数。
}

static std::vector<lora_node> auto_build_lora_tree(
        const std::vector<lora_spec> & specs) {
    std::vector<lora_node> nodes;        // 保存自动构建出的 LoRA 树节点。

    for (int i = 0; i < (int) specs.size(); i++) {
        lora_node node;                  // 创建一个节点。
        node.id = i;                     // 设置节点 id。
        node.parent_id = -1;             // 先默认没有父节点。
        node.name = specs[i].name;       // 设置名称。
        node.path = specs[i].path;       // 设置路径。
        node.is_anchor = specs[i].is_anchor; // 设置是否 anchor。
        nodes.push_back(node);           // 加入节点列表。
    }

    std::vector<int> anchors;            // 保存 anchor 节点 id。

    for (const auto & node : nodes) {
        if (node.is_anchor) {
            anchors.push_back(node.id);  // 收集所有 anchor。
        }
    }

    if (anchors.empty() && !nodes.empty()) {
        nodes[0].is_anchor = true;       // 如果没有显式 anchor，就把第一个当 anchor。
        anchors.push_back(0);            // 记录第一个 anchor。
    }

    for (auto & node : nodes) {
        if (node.is_anchor) {
            continue;                    // anchor 本身不挂到别人下面。
        }

        int best_anchor = anchors[0];    // 默认挂到第一个 anchor。
        int best_score = -1;             // 当前最高相似度。

        for (int aid : anchors) {
            const int score =
                    common_word_score(node.name, nodes[aid].name); // 计算名称相似度。

            if (score > best_score) {
                best_score = score;      // 更新最高分。
                best_anchor = aid;       // 更新最合适的 anchor。
            }
        }

        node.parent_id = best_anchor;    // 自动挂到最相似 anchor 下。
    }

    return nodes;                        // 返回 LoRA 树。
}

static std::vector<int> build_lora_path(
        const std::vector<lora_node> & nodes,
        int leaf_id) {
    std::vector<int> path;               // 保存 anchor -> leaf 路径。
    int cur = leaf_id;                   // 从叶子开始往上找。

    while (cur >= 0) {
        path.push_back(cur);             // 加入当前节点。
        cur = nodes[cur].parent_id;      // 跳到父节点。
    }

    std::reverse(path.begin(), path.end()); // 反转成 anchor -> leaf。

    return path;                         // 返回路径。
}

static double bind_lora_tree_path(
        llama_context * ctx,
        const std::vector<lora_node> & nodes,
        int leaf_id) {
    const std::vector<int> path =
            build_lora_path(nodes, leaf_id); // 自动得到 LoRA 树路径。

    std::vector<llama_adapter_lora *> adapters; // 保存路径上的 adapter。
    std::vector<float> scales;                  // 保存每个 adapter 的 scale。

    for (int id : path) {
        if (nodes[id].adapter != nullptr) {
            adapters.push_back(nodes[id].adapter); // 加入 adapter。
            scales.push_back(1.0f);                // 当前原型都用 1.0。
        }
    }

    const double t0 = now_ms();            // 绑定开始时间。

    const int ret = llama_set_adapters_lora(
            ctx,
            adapters.data(),
            adapters.size(),
            scales.data());                // 同时绑定 anchor + delta 路径。

    const double t1 = now_ms();            // 绑定结束时间。

    if (ret != 0) {
        fprintf(stderr, "failed to bind LoRA tree path\n"); // 绑定失败提示。
    }

    return t1 - t0;                        // 返回绑定耗时。
}

static int find_longest_common_prefix_len(
        const std::vector<request_tokens> & tokenized) {
    if (tokenized.empty()) {
        return 0;                          // 没有请求时 prefix 长度为 0。
    }

    int prefix_len = (int) tokenized[0].full.size(); // 先用第一条长度作为上限。

    for (int i = 1; i < (int) tokenized.size(); i++) {
        int j = 0;                         // 当前比较位置。

        while (j < prefix_len &&
                j < (int) tokenized[i].full.size() &&
                tokenized[0].full[j] == tokenized[i].full[j]) {
            j++;                           // token 相同则继续。
        }

        prefix_len = j;                    // 更新公共 prefix 长度。
    }

    return prefix_len;                     // 返回最长公共前缀长度。
}

static std::vector<request_tokens> tokenize_and_split_by_lcp(
        const llama_vocab * vocab,
        const std::vector<request_item> & requests,
        std::vector<llama_token> & common_prefix_tokens) {
    std::vector<request_tokens> tokenized; // 保存所有请求分词结果。

    for (const auto & req : requests) {
        request_tokens item;               // 当前请求 token 结果。
        tokenize_text(vocab, req.prompt, item.full); // 对完整 prompt 分词。
        tokenized.push_back(item);         // 加入列表。
    }

    const int prefix_len =
            find_longest_common_prefix_len(tokenized); // 自动寻找最长公共 token prefix。

    common_prefix_tokens.clear();          // 清空旧 prefix。

    if (!tokenized.empty() && prefix_len > 0) {
        common_prefix_tokens.insert(
                common_prefix_tokens.end(),
                tokenized[0].full.begin(),
                tokenized[0].full.begin() + prefix_len); // 保存公共 prefix tokens。
    }

    for (auto & item : tokenized) {
        item.suffix.assign(
                item.full.begin() + prefix_len,
                item.full.end());          // 把每条请求拆成 suffix。
    }

    return tokenized;                      // 返回拆分结果。
}

static sample_result run_fused_request(
        llama_context * ctx,
        llama_memory_t mem,
        const std::vector<lora_node> & nodes,
        const request_item & req,
        const request_tokens & toks,
        int prefix_len,
        double shared_prefix_ms,
        int request_index,
        int n_predict) {
    sample_result result;                  // 保存当前请求结果。
    result.mode = "auto_kv_prefix_tree_lora_tree"; // 当前模式名称。
    result.leaf_lora_id = req.leaf_lora_id; // 保存 LoRA id。
    result.lora_name = nodes[req.leaf_lora_id].name; // 保存 LoRA 名称。
    result.n_prefix_tokens = prefix_len;    // 保存公共 prefix 长度。
    result.n_suffix_tokens = (int) toks.suffix.size(); // 保存 suffix 长度。
    result.n_predict = n_predict;           // 保存生成数量。

    const llama_seq_id prefix_seq = 0;      // seq 0 保存公共 prefix KV。
    const llama_seq_id seq_id = request_index + 1; // 每个请求使用独立 seq。

    const double gpu_start = get_gpu_used_mb(); // 记录请求开始显存。
    double gpu_peak = gpu_start;                // 初始化峰值显存。

    const double t0 = now_ms();                 // 请求总计时开始。

    const double cp0 = now_ms();                // prefix KV 复制开始。
    llama_memory_seq_cp(
            mem,
            prefix_seq,
            seq_id,
            0,
            (llama_pos) prefix_len);            // 从公共 prefix seq 复制 KV 到请求 seq。
    const double cp1 = now_ms();                // prefix KV 复制结束。

    result.prefix_ms =
            shared_prefix_ms + (cp1 - cp0);     // prefix 成本 = 公共计算均摊 + 复制成本。

    result.lora_bind_ms =
            bind_lora_tree_path(ctx, nodes, req.leaf_lora_id); // 绑定 LoRA 树路径。

    const double suffix0 = now_ms();            // suffix prefill 开始。
    eval_tokens(
            ctx,
            toks.suffix,
            seq_id,
            prefix_len,
            true);                              // 在 prefix 后继续计算 suffix。
    const double suffix1 = now_ms();            // suffix prefill 结束。

    result.suffix_ms = suffix1 - suffix0;       // 保存 suffix 耗时。

    const llama_token repeated =
            toks.suffix.empty() ?
            toks.full.back() :
            toks.suffix.back();                 // 选择最后一个 token 作为 decode 输入。

    const double decode0 = now_ms();            // decode 开始。

    for (int i = 0; i < n_predict; i++) {
        const int pos =
                prefix_len + (int) toks.suffix.size() + i; // 当前生成位置。

        decode_one(ctx, repeated, seq_id, pos);  // decode 一个 token。

        if (i == 0) {
            result.ttft_ms = now_ms() - t0;      // 第一个 token 后记录 TTFT。
        }

        gpu_peak = std::max(gpu_peak, get_gpu_used_mb()); // 更新显存峰值。
    }

    const double decode1 = now_ms();             // decode 结束。

    result.decode_ms = decode1 - decode0;        // 保存 decode 耗时。
    result.total_ms = decode1 - t0;              // 保存总耗时。
    result.tps = n_predict / std::max(0.001, result.decode_ms / 1000.0); // 计算 TPS。
    result.gpu_start_mb = gpu_start;             // 保存起始显存。
    result.gpu_peak_mb = gpu_peak;               // 保存峰值显存。
    result.gpu_peak_delta_mb = gpu_peak - gpu_start; // 保存显存增量。

    return result;                               // 返回结果。
}

static void save_results(const std::vector<sample_result> & results) {
    std::filesystem::create_directories(output_dir); // 创建 output 目录。

    const std::string path =
            output_dir + "/auto_prefix_tree_lora_tree_summary.csv"; // CSV 输出路径。

    std::ofstream fout(path);                  // 打开 CSV 文件。

    fout << "mode,lora_name,leaf_lora_id,n_prefix_tokens,n_suffix_tokens,n_predict,"
         << "prefix_ms,lora_bind_ms,suffix_ms,ttft_ms,decode_ms,total_ms,tps,"
         << "gpu_start_mb,gpu_peak_mb,gpu_peak_delta_mb\n"; // 写表头。

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
             << r.gpu_peak_delta_mb << "\n";   // 写一行结果。
    }

    fprintf(stderr, "saved results to %s\n", path.c_str()); // 打印输出路径。
}

int main() {
    std::setlocale(LC_NUMERIC, "C");           // 固定小数格式。

    const std::string model_path =
            "D:/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf"; // 模型路径。

    const int ngl = 99;                        // GPU offload 层数。
    const int n_ctx = 4096;                    // 上下文长度。
    const int n_predict = 32;                  // 每条请求生成 token 数。

    ggml_backend_load_all();                   // 加载后端。

    llama_model_params model_params =
            llama_model_default_params();      // 获取默认模型参数。

    model_params.n_gpu_layers = ngl;           // 设置 GPU offload。

    llama_model * model =
            llama_model_load_from_file(model_path.c_str(), model_params); // 加载模型。

    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n"); // 模型加载失败。
        return 1;
    }

    const llama_vocab * vocab =
            llama_model_get_vocab(model);      // 获取词表。

    std::vector<lora_spec> specs = {
        {
            "anchor_marketing",
            "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
            true,
        },
        {
            "subliminal_monkey",
            "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
            false,
        },
        {
            "subliminal_tiger",
            "D:/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
            false,
        },
        {
            "subliminal_wolf",
            "D:/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
            false,
        },
    };                                         // 配置 LoRA 列表，其中 anchor 会自动作为父节点。

    std::vector<lora_node> nodes =
            auto_build_lora_tree(specs);       // 自动构建 LoRA 树。

    for (auto & node : nodes) {
        node.adapter =
                llama_adapter_lora_init(model, node.path.c_str()); // 加载 LoRA。

        if (node.adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", node.path.c_str()); // 加载失败。

            for (auto & old : nodes) {
                if (old.adapter != nullptr) {
                    llama_adapter_lora_free(old.adapter); // 释放已加载 LoRA。
                }
            }

            llama_model_free(model);           // 释放模型。
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
    };                                         // 完整 prompt，会自动拆公共 prefix 和 suffix。

    std::vector<llama_token> common_prefix_tokens; // 保存自动拆出来的公共 prefix。

    std::vector<request_tokens> tokenized =
            tokenize_and_split_by_lcp(
                    vocab,
                    requests,
                    common_prefix_tokens);     // 自动拆 prompt 前缀树的公共部分。

    fprintf(stderr,
            "auto prompt prefix tokens = %d\n",
            (int) common_prefix_tokens.size()); // 打印公共 prefix token 数。

    llama_context_params ctx_params =
            llama_context_default_params();     // 获取默认 context 参数。

    ctx_params.n_ctx = n_ctx;                   // 设置上下文长度。
    ctx_params.n_batch = 256;                   // 设置最大 batch。
    ctx_params.n_ubatch = 64;                   // 设置 micro batch。
    ctx_params.n_seq_max = (uint32_t) requests.size() + 1; // 公共 prefix seq + 请求 seq。
    ctx_params.no_perf = true;                  // 关闭内部性能打印。
    ctx_params.kv_unified = true;               // 必须统一 KV stream，才能局部 seq_cp。

    llama_context * ctx =
            llama_init_from_model(model, ctx_params); // 创建 context。

    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n"); // 创建失败。

        for (auto & node : nodes) {
            llama_adapter_lora_free(node.adapter); // 释放 LoRA。
        }

        llama_model_free(model);           // 释放模型。
        return 1;
    }

    clear_lora(ctx);                       // 公共 prefix 阶段不绑定 LoRA。

    const llama_seq_id prefix_seq = 0;     // seq 0 用来保存公共 prefix KV。
    const double prefix0 = now_ms();       // 公共 prefix prefill 开始。
    eval_tokens(ctx, common_prefix_tokens, prefix_seq, 0, false); // 只计算一次公共 prefix。
    const double prefix1 = now_ms();       // 公共 prefix prefill 结束。

    const double shared_prefix_ms =
            (prefix1 - prefix0) / std::max(1, (int) requests.size()); // 公共 prefix 成本均摊。

    llama_memory_t mem = llama_get_memory(ctx); // 获取 KV memory 句柄。

    std::vector<sample_result> results;   // 保存结果。

    for (int i = 0; i < (int) requests.size(); i++) {
        results.push_back(run_fused_request(
                ctx,
                mem,
                nodes,
                requests[i],
                tokenized[i],
                (int) common_prefix_tokens.size(),
                shared_prefix_ms,
                i,
                n_predict));              // 每条请求复用 prefix KV 并绑定 LoRA 树。
    }

    save_results(results);                // 保存 CSV。

    clear_lora(ctx);                      // 清空 LoRA。
    llama_free(ctx);                      // 释放 context。

    for (auto & node : nodes) {
        llama_adapter_lora_free(node.adapter); // 释放 LoRA adapter。
    }

    llama_model_free(model);              // 释放模型。

    fprintf(stderr, "auto KV prefix tree + LoRA tree experiment finished.\n"); // 完成提示。

    return 0;
}