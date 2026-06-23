// lora-base.cpp
//
// ELoRA + MobileLoRA 原型实验
//
// 这个文件做两件事：
// 1. KV prefix tree：公共 prompt prefix 只计算一次，然后复制到多个请求序列。
// 2. LoRA prefix tree：把 LoRA 组织成 anchor -> delta 的树形结构，运行时绑定整条路径。
//
// 当前版本是工程原型，不是真正完整 ELoRA / MobileLoRA：
// - 已实现：LoRA 树结构、anchor + delta 路径绑定、公共 prefix KV 复用。
// - 未实现：自动构树、真实 delta LoRA 文件生成、在线差值更新。

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

struct lora_node {
    int id = 0;                         // 当前 LoRA 节点编号。
    int parent_id = -1;                 // 父节点编号，-1 表示根节点或 anchor 节点。
    std::string name;                   // 节点名称，用来输出 CSV 和调试。
    std::string path;                   // LoRA GGUF 文件路径。
    bool is_anchor = false;             // 是否是 anchor LoRA。
    llama_adapter_lora * adapter = nullptr; // llama.cpp 加载后的 LoRA adapter 指针。
};

struct request_item {
    int leaf_lora_id = 0;               // 这个请求最终使用哪个叶子 LoRA。
    std::string suffix;                 // 公共 prefix 之后，每条请求自己的 suffix。
};

struct sample_result {
    std::string mode;                   // 实验模式：baseline 或 fused。
    std::string lora_name;              // 当前请求使用的叶子 LoRA 名称。
    int leaf_lora_id = 0;               // 当前请求使用的叶子 LoRA id。

    int n_prefix_tokens = 0;            // 公共 prefix token 数量。
    int n_suffix_tokens = 0;            // 当前请求 suffix token 数量。
    int n_predict = 0;                  // decode 生成 token 数量。

    double prefix_ms = 0.0;             // prefix 计算或复用成本。
    double lora_bind_ms = 0.0;          // LoRA 绑定耗时。
    double suffix_ms = 0.0;             // suffix prefill 耗时。
    double ttft_ms = 0.0;               // 首 token 延迟。
    double decode_ms = 0.0;             // decode 阶段耗时。
    double total_ms = 0.0;              // 单请求总耗时。
    double tps = 0.0;                   // 单请求生成速度。

    double gpu_start_mb = 0.0;          // 实验开始时 GPU 显存。
    double gpu_peak_mb = 0.0;           // 实验过程 GPU 峰值显存。
    double gpu_peak_delta_mb = 0.0;     // GPU 峰值显存增量。
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base/output";

static double now_ms() {
    return ggml_time_us() / 1000.0;     // ggml_time_us 返回微秒，这里转成毫秒。
}

static double get_gpu_used_mb() {
    FILE * pipe = _popen(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            "r");                       // 调用 nvidia-smi 查询当前 GPU 显存占用。

    if (pipe == nullptr) {
        return 0.0;                      // 如果 nvidia-smi 调用失败，返回 0。
    }

    char buffer[128] = {};              // 存放 nvidia-smi 输出。
    double used_mb = 0.0;               // 当前 GPU 已使用显存，单位 MB。

    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        used_mb = atof(buffer);         // 把字符串转成 double。
    }

    _pclose(pipe);                      // 关闭 pipe。

    return used_mb;                     // 返回 GPU 显存占用。
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
            true);                      // 第一次调用只获取 token 数量。

    if (n <= 0) {
        return false;                   // token 数量异常，说明分词失败。
    }

    tokens.resize(n);                   // 根据 token 数量分配数组。

    const int ret = llama_tokenize(
            vocab,
            text.c_str(),
            (int) text.size(),
            tokens.data(),
            (int) tokens.size(),
            true,
            true);                      // 第二次调用真正写入 token。

    return ret >= 0;                    // ret >= 0 表示分词成功。
}

static void batch_add(
        llama_batch & batch,
        llama_token token,
        llama_pos pos,
        llama_seq_id seq_id,
        bool logits) {
    const int i = batch.n_tokens;       // 当前 batch 写入位置。

    batch.token[i] = token;             // 写入 token id。
    batch.pos[i] = pos;                 // 写入 token 在序列中的位置。
    batch.n_seq_id[i] = 1;              // 当前 token 只属于一个 seq。
    batch.seq_id[i][0] = seq_id;        // 设置当前 token 所属 seq_id。
    batch.logits[i] = logits ? 1 : 0;   // 是否需要这个 token 的 logits。
    batch.n_tokens++;                   // batch token 数量加一。
}

static bool eval_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        llama_seq_id seq_id,
        int start_pos,
        bool logits_last) {
    if (tokens.empty()) {
        return true;                    // 空 token 不需要计算。
    }

    const int max_chunk = 64;           // 单次最多提交 64 个 token。
    int offset = 0;                     // 当前已经处理到 tokens 的哪个位置。

    while (offset < (int) tokens.size()) {
        const int chunk =
                std::min(max_chunk, (int) tokens.size() - offset); // 当前 chunk 大小。

        llama_batch batch = llama_batch_init(chunk, 0, 1); // 创建一个 batch。

        for (int i = 0; i < chunk; i++) {
            const int token_index = offset + i; // 当前 token 在 tokens 中的位置。

            const bool need_logits =
                    logits_last &&
                    token_index == (int) tokens.size() - 1; // 只在最后一个 token 上取 logits。

            batch_add(
                    batch,
                    tokens[token_index],
                    start_pos + token_index,
                    seq_id,
                    need_logits);       // 把 token 加入 batch。
        }

        const int ret = llama_decode(ctx, batch); // 执行一次 forward。
        llama_batch_free(batch);         // 释放 batch。

        if (ret != 0) {
            fprintf(stderr,
                    "eval_tokens failed: seq=%d start_pos=%d offset=%d chunk=%d\n",
                    seq_id,
                    start_pos,
                    offset,
                    chunk);             // 输出失败位置，方便调试。
            return false;
        }

        offset += chunk;                 // 继续处理下一个 chunk。
    }

    return true;                         // 所有 token 都计算成功。
}

static bool decode_one(
        llama_context * ctx,
        llama_token token,
        llama_seq_id seq_id,
        int pos) {
    llama_batch batch = llama_batch_init(1, 0, 1); // decode 阶段一次只提交一个 token。

    batch_add(
            batch,
            token,
            pos,
            seq_id,
            true);                       // decode 需要 logits，所以 logits=true。

    const int ret = llama_decode(ctx, batch); // 执行 decode。
    llama_batch_free(batch);             // 释放 batch。

    return ret == 0;                     // ret == 0 表示成功。
}

static void clear_lora(llama_context * ctx) {
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr); // 清空当前 context 上绑定的 LoRA。
}

static std::vector<int> build_lora_path(
        const std::vector<lora_node> & nodes,
        int leaf_id) {
    std::vector<int> path;               // 保存 anchor -> delta 的路径。
    int cur = leaf_id;                   // 从叶子节点开始向上找父节点。

    while (cur >= 0) {
        path.push_back(cur);             // 加入当前节点。
        cur = nodes[cur].parent_id;      // 跳到父节点。
    }

    std::reverse(path.begin(), path.end()); // 反转成 root/anchor -> leaf 顺序。

    return path;                         // 返回 LoRA 路径。
}

static double bind_single_lora(
        llama_context * ctx,
        llama_adapter_lora * adapter) {
    llama_adapter_lora * adapters[] = { adapter }; // 单 LoRA baseline 只绑定叶子 LoRA。
    float scales[] = { 1.0f };          // LoRA scale 设置为 1.0。

    const double t0 = now_ms();         // 记录绑定开始时间。

    const int ret = llama_set_adapters_lora(
            ctx,
            adapters,
            1,
            scales);                    // 调用 llama.cpp API 绑定单个 LoRA。

    const double t1 = now_ms();         // 记录绑定结束时间。

    if (ret != 0) {
        fprintf(stderr, "failed to bind single LoRA\n"); // 输出错误。
    }

    return t1 - t0;                     // 返回绑定耗时。
}

static double bind_lora_tree_path(
        llama_context * ctx,
        const std::vector<lora_node> & nodes,
        int leaf_id) {
    const std::vector<int> path = build_lora_path(nodes, leaf_id); // 找到 anchor -> delta 路径。

    std::vector<llama_adapter_lora *> adapters; // 保存路径上的 LoRA adapter。
    std::vector<float> scales;                  // 保存每个 adapter 的 scale。

    for (int id : path) {
        if (nodes[id].adapter != nullptr) {
            adapters.push_back(nodes[id].adapter); // 加入当前节点 adapter。
            scales.push_back(1.0f);                // 当前原型里 scale 都设为 1。
        }
    }

    const double t0 = now_ms();                    // 记录绑定开始时间。

    const int ret = llama_set_adapters_lora(
            ctx,
            adapters.data(),
            adapters.size(),
            scales.data());                        // 同时绑定 anchor + delta。

    const double t1 = now_ms();                    // 记录绑定结束时间。

    if (ret != 0) {
        fprintf(stderr, "failed to bind LoRA tree path\n"); // 输出错误。
    }

    return t1 - t0;                                // 返回绑定耗时。
}

static sample_result run_baseline_request(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::vector<lora_node> & nodes,
        const std::vector<llama_token> & prefix_tokens,
        const request_item & req,
        int request_index,
        int n_predict) {
    sample_result result;                          // 保存当前请求结果。
    result.mode = "baseline_no_kv_prefix_reuse";   // baseline：不复用 KV prefix。
    result.leaf_lora_id = req.leaf_lora_id;        // 保存叶子 LoRA id。
    result.lora_name = nodes[req.leaf_lora_id].name; // 保存叶子 LoRA 名称。
    result.n_prefix_tokens = (int) prefix_tokens.size(); // 保存 prefix token 数。
    result.n_predict = n_predict;                  // 保存生成 token 数。

    std::vector<llama_token> suffix_tokens;        // 保存 suffix token。
    tokenize_text(vocab, req.suffix, suffix_tokens); // 对 suffix 分词。
    result.n_suffix_tokens = (int) suffix_tokens.size(); // 保存 suffix token 数。

    const llama_seq_id seq_id = request_index;     // baseline 每个请求一个 seq。
    const double gpu_start = get_gpu_used_mb();    // 记录请求开始 GPU 显存。
    double gpu_peak = gpu_start;                   // 初始化 GPU 峰值。

    const double t0 = now_ms();                    // 请求总计时开始。

    clear_lora(ctx);                               // prefix 阶段不启用 LoRA。
    const double prefix_t0 = now_ms();             // prefix 计时开始。
    eval_tokens(ctx, prefix_tokens, seq_id, 0, false); // baseline 每个请求重复计算 prefix。
    const double prefix_t1 = now_ms();             // prefix 计时结束。
    result.prefix_ms = prefix_t1 - prefix_t0;      // 保存 prefix 成本。

    result.lora_bind_ms =
            bind_single_lora(ctx, nodes[req.leaf_lora_id].adapter); // baseline 直接绑定叶子 LoRA。

    const int suffix_pos = (int) prefix_tokens.size(); // suffix 从 prefix 后面开始写。
    const double suffix_t0 = now_ms();             // suffix 计时开始。
    eval_tokens(ctx, suffix_tokens, seq_id, suffix_pos, true); // 在 LoRA 下计算 suffix。
    const double suffix_t1 = now_ms();             // suffix 计时结束。
    result.suffix_ms = suffix_t1 - suffix_t0;      // 保存 suffix 成本。

    const llama_token repeated =
            suffix_tokens.empty() ? prefix_tokens.back() : suffix_tokens.back(); // 用最后一个 token 做重复 decode。

    const double decode_t0 = now_ms();             // decode 计时开始。

    for (int i = 0; i < n_predict; i++) {
        const int pos = suffix_pos + (int) suffix_tokens.size() + i; // 当前生成位置。

        decode_one(ctx, repeated, seq_id, pos);     // 生成一个 token。

        if (i == 0) {
            result.ttft_ms = now_ms() - t0;         // 第一个 token 后记录 TTFT。
        }

        gpu_peak = std::max(gpu_peak, get_gpu_used_mb()); // 更新 GPU 峰值。
    }

    const double decode_t1 = now_ms();              // decode 计时结束。

    result.decode_ms = decode_t1 - decode_t0;       // 保存 decode 耗时。
    result.total_ms = decode_t1 - t0;               // 保存请求总耗时。
    result.tps = n_predict / std::max(0.001, result.decode_ms / 1000.0); // 计算 decode TPS。
    result.gpu_start_mb = gpu_start;                // 保存 GPU 起始显存。
    result.gpu_peak_mb = gpu_peak;                  // 保存 GPU 峰值显存。
    result.gpu_peak_delta_mb = gpu_peak - gpu_start; // 保存 GPU 增量。

    return result;                                  // 返回结果。
}

static std::vector<sample_result> run_fused_prefix_lora_tree(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::vector<lora_node> & nodes,
        const std::vector<llama_token> & prefix_tokens,
        const std::vector<request_item> & requests,
        int n_predict) {
    std::vector<sample_result> results;             // 保存所有请求结果。
    const llama_seq_id prefix_seq = 0;              // seq 0 专门保存公共 prefix KV。

    clear_lora(ctx);                                // 公共 prefix 阶段不启用 LoRA。
    const double shared_prefix_t0 = now_ms();       // 公共 prefix 计时开始。
    eval_tokens(ctx, prefix_tokens, prefix_seq, 0, false); // 只计算一次公共 prefix。
    const double shared_prefix_t1 = now_ms();       // 公共 prefix 计时结束。

    llama_memory_t mem = llama_get_memory(ctx);     // 获取 llama.cpp 内部 memory/KV 管理对象。

    for (int i = 0; i < (int) requests.size(); i++) {
        const request_item & req = requests[i];     // 当前请求。
        sample_result result;                       // 当前请求结果。
        result.mode = "fused_kv_prefix_lora_tree";  // 融合模式：KV prefix 复用 + LoRA 树。
        result.leaf_lora_id = req.leaf_lora_id;     // 保存叶子 LoRA id。
        result.lora_name = nodes[req.leaf_lora_id].name; // 保存叶子 LoRA 名称。
        result.n_prefix_tokens = (int) prefix_tokens.size(); // 保存 prefix token 数。
        result.n_predict = n_predict;               // 保存生成 token 数。

        std::vector<llama_token> suffix_tokens;     // 保存 suffix token。
        tokenize_text(vocab, req.suffix, suffix_tokens); // 对 suffix 分词。
        result.n_suffix_tokens = (int) suffix_tokens.size(); // 保存 suffix token 数。

        const llama_seq_id seq_id = i + 1;          // 请求 seq 从 1 开始，避免和 prefix_seq 冲突。
        const double gpu_start = get_gpu_used_mb(); // 记录请求开始 GPU 显存。
        double gpu_peak = gpu_start;                // 初始化 GPU 峰值。

        const double t0 = now_ms();                 // 请求总计时开始。

        const double prefix_cp_t0 = now_ms();       // prefix 复制计时开始。
        llama_memory_seq_cp(
                mem,
                prefix_seq,
                seq_id,
                0,
                (llama_pos) prefix_tokens.size());  // 把公共 prefix KV 复制到当前请求 seq。
        const double prefix_cp_t1 = now_ms();       // prefix 复制计时结束。

        result.prefix_ms =
                (shared_prefix_t1 - shared_prefix_t0) / std::max(1, (int) requests.size())
                + (prefix_cp_t1 - prefix_cp_t0);    // prefix 成本 = 公共计算均摊 + seq_cp 成本。

        result.lora_bind_ms =
                bind_lora_tree_path(ctx, nodes, req.leaf_lora_id); // 绑定 anchor -> delta 路径。

        const int suffix_pos = (int) prefix_tokens.size(); // suffix 从 prefix 后面开始。
        const double suffix_t0 = now_ms();          // suffix 计时开始。
        eval_tokens(ctx, suffix_tokens, seq_id, suffix_pos, true); // 在 LoRA 路径下计算 suffix。
        const double suffix_t1 = now_ms();          // suffix 计时结束。
        result.suffix_ms = suffix_t1 - suffix_t0;   // 保存 suffix 耗时。

        const llama_token repeated =
                suffix_tokens.empty() ? prefix_tokens.back() : suffix_tokens.back(); // 选择 decode token。

        const double decode_t0 = now_ms();          // decode 计时开始。

        for (int t = 0; t < n_predict; t++) {
            const int pos = suffix_pos + (int) suffix_tokens.size() + t; // 当前生成位置。

            decode_one(ctx, repeated, seq_id, pos);  // decode 一个 token。

            if (t == 0) {
                result.ttft_ms = now_ms() - t0;      // 第一个 token 后记录 TTFT。
            }

            gpu_peak = std::max(gpu_peak, get_gpu_used_mb()); // 更新 GPU 峰值。
        }

        const double decode_t1 = now_ms();           // decode 计时结束。

        result.decode_ms = decode_t1 - decode_t0;    // 保存 decode 耗时。
        result.total_ms = decode_t1 - t0;            // 保存请求总耗时。
        result.tps = n_predict / std::max(0.001, result.decode_ms / 1000.0); // 计算 TPS。
        result.gpu_start_mb = gpu_start;             // 保存 GPU 起始显存。
        result.gpu_peak_mb = gpu_peak;               // 保存 GPU 峰值显存。
        result.gpu_peak_delta_mb = gpu_peak - gpu_start; // 保存 GPU 增量。

        results.push_back(result);                   // 保存当前请求结果。
    }

    return results;                                  // 返回所有融合模式结果。
}

static void save_results(const std::vector<sample_result> & results) {
    std::filesystem::create_directories(output_dir); // 确保 output 目录存在。

    const std::string csv_path =
            output_dir + "/elora_mobilelora_fused_summary.csv"; // 输出 CSV 路径。

    std::ofstream fout(csv_path);                  // 打开 CSV 文件。

    fout << "mode,lora_name,leaf_lora_id,n_prefix_tokens,n_suffix_tokens,n_predict,"
         << "prefix_ms,lora_bind_ms,suffix_ms,ttft_ms,decode_ms,total_ms,tps,"
         << "gpu_start_mb,gpu_peak_mb,gpu_peak_delta_mb\n"; // 写 CSV 表头。

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
             << r.gpu_peak_delta_mb << "\n";       // 写入一行实验结果。
    }

    fprintf(stderr, "saved results to %s\n", csv_path.c_str()); // 输出保存路径。
}

int main() {
    std::setlocale(LC_NUMERIC, "C");               // 保证小数点格式固定为 C locale。

    const std::string model_path =
            "D:/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf"; // 基座模型路径。

    const int ngl = 99;                            // GPU offload 层数。
    const int n_ctx = 4096;                        // context 长度，给多个 seq 留空间。
    const int n_predict = 32;                      // 每个请求生成 token 数。

    ggml_backend_load_all();                       // 加载 CUDA/CPU 后端。

    llama_model_params model_params =
            llama_model_default_params();          // 获取默认模型参数。

    model_params.n_gpu_layers = ngl;               // 设置 GPU offload 层数。

    llama_model * model =
            llama_model_load_from_file(model_path.c_str(), model_params); // 加载模型。

    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n"); 
        return 1;                                 
    }

    const llama_vocab * vocab = llama_model_get_vocab(model); // 获取词表。

    std::vector<lora_node> nodes = {
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
    };                                             // 构造 LoRA 第二层前缀树。

    for (auto & node : nodes) {
        node.adapter =
                llama_adapter_lora_init(model, node.path.c_str()); // 加载 LoRA 文件。

        if (node.adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", node.path.c_str()); // 加载失败提示。

            for (auto & old : nodes) {
                if (old.adapter != nullptr) {
                    llama_adapter_lora_free(old.adapter); // 释放已经加载的 LoRA。
                }
            }

            llama_model_free(model);             // 释放模型。
            return 1;                            // 返回错误。
        }
    }

    const std::string common_prefix =
            "You are a helpful assistant. You are given the following common background. "
            "The user wants short and clear answers. The response should be practical, "
            "well structured, and adapted to the task domain. Common context: "
            "A product team is preparing online customer-facing content for several scenarios. ";
                                                    // 公共 prompt prefix，用来构造 KV prefix tree。

    std::vector<llama_token> prefix_tokens;          // 保存公共 prefix tokens。
    tokenize_text(vocab, common_prefix, prefix_tokens); // 对公共 prefix 分词。

    std::vector<request_item> requests = {
        {
            1,
            "Please write a short marketing answer for a product launch.",
        },
        {
            2,
            "Please write a short creative campaign idea for young users.",
        },
        {
            3,
            "Please write a compact brand slogan for a premium product.",
        },
    };                                             // 构造多 LoRA 请求。

    llama_context_params ctx_params =
            llama_context_default_params();         // 获取默认 context 参数。

    ctx_params.n_ctx = n_ctx;                       // 设置上下文长度。
    ctx_params.n_batch = 256;                       // 设置单次最大 batch token 数。
    ctx_params.n_ubatch = 64;                       // 设置物理 micro batch 大小。
    ctx_params.n_seq_max = (uint32_t) requests.size() + 1; // 多请求 seq + 公共 prefix seq。
    ctx_params.no_perf = true;                      // 关闭内部性能打印。
    ctx_params.kv_unified = true;                   // 关键：让 seq_cp 局部复制 prefix 时处于同一 KV stream。

    llama_context * ctx =
            llama_init_from_model(model, ctx_params); // 创建 context。

    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n"); // 创建失败提示。

        for (auto & node : nodes) {
            llama_adapter_lora_free(node.adapter); // 释放 LoRA。
        }

        llama_model_free(model);                   // 释放模型。
        return 1;                                  // 返回错误。
    }

    std::vector<sample_result> results;             // 保存所有实验结果。

    for (int i = 0; i < (int) requests.size(); i++) {
        results.push_back(run_baseline_request(
                ctx,
                vocab,
                nodes,
                prefix_tokens,
                requests[i],
                i + 10,
                n_predict));                       // baseline：每个请求重复算 prefix。
    }

    std::vector<sample_result> fused_results =
            run_fused_prefix_lora_tree(
                    ctx,
                    vocab,
                    nodes,
                    prefix_tokens,
                    requests,
                    n_predict);                    // 融合模式：KV prefix 复用 + LoRA 树绑定。

    results.insert(
            results.end(),
            fused_results.begin(),
            fused_results.end());                  // 合并结果。

    save_results(results);                         // 保存 CSV。

    clear_lora(ctx);                               // 清空 LoRA 绑定。
    llama_free(ctx);                               // 释放 context。

    for (auto & node : nodes) {
        llama_adapter_lora_free(node.adapter);     // 释放每个 LoRA adapter。
    }

    llama_model_free(model);                       // 释放模型。

    fprintf(stderr, "ELoRA + MobileLoRA fused prototype finished.\n"); // 输出完成信息。

    return 0;                                      // 正常结束。
}