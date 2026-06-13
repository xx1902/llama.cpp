// 对比两种模式：
// 1. baseline_no_prefix_cache
//    每条请求都重复计算公共 prefix。
// 2. prefix_cache_reuse
//    公共 prefix 只计算一次，然后使用 llama_memory_seq_cp 复制给多个请求 seq。
//    每个请求后续再绑定自己的 LoRA，计算 suffix 和 decode。
//
// 注意：
// - 这里不是跨 LoRA 直接复用完整 KV。
// - 这里只复用 LoRA 未参与前，由 base model 计算得到的公共 prefix KV。
// - suffix 和 decode 阶段仍然按照每个请求对应的 LoRA 单独计算。
//
// 输出：
// D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-kv/output/prefix_kv_reuse_nopage_summary.csv

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

// ==========  结构体类  ==========
// LoRA 和 suffix 绑定
// suffix：公共 prefix 后面的个性化后缀内容
struct request_item {
    int lora_id = 0;
    std::string suffix;
};

// 请求耗时记录
struct request_latency {
    double prefix_ms = 0.0;
    double suffix_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    int generated_tokens = 0;   // 生成了多少 token
};

// GPU 显存追踪
struct gpu_tracker {
    double start_mb = 0.0;
    double peak_mb = 0.0;
    double end_mb = 0.0;
};

// 汇总实验表
struct experiment_result {
    std::string mode;               // 实验模式名，比如 "baseline_no_prefix_cache" 或 "prefix_cache_reuse"
    int use_prefix_reuse = 0;       // 是否 prefix 复用\

    int n_requests = 0;             // 请求总数
    int n_lora = 0;                 // 使用了多少个 LoRA
    int n_ctx = 0;                  // 上下文长度设置
    int n_prefix_tokens = 0;        // 公共 prefix 的 token 数
    int n_suffix_avg_tokens = 0;    // suffix 的平均 token 数
    int n_predict = 0;              // 每条请求预测生成多少 token
    int n_generated_tokens = 0;     // 这轮实验一共生成了多少 token

    double elapsed_ms = 0.0;        // 整轮实验耗时，单位毫秒
    double throughput_tps = 0.0;    // 吞吐率，单位 tokens/s

    double avg_prefix_ms = 0.0;     // 平均 prefix 耗时
    double avg_suffix_ms = 0.0;     // 平均 suffix 耗时
    double avg_ttft_ms = 0.0;       // 平均 TTFT
    double avg_decode_ms = 0.0;     // 平均 decode 耗时
    double avg_total_ms = 0.0;      // 平均总耗时

    double p95_ttft_ms = 0.0;       // TTFT 的 95 分位数
    double p95_total_ms = 0.0;      // 总耗时的 95 分位数

    double gpu_start_mb = 0.0;      // 实验开始时的 GPU 显存占用
    double gpu_peak_mb = 0.0;       // 实验过程中的 GPU 显存峰值
    double gpu_end_mb = 0.0;        // 实验结束时的 GPU 显存占用
    double gpu_peak_delta_mb = 0.0; // GPU 峰值相对起点增加量

    double process_start_mb = 0.0;  // 实验开始时的进程私有内存
    double process_end_mb = 0.0;    // 实验结束时的进程私有内存
    double process_delta_mb = 0.0;  // 进程私有内存变化量
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-kv/output";

// ==========  工具类  ==========
// 毫秒时间戳
static double now_ms() {
    return ggml_time_us() / 1000.0;
}

// Windows API读取当前进程的私有内存，单位 MB
static double get_process_private_mb() {
    PROCESS_MEMORY_COUNTERS_EX pmc;

    if (!GetProcessMemoryInfo(
                GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc),
                sizeof(pmc))) {
        return 0.0;
    }

    return (double) pmc.PrivateUsage / 1024.0 / 1024.0;
}

// Nvidia-smi 读当前 GPU 显存占用
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

// 维护 gpu_tracker 结构体
static void gpu_sample(gpu_tracker & gpu) {
    const double current = get_gpu_used_mb();

    if (gpu.start_mb == 0.0) {
        gpu.start_mb = current;
        gpu.peak_mb = current;
    }

    gpu.peak_mb = std::max(gpu.peak_mb, current);
    gpu.end_mb = current;
}

// 计算平均值
static double average(const std::vector<double> & values) {
    if (values.empty()) {
        return 0.0;
    }

    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

// 算分位数 P95等
static double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const double pos = q * (values.size() - 1);
    const int lo = (int) pos;
    const int hi = std::min(lo + 1, (int) values.size() - 1);
    const double w = pos - lo;

    return values[lo] * (1.0 - w) + values[hi] * w;
}

// ==========  分词和 batch 构造  ==========
// 分词
static bool tokenize_prompt(
        const llama_vocab * vocab,
        const std::string & prompt,
        std::vector<llama_token> & tokens) {
    const int n = -llama_tokenize( // 第一次调用，传入 NULL 获取 prompt 分词后的 token 数量
            vocab,
            prompt.c_str(),
            (int) prompt.size(),
            nullptr,
            0,
            true,
            true);

    if (n <= 0) {
        return false;
    }

    tokens.resize(n);

    const int ret = llama_tokenize( // 第二次调用，真正把 prompt 转换成 token 序列
            vocab,
            prompt.c_str(),
            (int) prompt.size(),
            tokens.data(),
            (int) tokens.size(),
            true,
            true);

    return ret >= 0;
}

// batch 构造
// 往 llama_batch 里追加一个 token
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

// 把一个 LoRA adapter 绑定到当前 context
static bool bind_lora(
        llama_context * ctx,
        llama_adapter_lora * adapter,
        float scale) {
    llama_adapter_lora * adapters[] = { adapter };
    float scales[] = { scale };

    return llama_set_adapters_lora(ctx, adapters, 1, scales) == 0;
}

// 把 context 上的 LoRA 清空
static void clear_lora(llama_context * ctx) {
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
}


// 评估 prefix / suffix 的通用函数
// 作用：把一段 token 序列分批喂给 llama.cpp，让模型真正执行前向计算
static bool eval_tokens(
        llama_context * ctx,                         // 当前推理上下文
        const std::vector<llama_token> & tokens,     // 要喂给模型的一串 token
        llama_seq_id seq_id,                         // 这串 token 属于哪个 sequence
        int start_pos,                               // 这串 token 在序列里的起始位置
        bool logits_last) {                          // 最后一个 token 是否需要 logits
    if (tokens.empty()) {                            // 如果没有 token，直接认为成功
        return true;
    }

    const int max_chunk = 64;                        // 每次最多处理 64 个 token
    int offset = 0;                                  // 当前处理到 tokens 的哪个位置

    while (offset < (int) tokens.size()) {           // 只要还没处理完，就继续分块
        const int chunk = std::min(max_chunk, (int) tokens.size() - offset);
                                                     // 本次实际处理的 token 数
                                                     // 末尾不足 64 个时就处理剩下的

        llama_batch batch = llama_batch_init(chunk, 0, 1);
                                                     // 创建一个 batch
                                                     // chunk = 这批 token 的数量
                                                     // 0 = 不预留 logits 额外空间
                                                     // 1 = 最大 sequence 数

        for (int i = 0; i < chunk; i++) {            // 把当前 chunk 的每个 token 填进 batch
            const int token_index = offset + i;      // 这个 token 在原始 tokens 中的全局下标

            const bool is_last =
                    logits_last &&
                    token_index == (int) tokens.size() - 1;
                                                     // 只有在以下两种条件都满足时才要 logits：
                                                     // 1. 调用者要求最后一个 token 输出 logits
                                                     // 2. 当前 token 正好是整段 tokens 的最后一个

            batch_add(
                    batch,                           // 往 batch 里追加一个 token
                    tokens[token_index],             // 当前 token 的 id
                    start_pos + token_index,         // 这个 token 的位置编号
                    seq_id,                          // 所属 sequence id
                    is_last);                        // 是否需要 logits
        }

        const int ret = llama_decode(ctx, batch);    // 真正让 llama.cpp 执行这批 token
        llama_batch_free(batch);                     // batch 用完就释放

        if (ret != 0) {                              // decode 失败
            fprintf(stderr,
                    "eval_tokens failed: seq=%d start_pos=%d offset=%d chunk=%d\n",
                    seq_id,
                    start_pos,
                    offset,
                    chunk);
            return false;
        }

        offset += chunk;                             // 继续处理下一批 token
    }

    return true;                                     // 所有 token 都成功喂完了
}

// 只 decode 一个 token
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

// ==========  汇总结果  ==========
// 一整组请求的统计结果汇总成 experiment_result
static experiment_result summarize(
        const std::string & mode,
        bool use_prefix_reuse,
        int n_requests,
        int n_lora,
        int n_ctx,
        int n_prefix_tokens,
        int n_suffix_avg_tokens,
        int n_predict,
        const std::vector<request_latency> & latencies,
        double elapsed_ms,
        const gpu_tracker & gpu,
        double process_start_mb,
        double process_end_mb) {
    experiment_result r;

    r.mode = mode;
    r.use_prefix_reuse = use_prefix_reuse ? 1 : 0;
    r.n_requests = n_requests;
    r.n_lora = n_lora;
    r.n_ctx = n_ctx;
    r.n_prefix_tokens = n_prefix_tokens;
    r.n_suffix_avg_tokens = n_suffix_avg_tokens;
    r.n_predict = n_predict;

    std::vector<double> prefix_values;
    std::vector<double> suffix_values;
    std::vector<double> ttft_values;
    std::vector<double> decode_values;
    std::vector<double> total_values;

    for (const auto & item : latencies) {
        prefix_values.push_back(item.prefix_ms);
        suffix_values.push_back(item.suffix_ms);
        ttft_values.push_back(item.ttft_ms);
        decode_values.push_back(item.decode_ms);
        total_values.push_back(item.total_ms);
        r.n_generated_tokens += item.generated_tokens;
    }

    r.elapsed_ms = elapsed_ms;
    r.throughput_tps = elapsed_ms > 0.0 ?
            r.n_generated_tokens / (elapsed_ms / 1000.0) :
            0.0;

    r.avg_prefix_ms = average(prefix_values);
    r.avg_suffix_ms = average(suffix_values);
    r.avg_ttft_ms = average(ttft_values);
    r.avg_decode_ms = average(decode_values);
    r.avg_total_ms = average(total_values);

    r.p95_ttft_ms = percentile(ttft_values, 0.95);
    r.p95_total_ms = percentile(total_values, 0.95);

    r.gpu_start_mb = gpu.start_mb;
    r.gpu_peak_mb = gpu.peak_mb;
    r.gpu_end_mb = gpu.end_mb;
    r.gpu_peak_delta_mb = gpu.peak_mb - gpu.start_mb;

    r.process_start_mb = process_start_mb;
    r.process_end_mb = process_end_mb;
    r.process_delta_mb = process_end_mb - process_start_mb;

    return r;
}

// 写 CSV
static void save_results(const std::vector<experiment_result> & results) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/prefix_kv_reuse_nopage_summary.csv";
    std::ofstream fout(path);

    fout << "mode,use_prefix_reuse,n_requests,n_lora,n_ctx,"
         << "n_prefix_tokens,n_suffix_avg_tokens,n_predict,n_generated_tokens,"
         << "elapsed_ms,throughput_tps,"
         << "avg_prefix_ms,avg_suffix_ms,avg_ttft_ms,avg_decode_ms,avg_total_ms,"
         << "p95_ttft_ms,p95_total_ms,"
         << "gpu_start_mb,gpu_peak_mb,gpu_end_mb,gpu_peak_delta_mb,"
         << "process_start_mb,process_end_mb,process_delta_mb\n";

    for (const auto & r : results) {
        fout << r.mode << ","
             << r.use_prefix_reuse << ","
             << r.n_requests << ","
             << r.n_lora << ","
             << r.n_ctx << ","
             << r.n_prefix_tokens << ","
             << r.n_suffix_avg_tokens << ","
             << r.n_predict << ","
             << r.n_generated_tokens << ","
             << r.elapsed_ms << ","
             << r.throughput_tps << ","
             << r.avg_prefix_ms << ","
             << r.avg_suffix_ms << ","
             << r.avg_ttft_ms << ","
             << r.avg_decode_ms << ","
             << r.avg_total_ms << ","
             << r.p95_ttft_ms << ","
             << r.p95_total_ms << ","
             << r.gpu_start_mb << ","
             << r.gpu_peak_mb << ","
             << r.gpu_end_mb << ","
             << r.gpu_peak_delta_mb << ","
             << r.process_start_mb << ","
             << r.process_end_mb << ","
             << r.process_delta_mb << "\n";
    }

    fprintf(stderr, "saved prefix KV reuse results to %s\n", path.c_str());
}


// ==========  baseline：不复用 prefix  ==========
static experiment_result run_baseline_no_prefix_cache(
        llama_model * model,
        const llama_vocab * vocab,
        const std::string & common_prefix,
        const std::vector<request_item> & requests,
        const std::vector<llama_adapter_lora *> & adapters,
        int n_ctx,
        int n_predict,
        float lora_scale) {
    // 请求条数
    const int n_seq = (int) requests.size();

    // 把公共 prefix 转成 token
    std::vector<llama_token> prefix_tokens;
    if (!tokenize_prompt(vocab, common_prefix, prefix_tokens)) {
        fprintf(stderr, "failed to tokenize common prefix\n");
    }

    // 每条请求自己的 suffix 也分别转成 token
    std::vector<std::vector<llama_token>> suffix_tokens(n_seq);
    int suffix_total = 0;

    for (int i = 0; i < n_seq; i++) {
        if (!tokenize_prompt(vocab, requests[i].suffix, suffix_tokens[i])) {
            fprintf(stderr, "failed to tokenize suffix %d\n", i);
        }

        suffix_total += (int) suffix_tokens[i].size();
    }

    // 构造 context 参数
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = 256;
    ctx_params.n_ubatch = 64;
    ctx_params.n_seq_max = (uint32_t) n_seq;
    ctx_params.no_perf = true;

    // 关键：设置成统一 KV
    // 局部 prefix seq_cp 需要同一个 KV stream。
    // baseline 虽然不用 seq_cp，但保持同样设置方便公平对比。
    ctx_params.kv_unified = true;

    // 记录 GPU 显存的起点
    gpu_tracker gpu;
    gpu_sample(gpu);

    // 记录进程内存起点
    const double process_start_mb = get_process_private_mb();
    // 创建 context
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    gpu_sample(gpu);

    if (ctx == nullptr) {
        fprintf(stderr, "failed to create baseline context\n");
        return experiment_result();
    }

    // 存每条请求的耗时
    std::vector<request_latency> latencies;
    // 整体开始计时
    const double t_all0 = now_ms();
    // 逐条请求执行
    for (int i = 0; i < n_seq; i++) {
        request_latency item;   // 当前请求的统计结果

        const llama_seq_id seq_id = i;  // 这条请求对应的 sequence id
        const double t0 = now_ms();     // 当前请求开始时间
        
        clear_lora(ctx);            // baseline 不复用 prefix，所以每条请求都先清掉 LoRA，确保 prefix 阶段只算 base model

        // ---------- 1) prefix 阶段 ----------
        const double tp0 = now_ms();
        eval_tokens(
                ctx,
                prefix_tokens,  // 公共 prefix 每条请求都重新算一遍
                seq_id,
                0,              // 从位置 0 开始
                false);         // prefix 阶段最后一个 token 不一定需要 logits
        const double tp1 = now_ms();

        item.prefix_ms = tp1 - tp0;     // 记录 prefix 耗时

        // ---------- 2) suffix 阶段 ----------
        // prefix 算完后，再绑定这条请求自己的 LoRA
        if (!bind_lora(ctx, adapters[requests[i].lora_id], lora_scale)) {
            fprintf(stderr, "failed to bind LoRA for request %d\n", i);
        }

        // suffix 从 prefix 后面继续接着算
        const int suffix_pos = (int) prefix_tokens.size();

        const double ts0 = now_ms();
        eval_tokens(
                ctx,
                suffix_tokens[i],   // 当前请求自己的 suffix
                seq_id,
                suffix_pos,         // suffix 起始位置接在 prefix 后面
                true);              // suffix 最后一个 token 需要 logits
        const double ts1 = now_ms();

        item.suffix_ms = ts1 - ts0; // 记录 suffix 耗时

        // ---------- 3) decode 阶段 ----------
        // 这里用 suffix 的最后一个 token 作为后续 decode 的重复输入
        // 如果 suffix 为空，就用 prefix 的最后一个 token
        const llama_token repeated = suffix_tokens[i].empty() ?
                prefix_tokens.back() :
                suffix_tokens[i].back();

        const double td0 = now_ms();

        for (int t = 0; t < n_predict; t++) {
            const int pos = suffix_pos + (int) suffix_tokens[i].size() + t;

            // 每次只 decode 一个 token
            if (!decode_one(ctx, repeated, seq_id, pos)) {
                fprintf(stderr, "decode failed for request %d token %d\n", i, t);
            }
            // 第一个生成 token 出来的时间，就是 TTFT
            if (t == 0) {
                item.ttft_ms = now_ms() - t0;
            }

            // 每步都采一次 GPU 显存，方便看峰值
            gpu_sample(gpu);
        }

        const double td1 = now_ms();

        item.decode_ms = td1 - td0;     // 后续生成阶段总耗时
        item.total_ms = td1 - t0;       // 当前请求从开始到结束的总耗时
        item.generated_tokens = n_predict;

        latencies.push_back(item);
        // 一条请求结束后再采一次显存
        gpu_sample(gpu);
    }

    // 整体结束时间
    const double t_all1 = now_ms();
    // 清掉 LoRA，释放绑定状态
    clear_lora(ctx);

    // 释放 context
    llama_free(ctx);
    gpu_sample(gpu);    // 记录释放 context 后的显存

    // 记录进程内存结束值
    const double process_end_mb = get_process_private_mb();

    // 汇总整组实验结果
    return summarize(
            "baseline_no_prefix_cache",
            false,
            n_seq,
            (int) adapters.size(),
            n_ctx,
            (int) prefix_tokens.size(),
            suffix_total / std::max(1, n_seq),
            n_predict,
            latencies,
            t_all1 - t_all0,
            gpu,
            process_start_mb,
            process_end_mb);
}

// prefix 复用
// 这个版本的核心思路是：公共 prefix 只计算一次，
// 然后把这段 prefix 的 KV 复制给每条请求，避免重复计算。
static experiment_result run_prefix_cache_reuse(
        llama_model * model,                          // 已加载好的模型
        const llama_vocab * vocab,                    // 模型词表，用来 tokenization
        const std::string & common_prefix,            // 所有请求共享的公共前缀
        const std::vector<request_item> & requests,    // 请求列表，每条请求有自己的 LoRA 和 suffix
        const std::vector<llama_adapter_lora *> & adapters, // 已预加载好的 LoRA adapter
        int n_ctx,                                    // 上下文长度
        int n_predict,                                // 每条请求生成多少 token
        float lora_scale) {                           // LoRA 缩放系数

    const int n_seq = (int) requests.size();          // 请求总数
    const llama_seq_id prefix_seq = 0;                // 0 号 sequence 专门放公共 prefix

    // 把公共 prefix 转成 token
    std::vector<llama_token> prefix_tokens;
    if (!tokenize_prompt(vocab, common_prefix, prefix_tokens)) {
        fprintf(stderr, "failed to tokenize common prefix\n");
    }

    // 把每条请求自己的 suffix 也转成 token
    std::vector<std::vector<llama_token>> suffix_tokens(n_seq);
    int suffix_total = 0;                             // 统计所有 suffix token 总数

    for (int i = 0; i < n_seq; i++) {
        if (!tokenize_prompt(vocab, requests[i].suffix, suffix_tokens[i])) {
            fprintf(stderr, "failed to tokenize suffix %d\n", i);
        }

        suffix_total += (int) suffix_tokens[i].size();
    }

    // 创建 context 参数
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;                         // 最大上下文长度
    ctx_params.n_batch = 256;                         // batch 大小
    ctx_params.n_ubatch = 64;                         // micro-batch 大小
    ctx_params.n_seq_max = (uint32_t) (n_seq + 1);    // 多一个 seq 给公共 prefix 用
    ctx_params.no_perf = true;                        // 不输出 llama.cpp 内部性能统计

    // 关键：必须统一 KV stream，否则 llama_memory_seq_cp 会因为局部复制 prefix 而断言失败
    ctx_params.kv_unified = true;

    // 记录 GPU 和进程内存起点
    gpu_tracker gpu;
    gpu_sample(gpu);

    const double process_start_mb = get_process_private_mb();

    // 创建 llama context
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    gpu_sample(gpu);                                  // 记录创建 context 后的显存

    if (ctx == nullptr) {                             // 如果 context 创建失败，直接返回空结果
        fprintf(stderr, "failed to create prefix reuse context\n");
        return experiment_result();
    }

    // 保存每条请求的耗时
    std::vector<request_latency> latencies;

    // 整轮实验开始计时
    const double t_all0 = now_ms();

    // 先清掉 LoRA，保证公共 prefix 阶段只跑 base model
    clear_lora(ctx);

    // ---------- 1) 公共 prefix 只算一次 ----------
    const double shared_prefix_t0 = now_ms();
    eval_tokens(
            ctx,
            prefix_tokens,     // 只算一次的公共 prefix
            prefix_seq,        // 用 0 号 sequence 存公共 prefix
            0,                 // 从位置 0 开始
            false);            // prefix 阶段这里不需要 logits
    const double shared_prefix_t1 = now_ms();

    gpu_sample(gpu);                                // prefix 算完后采样一次显存

    // 拿到 context 的 memory 对象，后面要做 seq 级别复制
    llama_memory_t mem = llama_get_memory(ctx);

    // ---------- 2) 逐条请求复用 prefix ----------
    for (int i = 0; i < n_seq; i++) {
        request_latency item;                       // 当前请求的耗时统计

        const llama_seq_id seq_id = i + 1;          // 当前请求使用 1..n_seq 的 seq id
        const double t0 = now_ms();                 // 当前请求开始时间

        // ---------- 2.1 复制公共 prefix KV 到当前 seq ----------
        const double cp0 = now_ms();

        llama_memory_seq_cp(
                mem,                                // 当前 context 的 memory
                prefix_seq,                         // 源 seq：公共 prefix
                seq_id,                             // 目标 seq：当前请求
                0,                                  // 复制起点
                (llama_pos) prefix_tokens.size());  // 复制整个 prefix 区间

        const double cp1 = now_ms();

        // 把“公共 prefix 只算一次”的成本均摊到每条请求上
        item.prefix_ms =
                (shared_prefix_t1 - shared_prefix_t0) / std::max(1, n_seq)
                + (cp1 - cp0);

        // ---------- 2.2 绑定当前请求自己的 LoRA ----------
        if (!bind_lora(ctx, adapters[requests[i].lora_id], lora_scale)) {
            fprintf(stderr, "failed to bind LoRA for request %d\n", i);
        }

        // suffix 接在 prefix 后面
        const int suffix_pos = (int) prefix_tokens.size();

        // ---------- 2.3 计算 suffix ----------
        const double ts0 = now_ms();
        eval_tokens(
                ctx,
                suffix_tokens[i],                    // 当前请求自己的 suffix
                seq_id,                              // 仍然用当前请求自己的 seq
                suffix_pos,                          // 从 prefix 后面开始接
                true);                               // suffix 最后一个 token 要 logits
        const double ts1 = now_ms();

        item.suffix_ms = ts1 - ts0;

        // 如果 suffix 为空，就用 prefix 最后一个 token；否则用 suffix 最后一个 token
        const llama_token repeated = suffix_tokens[i].empty() ?
                prefix_tokens.back() :
                suffix_tokens[i].back();

        // ---------- 2.4 decode 阶段 ----------
        const double td0 = now_ms();

        for (int t = 0; t < n_predict; t++) {
            const int pos = suffix_pos + (int) suffix_tokens[i].size() + t;

            // 每次 decode 一个 token
            if (!decode_one(ctx, repeated, seq_id, pos)) {
                fprintf(stderr, "decode failed for request %d token %d\n", i, t);
            }

            // 第一个 token 出来时，记录 TTFT
            if (t == 0) {
                item.ttft_ms = now_ms() - t0;
            }

            // 每一步都采样一次 GPU 显存
            gpu_sample(gpu);
        }

        const double td1 = now_ms();

        item.decode_ms = td1 - td0;                 // decode 阶段总耗时
        item.total_ms = td1 - t0;                   // 当前请求总耗时
        item.generated_tokens = n_predict;          // 当前请求生成 token 数

        latencies.push_back(item);                  // 保存当前请求结果

        gpu_sample(gpu);                            // 请求结束后再采一次显存
    }

    const double t_all1 = now_ms();                 // 整轮实验结束时间

    // 释放 LoRA 和 context
    clear_lora(ctx);
    llama_free(ctx);
    gpu_sample(gpu);                                // 记录释放后的显存

    const double process_end_mb = get_process_private_mb();

    // ---------- 3) 汇总整轮实验 ----------
    experiment_result result = summarize(
            "prefix_cache_reuse",                   // 模式名
            true,                                   // 表示启用了 prefix reuse
            n_seq,                                  // 请求总数
            (int) adapters.size(),                  // LoRA 数
            n_ctx,                                  // 上下文长度
            (int) prefix_tokens.size(),             // prefix token 数
            suffix_total / std::max(1, n_seq),      // 平均 suffix token 数
            n_predict,                              // 每条请求生成 token 数
            latencies,                              // 每条请求的详细耗时
            t_all1 - t_all0,                        // 整轮实验总耗时
            gpu,                                    // GPU 显存采样结果
            process_start_mb,                       // 进程内存起点
            process_end_mb);                        // 进程内存终点

    // 这里重新记录 prefix 成本，强调 prefix 只算一次，再平均摊到每条请求上
    result.avg_prefix_ms =
            (shared_prefix_t1 - shared_prefix_t0) / std::max(1, n_seq);

    return result;
}

int main() {
    std::setlocale(LC_NUMERIC, "C");

    const std::string model_path =
            "D:/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf";

    std::vector<std::string> lora_paths = {
        "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
        "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
        "D:/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
        "D:/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
    };

    const std::string common_prefix =
            "You are a helpful assistant. You are given the following common background. "
            "The user wants short and clear answers. The response should be practical, "
            "well structured, and adapted to the task domain. Common context: "
            "A product team is preparing online customer-facing content for several scenarios. "
            "The assistant should keep the tone concise, useful, and customer friendly. ";

    std::vector<request_item> requests = {
        {0, "Please write a short marketing slogan for a new coffee product."},
        {1, "Please answer like a concise assistant and summarize the customer need."},
        {2, "Please write a short creative response for an animal themed campaign."},
        {3, "Please produce a compact product description for a wolf themed brand."},
        {0, "Please rewrite the message as a social media advertisement."},
        {1, "Please give a short reply suitable for customer service."},
        {2, "Please create a playful one sentence campaign idea."},
        {3, "Please generate a brief brand positioning statement."},
    };

    const int ngl = 99;
    const int n_ctx = 4096;
    const int n_predict = 32;
    const float lora_scale = 1.0f;

    // 加载 llama.cpp 支持的后端，例如 CUDA、CPU 等
    ggml_backend_load_all();

    // 初始化模型加载参数
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    // 加载基座模型
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    // 获取模型词表，用于分词和 token 转文本
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ==============  加载多 LoRA 适配器  ==============
    std::vector<llama_adapter_lora *> adapters;

    for (const auto & path : lora_paths) {
        llama_adapter_lora * adapter = llama_adapter_lora_init(
                model,
                path.c_str());

        if (adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", path.c_str());

            for (auto * old : adapters) {
                llama_adapter_lora_free(old);
            }

            llama_model_free(model);
            return 1;
        }

        adapters.push_back(adapter);
    }


    // ==============  跑实验结果  ==============
    std::vector<experiment_result> results;

    results.push_back(run_baseline_no_prefix_cache(
            model,
            vocab,
            common_prefix,
            requests,
            adapters,
            n_ctx,
            n_predict,
            lora_scale));

    results.push_back(run_prefix_cache_reuse(
            model,
            vocab,
            common_prefix,
            requests,
            adapters,
            n_ctx,
            n_predict,
            lora_scale));

    save_results(results);

    // 释放
    for (auto * adapter : adapters) {
        llama_adapter_lora_free(adapter);
    }

    llama_model_free(model);

    fprintf(stderr, "prefix KV reuse nopage experiment finished.\n");

    return 0;
}