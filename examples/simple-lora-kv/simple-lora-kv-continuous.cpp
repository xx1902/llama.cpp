// simple-lora-kv.cpp
//
// 多 LoRA 公共 prefix KV 复用实验，连续 KV 版本
//
// 这个版本不启用分页 KV，只测试 prefix KV 复用本身的效果。
//
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

struct request_item {
    int lora_id = 0;
    std::string suffix;
};

struct request_latency {
    double prefix_ms = 0.0;
    double suffix_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    int generated_tokens = 0;
};

struct gpu_tracker {
    double start_mb = 0.0;
    double peak_mb = 0.0;
    double end_mb = 0.0;
};

struct experiment_result {
    std::string mode;
    int use_prefix_reuse = 0;

    int n_requests = 0;
    int n_lora = 0;
    int n_ctx = 0;
    int n_prefix_tokens = 0;
    int n_suffix_avg_tokens = 0;
    int n_predict = 0;
    int n_generated_tokens = 0;

    double elapsed_ms = 0.0;
    double throughput_tps = 0.0;

    double avg_prefix_ms = 0.0;
    double avg_suffix_ms = 0.0;
    double avg_ttft_ms = 0.0;
    double avg_decode_ms = 0.0;
    double avg_total_ms = 0.0;

    double p95_ttft_ms = 0.0;
    double p95_total_ms = 0.0;

    double gpu_start_mb = 0.0;
    double gpu_peak_mb = 0.0;
    double gpu_end_mb = 0.0;
    double gpu_peak_delta_mb = 0.0;

    double process_start_mb = 0.0;
    double process_end_mb = 0.0;
    double process_delta_mb = 0.0;
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-kv/output";

static double now_ms() {
    return ggml_time_us() / 1000.0;
}

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

static void gpu_sample(gpu_tracker & gpu) {
    const double current = get_gpu_used_mb();

    if (gpu.start_mb == 0.0) {
        gpu.start_mb = current;
        gpu.peak_mb = current;
    }

    gpu.peak_mb = std::max(gpu.peak_mb, current);
    gpu.end_mb = current;
}

static double average(const std::vector<double> & values) {
    if (values.empty()) {
        return 0.0;
    }

    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

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

static bool tokenize_prompt(
        const llama_vocab * vocab,
        const std::string & prompt,
        std::vector<llama_token> & tokens) {
    const int n = -llama_tokenize(
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

    const int ret = llama_tokenize(
            vocab,
            prompt.c_str(),
            (int) prompt.size(),
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

static bool bind_lora(
        llama_context * ctx,
        llama_adapter_lora * adapter,
        float scale) {
    llama_adapter_lora * adapters[] = { adapter };
    float scales[] = { scale };

    return llama_set_adapters_lora(ctx, adapters, 1, scales) == 0;
}

static void clear_lora(llama_context * ctx) {
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
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
        const int chunk = std::min(max_chunk, (int) tokens.size() - offset);

        llama_batch batch = llama_batch_init(chunk, 0, 1);

        for (int i = 0; i < chunk; i++) {
            const int token_index = offset + i;

            const bool is_last =
                    logits_last &&
                    token_index == (int) tokens.size() - 1;

            batch_add(
                    batch,
                    tokens[token_index],
                    start_pos + token_index,
                    seq_id,
                    is_last);
        }

        const int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            fprintf(stderr,
                    "eval_tokens failed: seq=%d start_pos=%d offset=%d chunk=%d\n",
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

static experiment_result run_baseline_no_prefix_cache(
        llama_model * model,
        const llama_vocab * vocab,
        const std::string & common_prefix,
        const std::vector<request_item> & requests,
        const std::vector<llama_adapter_lora *> & adapters,
        int n_ctx,
        int n_predict,
        float lora_scale) {
    const int n_seq = (int) requests.size();

    std::vector<llama_token> prefix_tokens;
    if (!tokenize_prompt(vocab, common_prefix, prefix_tokens)) {
        fprintf(stderr, "failed to tokenize common prefix\n");
    }

    std::vector<std::vector<llama_token>> suffix_tokens(n_seq);
    int suffix_total = 0;

    for (int i = 0; i < n_seq; i++) {
        if (!tokenize_prompt(vocab, requests[i].suffix, suffix_tokens[i])) {
            fprintf(stderr, "failed to tokenize suffix %d\n", i);
        }

        suffix_total += (int) suffix_tokens[i].size();
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = 256;
    ctx_params.n_ubatch = 64;
    ctx_params.n_seq_max = (uint32_t) n_seq;
    ctx_params.no_perf = true;

    // 关键：局部 prefix seq_cp 需要同一个 KV stream。
    // baseline 虽然不用 seq_cp，但保持同样设置方便公平对比。
    ctx_params.kv_unified = true;

    gpu_tracker gpu;
    gpu_sample(gpu);

    const double process_start_mb = get_process_private_mb();

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    gpu_sample(gpu);

    if (ctx == nullptr) {
        fprintf(stderr, "failed to create baseline context\n");
        return experiment_result();
    }

    std::vector<request_latency> latencies;

    const double t_all0 = now_ms();

    for (int i = 0; i < n_seq; i++) {
        request_latency item;

        const llama_seq_id seq_id = i;
        const double t0 = now_ms();

        clear_lora(ctx);

        const double tp0 = now_ms();
        eval_tokens(
                ctx,
                prefix_tokens,
                seq_id,
                0,
                false);
        const double tp1 = now_ms();

        item.prefix_ms = tp1 - tp0;

        if (!bind_lora(ctx, adapters[requests[i].lora_id], lora_scale)) {
            fprintf(stderr, "failed to bind LoRA for request %d\n", i);
        }

        const int suffix_pos = (int) prefix_tokens.size();

        const double ts0 = now_ms();
        eval_tokens(
                ctx,
                suffix_tokens[i],
                seq_id,
                suffix_pos,
                true);
        const double ts1 = now_ms();

        item.suffix_ms = ts1 - ts0;

        const llama_token repeated = suffix_tokens[i].empty() ?
                prefix_tokens.back() :
                suffix_tokens[i].back();

        const double td0 = now_ms();

        for (int t = 0; t < n_predict; t++) {
            const int pos = suffix_pos + (int) suffix_tokens[i].size() + t;

            if (!decode_one(ctx, repeated, seq_id, pos)) {
                fprintf(stderr, "decode failed for request %d token %d\n", i, t);
            }

            if (t == 0) {
                item.ttft_ms = now_ms() - t0;
            }

            gpu_sample(gpu);
        }

        const double td1 = now_ms();

        item.decode_ms = td1 - td0;
        item.total_ms = td1 - t0;
        item.generated_tokens = n_predict;

        latencies.push_back(item);

        gpu_sample(gpu);
    }

    const double t_all1 = now_ms();

    clear_lora(ctx);

    llama_free(ctx);
    gpu_sample(gpu);

    const double process_end_mb = get_process_private_mb();

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

static experiment_result run_prefix_cache_reuse(
        llama_model * model,
        const llama_vocab * vocab,
        const std::string & common_prefix,
        const std::vector<request_item> & requests,
        const std::vector<llama_adapter_lora *> & adapters,
        int n_ctx,
        int n_predict,
        float lora_scale) {
    const int n_seq = (int) requests.size();
    const llama_seq_id prefix_seq = 0;

    std::vector<llama_token> prefix_tokens;
    if (!tokenize_prompt(vocab, common_prefix, prefix_tokens)) {
        fprintf(stderr, "failed to tokenize common prefix\n");
    }

    std::vector<std::vector<llama_token>> suffix_tokens(n_seq);
    int suffix_total = 0;

    for (int i = 0; i < n_seq; i++) {
        if (!tokenize_prompt(vocab, requests[i].suffix, suffix_tokens[i])) {
            fprintf(stderr, "failed to tokenize suffix %d\n", i);
        }

        suffix_total += (int) suffix_tokens[i].size();
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = 256;
    ctx_params.n_ubatch = 64;
    ctx_params.n_seq_max = (uint32_t) (n_seq + 1);
    ctx_params.no_perf = true;

    // 关键：必须统一 KV stream，否则 llama_memory_seq_cp 局部复制 prefix 会断言。
    ctx_params.kv_unified = true;

    gpu_tracker gpu;
    gpu_sample(gpu);

    const double process_start_mb = get_process_private_mb();

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    gpu_sample(gpu);

    if (ctx == nullptr) {
        fprintf(stderr, "failed to create prefix reuse context\n");
        return experiment_result();
    }

    std::vector<request_latency> latencies;

    const double t_all0 = now_ms();

    clear_lora(ctx);

    const double shared_prefix_t0 = now_ms();
    eval_tokens(
            ctx,
            prefix_tokens,
            prefix_seq,
            0,
            false);
    const double shared_prefix_t1 = now_ms();

    gpu_sample(gpu);

    llama_memory_t mem = llama_get_memory(ctx);

    for (int i = 0; i < n_seq; i++) {
        request_latency item;

        const llama_seq_id seq_id = i + 1;
        const double t0 = now_ms();

        const double cp0 = now_ms();

        llama_memory_seq_cp(
                mem,
                prefix_seq,
                seq_id,
                0,
                (llama_pos) prefix_tokens.size());

        const double cp1 = now_ms();

        // 把公共 prefix 的一次计算成本均摊到每条请求。
        item.prefix_ms =
                (shared_prefix_t1 - shared_prefix_t0) / std::max(1, n_seq)
                + (cp1 - cp0);

        if (!bind_lora(ctx, adapters[requests[i].lora_id], lora_scale)) {
            fprintf(stderr, "failed to bind LoRA for request %d\n", i);
        }

        const int suffix_pos = (int) prefix_tokens.size();

        const double ts0 = now_ms();
        eval_tokens(
                ctx,
                suffix_tokens[i],
                seq_id,
                suffix_pos,
                true);
        const double ts1 = now_ms();

        item.suffix_ms = ts1 - ts0;

        const llama_token repeated = suffix_tokens[i].empty() ?
                prefix_tokens.back() :
                suffix_tokens[i].back();

        const double td0 = now_ms();

        for (int t = 0; t < n_predict; t++) {
            const int pos = suffix_pos + (int) suffix_tokens[i].size() + t;

            if (!decode_one(ctx, repeated, seq_id, pos)) {
                fprintf(stderr, "decode failed for request %d token %d\n", i, t);
            }

            if (t == 0) {
                item.ttft_ms = now_ms() - t0;
            }

            gpu_sample(gpu);
        }

        const double td1 = now_ms();

        item.decode_ms = td1 - td0;
        item.total_ms = td1 - t0;
        item.generated_tokens = n_predict;

        latencies.push_back(item);

        gpu_sample(gpu);
    }

    const double t_all1 = now_ms();

    clear_lora(ctx);

    llama_free(ctx);
    gpu_sample(gpu);

    const double process_end_mb = get_process_private_mb();

    experiment_result result = summarize(
            "prefix_cache_reuse",
            true,
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

    // 这里重新记录平均 prefix 成本，强调 prefix 只算一次后均摊。
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

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(
            model_path.c_str(),
            model_params);

    if (model == nullptr) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

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

    for (auto * adapter : adapters) {
        llama_adapter_lora_free(adapter);
    }

    llama_model_free(model);

    fprintf(stderr, "prefix KV reuse nopage experiment finished.\n");

    return 0;
}