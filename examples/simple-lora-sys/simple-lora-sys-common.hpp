#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "llama.h"

#include <windows.h>
#include <psapi.h>

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
    std::string prompt;
};

struct request_latency {
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    int generated_tokens = 0;
};

struct gpu_tracker {
    double start_mb = 0.0;
    double peak_mb = 0.0;
    double end_mb = 0.0;
};

struct experiment_result {
    std::string config;
    int use_paged_kv = 0;
    int use_multilora_batch = 0;

    int n_requests = 0;
    int n_lora = 0;
    int n_predict = 0;
    int n_tokens = 0;

    int n_ctx = 0;
    int kv_page_size = 0;

    double elapsed_ms = 0.0;
    double throughput_tps = 0.0;

    double avg_ttft_ms = 0.0;
    double p50_ttft_ms = 0.0;
    double p95_ttft_ms = 0.0;
    double p99_ttft_ms = 0.0;

    double avg_request_total_ms = 0.0;
    double p95_request_total_ms = 0.0;

    double gpu_start_mb = 0.0;
    double gpu_peak_mb = 0.0;
    double gpu_end_mb = 0.0;
    double gpu_peak_delta_mb = 0.0;

    double process_start_mb = 0.0;
    double process_end_mb = 0.0;
    double process_delta_mb = 0.0;
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-sys/output";

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

static void gpu_sample(gpu_tracker & tracker) {
    const double current = get_gpu_used_mb();

    if (tracker.start_mb == 0.0) {
        tracker.start_mb = current;
        tracker.peak_mb = current;
    }

    tracker.peak_mb = std::max(tracker.peak_mb, current);
    tracker.end_mb = current;
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

static double average(const std::vector<double> & values) {
    if (values.empty()) {
        return 0.0;
    }

    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
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

static void apply_kv_params(
        llama_context_params & ctx_params,
        bool use_paged_kv,
        int kv_page_size) {
    if (use_paged_kv) {
        ctx_params.experimental_physical_paged_kv = true;
        ctx_params.physical_kv_page_size = kv_page_size;

        // 当前实验性 physical paged KV 的 get_k/get_v 读取路径只支持单 stream。
        // 多 LoRA batch 会设置 n_seq_max > 1，如果 kv_unified=false，会产生多个 KV stream，
        // 从而触发 GGML_ASSERT(sinfo.n_stream() == 1)。
        //
        // 因此分页 KV 实验路径下强制启用 unified KV，让多个 seq 共用一个 stream。
        // 这是当前实验代码的兼容方案；真正完整的多 stream paged KV 需要继续改 get_k/get_v。
        ctx_params.kv_unified = true;
    } else {
        ctx_params.experimental_physical_paged_kv = false;
        ctx_params.physical_kv_page_size = 0;
    }
}

static request_latency run_one_request(
        llama_model * model,
        const llama_vocab * vocab,
        const request_item & req,
        const std::vector<llama_adapter_lora *> & adapters,
        bool use_paged_kv,
        int kv_page_size,
        int n_ctx,
        int n_predict,
        float lora_scale,
        gpu_tracker & gpu) {
    request_latency latency;

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_prompt(vocab, req.prompt, prompt_tokens)) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return latency;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = std::max(64, (int) prompt_tokens.size());
    ctx_params.n_seq_max = 1;
    ctx_params.no_perf = true;
    apply_kv_params(ctx_params, use_paged_kv, kv_page_size);

    const double t0 = now_ms();

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    gpu_sample(gpu);

    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        return latency;
    }

    if (!bind_lora(ctx, adapters[req.lora_id], lora_scale)) {
        fprintf(stderr, "failed to bind LoRA\n");
    }

    gpu_sample(gpu);

    llama_batch prompt_batch = llama_batch_init((int) prompt_tokens.size(), 0, 1);

    for (int i = 0; i < (int) prompt_tokens.size(); i++) {
        batch_add(
                prompt_batch,
                prompt_tokens[i],
                i,
                0,
                i == (int) prompt_tokens.size() - 1);
    }

    if (llama_decode(ctx, prompt_batch) != 0) {
        fprintf(stderr, "prompt eval failed\n");
    }

    llama_batch_free(prompt_batch);
    gpu_sample(gpu);

    const llama_token repeated_token = prompt_tokens.back();

    for (int t = 0; t < n_predict; t++) {
        llama_batch decode_batch = llama_batch_init(1, 0, 1);

        batch_add(
                decode_batch,
                repeated_token,
                (int) prompt_tokens.size() + t,
                0,
                true);

        if (llama_decode(ctx, decode_batch) != 0) {
            fprintf(stderr, "decode failed\n");
        }

        llama_batch_free(decode_batch);

        if (t == 0) {
            latency.ttft_ms = now_ms() - t0;
        }

        gpu_sample(gpu);
    }

    const double t1 = now_ms();

    latency.total_ms = t1 - t0;
    latency.generated_tokens = n_predict;

    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
    llama_free(ctx);
    gpu_sample(gpu);

    return latency;
}

static request_latency run_lora_group_batch(
        llama_model * model,
        const llama_vocab * vocab,
        const std::vector<request_item> & group,
        int lora_id,
        const std::vector<llama_adapter_lora *> & adapters,
        bool use_paged_kv,
        int kv_page_size,
        int n_ctx,
        int n_predict,
        float lora_scale,
        gpu_tracker & gpu) {
    request_latency latency;

    std::vector<std::vector<llama_token>> prompt_tokens(group.size());
    int n_prompt_total = 0;

    for (int i = 0; i < (int) group.size(); i++) {
        if (!tokenize_prompt(vocab, group[i].prompt, prompt_tokens[i])) {
            fprintf(stderr, "failed to tokenize prompt in batch\n");
            return latency;
        }

        n_prompt_total += (int) prompt_tokens[i].size();
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = std::max(64, n_prompt_total);
    ctx_params.n_seq_max = (uint32_t) group.size();
    ctx_params.no_perf = true;
    apply_kv_params(ctx_params, use_paged_kv, kv_page_size);

    const double t0 = now_ms();

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    gpu_sample(gpu);

    if (ctx == nullptr) {
        fprintf(stderr, "failed to create batch context\n");
        return latency;
    }

    if (!bind_lora(ctx, adapters[lora_id], lora_scale)) {
        fprintf(stderr, "failed to bind LoRA in batch\n");
    }

    gpu_sample(gpu);

    llama_batch prompt_batch = llama_batch_init(n_prompt_total, 0, (int) group.size());

    for (int s = 0; s < (int) group.size(); s++) {
        for (int p = 0; p < (int) prompt_tokens[s].size(); p++) {
            batch_add(
                    prompt_batch,
                    prompt_tokens[s][p],
                    p,
                    s,
                    p == (int) prompt_tokens[s].size() - 1);
        }
    }

    if (llama_decode(ctx, prompt_batch) != 0) {
        fprintf(stderr, "batch prompt eval failed\n");
    }

    llama_batch_free(prompt_batch);
    gpu_sample(gpu);

    for (int t = 0; t < n_predict; t++) {
        llama_batch decode_batch = llama_batch_init((int) group.size(), 0, (int) group.size());

        for (int s = 0; s < (int) group.size(); s++) {
            batch_add(
                    decode_batch,
                    prompt_tokens[s].back(),
                    (int) prompt_tokens[s].size() + t,
                    s,
                    true);
        }

        if (llama_decode(ctx, decode_batch) != 0) {
            fprintf(stderr, "batch decode failed\n");
        }

        llama_batch_free(decode_batch);

        if (t == 0) {
            latency.ttft_ms = now_ms() - t0;
        }

        gpu_sample(gpu);
    }

    const double t1 = now_ms();

    latency.total_ms = t1 - t0;
    latency.generated_tokens = n_predict * (int) group.size();

    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
    llama_free(ctx);
    gpu_sample(gpu);

    return latency;
}

static std::vector<request_latency> run_sequential_requests(
        llama_model * model,
        const llama_vocab * vocab,
        const std::vector<request_item> & requests,
        const std::vector<llama_adapter_lora *> & adapters,
        bool use_paged_kv,
        int kv_page_size,
        int n_ctx,
        int n_predict,
        float lora_scale,
        gpu_tracker & gpu) {
    std::vector<request_latency> latencies;

    for (const auto & req : requests) {
        latencies.push_back(run_one_request(
                model,
                vocab,
                req,
                adapters,
                use_paged_kv,
                kv_page_size,
                n_ctx,
                n_predict,
                lora_scale,
                gpu));
    }

    return latencies;
}

static std::vector<request_latency> run_multilora_grouped_requests(
        llama_model * model,
        const llama_vocab * vocab,
        std::vector<request_item> requests,
        const std::vector<llama_adapter_lora *> & adapters,
        bool use_paged_kv,
        int kv_page_size,
        int n_ctx,
        int n_predict,
        int batch_width,
        float lora_scale,
        gpu_tracker & gpu) {
    std::stable_sort(
            requests.begin(),
            requests.end(),
            [](const request_item & a, const request_item & b) {
                return a.lora_id < b.lora_id;
            });

    std::vector<request_latency> latencies;

    int i = 0;

    while (i < (int) requests.size()) {
        const int lora_id = requests[i].lora_id;

        std::vector<request_item> group;

        while (i < (int) requests.size()
                && requests[i].lora_id == lora_id
                && (int) group.size() < batch_width) {
            group.push_back(requests[i]);
            i++;
        }

        const request_latency group_latency = run_lora_group_batch(
                model,
                vocab,
                group,
                lora_id,
                adapters,
                use_paged_kv,
                kv_page_size,
                n_ctx,
                n_predict,
                lora_scale,
                gpu);

        for (int j = 0; j < (int) group.size(); j++) {
            request_latency one;
            one.ttft_ms = group_latency.ttft_ms;
            one.total_ms = group_latency.total_ms;
            one.generated_tokens = n_predict;
            latencies.push_back(one);
        }
    }

    return latencies;
}

static experiment_result summarize_result(
        const std::string & config,
        bool use_paged_kv,
        bool use_multilora_batch,
        int n_lora,
        int n_ctx,
        int kv_page_size,
        int n_predict,
        const std::vector<request_latency> & latencies,
        double elapsed_ms,
        const gpu_tracker & gpu,
        double process_start_mb,
        double process_end_mb) {
    experiment_result result;

    result.config = config;
    result.use_paged_kv = use_paged_kv ? 1 : 0;
    result.use_multilora_batch = use_multilora_batch ? 1 : 0;
    result.n_lora = n_lora;
    result.n_ctx = n_ctx;
    result.kv_page_size = use_paged_kv ? kv_page_size : 0;
    result.n_predict = n_predict;
    result.n_requests = (int) latencies.size();
    result.elapsed_ms = elapsed_ms;

    std::vector<double> ttft_values;
    std::vector<double> total_values;

    for (const auto & item : latencies) {
        result.n_tokens += item.generated_tokens;
        ttft_values.push_back(item.ttft_ms);
        total_values.push_back(item.total_ms);
    }

    result.throughput_tps = result.n_tokens / (elapsed_ms / 1000.0);

    result.avg_ttft_ms = average(ttft_values);
    result.p50_ttft_ms = percentile(ttft_values, 0.50);
    result.p95_ttft_ms = percentile(ttft_values, 0.95);
    result.p99_ttft_ms = percentile(ttft_values, 0.99);

    result.avg_request_total_ms = average(total_values);
    result.p95_request_total_ms = percentile(total_values, 0.95);

    result.gpu_start_mb = gpu.start_mb;
    result.gpu_peak_mb = gpu.peak_mb;
    result.gpu_end_mb = gpu.end_mb;
    result.gpu_peak_delta_mb = gpu.peak_mb - gpu.start_mb;

    result.process_start_mb = process_start_mb;
    result.process_end_mb = process_end_mb;
    result.process_delta_mb = process_end_mb - process_start_mb;

    return result;
}

static void save_result_csv(
        const experiment_result & r,
        const std::string & file_name) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/" + file_name;

    std::ofstream fout(path);

    fout << "config,use_paged_kv,use_multilora_batch,n_requests,n_lora,n_ctx,kv_page_size,"
         << "n_predict,n_tokens,elapsed_ms,throughput_tps,"
         << "avg_ttft_ms,p50_ttft_ms,p95_ttft_ms,p99_ttft_ms,"
         << "avg_request_total_ms,p95_request_total_ms,"
         << "gpu_start_mb,gpu_peak_mb,gpu_end_mb,gpu_peak_delta_mb,"
         << "process_start_mb,process_end_mb,process_delta_mb\n";

    fout << r.config << ","
         << r.use_paged_kv << ","
         << r.use_multilora_batch << ","
         << r.n_requests << ","
         << r.n_lora << ","
         << r.n_ctx << ","
         << r.kv_page_size << ","
         << r.n_predict << ","
         << r.n_tokens << ","
         << r.elapsed_ms << ","
         << r.throughput_tps << ","
         << r.avg_ttft_ms << ","
         << r.p50_ttft_ms << ","
         << r.p95_ttft_ms << ","
         << r.p99_ttft_ms << ","
         << r.avg_request_total_ms << ","
         << r.p95_request_total_ms << ","
         << r.gpu_start_mb << ","
         << r.gpu_peak_mb << ","
         << r.gpu_end_mb << ","
         << r.gpu_peak_delta_mb << ","
         << r.process_start_mb << ","
         << r.process_end_mb << ","
         << r.process_delta_mb << "\n";

    fprintf(stderr, "saved result to %s\n", path.c_str());
}

static int run_sys_experiment(
        const std::string & config_name,
        const std::string & output_csv,
        bool use_paged_kv,
        bool use_multilora_batch) {
    std::setlocale(LC_NUMERIC, "C");

    const std::string model_path =
            "D:/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf";

    std::vector<std::string> lora_paths = {
        "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
        "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
        "D:/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
        "D:/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
    };

    std::vector<std::string> prompt_pool = {
        "Hello my name is",
        "The future of marketing is",
        "A good product slogan is",
        "To attract customers, we should",
        "A successful advertising campaign should",
        "The best way to build a brand is",
        "When launching a new product, marketers should",
        "Customer loyalty can be improved by",
    };

    const int n_requests = 16;
    const int n_ctx = 512;
    const int kv_page_size = 256;
    const int n_predict = 32;
    const int batch_width = 4;
    const int ngl = 99;
    const float lora_scale = 1.0f;

    std::vector<request_item> requests;

    for (int i = 0; i < n_requests; i++) {
        request_item item;
        item.lora_id = i % (int) lora_paths.size();
        item.prompt = prompt_pool[i % (int) prompt_pool.size()];
        requests.push_back(item);
    }

    ggml_backend_load_all();

    gpu_tracker gpu;
    gpu_sample(gpu);

    const double process_start_mb = get_process_private_mb();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(
            model_path.c_str(),
            model_params);

    gpu_sample(gpu);

    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_adapter_lora *> adapters;

    for (const auto & path : lora_paths) {
        llama_adapter_lora * adapter = llama_adapter_lora_init(
                model,
                path.c_str());

        gpu_sample(gpu);

        if (adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", path.c_str());
            llama_model_free(model);
            return 1;
        }

        adapters.push_back(adapter);
    }

    const double t0 = now_ms();

    std::vector<request_latency> latencies;

    if (use_multilora_batch) {
        latencies = run_multilora_grouped_requests(
                model,
                vocab,
                requests,
                adapters,
                use_paged_kv,
                kv_page_size,
                n_ctx,
                n_predict,
                batch_width,
                lora_scale,
                gpu);
    } else {
        latencies = run_sequential_requests(
                model,
                vocab,
                requests,
                adapters,
                use_paged_kv,
                kv_page_size,
                n_ctx,
                n_predict,
                lora_scale,
                gpu);
    }

    const double t1 = now_ms();

    for (auto * adapter : adapters) {
        llama_adapter_lora_free(adapter);
        gpu_sample(gpu);
    }

    llama_model_free(model);
    gpu_sample(gpu);

    const double process_end_mb = get_process_private_mb();

    const experiment_result result = summarize_result(
            config_name,
            use_paged_kv,
            use_multilora_batch,
            (int) lora_paths.size(),
            n_ctx,
            kv_page_size,
            n_predict,
            latencies,
            t1 - t0,
            gpu,
            process_start_mb,
            process_end_mb);

    save_result_csv(result, output_csv);

    fprintf(stderr,
            "[%s] elapsed=%.2f ms, tps=%.2f, p95_ttft=%.2f ms, gpu_peak_delta=%.2f MB\n",
            result.config.c_str(),
            result.elapsed_ms,
            result.throughput_tps,
            result.p95_ttft_ms,
            result.gpu_peak_delta_mb);

    return 0;
}