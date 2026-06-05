// simple-system-ablation.cpp
//
// PreCoLoRA 消融实验
//
// 本程序只测试端到端吞吐量，不做负载扫描。
// 所有吞吐量来自真实 llama.cpp API 计时：
// - llama_adapter_lora_init
// - llama_init_from_model
// - llama_set_adapters_lora
// - llama_decode
// - llama_free
//
// 消融配置：
// 1. native_llama.cpp：无缓存、无批处理、无分组。
// 2. w/o cache：不复用 LoRA，但按 LoRA 分组批处理。
// 3. w/o batch：复用 LoRA，但请求逐个顺序执行。
// 4. w/o grouping：复用 LoRA，但不按 LoRA 重新排序，只利用到达序列中的连续同 LoRA 请求。
// 5. PreCoLoRA：复用 LoRA，并按 LoRA 分组批处理。
//
// 输出：
// D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output/system_ablation_summary.csv

#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

struct request_item {
    int lora_id = 0;
    std::string prompt;
};

struct request_latency {
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    int generated_tokens = 0;
};

struct ablation_result {
    std::string config;
    int n_requests = 0;
    int n_tokens = 0;
    double elapsed_ms = 0.0;
    double throughput_tps = 0.0;
    double speedup_vs_native = 1.0;
    double speedup_vs_full = 1.0;
    double avg_ttft_ms = 0.0;
    double p95_ttft_ms = 0.0;
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output";

static double now_ms() {
    return ggml_time_us() / 1000.0;
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

static request_latency run_one_request(
        llama_model * model,
        const llama_vocab * vocab,
        const request_item & req,
        const std::vector<std::string> & lora_paths,
        const std::vector<llama_adapter_lora *> & cached_adapters,
        bool use_cache,
        int n_predict,
        float lora_scale) {
    request_latency latency;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 64;
    ctx_params.n_seq_max = 1;
    ctx_params.no_perf = true;

    const double t0 = now_ms();

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create context\n");
        return latency;
    }

    llama_adapter_lora * adapter = nullptr;

    if (use_cache) {
        adapter = cached_adapters[req.lora_id];
    } else {
        adapter = llama_adapter_lora_init(model, lora_paths[req.lora_id].c_str());

        if (adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA\n");
            llama_free(ctx);
            return latency;
        }
    }

    if (!bind_lora(ctx, adapter, lora_scale)) {
        fprintf(stderr, "failed to bind LoRA\n");
    }

    std::vector<llama_token> prompt_tokens;

    if (!tokenize_prompt(vocab, req.prompt, prompt_tokens)) {
        fprintf(stderr, "failed to tokenize prompt\n");

        if (!use_cache) {
            llama_adapter_lora_free(adapter);
        }

        llama_free(ctx);
        return latency;
    }

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
    }

    const double t1 = now_ms();

    latency.total_ms = t1 - t0;
    latency.generated_tokens = n_predict;

    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);

    if (!use_cache) {
        llama_adapter_lora_free(adapter);
    }

    llama_free(ctx);

    return latency;
}

static request_latency run_batch_group(
        llama_model * model,
        const llama_vocab * vocab,
        const std::vector<request_item> & group,
        int lora_id,
        const std::vector<std::string> & lora_paths,
        const std::vector<llama_adapter_lora *> & cached_adapters,
        bool use_cache,
        int n_predict,
        float lora_scale) {
    request_latency latency;

    std::vector<std::vector<llama_token>> prompt_tokens(group.size());
    int n_prompt_total = 0;

    for (int i = 0; i < (int) group.size(); i++) {
        tokenize_prompt(vocab, group[i].prompt, prompt_tokens[i]);
        n_prompt_total += (int) prompt_tokens[i].size();
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = std::max(64, n_prompt_total);
    ctx_params.n_seq_max = (uint32_t) group.size();
    ctx_params.no_perf = true;

    const double t0 = now_ms();

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create batch context\n");
        return latency;
    }

    llama_adapter_lora * adapter = nullptr;

    if (use_cache) {
        adapter = cached_adapters[lora_id];
    } else {
        adapter = llama_adapter_lora_init(model, lora_paths[lora_id].c_str());

        if (adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA for batch\n");
            llama_free(ctx);
            return latency;
        }
    }

    bind_lora(ctx, adapter, lora_scale);

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
    }

    const double t1 = now_ms();

    latency.total_ms = t1 - t0;
    latency.generated_tokens = n_predict * (int) group.size();

    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);

    if (!use_cache) {
        llama_adapter_lora_free(adapter);
    }

    llama_free(ctx);

    return latency;
}

static std::vector<request_latency> run_sequential(
        llama_model * model,
        const llama_vocab * vocab,
        const std::vector<request_item> & requests,
        const std::vector<std::string> & lora_paths,
        const std::vector<llama_adapter_lora *> & cached_adapters,
        bool use_cache,
        int n_predict,
        float lora_scale) {
    std::vector<request_latency> latencies;

    for (const auto & req : requests) {
        latencies.push_back(run_one_request(
                model,
                vocab,
                req,
                lora_paths,
                cached_adapters,
                use_cache,
                n_predict,
                lora_scale));
    }

    return latencies;
}

static std::vector<request_latency> run_batched(
        llama_model * model,
        const llama_vocab * vocab,
        std::vector<request_item> requests,
        const std::vector<std::string> & lora_paths,
        const std::vector<llama_adapter_lora *> & cached_adapters,
        bool use_cache,
        bool sort_by_lora,
        int batch_width,
        int n_predict,
        float lora_scale) {
    if (sort_by_lora) {
        std::stable_sort(
                requests.begin(),
                requests.end(),
                [](const request_item & a, const request_item & b) {
                    return a.lora_id < b.lora_id;
                });
    }

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

        request_latency group_latency = run_batch_group(
                model,
                vocab,
                group,
                lora_id,
                lora_paths,
                cached_adapters,
                use_cache,
                n_predict,
                lora_scale);

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

static ablation_result summarize(
        const std::string & name,
        const std::vector<request_latency> & latencies,
        double elapsed_ms) {
    ablation_result result;

    result.config = name;
    result.n_requests = (int) latencies.size();
    result.elapsed_ms = elapsed_ms;

    std::vector<double> ttft_values;

    for (const auto & item : latencies) {
        result.n_tokens += item.generated_tokens;
        ttft_values.push_back(item.ttft_ms);
    }

    result.throughput_tps = result.n_tokens / (elapsed_ms / 1000.0);

    if (!ttft_values.empty()) {
        result.avg_ttft_ms =
                std::accumulate(ttft_values.begin(), ttft_values.end(), 0.0) /
                (double) ttft_values.size();
        result.p95_ttft_ms = percentile(ttft_values, 0.95);
    }

    return result;
}

static void save_summary(const std::vector<ablation_result> & results) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/system_ablation_summary.csv";
    std::ofstream fout(path);

    fout << "config,n_requests,n_tokens,elapsed_ms,throughput_tps,"
         << "speedup_vs_native,speedup_vs_full,avg_ttft_ms,p95_ttft_ms\n";

    for (const auto & r : results) {
        fout << r.config << ","
             << r.n_requests << ","
             << r.n_tokens << ","
             << r.elapsed_ms << ","
             << r.throughput_tps << ","
             << r.speedup_vs_native << ","
             << r.speedup_vs_full << ","
             << r.avg_ttft_ms << ","
             << r.p95_ttft_ms << "\n";
    }

    fprintf(stderr, "saved ablation summary to %s\n", path.c_str());
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
    const int n_predict = 16;
    const int batch_width = 4;
    const float lora_scale = 1.0f;

    std::vector<request_item> requests;

    for (int i = 0; i < n_requests; i++) {
        request_item item;
        item.lora_id = i % (int) lora_paths.size();
        item.prompt = prompt_pool[i % (int) prompt_pool.size()];
        requests.push_back(item);
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == nullptr) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_adapter_lora *> cached_adapters;

    for (const auto & path : lora_paths) {
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, path.c_str());

        if (adapter == nullptr) {
            fprintf(stderr, "failed to load cached LoRA: %s\n", path.c_str());
            llama_model_free(model);
            return 1;
        }

        cached_adapters.push_back(adapter);
    }

    std::vector<ablation_result> results;

    {
        const double t0 = now_ms();

        auto latencies = run_sequential(
                model,
                vocab,
                requests,
                lora_paths,
                cached_adapters,
                false,
                n_predict,
                lora_scale);

        const double t1 = now_ms();

        results.push_back(summarize("native_llama.cpp", latencies, t1 - t0));
    }

    {
        const double t0 = now_ms();

        auto latencies = run_batched(
                model,
                vocab,
                requests,
                lora_paths,
                cached_adapters,
                false,
                true,
                batch_width,
                n_predict,
                lora_scale);

        const double t1 = now_ms();

        results.push_back(summarize("w/o cache", latencies, t1 - t0));
    }

    {
        const double t0 = now_ms();

        auto latencies = run_sequential(
                model,
                vocab,
                requests,
                lora_paths,
                cached_adapters,
                true,
                n_predict,
                lora_scale);

        const double t1 = now_ms();

        results.push_back(summarize("w/o batch", latencies, t1 - t0));
    }

    {
        const double t0 = now_ms();

        auto latencies = run_batched(
                model,
                vocab,
                requests,
                lora_paths,
                cached_adapters,
                true,
                false,
                batch_width,
                n_predict,
                lora_scale);

        const double t1 = now_ms();

        results.push_back(summarize("w/o grouping", latencies, t1 - t0));
    }

    {
        const double t0 = now_ms();

        auto latencies = run_batched(
                model,
                vocab,
                requests,
                lora_paths,
                cached_adapters,
                true,
                true,
                batch_width,
                n_predict,
                lora_scale);

        const double t1 = now_ms();

        results.push_back(summarize("PreCoLoRA", latencies, t1 - t0));
    }

    const double native_tps = results.front().throughput_tps;
    const double full_tps = results.back().throughput_tps;

    for (auto & r : results) {
        r.speedup_vs_native = r.throughput_tps / native_tps;
        r.speedup_vs_full = r.throughput_tps / full_tps;
    }

    save_summary(results);

    for (auto * adapter : cached_adapters) {
        llama_adapter_lora_free(adapter);
    }

    llama_model_free(model);

    fprintf(stderr, "ablation experiment finished.\n");

    return 0;
}