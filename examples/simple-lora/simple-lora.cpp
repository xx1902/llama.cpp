// simple-lora.cpp
//
// 图 5-9 延迟分布实验代码
//
// 本程序用于测量多 LoRA 推理场景下三种执行方式的端到端延迟分布：
// 1. sequential：4 个请求同时到达，但系统按顺序逐个处理。
//    因此第 2、3、4 个请求的 TTFT 和总延迟需要包含前面请求的等待时间。
// 2. batch_unfused：4 个请求组成 batch，但 LoRA 交错排列：0,1,0,1。
// 3. batch_grouped：4 个请求组成 batch，并按 LoRA 分组排列：0,0,1,1。
//
// 主要统计指标：
// - TTFT：Time To First Token，从请求到达系统到生成第一个 token 的时间。
// - TGI：Token Generation Interval，同一请求相邻生成 token 之间的平均间隔。
// - Total Latency：从请求到达系统到该请求生成结束的总时间。
// - P50 / P95 / P99：延迟分位值。
//
// 输出文件：
// D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output/latency_samples.csv
// D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output/latency_summary.csv

#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

enum class experiment_mode {
    sequential,
    batch_unfused,
    batch_grouped,
};

struct request_latency {
    std::string scheme;
    int round = 0;
    int request_id = 0;
    int adapter_id = 0;
    int n_tokens = 0;

    // TTFT：从请求共同到达到生成第一个 token 的时间。
    double ttft_ms = 0.0;

    // 用户感知 TGI：同一个请求内部，相邻 token 之间的平均间隔。
    // 这个指标反映用户看到流式输出时是否平滑。
    double user_tgi_ms = 0.0;

    // 系统级 TGI：系统整体每生成一个 token 的平均耗时。
    // 批处理每一步会同时为多个请求生成 token，因此需要用总耗时 / 总 token 数来衡量系统效率。
    double system_tgi_ms = 0.0;

    // 总延迟：从请求共同到达到该请求生成结束的时间。
    double total_latency_ms = 0.0;
};

struct latency_summary {
    std::string scheme;
    int n_samples = 0;
    double ttft_p50 = 0.0;
    double ttft_p95 = 0.0;
    double ttft_p99 = 0.0;

    double user_tgi_p50 = 0.0;
    double user_tgi_p95 = 0.0;
    double user_tgi_p99 = 0.0;

    double system_tgi_p50 = 0.0;
    double system_tgi_p95 = 0.0;
    double system_tgi_p99 = 0.0;
    
    double total_p50 = 0.0;
    double total_p95 = 0.0;
    double total_p99 = 0.0;
};

static const char * mode_name(experiment_mode mode) {
    switch (mode) {
        case experiment_mode::sequential:
            return "sequential";
        case experiment_mode::batch_unfused:
            return "batch_unfused";
        case experiment_mode::batch_grouped:
            return "batch_grouped";
    }

    return "unknown";
}

static double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const double rank = (p / 100.0) * (values.size() - 1);
    const size_t lo = (size_t) rank;
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = rank - lo;

    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

// 4 个请求、2 个 LoRA。
// batch_unfused: 0,1,0,1
// batch_grouped: 0,0,1,1
// static std::vector<int> build_seq_to_lora(experiment_mode mode) {
//     if (mode == experiment_mode::batch_grouped) {
//         return { 0, 0, 1, 1 };
//     }

//     return { 0, 1, 0, 1 };
// }

// 8 个请求、4 个 LoRA。
// batch_unfused: 0,1,2,3,0,1,2,3
// batch_grouped: 0,0,1,1,2,2,3,3
//
// 这样设计的目的：
// - batch_unfused 表示 LoRA 请求交错排列，底层会产生更多 LoRA group。
// - batch_grouped 表示相同 LoRA 的请求连续排列，便于按 adapter 分组执行。
static std::vector<int> build_seq_to_lora(experiment_mode mode) {
    if (mode == experiment_mode::batch_grouped) {
        return { 0, 0, 1, 1, 2, 2, 3, 3 };
    }

    return { 0, 1, 2, 3, 0, 1, 2, 3 };
}

static void batch_add(
        llama_batch & batch,
        llama_token token,
        llama_pos pos,
        const std::vector<llama_seq_id> & seq_ids,
        bool logits) {
    const int i = batch.n_tokens;

    batch.token[i] = token;
    batch.pos[i] = pos;
    batch.n_seq_id[i] = (int32_t) seq_ids.size();

    for (int j = 0; j < (int) seq_ids.size(); j++) {
        batch.seq_id[i][j] = seq_ids[j];
    }

    batch.logits[i] = logits ? 1 : 0;
    batch.n_tokens++;
}

static bool tokenize_prompts(
        const llama_vocab * vocab,
        const std::vector<std::string> & prompts,
        std::vector<std::vector<llama_token>> & prompt_tokens,
        int & n_prompt_total) {
    prompt_tokens.resize(prompts.size());
    n_prompt_total = 0;

    for (int i = 0; i < (int) prompts.size(); i++) {
        const int n = -llama_tokenize(
                vocab,
                prompts[i].c_str(),
                prompts[i].size(),
                nullptr,
                0,
                true,
                true);

        prompt_tokens[i].resize(n);

        if (llama_tokenize(
                    vocab,
                    prompts[i].c_str(),
                    prompts[i].size(),
                    prompt_tokens[i].data(),
                    prompt_tokens[i].size(),
                    true,
                    true) < 0) {
            fprintf(stderr, "failed to tokenize prompt %d\n", i);
            return false;
        }

        n_prompt_total += n;
    }

    return true;
}

// 批处理延迟实验。
// 4 个请求同时进入 batch，因此每个请求的 TTFT 和 total latency 都从 t_arrival 开始算。
static std::vector<request_latency> run_batch_latency_experiment(
        llama_model * model,
        const llama_vocab * vocab,
        const std::vector<std::string> & prompts,
        const std::vector<std::vector<llama_token>> & prompt_tokens,
        const std::vector<llama_adapter_lora *> & lora_adapters,
        experiment_mode mode,
        int n_prompt_total,
        int n_predict,
        float lora_scale,
        int round_id) {
    std::vector<request_latency> samples;

    const int batch_size = (int) prompts.size();
    std::vector<int> seq_to_lora = build_seq_to_lora(mode);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 256;
    ctx_params.n_seq_max = batch_size;
    ctx_params.n_batch = n_prompt_total;
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        return samples;
    }

    std::vector<llama_seq_id> seq_lora_ids;
    std::vector<llama_adapter_lora *> seq_lora_adapters;
    std::vector<float> seq_lora_scales;

    for (int s = 0; s < batch_size; s++) {
        const int adapter_id = seq_to_lora[s];

        seq_lora_ids.push_back((llama_seq_id) s);
        seq_lora_adapters.push_back(lora_adapters[adapter_id]);
        seq_lora_scales.push_back(lora_scale);
    }

    if (llama_set_seq_adapters_lora(
                ctx,
                seq_lora_ids.data(),
                seq_lora_adapters.data(),
                seq_lora_scales.data(),
                seq_lora_ids.size()) != 0) {
        llama_free(ctx);
        return samples;
    }

    std::vector<llama_sampler *> samplers(batch_size);

    for (int s = 0; s < batch_size; s++) {
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = true;
        samplers[s] = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(samplers[s], llama_sampler_init_greedy());
    }

    llama_batch batch = llama_batch_init(n_prompt_total, 0, batch_size);
    batch.n_tokens = 0;

    std::vector<std::vector<llama_seq_id>> seq_ids(batch_size);
    for (int s = 0; s < batch_size; s++) {
        seq_ids[s] = { s };
    }

    std::vector<int> last_logits_idx(batch_size, -1);

    for (int s = 0; s < batch_size; s++) {
        for (int j = 0; j < (int) prompt_tokens[s].size(); j++) {
            const bool need_logits = j == (int) prompt_tokens[s].size() - 1;

            if (need_logits) {
                last_logits_idx[s] = batch.n_tokens;
            }

            batch_add(batch, prompt_tokens[s][j], j, seq_ids[s], need_logits);
        }
    }

    std::vector<int> seq_pos(batch_size);
    std::vector<bool> done(batch_size, false);
    std::vector<int> generated_tokens(batch_size, 0);
    std::vector<int64_t> ttft_us(batch_size, -1);
    std::vector<int64_t> last_token_us(batch_size, -1);
    std::vector<int64_t> tgi_sum_us(batch_size, 0);
    std::vector<int> tgi_count(batch_size, 0);
    std::vector<int64_t> finish_us(batch_size, -1);

    for (int s = 0; s < batch_size; s++) {
        seq_pos[s] = (int) prompt_tokens[s].size();
    }

    const int64_t t_arrival = ggml_time_us();
    const int64_t t_system_start = t_arrival;

    int system_generated_tokens = 0;

    if (llama_decode(ctx, batch)) {
        llama_batch_free(batch);
        for (auto * sampler : samplers) {
            llama_sampler_free(sampler);
        }
        llama_free(ctx);
        return samples;
    }

    for (int step = 0; step < n_predict; step++) {
        llama_batch next_batch = llama_batch_init(batch_size, 0, batch_size);
        next_batch.n_tokens = 0;

        int active_count = 0;

        for (int s = 0; s < batch_size; s++) {
            if (done[s]) {
                continue;
            }

            llama_token new_token_id = llama_sampler_sample(samplers[s], ctx, last_logits_idx[s]);
            const int64_t now_us = ggml_time_us();

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                done[s] = true;
                finish_us[s] = now_us;
                continue;
            }

            if (generated_tokens[s] == 0) {
                ttft_us[s] = now_us - t_arrival;
            } else {
                tgi_sum_us[s] += now_us - last_token_us[s];
                tgi_count[s]++;
            }

            last_token_us[s] = now_us;
            generated_tokens[s]++;
            system_generated_tokens++;

            last_logits_idx[s] = next_batch.n_tokens;
            batch_add(next_batch, new_token_id, seq_pos[s], seq_ids[s], true);

            seq_pos[s]++;
            active_count++;
        }

        if (active_count == 0) {
            llama_batch_free(next_batch);
            break;
        }

        if (llama_decode(ctx, next_batch)) {
            llama_batch_free(next_batch);
            break;
        }

        const int64_t after_decode_us = ggml_time_us();
        for (int s = 0; s < batch_size; s++) {
            if (!done[s] && generated_tokens[s] > 0) {
                finish_us[s] = after_decode_us;
            }
        }

        llama_batch_free(next_batch);
    }

    const int64_t t_system_end = ggml_time_us();

    const double system_tgi_ms =
            system_generated_tokens > 0
            ? (t_system_end - t_system_start) / 1000.0 / system_generated_tokens
            : 0.0;

    for (int s = 0; s < batch_size; s++) {
        request_latency item;
        item.scheme = mode_name(mode);
        item.round = round_id;
        item.request_id = s;
        item.adapter_id = seq_to_lora[s];
        item.n_tokens = generated_tokens[s];
        item.ttft_ms = ttft_us[s] >= 0 ? ttft_us[s] / 1000.0 : 0.0;
        item.user_tgi_ms = tgi_count[s] > 0 ? (tgi_sum_us[s] / 1000.0) / tgi_count[s] : 0.0;
        item.system_tgi_ms = system_tgi_ms;
        item.total_latency_ms = finish_us[s] >= 0 ? (finish_us[s] - t_arrival) / 1000.0 : 0.0;

        samples.push_back(item);
    }

    llama_batch_free(batch);

    for (auto * sampler : samplers) {
        llama_sampler_free(sampler);
    }

    llama_free(ctx);
    return samples;
}

// 顺序延迟实验。
// 关键点：4 个请求视为同时到达，因此 t_arrival 在处理第一个请求之前记录。
// 第 2、3、4 个请求的 TTFT 和 total latency 会自然包含排队等待时间。
static std::vector<request_latency> run_sequential_latency_experiment(
        llama_model * model,
        const llama_vocab * vocab,
        const std::vector<std::string> & prompts,
        const std::vector<std::vector<llama_token>> & prompt_tokens,
        const std::vector<llama_adapter_lora *> & lora_adapters,
        int n_predict,
        float lora_scale,
        int round_id) {
    std::vector<request_latency> samples;

    const int n_requests = (int) prompts.size();
    std::vector<int> seq_to_lora = build_seq_to_lora(experiment_mode::batch_unfused);

    const int64_t t_arrival = ggml_time_us();
    const int64_t t_system_start = t_arrival;

    int system_generated_tokens = 0;

    for (int s = 0; s < n_requests; s++) {
        const int adapter_id = seq_to_lora[s];
        const int n_prompt = (int) prompt_tokens[s].size();

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 256;
        ctx_params.n_seq_max = 1;
        ctx_params.n_batch = n_prompt;
        ctx_params.no_perf = true;

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            continue;
        }

        std::vector<llama_adapter_lora *> adapters = { lora_adapters[adapter_id] };
        std::vector<float> scales = { lora_scale };

        if (llama_set_adapters_lora(ctx, adapters.data(), adapters.size(), scales.data()) != 0) {
            llama_free(ctx);
            continue;
        }

        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = true;

        llama_sampler * sampler = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        llama_batch batch = llama_batch_get_one(
                const_cast<llama_token *>(prompt_tokens[s].data()),
                prompt_tokens[s].size());

        int generated = 0;
        int64_t ttft_us = -1;
        int64_t last_token_us = -1;
        int64_t tgi_sum_us = 0;
        int tgi_count = 0;
        int64_t finish_us = -1;

        llama_token new_token_id;

        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
            if (llama_decode(ctx, batch)) {
                break;
            }

            n_pos += batch.n_tokens;

            new_token_id = llama_sampler_sample(sampler, ctx, -1);
            const int64_t now_us = ggml_time_us();

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                finish_us = now_us;
                break;
            }

            if (generated == 0) {
                ttft_us = now_us - t_arrival;
            } else {
                tgi_sum_us += now_us - last_token_us;
                tgi_count++;
            }

            last_token_us = now_us;
            generated++;
            system_generated_tokens++;

            batch = llama_batch_get_one(&new_token_id, 1);
            finish_us = now_us;
        }

        request_latency item;
        item.scheme = mode_name(experiment_mode::sequential);
        item.round = round_id;
        item.request_id = s;
        item.adapter_id = adapter_id;
        item.n_tokens = generated;
        item.ttft_ms = ttft_us >= 0 ? ttft_us / 1000.0 : 0.0;
        item.user_tgi_ms = tgi_count > 0 ? (tgi_sum_us / 1000.0) / tgi_count : 0.0;
        item.system_tgi_ms = 0.0; // 循环结束后统一回填
        item.total_latency_ms = finish_us >= 0 ? (finish_us - t_arrival) / 1000.0 : 0.0;

        samples.push_back(item);

        llama_sampler_free(sampler);
        llama_free(ctx);
    }

    const int64_t t_system_end = ggml_time_us();

    const double system_tgi_ms =
            system_generated_tokens > 0
            ? (t_system_end - t_system_start) / 1000.0 / system_generated_tokens
            : 0.0;

    for (auto & item : samples) {
        item.system_tgi_ms = system_tgi_ms;
    }

    return samples;
}

static latency_summary build_summary(
        const std::vector<request_latency> & samples,
        const std::string & scheme) {
    std::vector<double> ttft_values;
    std::vector<double> user_tgi_values;
    std::vector<double> system_tgi_values;
    std::vector<double> total_values;

    for (const auto & s : samples) {
        if (s.scheme == scheme) {
            if (s.ttft_ms > 0.0) {
                ttft_values.push_back(s.ttft_ms);
            }

            if (s.user_tgi_ms > 0.0) {
                user_tgi_values.push_back(s.user_tgi_ms);
            }

            if (s.system_tgi_ms > 0.0) {
                system_tgi_values.push_back(s.system_tgi_ms);
            }

            if (s.total_latency_ms > 0.0) {
                total_values.push_back(s.total_latency_ms);
            }
        }
    }

    latency_summary summary;
    summary.scheme = scheme;
    summary.n_samples = (int) ttft_values.size();
    summary.ttft_p50 = percentile(ttft_values, 50);
    summary.ttft_p95 = percentile(ttft_values, 95);
    summary.ttft_p99 = percentile(ttft_values, 99);

    summary.user_tgi_p50 = percentile(user_tgi_values, 50);
    summary.user_tgi_p95 = percentile(user_tgi_values, 95);
    summary.user_tgi_p99 = percentile(user_tgi_values, 99);

    summary.system_tgi_p50 = percentile(system_tgi_values, 50);
    summary.system_tgi_p95 = percentile(system_tgi_values, 95);
    summary.system_tgi_p99 = percentile(system_tgi_values, 99);

    summary.total_p50 = percentile(total_values, 50);
    summary.total_p95 = percentile(total_values, 95);
    summary.total_p99 = percentile(total_values, 99);

    return summary;
}

static void save_latency_samples(const std::vector<request_latency> & samples) {
    const std::string output_dir = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output";
    const std::string csv_path = output_dir + "/latency_samples.csv";

    std::filesystem::create_directories(output_dir);

    std::ofstream fout(csv_path);
    fout << "scheme,round,request_id,adapter_id,n_tokens,ttft_ms,user_tgi_ms,system_tgi_ms,total_latency_ms\n";

    for (const auto & s : samples) {
        fout << s.scheme << ","
             << s.round << ","
             << s.request_id << ","
             << s.adapter_id << ","
             << s.n_tokens << ","
             << s.ttft_ms << ","
            << s.user_tgi_ms << ","
            << s.system_tgi_ms << ","
             << s.total_latency_ms << "\n";
    }

    fout.close();
    fprintf(stderr, "saved latency samples to %s\n", csv_path.c_str());
}

static void save_latency_summary(const std::vector<latency_summary> & summaries) {
    const std::string output_dir = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output";
    const std::string csv_path = output_dir + "/latency_summary.csv";

    std::filesystem::create_directories(output_dir);

    std::ofstream fout(csv_path);
    fout << "scheme,n_samples,"
        << "ttft_p50,ttft_p95,ttft_p99,"
        << "user_tgi_p50,user_tgi_p95,user_tgi_p99,"
        << "system_tgi_p50,system_tgi_p95,system_tgi_p99,"
        << "total_p50,total_p95,total_p99\n";

    for (const auto & s : summaries) {
        fout << s.scheme << ","
            << s.n_samples << ","
            << s.ttft_p50 << ","
            << s.ttft_p95 << ","
            << s.ttft_p99 << ","
            << s.user_tgi_p50 << ","
            << s.user_tgi_p95 << ","
            << s.user_tgi_p99 << ","
            << s.system_tgi_p50 << ","
            << s.system_tgi_p95 << ","
            << s.system_tgi_p99 << ","
            << s.total_p50 << ","
            << s.total_p95 << ","
            << s.total_p99 << "\n";
    }

    fout.close();
    fprintf(stderr, "saved latency summary to %s\n", csv_path.c_str());
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::setlocale(LC_NUMERIC, "C");

    std::string model_path = "D:/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf";

    // std::vector<std::string> lora_paths = {
    //     "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
    //     "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
    // };
    std::vector<std::string> lora_paths = {
        "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
        "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
        "D:/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
        "D:/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
    };

    // std::vector<std::string> prompts = {
    //     "Hello my name is",
    //     "The future of marketing is",
    //     "A good product slogan is",
    //     "To attract customers, we should",
    // };
    std::vector<std::string> prompts = {
        "Hello my name is",
        "The future of marketing is",
        "A good product slogan is",
        "To attract customers, we should",
        "A successful advertising campaign should",
        "The best way to build a brand is",
        "When launching a new product, marketers should",
        "Customer loyalty can be improved by",
    };

    const int ngl = 99;
    const int n_predict = 32;
    const float lora_scale = 1.0f;
    const int n_rounds = 30;

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    std::vector<llama_adapter_lora *> lora_adapters;

    for (const auto & path : lora_paths) {
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, path.c_str());
        if (adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", path.c_str());
            llama_model_free(model);
            return 1;
        }

        lora_adapters.push_back(adapter);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<std::vector<llama_token>> prompt_tokens;
    int n_prompt_total = 0;

    if (!tokenize_prompts(vocab, prompts, prompt_tokens, n_prompt_total)) {
        for (auto * adapter : lora_adapters) {
            llama_adapter_lora_free(adapter);
        }
        llama_model_free(model);
        return 1;
    }

    std::vector<request_latency> all_samples;

    for (int round = 0; round < n_rounds; round++) {
        fprintf(stderr, "\n========== LATENCY ROUND %d / %d ==========\n", round + 1, n_rounds);

        auto seq_samples = run_sequential_latency_experiment(
                model,
                vocab,
                prompts,
                prompt_tokens,
                lora_adapters,
                n_predict,
                lora_scale,
                round);

        all_samples.insert(all_samples.end(), seq_samples.begin(), seq_samples.end());

        auto unfused_samples = run_batch_latency_experiment(
                model,
                vocab,
                prompts,
                prompt_tokens,
                lora_adapters,
                experiment_mode::batch_unfused,
                n_prompt_total,
                n_predict,
                lora_scale,
                round);

        all_samples.insert(all_samples.end(), unfused_samples.begin(), unfused_samples.end());

        auto grouped_samples = run_batch_latency_experiment(
                model,
                vocab,
                prompts,
                prompt_tokens,
                lora_adapters,
                experiment_mode::batch_grouped,
                n_prompt_total,
                n_predict,
                lora_scale,
                round);

        all_samples.insert(all_samples.end(), grouped_samples.begin(), grouped_samples.end());
    }

    save_latency_samples(all_samples);

    std::vector<latency_summary> summaries;
    summaries.push_back(build_summary(all_samples, "sequential"));
    summaries.push_back(build_summary(all_samples, "batch_unfused"));
    summaries.push_back(build_summary(all_samples, "batch_grouped"));

    save_latency_summary(summaries);

    for (auto * adapter : lora_adapters) {
        llama_adapter_lora_free(adapter);
    }

    llama_model_free(model);

    return 0;
}