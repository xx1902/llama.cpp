// simple-lora.cpp
//
// Profiling 时间分布实验
//
// 本程序用于单独测试多 LoRA 批处理推理中的时间组成。
// 对比两种方案：
// 1. batch_unfused：批处理但不进行 LoRA 分组融合，请求中的 LoRA 交错排列。
// 2. batch_grouped：批处理融合，同一 LoRA 的请求连续排列，便于底层按 adapter 分组执行 LoRA 增量。
//
// 统计的时间组成包括：
// - LoRA 映射时间：调用 llama_set_seq_adapters_lora 的时间。
// - Prefill 时间：prompt batch 的 llama_decode 时间。
// - Decode 时间：生成阶段每一步 llama_decode 的累计时间。
// - Sampling 时间：每一步 llama_sampler_sample 的累计时间。
// - Other 时间：总时间减去上述已知时间后的剩余部分。
//
// 注意：
// 这里统计的是代码层时间分布，不是严格的 CUDA kernel profiling。
// 更细粒度的 kernel 启动开销和显存 IO 时间需要使用 Nsight Systems / Nsight Compute。

#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

enum class experiment_mode {
    batch_unfused,
    batch_grouped,
};

struct profiling_result {
    std::string scheme;
    int n_lora = 0;
    int n_requests = 0;
    int n_tokens = 0;

    double total_ms = 0.0;
    double lora_mapping_ms = 0.0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double sampling_ms = 0.0;
    double other_ms = 0.0;
};

static const char * mode_name(experiment_mode mode) {
    switch (mode) {
        case experiment_mode::batch_unfused:
            return "batch_unfused";
        case experiment_mode::batch_grouped:
            return "batch_grouped";
    }

    return "unknown";
}

// 构造 seq_id -> LoRA id 的映射。
// batch_unfused 示例：0,1,2,3,0,1,2,3
// batch_grouped 示例：0,0,1,1,2,2,3,3
static std::vector<int> build_seq_to_lora(
        experiment_mode mode,
        int logical_n_lora,
        int n_real_lora) {
    std::vector<int> seq_to_lora;
    seq_to_lora.reserve(logical_n_lora);

    if (mode == experiment_mode::batch_grouped) {
        for (int adapter_id = 0; adapter_id < n_real_lora; adapter_id++) {
            for (int i = 0; i < logical_n_lora; i++) {
                if (i % n_real_lora == adapter_id) {
                    seq_to_lora.push_back(adapter_id);
                }
            }
        }
    } else {
        for (int i = 0; i < logical_n_lora; i++) {
            seq_to_lora.push_back(i % n_real_lora);
        }
    }

    return seq_to_lora;
}

// 每个逻辑 LoRA 对应一个请求。
// 如果请求数量超过 prompt 数量，则循环复用 prompt。
static std::vector<std::string> build_prompts(
        const std::vector<std::string> & prompt_pool,
        int logical_n_lora) {
    std::vector<std::string> prompts;
    prompts.reserve(logical_n_lora);

    for (int i = 0; i < logical_n_lora; i++) {
        prompts.push_back(prompt_pool[i % prompt_pool.size()]);
    }

    return prompts;
}

// 向 llama_batch 中加入一个 token。
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

// 对多个 prompt 进行分词，并返回 prompt token 总数。
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
            fprintf(stderr, "error: failed to tokenize prompt %d\n", i);
            return false;
        }

        n_prompt_total += n;
    }

    return true;
}

// 执行一次 batch Profiling 实验。
// 该函数会拆分统计 LoRA 映射、prefill、decode、sampling 和 other 的时间。
static profiling_result run_profiling_experiment(
        llama_model * model,
        const llama_vocab * vocab,
        const std::vector<std::string> & prompts,
        const std::vector<std::vector<llama_token>> & prompt_tokens,
        const std::vector<llama_adapter_lora *> & lora_adapters,
        const std::vector<std::string> & lora_paths,
        experiment_mode mode,
        int logical_n_lora,
        int n_prompt_total,
        int n_predict,
        float lora_scale) {
    profiling_result result;

    const int batch_size = (int) prompts.size();
    const int n_real_lora = (int) lora_adapters.size();

    result.scheme = mode_name(mode);
    result.n_lora = logical_n_lora;
    result.n_requests = batch_size;

    const int64_t t_total_start = ggml_time_us();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 256;
    ctx_params.n_seq_max = batch_size;
    ctx_params.n_batch = n_prompt_total;
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return result;
    }

    std::vector<int> seq_to_lora = build_seq_to_lora(mode, logical_n_lora, n_real_lora);

    std::vector<llama_seq_id> seq_lora_ids;
    std::vector<llama_adapter_lora *> seq_lora_adapters;
    std::vector<float> seq_lora_scales;

    for (int s = 0; s < batch_size; s++) {
        const int adapter_id = seq_to_lora[s];

        seq_lora_ids.push_back((llama_seq_id) s);
        seq_lora_adapters.push_back(lora_adapters[adapter_id]);
        seq_lora_scales.push_back(lora_scale);

        fprintf(stderr, "[%s] seq %d -> LoRA %d (%s)\n",
                mode_name(mode),
                s,
                adapter_id,
                lora_paths[adapter_id].c_str());
    }

    // 统计 LoRA 映射时间。
    const int64_t t_lora_mapping_start = ggml_time_us();

    if (llama_set_seq_adapters_lora(
                ctx,
                seq_lora_ids.data(),
                seq_lora_adapters.data(),
                seq_lora_scales.data(),
                seq_lora_ids.size()) != 0) {
        fprintf(stderr, "%s: failed to set seq LoRA mapping\n", __func__);
        llama_free(ctx);
        return result;
    }

    const int64_t t_lora_mapping_end = ggml_time_us();
    result.lora_mapping_ms = (t_lora_mapping_end - t_lora_mapping_start) / 1000.0;

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

    // 统计 prefill 时间，即 prompt batch 的一次 decode。
    const int64_t t_prefill_start = ggml_time_us();

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s: failed to eval prompt batch\n", __func__);

        llama_batch_free(batch);
        for (auto * sampler : samplers) {
            llama_sampler_free(sampler);
        }
        llama_free(ctx);

        return result;
    }

    const int64_t t_prefill_end = ggml_time_us();
    result.prefill_ms = (t_prefill_end - t_prefill_start) / 1000.0;

    std::vector<int> seq_pos(batch_size);
    std::vector<bool> done(batch_size, false);

    for (int s = 0; s < batch_size; s++) {
        seq_pos[s] = (int) prompt_tokens[s].size();
    }

    int n_decode_total = 0;

    for (int step = 0; step < n_predict; step++) {
        llama_batch next_batch = llama_batch_init(batch_size, 0, batch_size);
        next_batch.n_tokens = 0;

        int active_count = 0;

        for (int s = 0; s < batch_size; s++) {
            if (done[s]) {
                continue;
            }

            // 统计采样时间。
            const int64_t t_sampling_start = ggml_time_us();

            llama_token new_token_id = llama_sampler_sample(samplers[s], ctx, last_logits_idx[s]);

            const int64_t t_sampling_end = ggml_time_us();
            result.sampling_ms += (t_sampling_end - t_sampling_start) / 1000.0;

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                done[s] = true;
                continue;
            }

            last_logits_idx[s] = next_batch.n_tokens;
            batch_add(next_batch, new_token_id, seq_pos[s], seq_ids[s], true);

            seq_pos[s]++;
            n_decode_total++;
            active_count++;
        }

        if (active_count == 0) {
            llama_batch_free(next_batch);
            break;
        }

        // 统计生成阶段 decode 时间。
        const int64_t t_decode_start = ggml_time_us();

        if (llama_decode(ctx, next_batch)) {
            fprintf(stderr, "%s: failed to eval decode batch\n", __func__);
            llama_batch_free(next_batch);
            break;
        }

        const int64_t t_decode_end = ggml_time_us();
        result.decode_ms += (t_decode_end - t_decode_start) / 1000.0;

        llama_batch_free(next_batch);
    }

    const int64_t t_total_end = ggml_time_us();

    result.n_tokens = n_decode_total;
    result.total_ms = (t_total_end - t_total_start) / 1000.0;

    const double known_ms =
        result.lora_mapping_ms +
        result.prefill_ms +
        result.decode_ms +
        result.sampling_ms;

    result.other_ms = std::max(0.0, result.total_ms - known_ms);

    fprintf(stderr,
            "[%s] total = %.2f ms, mapping = %.2f ms, prefill = %.2f ms, decode = %.2f ms, sampling = %.2f ms, other = %.2f ms\n",
            mode_name(mode),
            result.total_ms,
            result.lora_mapping_ms,
            result.prefill_ms,
            result.decode_ms,
            result.sampling_ms,
            result.other_ms);

    llama_batch_free(batch);

    for (auto * sampler : samplers) {
        llama_sampler_free(sampler);
    }

    llama_free(ctx);
    return result;
}

static void save_profiling_results(const std::vector<profiling_result> & results) {
    const std::string output_dir = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output";
    const std::string csv_path = output_dir + "/profiling_results.csv";

    std::filesystem::create_directories(output_dir);

    std::ofstream fout(csv_path);

    fout << "scheme,n_lora,n_requests,n_tokens,total_ms,"
         << "lora_mapping_ms,prefill_ms,decode_ms,sampling_ms,other_ms\n";

    for (const auto & r : results) {
        fout << r.scheme << ","
             << r.n_lora << ","
             << r.n_requests << ","
             << r.n_tokens << ","
             << r.total_ms << ","
             << r.lora_mapping_ms << ","
             << r.prefill_ms << ","
             << r.decode_ms << ","
             << r.sampling_ms << ","
             << r.other_ms << "\n";
    }

    fout.close();

    fprintf(stderr, "saved profiling results to %s\n", csv_path.c_str());
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    std::setlocale(LC_NUMERIC, "C");

    std::string model_path = "D:/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf";

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

    const int ngl = 99;
    const int n_predict = 32;
    const float lora_scale = 1.0f;

    // Profiling 选择一个固定并发规模即可。
    // 这里使用 8 个逻辑 LoRA 请求，超过 4 个真实 LoRA 时复用已有 adapter。
    const int logical_n_lora = 8;

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
        fprintf(stderr, "\n========== LoRA loaded ==========\n");

        const auto t0 = ggml_time_us();
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, path.c_str());
        const auto t1 = ggml_time_us();

        if (adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", path.c_str());
            llama_model_free(model);
            return 1;
        }

        fprintf(stderr, "[LoRA] loaded %s in %.3f ms\n",
                path.c_str(),
                (t1 - t0) / 1000.0);

        lora_adapters.push_back(adapter);
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<std::string> prompts = build_prompts(prompt_pool, logical_n_lora);

    std::vector<std::vector<llama_token>> prompt_tokens;
    int n_prompt_total = 0;

    if (!tokenize_prompts(vocab, prompts, prompt_tokens, n_prompt_total)) {
        for (auto * adapter : lora_adapters) {
            llama_adapter_lora_free(adapter);
        }
        llama_model_free(model);
        return 1;
    }

    std::vector<profiling_result> results;

    results.push_back(run_profiling_experiment(
            model,
            vocab,
            prompts,
            prompt_tokens,
            lora_adapters,
            lora_paths,
            experiment_mode::batch_unfused,
            logical_n_lora,
            n_prompt_total,
            n_predict,
            lora_scale));

    results.push_back(run_profiling_experiment(
            model,
            vocab,
            prompts,
            prompt_tokens,
            lora_adapters,
            lora_paths,
            experiment_mode::batch_grouped,
            logical_n_lora,
            n_prompt_total,
            n_predict,
            lora_scale));

    save_profiling_results(results);

    for (auto * adapter : lora_adapters) {
        llama_adapter_lora_free(adapter);
    }

    llama_model_free(model);

    return 0;
}