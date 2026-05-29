// simple-lora.cpp
//
// 本程序用于测试多 LoRA 推理场景下三种执行方式的吞吐量：
// 1. sequential：顺序推理，每个请求单独绑定一个 LoRA 并独立 decode。
// 2. batch_unfused：批处理但不进行 LoRA 分组，请求中的 LoRA 交错排列。
// 3. batch_grouped：批处理加分组融合，同一 LoRA adapter 的请求连续排列。
//
// 实验口径：
// - 横轴为逻辑并行 LoRA 数量：1, 2, 4, 8, 16。
// - 每个逻辑 LoRA 对应 1 个请求，因此请求数等于横轴数值。
// - 当前只有 4 个真实 LoRA 文件，超过 4 时通过取模复用已有 LoRA。
// - 因此 8、16 表示更大的逻辑并发请求规模，不表示 8、16 个真实不同 adapter 文件。
//
// 输出 CSV：
// D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output/multilora_results.csv

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

struct experiment_result {
    std::string name;
    int n_lora = 0;      // 逻辑 LoRA 数量，也等于请求数量
    int n_requests = 0;
    int n_tokens = 0;
    double elapsed_s = 0.0;
    double tps = 0.0;
    double rps = 0.0;
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

// 生成 seq_id -> 真实 LoRA adapter id 的映射。
// 假设真实 LoRA 文件数为 4：
//
// batch_unfused:
// n_lora = 8  -> 0,1,2,3,0,1,2,3
// n_lora = 16 -> 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3
//
// batch_grouped:
// n_lora = 8  -> 0,0,1,1,2,2,3,3
// n_lora = 16 -> 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3
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

// 对多个 prompt 分词。
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

// 构造当前逻辑 LoRA 数量下的 prompt。
// 每个逻辑 LoRA 对应 1 个请求。
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

// 批处理实验：batch_unfused 或 batch_grouped。
static experiment_result run_batch_experiment(
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
    const int batch_size = (int) prompts.size();
    const int n_real_lora = (int) lora_adapters.size();

    experiment_result result;
    result.name = mode_name(mode);
    result.n_lora = logical_n_lora;
    result.n_requests = batch_size;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 256;
    ctx_params.n_seq_max = batch_size;
    ctx_params.n_batch = n_prompt_total;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to create context\n", __func__);
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

    if (llama_set_seq_adapters_lora(
                ctx,
                seq_lora_ids.data(),
                seq_lora_adapters.data(),
                seq_lora_scales.data(),
                seq_lora_ids.size()) != 0) {
        fprintf(stderr, "%s: error: failed to set seq LoRA mapping\n", __func__);
        llama_free(ctx);
        return result;
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

    const auto t_start = ggml_time_us();

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s: failed to eval prompt batch\n", __func__);

        llama_batch_free(batch);
        for (auto * sampler : samplers) {
            llama_sampler_free(sampler);
        }
        llama_free(ctx);

        return result;
    }

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

            llama_token new_token_id = llama_sampler_sample(samplers[s], ctx, last_logits_idx[s]);

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

        if (llama_decode(ctx, next_batch)) {
            fprintf(stderr, "%s: failed to eval decode batch\n", __func__);
            llama_batch_free(next_batch);
            break;
        }

        llama_batch_free(next_batch);
    }

    const auto t_end = ggml_time_us();
    const double elapsed_s = (t_end - t_start) / 1000000.0;

    result.n_tokens = n_decode_total;
    result.elapsed_s = elapsed_s;
    result.tps = n_decode_total / elapsed_s;
    result.rps = batch_size / elapsed_s;

    fprintf(stderr, "[%s][logical_n_lora=%d] decoded %d tokens in %.2f s, TPS = %.2f, RPS = %.2f\n",
            mode_name(mode),
            logical_n_lora,
            result.n_tokens,
            result.elapsed_s,
            result.tps,
            result.rps);

    llama_batch_free(batch);

    for (auto * sampler : samplers) {
        llama_sampler_free(sampler);
    }

    llama_free(ctx);
    return result;
}

// 顺序推理实验。
// 每个请求单独创建 context，单独绑定 LoRA，独立 decode。
static experiment_result run_sequential_experiment(
        llama_model * model,
        const llama_vocab * vocab,
        const std::vector<std::string> & prompts,
        const std::vector<std::vector<llama_token>> & prompt_tokens,
        const std::vector<llama_adapter_lora *> & lora_adapters,
        int logical_n_lora,
        int n_predict,
        float lora_scale) {
    const int n_requests = (int) prompts.size();
    const int n_real_lora = (int) lora_adapters.size();

    experiment_result result;
    result.name = mode_name(experiment_mode::sequential);
    result.n_lora = logical_n_lora;
    result.n_requests = n_requests;

    std::vector<int> seq_to_lora = build_seq_to_lora(
            experiment_mode::batch_unfused,
            logical_n_lora,
            n_real_lora);

    const auto t_start = ggml_time_us();

    int n_decode_total = 0;

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
            fprintf(stderr, "%s: failed to create context for request %d\n", __func__, s);
            continue;
        }

        std::vector<llama_adapter_lora *> adapters = { lora_adapters[adapter_id] };
        std::vector<float> scales = { lora_scale };

        if (llama_set_adapters_lora(ctx, adapters.data(), adapters.size(), scales.data()) != 0) {
            fprintf(stderr, "%s: failed to set LoRA for request %d\n", __func__, s);
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

        int n_decode = 0;
        llama_token new_token_id;

        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s: failed to eval request %d\n", __func__, s);
                break;
            }

            n_pos += batch.n_tokens;

            new_token_id = llama_sampler_sample(sampler, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            batch = llama_batch_get_one(&new_token_id, 1);
            n_decode++;
        }

        n_decode_total += n_decode;

        llama_sampler_free(sampler);
        llama_free(ctx);
    }

    const auto t_end = ggml_time_us();
    const double elapsed_s = (t_end - t_start) / 1000000.0;

    result.n_tokens = n_decode_total;
    result.elapsed_s = elapsed_s;
    result.tps = n_decode_total / elapsed_s;
    result.rps = n_requests / elapsed_s;

    fprintf(stderr, "[sequential][logical_n_lora=%d] decoded %d tokens in %.2f s, TPS = %.2f, RPS = %.2f\n",
            logical_n_lora,
            result.n_tokens,
            result.elapsed_s,
            result.tps,
            result.rps);

    return result;
}

static void save_results(const std::vector<experiment_result> & results) {
    const std::string csv_path = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output/multilora_results.csv";

    std::filesystem::create_directories("D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora/output");

    std::ofstream fout(csv_path);
    fout << "scheme,n_lora,n_requests,n_tokens,elapsed_s,tps,rps\n";

    for (const auto & r : results) {
        fout << r.name << ","
             << r.n_lora << ","
             << r.n_requests << ","
             << r.n_tokens << ","
             << r.elapsed_s << ","
             << r.tps << ","
             << r.rps << "\n";
    }

    fout.close();

    fprintf(stderr, "saved results to %s\n", csv_path.c_str());
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

    // 横轴：逻辑并行 LoRA 数量。
    // 其中 8 和 16 会复用已有 4 个真实 LoRA 文件。
    // std::vector<int> logical_lora_counts = { 1, 2, 4, 8, 16 };
    
    // 横轴：逻辑并行 LoRA 请求数量。
    // 当前只有 4 个真实 LoRA 文件，超过 4 时通过取模复用已有 LoRA。
    // 这里测试 1 到 15 个逻辑 LoRA 请求的吞吐变化。
    std::vector<int> logical_lora_counts = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    };

    std::vector<experiment_result> results;

    for (int logical_n_lora : logical_lora_counts) {
        fprintf(stderr, "\n========== EXPERIMENT logical_n_lora = %d ==========\n", logical_n_lora);

        std::vector<std::string> prompts = build_prompts(prompt_pool, logical_n_lora);

        std::vector<std::vector<llama_token>> prompt_tokens;
        int n_prompt_total = 0;

        if (!tokenize_prompts(vocab, prompts, prompt_tokens, n_prompt_total)) {
            continue;
        }

        results.push_back(run_sequential_experiment(
                model,
                vocab,
                prompts,
                prompt_tokens,
                lora_adapters,
                logical_n_lora,
                n_predict,
                lora_scale));

        results.push_back(run_batch_experiment(
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

        results.push_back(run_batch_experiment(
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
    }

    save_results(results);

    for (auto * adapter : lora_adapters) {
        llama_adapter_lora_free(adapter);
    }

    llama_model_free(model);

    return 0;
}