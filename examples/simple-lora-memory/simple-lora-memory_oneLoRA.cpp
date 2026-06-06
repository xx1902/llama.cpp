// simple-lora-memory_oneLoRA.cpp
//
// 单 LoRA 长上下文 KV 显存实验
//
// 实验目的：
// 1. 使用同一个 LoRA adapter。
// 2. 在同一个 llama_context 中连续模拟多轮请求。
// 3. 不清空 KV cache，让上下文逐步增长直到接近 n_ctx 上限。
// 4. 分别对比：
//    - 原生连续 KV cache
//    - 实验性物理分页 KV cache
//
// 预期现象：
// - 连续 KV 在 context 创建时就按 n_ctx 申请完整 KV buffer。
// - 分页 KV 在 context 创建时不申请完整 KV，而是随着 token 写入逐页申请。
// - 当上下文没有填满时，分页 KV 显存占用更低。
// - 当上下文逐渐接近 n_ctx 上限时，分页 KV 需要申请越来越多页，优势会逐渐缩小。
//
// 输出：
// output/one_lora_context_limit_continuous.csv
// output/one_lora_context_limit_paged.csv
// output/one_lora_context_limit_summary.csv

#include "llama.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <psapi.h>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#pragma comment(lib, "Psapi.lib")

enum class kv_mode {
    continuous,
    paged,
};

struct memory_sample {
    std::string mode;
    int step = 0;
    std::string event;
    int request_id = 0;
    int position = 0;
    int generated_tokens = 0;
    int n_ctx = 0;
    int kv_page_size = 0;
    int theoretical_used_pages = 0;
    double context_used_rate = 0.0;
    double page_used_rate = 0.0;
    double process_private_mb = 0.0;
    double gpu_used_mb = 0.0;
    double gpu_delta_mb = 0.0;
    double gpu_ready_delta_mb = 0.0;
};

struct experiment_summary {
    std::string mode;
    int n_ctx = 0;
    int kv_page_size = 0;
    int n_requests = 0;
    int final_position = 0;
    int generated_tokens = 0;
    int theoretical_used_pages = 0;
    double elapsed_ms = 0.0;
    double tps = 0.0;
    double gpu_start_mb = 0.0;
    double gpu_ready_mb = 0.0;
    double gpu_peak_mb = 0.0;
    double gpu_end_mb = 0.0;
    double gpu_peak_delta_mb = 0.0;
    double gpu_peak_ready_delta_mb = 0.0;
    double process_start_mb = 0.0;
    double process_end_mb = 0.0;
    double process_delta_mb = 0.0;
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output";

static const char * mode_name(kv_mode mode) {
    switch (mode) {
        case kv_mode::continuous:
            return "continuous_kv";
        case kv_mode::paged:
            return "paged_kv";
    }

    return "unknown";
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

static int ceil_div_int(int a, int b) {
    if (b <= 0) {
        return 0;
    }

    return (a + b - 1) / b;
}

static void record_sample(
        std::vector<memory_sample> & samples,
        kv_mode mode,
        int step,
        const std::string & event,
        int request_id,
        int position,
        int generated_tokens,
        int n_ctx,
        int kv_page_size,
        double gpu_start_mb,
        double gpu_ready_mb) {
    memory_sample s;

    s.mode = mode_name(mode);
    s.step = step;
    s.event = event;
    s.request_id = request_id;
    s.position = position;
    s.generated_tokens = generated_tokens;
    s.n_ctx = n_ctx;
    s.kv_page_size = kv_page_size;

    if (kv_page_size > 0) {
        s.theoretical_used_pages = ceil_div_int(position, kv_page_size);
    } else {
        s.theoretical_used_pages = 0;
    }

    s.context_used_rate = n_ctx > 0 ? (double) position / (double) n_ctx : 0.0;

    if (kv_page_size > 0 && n_ctx > 0) {
        const int total_pages = ceil_div_int(n_ctx, kv_page_size);
        s.page_used_rate = total_pages > 0 ?
                (double) s.theoretical_used_pages / (double) total_pages :
                0.0;
    }

    s.process_private_mb = get_process_private_mb();
    s.gpu_used_mb = get_gpu_used_mb();
    s.gpu_delta_mb = s.gpu_used_mb - gpu_start_mb;
    s.gpu_ready_delta_mb = s.gpu_used_mb - gpu_ready_mb;

    samples.push_back(s);

    fprintf(stderr,
            "[%s] step=%d event=%s req=%d pos=%d/%d pages=%d gpu=%.2f MB delta=%.2f MB ready_delta=%.2f MB\n",
            s.mode.c_str(),
            s.step,
            s.event.c_str(),
            s.request_id,
            s.position,
            s.n_ctx,
            s.theoretical_used_pages,
            s.gpu_used_mb,
            s.gpu_delta_mb,
            s.gpu_ready_delta_mb);
}

static void save_samples(
        const std::vector<memory_sample> & samples,
        const std::string & csv_name) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/" + csv_name;
    std::ofstream fout(path);

    fout << "mode,step,event,request_id,position,generated_tokens,"
         << "n_ctx,kv_page_size,theoretical_used_pages,"
         << "context_used_rate,page_used_rate,"
         << "process_private_mb,gpu_used_mb,gpu_delta_mb,gpu_ready_delta_mb\n";

    for (const auto & s : samples) {
        fout << s.mode << ","
             << s.step << ","
             << s.event << ","
             << s.request_id << ","
             << s.position << ","
             << s.generated_tokens << ","
             << s.n_ctx << ","
             << s.kv_page_size << ","
             << s.theoretical_used_pages << ","
             << s.context_used_rate << ","
             << s.page_used_rate << ","
             << s.process_private_mb << ","
             << s.gpu_used_mb << ","
             << s.gpu_delta_mb << ","
             << s.gpu_ready_delta_mb << "\n";
    }

    fprintf(stderr, "saved memory trace to %s\n", path.c_str());
}

static void save_summary(const std::vector<experiment_summary> & summaries) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/one_lora_context_limit_summary.csv";
    std::ofstream fout(path);

    fout << "mode,n_ctx,kv_page_size,n_requests,final_position,generated_tokens,"
         << "theoretical_used_pages,elapsed_ms,tps,"
         << "gpu_start_mb,gpu_ready_mb,gpu_peak_mb,gpu_end_mb,"
         << "gpu_peak_delta_mb,gpu_peak_ready_delta_mb,"
         << "process_start_mb,process_end_mb,process_delta_mb\n";

    for (const auto & s : summaries) {
        fout << s.mode << ","
             << s.n_ctx << ","
             << s.kv_page_size << ","
             << s.n_requests << ","
             << s.final_position << ","
             << s.generated_tokens << ","
             << s.theoretical_used_pages << ","
             << s.elapsed_ms << ","
             << s.tps << ","
             << s.gpu_start_mb << ","
             << s.gpu_ready_mb << ","
             << s.gpu_peak_mb << ","
             << s.gpu_end_mb << ","
             << s.gpu_peak_delta_mb << ","
             << s.gpu_peak_ready_delta_mb << ","
             << s.process_start_mb << ","
             << s.process_end_mb << ","
             << s.process_delta_mb << "\n";
    }

    fprintf(stderr, "saved summary to %s\n", path.c_str());
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

static bool decode_one_token(
        llama_context * ctx,
        llama_token token,
        int position) {
    llama_batch batch = llama_batch_init(1, 0, 1);

    batch_add(
            batch,
            token,
            position,
            0,
            true);

    const int ret = llama_decode(ctx, batch);

    llama_batch_free(batch);

    return ret == 0;
}

static double bind_adapter(
        llama_context * ctx,
        llama_adapter_lora * adapter,
        float scale) {
    std::vector<llama_adapter_lora *> adapters = { adapter };
    std::vector<float> scales = { scale };

    const auto t0 = ggml_time_us();

    const int ret = llama_set_adapters_lora(
            ctx,
            adapters.data(),
            adapters.size(),
            scales.data());

    const auto t1 = ggml_time_us();

    if (ret != 0) {
        fprintf(stderr, "failed to bind LoRA adapter\n");
    }

    return (t1 - t0) / 1000.0;
}

static experiment_summary run_one_mode(
        kv_mode mode,
        const std::string & csv_name,
        const std::string & model_path,
        const std::string & lora_path) {
    const int ngl = 99;
    const int n_ctx = 512;
    const int n_batch = 1;
    const int n_ubatch = 1;
    const int n_predict_per_request = 32;
    const int max_requests = 128;
    const int physical_kv_page_size = 16;
    const float lora_scale = 1.0f;

    std::vector<memory_sample> samples;

    experiment_summary summary;
    summary.mode = mode_name(mode);
    summary.n_ctx = n_ctx;
    summary.kv_page_size = mode == kv_mode::paged ? physical_kv_page_size : 0;

    const double gpu_start_mb = get_gpu_used_mb();
    const double process_start_mb = get_process_private_mb();

    summary.gpu_start_mb = gpu_start_mb;
    summary.process_start_mb = process_start_mb;

    int step = 0;
    int position = 0;
    int generated_tokens = 0;
    int request_count = 0;
    double gpu_ready_mb = gpu_start_mb;

    record_sample(
            samples,
            mode,
            step++,
            "start",
            0,
            position,
            generated_tokens,
            n_ctx,
            summary.kv_page_size,
            gpu_start_mb,
            gpu_ready_mb);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(
            model_path.c_str(),
            model_params);

    if (model == nullptr) {
        fprintf(stderr, "[%s] failed to load model: %s\n",
                mode_name(mode),
                model_path.c_str());
        save_samples(samples, csv_name);
        return summary;
    }

    record_sample(
            samples,
            mode,
            step++,
            "model_loaded",
            0,
            position,
            generated_tokens,
            n_ctx,
            summary.kv_page_size,
            gpu_start_mb,
            gpu_ready_mb);

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.n_ubatch = n_ubatch;
    ctx_params.n_seq_max = 1;
    ctx_params.no_perf = true;
    ctx_params.kv_unified = true;

    if (mode == kv_mode::paged) {
        ctx_params.experimental_physical_paged_kv = true;
        ctx_params.physical_kv_page_size = physical_kv_page_size;
    } else {
        ctx_params.experimental_physical_paged_kv = false;
        ctx_params.physical_kv_page_size = 0;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == nullptr) {
        fprintf(stderr, "[%s] failed to create llama context\n", mode_name(mode));
        llama_model_free(model);
        save_samples(samples, csv_name);
        return summary;
    }

    record_sample(
            samples,
            mode,
            step++,
            "context_created",
            0,
            position,
            generated_tokens,
            n_ctx,
            summary.kv_page_size,
            gpu_start_mb,
            gpu_ready_mb);

    llama_adapter_lora * adapter =
            llama_adapter_lora_init(model, lora_path.c_str());

    if (adapter == nullptr) {
        fprintf(stderr, "[%s] failed to load LoRA adapter: %s\n",
                mode_name(mode),
                lora_path.c_str());
        llama_free(ctx);
        llama_model_free(model);
        save_samples(samples, csv_name);
        return summary;
    }

    bind_adapter(ctx, adapter, lora_scale);

    gpu_ready_mb = get_gpu_used_mb();
    summary.gpu_ready_mb = gpu_ready_mb;

    record_sample(
            samples,
            mode,
            step++,
            "lora_bound_ready",
            0,
            position,
            generated_tokens,
            n_ctx,
            summary.kv_page_size,
            gpu_start_mb,
            gpu_ready_mb);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> request_tokens;
    const std::string request_prompt =
            "User: please continue this answer. Assistant:";

    if (!tokenize_prompt(vocab, request_prompt, request_tokens)) {
        fprintf(stderr, "[%s] failed to tokenize request prompt\n", mode_name(mode));
        llama_adapter_lora_free(adapter);
        llama_free(ctx);
        llama_model_free(model);
        save_samples(samples, csv_name);
        return summary;
    }

    const llama_token repeat_token = request_tokens.empty() ?
            llama_vocab_bos(vocab) :
            request_tokens.back();

    double gpu_peak_mb = gpu_ready_mb;

    const auto t0 = ggml_time_us();

    while (position < n_ctx - 1 && request_count < max_requests) {
        request_count++;

        record_sample(
                samples,
                mode,
                step++,
                "request_start",
                request_count,
                position,
                generated_tokens,
                n_ctx,
                summary.kv_page_size,
                gpu_start_mb,
                gpu_ready_mb);

        for (int i = 0; i < (int) request_tokens.size() && position < n_ctx - 1; i++) {
            if (!decode_one_token(ctx, request_tokens[i], position)) {
                fprintf(stderr,
                        "[%s] llama_decode failed at prompt token, pos=%d\n",
                        mode_name(mode),
                        position);
                break;
            }

            position++;

            if (position % physical_kv_page_size == 0 || position >= n_ctx - 1) {
                record_sample(
                        samples,
                        mode,
                        step++,
                        "prompt_page_boundary",
                        request_count,
                        position,
                        generated_tokens,
                        n_ctx,
                        summary.kv_page_size,
                        gpu_start_mb,
                        gpu_ready_mb);
            }
        }

        for (int i = 0; i < n_predict_per_request && position < n_ctx - 1; i++) {
            if (!decode_one_token(ctx, repeat_token, position)) {
                fprintf(stderr,
                        "[%s] llama_decode failed at generated token, pos=%d\n",
                        mode_name(mode),
                        position);
                break;
            }

            position++;
            generated_tokens++;

            if (generated_tokens % 8 == 0 ||
                    position % physical_kv_page_size == 0 ||
                    position >= n_ctx - 1) {
                record_sample(
                        samples,
                        mode,
                        step++,
                        "decode_progress",
                        request_count,
                        position,
                        generated_tokens,
                        n_ctx,
                        summary.kv_page_size,
                        gpu_start_mb,
                        gpu_ready_mb);
            }

            const double now_gpu = get_gpu_used_mb();
            if (now_gpu > gpu_peak_mb) {
                gpu_peak_mb = now_gpu;
            }
        }

        record_sample(
                samples,
                mode,
                step++,
                "request_end",
                request_count,
                position,
                generated_tokens,
                n_ctx,
                summary.kv_page_size,
                gpu_start_mb,
                gpu_ready_mb);
    }

    const auto t1 = ggml_time_us();

    record_sample(
            samples,
            mode,
            step++,
            "context_limit_reached",
            request_count,
            position,
            generated_tokens,
            n_ctx,
            summary.kv_page_size,
            gpu_start_mb,
            gpu_ready_mb);

    summary.n_requests = request_count;
    summary.final_position = position;
    summary.generated_tokens = generated_tokens;
    summary.theoretical_used_pages = summary.kv_page_size > 0 ?
            ceil_div_int(position, summary.kv_page_size) :
            0;

    summary.elapsed_ms = (t1 - t0) / 1000.0;
    summary.tps = summary.elapsed_ms > 0.0 ?
            generated_tokens / (summary.elapsed_ms / 1000.0) :
            0.0;

    summary.gpu_peak_mb = gpu_peak_mb;
    summary.gpu_peak_delta_mb = gpu_peak_mb - gpu_start_mb;
    summary.gpu_peak_ready_delta_mb = gpu_peak_mb - gpu_ready_mb;

    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
    llama_adapter_lora_free(adapter);

    record_sample(
            samples,
            mode,
            step++,
            "lora_freed",
            request_count,
            position,
            generated_tokens,
            n_ctx,
            summary.kv_page_size,
            gpu_start_mb,
            gpu_ready_mb);

    llama_free(ctx);

    record_sample(
            samples,
            mode,
            step++,
            "context_freed",
            request_count,
            position,
            generated_tokens,
            n_ctx,
            summary.kv_page_size,
            gpu_start_mb,
            gpu_ready_mb);

    llama_model_free(model);

    record_sample(
            samples,
            mode,
            step++,
            "model_freed",
            request_count,
            position,
            generated_tokens,
            n_ctx,
            summary.kv_page_size,
            gpu_start_mb,
            gpu_ready_mb);

    summary.gpu_end_mb = get_gpu_used_mb();
    summary.process_end_mb = get_process_private_mb();
    summary.process_delta_mb = summary.process_end_mb - process_start_mb;

    save_samples(samples, csv_name);

    fprintf(stderr,
            "[%s] final_pos=%d/%d requests=%d generated=%d elapsed=%.2f ms TPS=%.2f peak_ready_delta=%.2f MB\n",
            mode_name(mode),
            summary.final_position,
            summary.n_ctx,
            summary.n_requests,
            summary.generated_tokens,
            summary.elapsed_ms,
            summary.tps,
            summary.gpu_peak_ready_delta_mb);

    return summary;
}

int main() {
    std::setlocale(LC_NUMERIC, "C");

    const std::string model_path =
            "D:/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf";

    const std::string lora_path =
            "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf";

    ggml_backend_load_all();

    std::vector<experiment_summary> summaries;

    summaries.push_back(run_one_mode(
            kv_mode::continuous,
            "one_lora_context_limit_continuous.csv",
            model_path,
            lora_path));

    summaries.push_back(run_one_mode(
            kv_mode::paged,
            "one_lora_context_limit_paged.csv",
            model_path,
            lora_path));

    save_summary(summaries);

    fprintf(stderr, "one LoRA context limit KV memory experiment finished.\n");

    return 0;
}