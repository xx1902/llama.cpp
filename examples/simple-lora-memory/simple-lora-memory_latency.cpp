// simple-lora-memory.cpp
// 本程序用于测量 LoRA adapter 从不同驻留状态到“完成绑定、可被当前 context 使用”的路径延迟。
// 本实验不包含 token decode 前向计算，只统计 LoRA adapter 的加载、CPU->GPU 提升和绑定开销。

// 测量路径：
// 1. GPU 热区：adapter 已经位于模型所在 backend buffer 中，测绑定开销。
// 2. CPU 温区：adapter 已经加载到 CPU backend buffer 中，测克隆/提升到模型所在 backend buffer + 绑定开销。
// 3. SSD 冷区：adapter 不在内存中，测从 LoRA GGUF 文件初始化到模型所在 backend buffer + 绑定开销。

#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct latency_sample {
    std::string path_type;
    int round = 0;
    int adapter_id = 0;
    double load_ms = 0.0;
    double bind_ms = 0.0;
    double total_ms = 0.0;
};

static const std::string output_dir =
        "/home/agx_orin/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output";

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

static void clear_adapter(llama_context * ctx) {
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
}

// 执行一次最小 1-token decode。
// 这一步用于触发 adapter 在当前 context 下真正参与一次前向计算。
// static double run_one_token_decode(
//         llama_context * ctx,
//         const llama_vocab * vocab) {
//     llama_memory_clear(llama_get_memory(ctx), true);

//     const char * prompt = "Hello";
//     const int n_prompt = -llama_tokenize(
//             vocab,
//             prompt,
//             5,
//             nullptr,
//             0,
//             true,
//             true);

//     std::vector<llama_token> tokens(n_prompt);

//     if (llama_tokenize(
//                 vocab,
//                 prompt,
//                 5,
//                 tokens.data(),
//                 tokens.size(),
//                 true,
//                 true) < 0) {
//         fprintf(stderr, "failed to tokenize warmup prompt\n");
//         return 0.0;
//     }

//     llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

//     const auto t0 = ggml_time_us();

//     if (llama_decode(ctx, batch)) {
//         fprintf(stderr, "failed to run one-token decode\n");
//     }

//     const auto t1 = ggml_time_us();

//     return (t1 - t0) / 1000.0;
// }

static latency_sample measure_gpu_hot(
        llama_context * ctx,
        llama_adapter_lora * gpu_adapter,
        int round,
        int adapter_id,
        float scale) {
    latency_sample sample;
    sample.path_type = "gpu_hot";
    sample.round = round;
    sample.adapter_id = adapter_id;

    sample.load_ms = 0.0;
    sample.bind_ms = bind_adapter(ctx, gpu_adapter, scale);
    sample.total_ms = sample.bind_ms;

    return sample;
}

static latency_sample measure_cpu_warm(
        llama_context * ctx,
        llama_adapter_lora * cpu_adapter,
        int round,
        int adapter_id,
        float scale) {
    latency_sample sample;
    sample.path_type = "cpu_warm";
    sample.round = round;
    sample.adapter_id = adapter_id;

    const auto t0 = ggml_time_us();
    llama_adapter_lora * gpu_adapter =
            llama_adapter_lora_clone_to_model_buft(cpu_adapter);
    const auto t1 = ggml_time_us();

    if (gpu_adapter == nullptr) {
        fprintf(stderr, "failed to promote CPU LoRA adapter\n");
        return sample;
    }

    sample.load_ms = (t1 - t0) / 1000.0;
    sample.bind_ms = bind_adapter(ctx, gpu_adapter, scale);
    sample.total_ms = sample.load_ms + sample.bind_ms;

    clear_adapter(ctx);
    llama_adapter_lora_free(gpu_adapter);

    return sample;
}

static latency_sample measure_file_load(
        llama_model * model,
        llama_context * ctx,
        const std::string & path,
        int round,
        int adapter_id,
        float scale) {
    latency_sample sample;
    sample.path_type = "file_load";
    sample.round = round;
    sample.adapter_id = adapter_id;

    const auto t0 = ggml_time_us();
    llama_adapter_lora * gpu_adapter =
            llama_adapter_lora_init(model, path.c_str());
    const auto t1 = ggml_time_us();

    if (gpu_adapter == nullptr) {
        fprintf(stderr, "failed to load LoRA adapter from file: %s\n", path.c_str());
        return sample;
    }

    sample.load_ms = (t1 - t0) / 1000.0;
    sample.bind_ms = bind_adapter(ctx, gpu_adapter, scale);
    sample.total_ms = sample.load_ms + sample.bind_ms;

    clear_adapter(ctx);
    llama_adapter_lora_free(gpu_adapter);

    return sample;
}

static void save_samples(const std::vector<latency_sample> & samples) {
    std::filesystem::create_directories(output_dir);

    const std::string csv_path = output_dir + "/lora_load_path_latency.csv";
    std::ofstream fout(csv_path);

    fout << "path_type,round,adapter_id,load_ms,bind_ms,total_ms\n";

    for (const auto & s : samples) {
        fout << s.path_type << ","
            << s.round << ","
            << s.adapter_id << ","
            << s.load_ms << ","
            << s.bind_ms << ","
            << s.total_ms << "\n";
    }

    fout.close();

    fprintf(stderr, "saved latency samples to %s\n", csv_path.c_str());
}

int main() {
    std::setlocale(LC_NUMERIC, "C");

    const std::string model_path =
            "/home/agx_orin/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf";

    std::vector<std::string> lora_paths = {
        "/home/agx_orin/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
        "/home/agx_orin/ecnu_experiment/Model/subliminal-monkey.gguf",
        "/home/agx_orin/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
        "/home/agx_orin/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
    };

    const int ngl = 99;
    const int n_ctx = 256;
    const int n_batch = 8;
    const int n_rounds = 60;
    const float lora_scale = 1.0f;

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_batch;
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "failed to create llama context\n");
        llama_model_free(model);
        return 1;
    }


    std::vector<latency_sample> samples;

    std::vector<llama_adapter_lora *> gpu_hot_adapters;
    std::vector<llama_adapter_lora *> cpu_warm_adapters;

    // 预加载 adapter，作为 CPU 温区和 GPU 热区实验的基础。
    // 这些 adapter 已经完成 llama_adapter_lora_init，不再重复从文件解析。
    for (int i = 0; i < (int) lora_paths.size(); i++) {
        llama_adapter_lora * gpu_adapter =
                llama_adapter_lora_init(model, lora_paths[i].c_str());

        llama_adapter_lora * cpu_adapter =
                llama_adapter_lora_init_cpu(model, lora_paths[i].c_str());

        if (gpu_adapter == nullptr || cpu_adapter == nullptr) {
            fprintf(stderr, "failed to preload LoRA adapter %d\n", i);
            return 1;
        }

        gpu_hot_adapters.push_back(gpu_adapter);
        cpu_warm_adapters.push_back(cpu_adapter);
    }

    // 预热：避免把首次 set adapter 的一些一次性开销全部算进第一轮样本。
    for (int i = 0; i < (int) gpu_hot_adapters.size(); i++) {
        bind_adapter(ctx, gpu_hot_adapters[i], lora_scale);
        clear_adapter(ctx);
    }

    for (int round = 0; round < n_rounds; round++) {
        for (int adapter_id = 0; adapter_id < (int) lora_paths.size(); adapter_id++) {
            fprintf(stderr,
                    "round %d/%d, adapter %d\n",
                    round + 1,
                    n_rounds,
                    adapter_id);

            samples.push_back(measure_gpu_hot(
                    ctx,
                    gpu_hot_adapters[adapter_id],
                    round,
                    adapter_id,
                    lora_scale));

            samples.push_back(measure_cpu_warm(
                    ctx,
                    cpu_warm_adapters[adapter_id],
                    round,
                    adapter_id,
                    lora_scale));

            samples.push_back(measure_file_load(
                    model,
                    ctx,
                    lora_paths[adapter_id],
                    round,
                    adapter_id,
                    lora_scale));
        }
    }

    save_samples(samples);

    clear_adapter(ctx);

    for (auto * adapter : gpu_hot_adapters) {
        llama_adapter_lora_free(adapter);
    }

    for (auto * adapter : cpu_warm_adapters) {
        llama_adapter_lora_free(adapter);
    }

    llama_free(ctx);
    llama_model_free(model);

    fprintf(stderr, "LoRA load path latency experiment finished.\n");

    return 0;
}