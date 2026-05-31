// simple-lora-memory.cpp
//
// LoRA 不同加载路径延迟真实测量实验
//
// 本程序用于测量 LoRA adapter 从不同状态到“可以被当前 context 使用”的时间。
// 这里不做文本生成，只测 LoRA 加载与绑定路径的真实 API 耗时。
//
// 测量路径：
// 1. GPU热区：LoRA adapter 已经通过 llama_adapter_lora_init 初始化完成。
//    此时只测 llama_set_adapters_lora 的绑定耗时。
// 2. CPU温区近似：先把 LoRA GGUF 文件预读到系统内存/文件页缓存，
//    然后测 llama_adapter_lora_init + llama_set_adapters_lora。
//    注意：这不是严格的“已反序列化 CPU adapter -> GPU adapter”迁移，
//    而是“文件内容已经在内存缓存中，再初始化到可运行状态”的近似路径。
// 3. 文件加载：直接从 LoRA GGUF 路径调用 llama_adapter_lora_init，
//    然后绑定到当前 context。
//    注意：如果操作系统文件缓存已经命中，它不等价于严格 SSD cold load。
//
// 输出文件：
// D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output/lora_load_path_latency.csv

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
    double init_ms = 0.0;
    double bind_ms = 0.0;
    double total_ms = 0.0;
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output";

static bool preload_file_to_os_cache(const std::string & path) {
    std::ifstream fin(path, std::ios::binary);

    if (!fin) {
        fprintf(stderr, "failed to open LoRA file for preload: %s\n", path.c_str());
        return false;
    }

    std::vector<char> buffer(4 * 1024 * 1024);

    while (fin.read(buffer.data(), buffer.size()) || fin.gcount() > 0) {
        // 读取即可，目的是让操作系统把文件页放入内存缓存。
    }

    return true;
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

static void clear_adapter(llama_context * ctx) {
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
}

static latency_sample measure_gpu_hot_bind(
        llama_context * ctx,
        llama_adapter_lora * adapter,
        int round,
        int adapter_id,
        float scale) {
    latency_sample sample;
    sample.path_type = "gpu_hot";
    sample.round = round;
    sample.adapter_id = adapter_id;

    sample.init_ms = 0.0;
    sample.bind_ms = bind_adapter(ctx, adapter, scale);
    sample.total_ms = sample.bind_ms;

    return sample;
}

static latency_sample measure_file_init_bind(
        llama_model * model,
        llama_context * ctx,
        const std::string & path,
        const std::string & path_type,
        int round,
        int adapter_id,
        float scale) {
    latency_sample sample;
    sample.path_type = path_type;
    sample.round = round;
    sample.adapter_id = adapter_id;

    const auto t_init0 = ggml_time_us();
    llama_adapter_lora * adapter = llama_adapter_lora_init(model, path.c_str());
    const auto t_init1 = ggml_time_us();

    if (adapter == nullptr) {
        fprintf(stderr, "failed to initialize LoRA adapter: %s\n", path.c_str());
        return sample;
    }

    sample.init_ms = (t_init1 - t_init0) / 1000.0;
    sample.bind_ms = bind_adapter(ctx, adapter, scale);
    sample.total_ms = sample.init_ms + sample.bind_ms;

    clear_adapter(ctx);
    llama_adapter_lora_free(adapter);

    return sample;
}

static void save_samples(const std::vector<latency_sample> & samples) {
    std::filesystem::create_directories(output_dir);

    const std::string csv_path = output_dir + "/lora_load_path_latency.csv";
    std::ofstream fout(csv_path);

    fout << "path_type,round,adapter_id,init_ms,bind_ms,total_ms\n";

    for (const auto & s : samples) {
        fout << s.path_type << ","
             << s.round << ","
             << s.adapter_id << ","
             << s.init_ms << ","
             << s.bind_ms << ","
             << s.total_ms << "\n";
    }

    fout.close();

    fprintf(stderr, "saved latency samples to %s\n", csv_path.c_str());
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

    const int ngl = 99;
    const int n_ctx = 256;
    const int n_batch = 8;
    const int n_rounds = 30;
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

    // 先初始化一组常驻 adapter，用于测 GPU 热区复用绑定路径。
    std::vector<llama_adapter_lora *> hot_adapters;

    for (int i = 0; i < (int) lora_paths.size(); i++) {
        fprintf(stderr, "preloading hot LoRA adapter %d: %s\n", i, lora_paths[i].c_str());

        llama_adapter_lora * adapter = llama_adapter_lora_init(model, lora_paths[i].c_str());

        if (adapter == nullptr) {
            fprintf(stderr, "failed to preload LoRA adapter: %s\n", lora_paths[i].c_str());

            for (auto * a : hot_adapters) {
                llama_adapter_lora_free(a);
            }

            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }

        hot_adapters.push_back(adapter);
    }

    // 测量三类路径。
    // round 多跑几次是为了得到延迟分布，而不是只看单点值。
    for (int round = 0; round < n_rounds; round++) {
        for (int adapter_id = 0; adapter_id < (int) lora_paths.size(); adapter_id++) {
            const std::string & path = lora_paths[adapter_id];

            fprintf(stderr,
                    "round %d/%d, adapter %d\n",
                    round + 1,
                    n_rounds,
                    adapter_id);

            // 路径 1：GPU 热区。
            // adapter 已经初始化完成，只测绑定到 context 的时间。
            samples.push_back(measure_gpu_hot_bind(
                    ctx,
                    hot_adapters[adapter_id],
                    round,
                    adapter_id,
                    lora_scale));

            clear_adapter(ctx);

            // 路径 2：CPU 温区近似。
            // 先预读文件，让 LoRA GGUF 内容尽量进入系统页缓存，
            // 然后测 init + bind。
            preload_file_to_os_cache(path);

            samples.push_back(measure_file_init_bind(
                    model,
                    ctx,
                    path,
                    "cpu_warm_approx",
                    round,
                    adapter_id,
                    lora_scale));

            // 路径 3：文件加载。
            // 直接调用 init + bind。
            // 由于 OS 文件缓存可能存在，该路径不是严格 SSD cold load。
            samples.push_back(measure_file_init_bind(
                    model,
                    ctx,
                    path,
                    "file_load",
                    round,
                    adapter_id,
                    lora_scale));
        }
    }

    save_samples(samples);

    clear_adapter(ctx);

    for (auto * adapter : hot_adapters) {
        llama_adapter_lora_free(adapter);
    }

    llama_free(ctx);
    llama_model_free(model);

    fprintf(stderr, "LoRA load path latency experiment finished.\n");

    return 0;
}