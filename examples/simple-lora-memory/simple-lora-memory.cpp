// simple-real-memory-trace.cpp
//
// 真实内存/显存变化测量实验
//
// 本程序会真实调用 llama.cpp API：
// - llama_model_load_from_file
// - llama_init_from_model
// - llama_free
// - llama_adapter_lora_init
// - llama_adapter_lora_free
//
// 每一步记录：
// - GPU 显存占用：通过 nvidia-smi 查询
// - 进程 RAM 占用：通过 Windows GetProcessMemoryInfo 查询
// - 当前活跃 context 数
// - 当前活跃 LoRA adapter 数
//
// 注意：
// 本程序测量的是 llama.cpp 当前实现下的真实资源变化。
// 它没有实现 paged KV cache，因此不能直接声称已经完成真实页块池化替换。

#include "llama.h"

#include <windows.h>
#include <psapi.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#pragma comment(lib, "Psapi.lib")

struct memory_sample {
    int step = 0;
    std::string event;
    int live_contexts = 0;
    int live_loras = 0;
    double process_private_mb = 0.0;
    double gpu_used_mb = 0.0;
    double gpu_delta_mb = 0.0;
};

static const std::string output_dir =
        "/home/agx_orin/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output";

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

    if (!pipe) {
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

static void record_sample(
        std::vector<memory_sample> & samples,
        int step,
        const std::string & event,
        int live_contexts,
        int live_loras,
        double gpu_baseline_mb) {
    memory_sample s;
    s.step = step;
    s.event = event;
    s.live_contexts = live_contexts;
    s.live_loras = live_loras;
    s.process_private_mb = get_process_private_mb();
    s.gpu_used_mb = get_gpu_used_mb();
    s.gpu_delta_mb = s.gpu_used_mb - gpu_baseline_mb;

    samples.push_back(s);

    fprintf(stderr,
            "step=%d event=%s contexts=%d loras=%d ram=%.2f MB gpu=%.2f MB delta=%.2f MB\n",
            s.step,
            s.event.c_str(),
            s.live_contexts,
            s.live_loras,
            s.process_private_mb,
            s.gpu_used_mb,
            s.gpu_delta_mb);
}

static void save_samples(const std::vector<memory_sample> & samples) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/real_memory_trace.csv";
    std::ofstream fout(path);

    fout << "step,event,live_contexts,live_loras,process_private_mb,gpu_used_mb,gpu_delta_mb\n";

    for (const auto & s : samples) {
        fout << s.step << ","
             << s.event << ","
             << s.live_contexts << ","
             << s.live_loras << ","
             << s.process_private_mb << ","
             << s.gpu_used_mb << ","
             << s.gpu_delta_mb << "\n";
    }

    fprintf(stderr, "saved memory trace to %s\n", path.c_str());
}

int main() {
    const std::string model_path =
            "/home/agx_orin/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf";

    std::vector<std::string> lora_paths = {
        "/home/agx_orin/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
        "/home/agx_orin/ecnu_experiment/Model/subliminal-monkey.gguf",
        "/home/agx_orin/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
        "/home/agx_orin/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
    };

    const int ngl = 99;
    const int n_steps = 120;

    ggml_backend_load_all();

    std::vector<memory_sample> samples;

    const double gpu_baseline_mb = get_gpu_used_mb();

    record_sample(samples, 0, "start", 0, 0, gpu_baseline_mb);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == nullptr) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    record_sample(samples, 1, "model_loaded", 0, 0, gpu_baseline_mb);

    std::vector<llama_context *> contexts;
    std::vector<llama_adapter_lora *> loras;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    std::uniform_int_distribution<int> ctx_dist(0, 2);
    std::uniform_int_distribution<int> lora_dist(0, (int) lora_paths.size() - 1);

    std::vector<int> ctx_sizes = {
        128,
        256,
        512,
    };

    for (int step = 2; step < n_steps + 2; step++) {
        const double op = op_dist(rng);

        if (op < 0.35 || contexts.empty()) {
            const int n_ctx = ctx_sizes[ctx_dist(rng)];

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = n_ctx;
            ctx_params.n_batch = 16;
            ctx_params.no_perf = true;

            llama_context * ctx = llama_init_from_model(model, ctx_params);

            if (ctx != nullptr) {
                contexts.push_back(ctx);
                record_sample(
                        samples,
                        step,
                        "context_create_nctx_" + std::to_string(n_ctx),
                        (int) contexts.size(),
                        (int) loras.size(),
                        gpu_baseline_mb);
            } else {
                record_sample(
                        samples,
                        step,
                        "context_create_failed",
                        (int) contexts.size(),
                        (int) loras.size(),
                        gpu_baseline_mb);
            }
        } else if (op < 0.55 && !contexts.empty()) {
            const int idx = (int) (rng() % contexts.size());

            llama_free(contexts[idx]);
            contexts.erase(contexts.begin() + idx);

            record_sample(
                    samples,
                    step,
                    "context_free",
                    (int) contexts.size(),
                    (int) loras.size(),
                    gpu_baseline_mb);
        } else if (op < 0.78) {
            const int lora_id = lora_dist(rng);

            llama_adapter_lora * adapter =
                    llama_adapter_lora_init(model, lora_paths[lora_id].c_str());

            if (adapter != nullptr) {
                loras.push_back(adapter);

                record_sample(
                        samples,
                        step,
                        "lora_load_" + std::to_string(lora_id),
                        (int) contexts.size(),
                        (int) loras.size(),
                        gpu_baseline_mb);
            } else {
                record_sample(
                        samples,
                        step,
                        "lora_load_failed",
                        (int) contexts.size(),
                        (int) loras.size(),
                        gpu_baseline_mb);
            }
        } else if (!loras.empty()) {
            const int idx = (int) (rng() % loras.size());

            llama_adapter_lora_free(loras[idx]);
            loras.erase(loras.begin() + idx);

            record_sample(
                    samples,
                    step,
                    "lora_free",
                    (int) contexts.size(),
                    (int) loras.size(),
                    gpu_baseline_mb);
        } else {
            record_sample(
                    samples,
                    step,
                    "idle",
                    (int) contexts.size(),
                    (int) loras.size(),
                    gpu_baseline_mb);
        }
    }

    for (auto * ctx : contexts) {
        llama_free(ctx);
    }

    contexts.clear();

    for (auto * adapter : loras) {
        llama_adapter_lora_free(adapter);
    }

    loras.clear();

    record_sample(samples, n_steps + 2, "cleanup", 0, 0, gpu_baseline_mb);

    llama_model_free(model);

    record_sample(samples, n_steps + 3, "model_free", 0, 0, gpu_baseline_mb);

    save_samples(samples);

    return 0;
}