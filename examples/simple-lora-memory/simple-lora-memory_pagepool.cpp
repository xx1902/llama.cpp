// simple-lora-memory.cpp
//
// 连续 KV 与分页 KV 的真实 GPU 显存生命周期采样实验
//
// 本程序会分别运行两组真实实验：
//
// 1. continuous_kv：原始连续 KV cache。
//    输出：real_lora_kv_gpu_memory.csv
//
// 2. paged_kv：实验性分页 KV cache。
//    输出：real_lora_kv_page_memory.csv
//
// 数据来源：
// - GPU 显存：通过 nvidia-smi 读取当前 GPU memory.used。
// - GPU 显存增量：当前 gpu_used_mb - 当前实验开始时 gpu_used_mb。
// - 进程内存：Windows API GetProcessMemoryInfo。
// - 每完成一个生命周期事件，就写一条采样记录。
//
// 生命周期事件：
// - start
// - model_loaded
// - context_created
// - lora_loaded
// - lora_bound
// - prompt_eval
// - decode_step
// - before_release
// - released
// - model_released
//
// 注意：
// - 为了让显存曲线更明显，这里会保留多个 context / LoRA adapter，最后统一释放。
// - 如果显存不足，可以降低 n_requests。
// - 分页 KV 是否真正节省驱动层显存，还取决于你底层 paged KV 是否真的按页申请 GPU buffer。

#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

enum class kv_mode {
    continuous,
    paged,
};

struct memory_sample {
    std::string mode;

    int step = 0;
    int request_id = -1;
    int generated_tokens = 0;

    std::string event;

    double gpu_used_mb = 0.0;
    double gpu_delta_mb = 0.0;

    double process_used_mb = 0.0;
    double process_delta_mb = 0.0;
};

struct live_request {
    llama_context * ctx = nullptr;
    llama_adapter_lora * adapter = nullptr;
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
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;

    if (GetProcessMemoryInfo(
                GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc),
                sizeof(pmc))) {
        return (double) pmc.PrivateUsage / 1024.0 / 1024.0;
    }
#endif

    return 0.0;
}

static double get_gpu_used_mb() {
#ifdef _WIN32
    FILE * pipe = _popen(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            "r");
#else
    FILE * pipe = popen(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            "r");
#endif

    if (pipe == nullptr) {
        return 0.0;
    }

    char buffer[256];
    double value = 0.0;

    if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        value = atof(buffer);
    }

#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif

    return value;
}

static void add_sample(
        std::vector<memory_sample> & samples,
        kv_mode mode,
        int & step,
        int request_id,
        const std::string & event,
        int generated_tokens,
        double gpu_base_mb,
        double process_base_mb) {
    memory_sample s;

    s.mode = mode_name(mode);
    s.step = step++;
    s.request_id = request_id;
    s.event = event;
    s.generated_tokens = generated_tokens;

    s.gpu_used_mb = get_gpu_used_mb();
    s.gpu_delta_mb = s.gpu_used_mb - gpu_base_mb;

    s.process_used_mb = get_process_private_mb();
    s.process_delta_mb = s.process_used_mb - process_base_mb;

    samples.push_back(s);

    fprintf(stderr,
            "[%s] step=%d event=%s request=%d gpu=%.2f MB gpu_delta=%.2f MB process=%.2f MB\n",
            s.mode.c_str(),
            s.step,
            s.event.c_str(),
            s.request_id,
            s.gpu_used_mb,
            s.gpu_delta_mb,
            s.process_used_mb);
}

static bool tokenize_prompt(
        const llama_vocab * vocab,
        const std::string & prompt,
        std::vector<llama_token> & tokens) {
    const int n_tokens = -llama_tokenize(
            vocab,
            prompt.c_str(),
            (int) prompt.size(),
            nullptr,
            0,
            true,
            true);

    if (n_tokens <= 0) {
        fprintf(stderr, "failed to get prompt token count\n");
        return false;
    }

    tokens.resize(n_tokens);

    const int ret = llama_tokenize(
            vocab,
            prompt.c_str(),
            (int) prompt.size(),
            tokens.data(),
            (int) tokens.size(),
            true,
            true);

    if (ret < 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        return false;
    }

    return true;
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

static bool eval_prompt(
        llama_context * ctx,
        const std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);

    for (int i = 0; i < (int) tokens.size(); i++) {
        batch_add(
                batch,
                tokens[i],
                i,
                0,
                i == (int) tokens.size() - 1);
    }

    const int ret = llama_decode(ctx, batch);

    llama_batch_free(batch);

    if (ret != 0) {
        fprintf(stderr, "prompt eval failed\n");
        return false;
    }

    return true;
}

static bool decode_one_token(
        llama_context * ctx,
        llama_token token,
        int pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);

    batch_add(
            batch,
            token,
            pos,
            0,
            true);

    const int ret = llama_decode(ctx, batch);

    llama_batch_free(batch);

    if (ret != 0) {
        fprintf(stderr, "decode one token failed\n");
        return false;
    }

    return true;
}

static bool bind_lora(
        llama_context * ctx,
        llama_adapter_lora * adapter,
        float scale) {
    llama_adapter_lora * adapters[] = { adapter };
    float scales[] = { scale };

    const int ret = llama_set_adapters_lora(
            ctx,
            adapters,
            1,
            scales);

    if (ret != 0) {
        fprintf(stderr, "failed to bind LoRA adapter\n");
        return false;
    }

    return true;
}

static void save_samples(
        const std::vector<memory_sample> & samples,
        const std::string & file_name) {
    std::filesystem::create_directories(output_dir);

    const std::string csv_path = output_dir + "/" + file_name;

    std::ofstream fout(csv_path);

    fout << "mode,step,request_id,event,generated_tokens,"
         << "gpu_used_mb,gpu_delta_mb,"
         << "process_used_mb,process_delta_mb\n";

    for (const auto & s : samples) {
        fout << s.mode << ","
             << s.step << ","
             << s.request_id << ","
             << s.event << ","
             << s.generated_tokens << ","
             << s.gpu_used_mb << ","
             << s.gpu_delta_mb << ","
             << s.process_used_mb << ","
             << s.process_delta_mb << "\n";
    }

    fout.close();

    fprintf(stderr, "saved memory trace to %s\n", csv_path.c_str());
}

static void release_live_requests(std::vector<live_request> & live_requests) {
    for (auto & req : live_requests) {
        if (req.ctx != nullptr) {
            llama_set_adapters_lora(req.ctx, nullptr, 0, nullptr);
        }

        if (req.adapter != nullptr) {
            llama_adapter_lora_free(req.adapter);
            req.adapter = nullptr;
        }

        if (req.ctx != nullptr) {
            llama_free(req.ctx);
            req.ctx = nullptr;
        }
    }

    live_requests.clear();
}

static bool run_memory_experiment(
        kv_mode mode,
        const std::string & csv_name,
        const std::string & model_path,
        const std::vector<std::string> & lora_paths,
        const std::vector<std::string> & prompts) {
    const int ngl = 99;
    const int n_ctx = 256;
    const int n_batch = 32;
    const int n_seq_max = 1;

    // 请求越多，保留的 context / LoRA 越多，显存变化越明显。
    // 如果显存不足，改成 4。
    const int n_requests = 8;

    // 每个请求生成几个 token，用于触发真实 KV 写入路径。
    const int n_decode_per_request = 4;

    // 分页 KV 的页大小。
    // 如果你之前 multi-page 已经跑通，可以用 16。
    // 如果还不稳定，可以临时改成 32。
    const int physical_kv_page_size = 16;

    const float lora_scale = 1.0f;

    std::vector<memory_sample> samples;
    std::vector<live_request> live_requests;

    int step = 0;

    // 每个实验单独取 baseline。
    // 这样 continuous_kv 和 paged_kv 的 gpu_delta_mb 可以公平对比。
    const double gpu_base_mb = get_gpu_used_mb();
    const double process_base_mb = get_process_private_mb();

    add_sample(
            samples,
            mode,
            step,
            -1,
            "start",
            0,
            gpu_base_mb,
            process_base_mb);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(
            model_path.c_str(),
            model_params);

    if (model == nullptr) {
        fprintf(stderr, "[%s] failed to load model: %s\n", mode_name(mode), model_path.c_str());
        return false;
    }

    add_sample(
            samples,
            mode,
            step,
            -1,
            "model_loaded",
            0,
            gpu_base_mb,
            process_base_mb);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    live_requests.reserve(n_requests);

    for (int request_id = 0; request_id < n_requests; request_id++) {
        fprintf(stderr,
                "\n========== %s request %d ==========\n",
                mode_name(mode),
                request_id);

        const int lora_id = request_id % (int) lora_paths.size();
        const int prompt_id = request_id % (int) prompts.size();

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_batch;
        ctx_params.n_seq_max = n_seq_max;
        ctx_params.no_perf = true;

        if (mode == kv_mode::paged) {
            // 这里使用你已经在 llama.cpp 中新增的实验性分页 KV 参数。
            ctx_params.experimental_physical_paged_kv = true;
            ctx_params.physical_kv_page_size = physical_kv_page_size;
        } else {
            ctx_params.experimental_physical_paged_kv = false;
            ctx_params.physical_kv_page_size = 0;
        }

        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) {
            fprintf(stderr, "[%s] failed to create context for request %d\n",
                    mode_name(mode),
                    request_id);
            break;
        }

        add_sample(
                samples,
                mode,
                step,
                request_id,
                "context_created",
                0,
                gpu_base_mb,
                process_base_mb);

        llama_adapter_lora * adapter =
                llama_adapter_lora_init(model, lora_paths[lora_id].c_str());

        if (adapter == nullptr) {
            fprintf(stderr, "[%s] failed to load LoRA adapter: %s\n",
                    mode_name(mode),
                    lora_paths[lora_id].c_str());

            llama_free(ctx);
            break;
        }

        add_sample(
                samples,
                mode,
                step,
                request_id,
                "lora_loaded",
                0,
                gpu_base_mb,
                process_base_mb);

        if (!bind_lora(ctx, adapter, lora_scale)) {
            llama_adapter_lora_free(adapter);
            llama_free(ctx);
            break;
        }

        add_sample(
                samples,
                mode,
                step,
                request_id,
                "lora_bound",
                0,
                gpu_base_mb,
                process_base_mb);

        std::vector<llama_token> prompt_tokens;
        if (!tokenize_prompt(vocab, prompts[prompt_id], prompt_tokens)) {
            llama_adapter_lora_free(adapter);
            llama_free(ctx);
            break;
        }

        if (!eval_prompt(ctx, prompt_tokens)) {
            llama_adapter_lora_free(adapter);
            llama_free(ctx);
            break;
        }

        add_sample(
                samples,
                mode,
                step,
                request_id,
                "prompt_eval",
                0,
                gpu_base_mb,
                process_base_mb);

        const llama_token repeated_token = prompt_tokens.back();

        for (int i = 0; i < n_decode_per_request; i++) {
            const int pos = (int) prompt_tokens.size() + i;

            if (!decode_one_token(ctx, repeated_token, pos)) {
                break;
            }

            add_sample(
                    samples,
                    mode,
                    step,
                    request_id,
                    "decode_step",
                    i + 1,
                    gpu_base_mb,
                    process_base_mb);
        }

        // 这里故意不立即释放。
        // 目的是让多个 LoRA/KV 生命周期同时存在，从真实显存上看到累积效果。
        live_request req;
        req.ctx = ctx;
        req.adapter = adapter;
        live_requests.push_back(req);
    }

    add_sample(
            samples,
            mode,
            step,
            -1,
            "before_release",
            0,
            gpu_base_mb,
            process_base_mb);

    release_live_requests(live_requests);

    add_sample(
            samples,
            mode,
            step,
            -1,
            "released",
            0,
            gpu_base_mb,
            process_base_mb);

    llama_model_free(model);

    add_sample(
            samples,
            mode,
            step,
            -1,
            "model_released",
            0,
            gpu_base_mb,
            process_base_mb);

    save_samples(samples, csv_name);

    return true;
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

    std::vector<std::string> prompts = {
        "Hello my name is",
        "The future of marketing is",
        "A good product slogan is",
        "To attract customers, we should",
    };

    ggml_backend_load_all();

    fprintf(stderr, "\n========== running continuous KV experiment ==========\n");

    if (!run_memory_experiment(
                kv_mode::continuous,
                "real_lora_kv_gpu_memory.csv",
                model_path,
                lora_paths,
                prompts)) {
        fprintf(stderr, "continuous KV experiment failed\n");
        return 1;
    }

    fprintf(stderr, "\n========== running paged KV experiment ==========\n");

    if (!run_memory_experiment(
                kv_mode::paged,
                "real_lora_kv_page_memory.csv",
                model_path,
                lora_paths,
                prompts)) {
        fprintf(stderr, "paged KV experiment failed\n");
        return 1;
    }

    fprintf(stderr, "real continuous/paged KV GPU memory experiments finished.\n");

    return 0;
}