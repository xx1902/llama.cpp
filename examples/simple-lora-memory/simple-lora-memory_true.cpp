// simple-lora-memory.cpp
//
// 连续 KV 与分页 KV 的真实 GPU 显存生命周期实验
//
// 这个实验复用你之前“波浪图”的思路：
// - 循环创建 context
// - 循环释放 context
// - 循环加载 LoRA
// - 循环释放 LoRA
// - 每一步都通过 nvidia-smi 查询当前真实 GPU 显存
//
// 不同点：
// - 同一套随机操作会运行两次。
// - 第一次使用原始连续 KV。
// - 第二次使用实验性分页 KV。
// - 两次都输出真实 GPU 显存变化 CSV。
//
// 输出：
// output/real_lora_kv_continuous_memory.csv
// output/real_lora_kv_page_memory.csv

#include "llama.h"

#include <windows.h>
#include <psapi.h>

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#pragma comment(lib, "Psapi.lib")

enum class kv_mode {
    continuous,
    paged,
};

enum class op_type {
    create_context,
    free_context,
    load_lora,
    free_lora,
    idle,
};

struct memory_sample {
    std::string mode;
    int step = 0;
    std::string event;
    int live_contexts = 0;
    int live_loras = 0;
    int n_ctx = 0;
    double process_private_mb = 0.0;
    double gpu_used_mb = 0.0;
    double gpu_delta_mb = 0.0;
};

struct planned_op {
    op_type type = op_type::idle;
    int n_ctx = 0;
    int lora_id = 0;
};

struct live_context {
    llama_context * ctx = nullptr;
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

static void record_sample(
        std::vector<memory_sample> & samples,
        kv_mode mode,
        int step,
        const std::string & event,
        int live_contexts,
        int live_loras,
        int n_ctx,
        double gpu_baseline_mb) {
    memory_sample s;

    s.mode = mode_name(mode);
    s.step = step;
    s.event = event;
    s.live_contexts = live_contexts;
    s.live_loras = live_loras;
    s.n_ctx = n_ctx;
    s.process_private_mb = get_process_private_mb();
    s.gpu_used_mb = get_gpu_used_mb();
    s.gpu_delta_mb = s.gpu_used_mb - gpu_baseline_mb;

    samples.push_back(s);

    fprintf(stderr,
            "[%s] step=%d event=%s contexts=%d loras=%d n_ctx=%d ram=%.2f MB gpu=%.2f MB delta=%.2f MB\n",
            s.mode.c_str(),
            s.step,
            s.event.c_str(),
            s.live_contexts,
            s.live_loras,
            s.n_ctx,
            s.process_private_mb,
            s.gpu_used_mb,
            s.gpu_delta_mb);
}

static void save_samples(
        const std::vector<memory_sample> & samples,
        const std::string & csv_name) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/" + csv_name;
    std::ofstream fout(path);

    fout << "mode,step,event,live_contexts,live_loras,n_ctx,"
         << "process_private_mb,gpu_used_mb,gpu_delta_mb\n";

    for (const auto & s : samples) {
        fout << s.mode << ","
             << s.step << ","
             << s.event << ","
             << s.live_contexts << ","
             << s.live_loras << ","
             << s.n_ctx << ","
             << s.process_private_mb << ","
             << s.gpu_used_mb << ","
             << s.gpu_delta_mb << "\n";
    }

    fprintf(stderr, "saved memory trace to %s\n", path.c_str());
}

static std::vector<planned_op> build_workload(
        int n_steps,
        int n_lora) {
    std::vector<planned_op> ops;
    ops.reserve(n_steps);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> op_dist(0.0, 1.0);
    std::uniform_int_distribution<int> ctx_dist(0, 2);
    std::uniform_int_distribution<int> lora_dist(0, n_lora - 1);

    std::vector<int> ctx_sizes = {
        512,
        1024,
        2048,
    };

    for (int i = 0; i < n_steps; i++) {
        const double op = op_dist(rng);

        planned_op item;

        if (op < 0.38) {
            item.type = op_type::create_context;
            item.n_ctx = ctx_sizes[ctx_dist(rng)];
        } else if (op < 0.58) {
            item.type = op_type::free_context;
        } else if (op < 0.80) {
            item.type = op_type::load_lora;
            item.lora_id = lora_dist(rng);
        } else if (op < 0.95) {
            item.type = op_type::free_lora;
        } else {
            item.type = op_type::idle;
        }

        ops.push_back(item);
    }

    return ops;
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

static bool run_small_decode(
        llama_context * ctx,
        const llama_vocab * vocab) {
    std::vector<llama_token> prompt_tokens;

    if (!tokenize_prompt(vocab, "Hello my name is", prompt_tokens)) {
        return false;
    }

    llama_batch batch = llama_batch_init((int) prompt_tokens.size(), 0, 1);

    for (int i = 0; i < (int) prompt_tokens.size(); i++) {
        batch_add(
                batch,
                prompt_tokens[i],
                i,
                0,
                i == (int) prompt_tokens.size() - 1);
    }

    int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        return false;
    }

    const llama_token token = prompt_tokens.back();

    for (int i = 0; i < 8; i++) {
        llama_batch one = llama_batch_init(1, 0, 1);

        batch_add(
                one,
                token,
                (int) prompt_tokens.size() + i,
                0,
                true);

        ret = llama_decode(ctx, one);
        llama_batch_free(one);

        if (ret != 0) {
            return false;
        }
    }

    return true;
}

static void free_all(
        std::vector<live_context> & contexts,
        std::vector<llama_adapter_lora *> & loras) {
    for (auto & item : contexts) {
        if (item.ctx != nullptr) {
            llama_free(item.ctx);
            item.ctx = nullptr;
        }
    }

    contexts.clear();

    for (auto * adapter : loras) {
        if (adapter != nullptr) {
            llama_adapter_lora_free(adapter);
        }
    }

    loras.clear();
}

static bool run_one_mode(
        kv_mode mode,
        const std::string & csv_name,
        const std::string & model_path,
        const std::vector<std::string> & lora_paths,
        const std::vector<planned_op> & ops) {
    const int ngl = 99;
    const int n_batch = 32;
    const int physical_kv_page_size = 16;

    std::vector<memory_sample> samples;
    std::vector<live_context> contexts;
    std::vector<llama_adapter_lora *> loras;

    const double gpu_baseline_mb = get_gpu_used_mb();

    record_sample(
            samples,
            mode,
            0,
            "start",
            0,
            0,
            0,
            gpu_baseline_mb);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(
            model_path.c_str(),
            model_params);

    if (model == nullptr) {
        fprintf(stderr, "[%s] failed to load model: %s\n",
                mode_name(mode),
                model_path.c_str());
        return false;
    }

    record_sample(
            samples,
            mode,
            1,
            "model_loaded",
            0,
            0,
            0,
            gpu_baseline_mb);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    for (int i = 0; i < (int) ops.size(); i++) {
        const int step = i + 2;
        const planned_op & op = ops[i];

        if (op.type == op_type::create_context) {
            llama_context_params ctx_params = llama_context_default_params();

            ctx_params.n_ctx = op.n_ctx;
            ctx_params.n_batch = n_batch;
            ctx_params.n_seq_max = 1;
            ctx_params.no_perf = true;

            if (mode == kv_mode::paged) {
                ctx_params.experimental_physical_paged_kv = true;
                ctx_params.physical_kv_page_size = physical_kv_page_size;
            } else {
                ctx_params.experimental_physical_paged_kv = false;
                ctx_params.physical_kv_page_size = 0;
            }

            llama_context * ctx = llama_init_from_model(model, ctx_params);

            if (ctx != nullptr) {
                // 运行一次小 decode。
                // 这样分页 KV 不只是创建 context，而是真的触发页分配。
                run_small_decode(ctx, vocab);

                live_context item;
                item.ctx = ctx;
                contexts.push_back(item);

                record_sample(
                        samples,
                        mode,
                        step,
                        "context_create_decode_nctx_" + std::to_string(op.n_ctx),
                        (int) contexts.size(),
                        (int) loras.size(),
                        op.n_ctx,
                        gpu_baseline_mb);
            } else {
                record_sample(
                        samples,
                        mode,
                        step,
                        "context_create_failed",
                        (int) contexts.size(),
                        (int) loras.size(),
                        op.n_ctx,
                        gpu_baseline_mb);
            }
        } else if (op.type == op_type::free_context) {
            if (!contexts.empty()) {
                const int idx = step % (int) contexts.size();

                llama_free(contexts[idx].ctx);
                contexts.erase(contexts.begin() + idx);

                record_sample(
                        samples,
                        mode,
                        step,
                        "context_free",
                        (int) contexts.size(),
                        (int) loras.size(),
                        0,
                        gpu_baseline_mb);
            } else {
                record_sample(
                        samples,
                        mode,
                        step,
                        "context_free_skip",
                        0,
                        (int) loras.size(),
                        0,
                        gpu_baseline_mb);
            }
        } else if (op.type == op_type::load_lora) {
            llama_adapter_lora * adapter =
                    llama_adapter_lora_init(model, lora_paths[op.lora_id].c_str());

            if (adapter != nullptr) {
                loras.push_back(adapter);

                record_sample(
                        samples,
                        mode,
                        step,
                        "lora_load_" + std::to_string(op.lora_id),
                        (int) contexts.size(),
                        (int) loras.size(),
                        0,
                        gpu_baseline_mb);
            } else {
                record_sample(
                        samples,
                        mode,
                        step,
                        "lora_load_failed",
                        (int) contexts.size(),
                        (int) loras.size(),
                        0,
                        gpu_baseline_mb);
            }
        } else if (op.type == op_type::free_lora) {
            if (!loras.empty()) {
                const int idx = step % (int) loras.size();

                llama_adapter_lora_free(loras[idx]);
                loras.erase(loras.begin() + idx);

                record_sample(
                        samples,
                        mode,
                        step,
                        "lora_free",
                        (int) contexts.size(),
                        (int) loras.size(),
                        0,
                        gpu_baseline_mb);
            } else {
                record_sample(
                        samples,
                        mode,
                        step,
                        "lora_free_skip",
                        (int) contexts.size(),
                        0,
                        0,
                        gpu_baseline_mb);
            }
        } else {
            record_sample(
                    samples,
                    mode,
                    step,
                    "idle",
                    (int) contexts.size(),
                    (int) loras.size(),
                    0,
                    gpu_baseline_mb);
        }
    }

    free_all(contexts, loras);

    record_sample(
            samples,
            mode,
            (int) ops.size() + 2,
            "cleanup",
            0,
            0,
            0,
            gpu_baseline_mb);

    llama_model_free(model);

    record_sample(
            samples,
            mode,
            (int) ops.size() + 3,
            "model_free",
            0,
            0,
            0,
            gpu_baseline_mb);

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

    const int n_steps = 120;

    ggml_backend_load_all();

    const std::vector<planned_op> ops =
            build_workload(n_steps, (int) lora_paths.size());

    if (!run_one_mode(
                kv_mode::continuous,
                "real_lora_kv_continuous_memory.csv",
                model_path,
                lora_paths,
                ops)) {
        fprintf(stderr, "continuous KV memory experiment failed\n");
        return 1;
    }

    if (!run_one_mode(
                kv_mode::paged,
                "real_lora_kv_page_memory.csv",
                model_path,
                lora_paths,
                ops)) {
        fprintf(stderr, "paged KV memory experiment failed\n");
        return 1;
    }

    fprintf(stderr, "continuous/paged real memory trace finished.\n");

    return 0;
}