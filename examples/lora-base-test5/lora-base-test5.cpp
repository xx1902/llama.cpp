// Focused KV-delta scheduling experiment.
//
// The implementation reuses the data/model helpers from test4, but keeps the
// measured region small and explicit.  "cuda-overlap" is a real overlap
// experiment: the fused delta-Q8 CUDA kernel is submitted on a low-priority
// stream while child suffix inference runs on the foreground stream.  It is
// deliberately not called a single inference+delta kernel because llama's
// inference graph contains multiple independent CUDA/cuBLAS kernels.  The
// "inference-only" mode is the no-delta baseline for the same child suffix.

#define main llama_lora_base_test4_embedded_main
#include "../lora-base-test4/lora-base-test4.cpp"
#undef main

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>

namespace fs = std::filesystem;

struct test5_options {
    std::string model_path = "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/Qwen2.5-1.5B-Instruct-f16.gguf";
    std::string workload_dir = "D:/ecnu_experiment/datasets/mobilora_workloads_87_original";
    std::string lora_config_path;
    std::string output_dir = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base-test5/output";
    int max_pairs = 4;
    int repeats = 2;
    int prefix_chunk = 128;
    int suffix_tokens = 16;
    int n_ctx = 4096;
    int n_batch = 256;
    int n_ubatch = 64;
    int n_gpu_layers = 99;
    std::string mode = "all";
};

struct test5_row {
    std::string mode;
    int repeat = 0;
    std::string pair_name;
    std::string group_name;
    int anchor_lora_id = -1;
    int child_lora_id = -1;
    int prefix_tokens = 0;
    int suffix_tokens = 0;
    double anchor_eval_ms = 0.0;
    double child_eval_ms = 0.0;
    double probe_ms = 0.0;
    double delta_submit_ms = 0.0;
    double delta_build_ms = 0.0;
    double delta_finish_wait_ms = 0.0;
    double foreground_ms = 0.0;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    double overlap_window_ms = 0.0;
    double saved_rate = 0.0;
    double reconstruction_cos = 0.0;
    double reconstruction_l2 = 0.0;
    unsigned long long full_kv_bytes = 0;
    unsigned long long delta_bytes = 0;
    int delta_build_ok = 0;
    int reconstruction_ok = 0;
    std::string status;
};

static bool test5_int(const char * value, int & out) {
    try {
        out = std::stoi(value);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_test5_options(int argc, char ** argv, test5_options & options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            fprintf(stderr,
                    "Usage: %s [--model PATH] [--workload-dir DIR] [--lora-config FILE]\n"
                    "  [--output-dir DIR] [--max-pairs N] [--repeats N]\n"
                    "  [--prefix-chunk N] [--suffix-tokens N] [--n-ctx N]\n"
                    "  [--n-batch N] [--n-ubatch N] [--n-gpu-layers N]\n"
                    "  [--mode all|inference-only|cpu-sync|cuda-sync|cuda-overlap]\n",
                    argv[0]);
            return false;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", arg.c_str());
            return false;
        }
        const char * value = argv[++i];
        if (arg == "--model") options.model_path = value;
        else if (arg == "--workload-dir") options.workload_dir = value;
        else if (arg == "--lora-config") options.lora_config_path = value;
        else if (arg == "--output-dir") options.output_dir = value;
        else if (arg == "--max-pairs") { if (!test5_int(value, options.max_pairs)) return false; }
        else if (arg == "--repeats") { if (!test5_int(value, options.repeats)) return false; }
        else if (arg == "--prefix-chunk") { if (!test5_int(value, options.prefix_chunk)) return false; }
        else if (arg == "--suffix-tokens") { if (!test5_int(value, options.suffix_tokens)) return false; }
        else if (arg == "--n-ctx") { if (!test5_int(value, options.n_ctx)) return false; }
        else if (arg == "--n-batch") { if (!test5_int(value, options.n_batch)) return false; }
        else if (arg == "--n-ubatch") { if (!test5_int(value, options.n_ubatch)) return false; }
        else if (arg == "--n-gpu-layers") { if (!test5_int(value, options.n_gpu_layers)) return false; }
        else if (arg == "--mode") options.mode = value;
        else {
            fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return false;
        }
    }

    if (options.lora_config_path.empty()) {
        options.lora_config_path = options.workload_dir + "/lora_groups.json";
    }
    if (options.max_pairs <= 0 || options.repeats <= 0 || options.prefix_chunk <= 0 ||
            options.suffix_tokens < 0 || options.n_ctx <= 0 || options.n_batch <= 0 ||
            options.n_ubatch <= 0 || (options.mode != "all" &&
            options.mode != "inference-only" &&
            options.mode != "cpu-sync" && options.mode != "cuda-sync" &&
            options.mode != "cuda-overlap")) {
        fprintf(stderr, "invalid test5 options\n");
        return false;
    }
    return true;
}

static experiment_options make_experiment_options(const test5_options & options,
                                                   const std::string & mode) {
    experiment_options result;
    result.model_path = options.model_path;
    result.workload_dir = options.workload_dir;
    result.lora_config_path = options.lora_config_path;
    result.output_dir = options.output_dir;
    result.n_ctx = options.n_ctx;
    result.n_batch = options.n_batch;
    result.n_ubatch = options.n_ubatch;
    result.n_gpu_layers = options.n_gpu_layers;
    result.n_predict = 1;
    result.delta_execution = mode;
    return result;
}

static std::vector<std::string> test5_modes(const std::string & mode) {
    if (mode == "all") return { "inference-only", "cpu-sync", "cuda-sync", "cuda-overlap" };
    return { mode };
}

static std::vector<llama_token> make_suffix(const tokenized_request & tokens, int count) {
    std::vector<llama_token> suffix = tokens.suffix;
    if ((int) suffix.size() > count) suffix.resize(count);
    if (count > 0 && suffix.empty()) {
        const llama_token filler = tokens.prefix.empty() ? 0 : tokens.prefix.back();
        suffix.assign(count, filler);
    }
    return suffix;
}

static bool finish_and_validate_delta(
        llama_context * context,
        llama_seq_id anchor_seq,
        llama_seq_id child_full_seq,
        llama_seq_id delta_seq,
        llama_seq_id recon_seq,
        int prefix_tokens,
        test5_row & row) {
    llama_kv_delta_materialize_stats materialized = {};
    const bool materialize_ok = llama_kv_seq_delta_materialize_branch(
            context, anchor_seq, delta_seq, recon_seq, 0, prefix_tokens, &materialized);
    if (!materialize_ok) {
        row.status = "reconstruction_failed";
        return false;
    }

    llama_kv_delta_probe_stats probe = {};
    const bool probe_ok = llama_kv_seq_delta_probe(
            context, child_full_seq, recon_seq, 0, prefix_tokens, &probe);
    row.reconstruction_ok = probe_ok && std::string(probe.probe_status) == "ok";
    row.reconstruction_cos = probe.kv_cos_avg;
    row.reconstruction_l2 = probe.kv_l2_avg;
    if (!row.reconstruction_ok) row.status = "reconstruction_probe_failed";
    return row.reconstruction_ok;
}

static test5_row run_test5_trial(
        llama_model * model,
        const llama_vocab * vocab,
        const test5_options & options,
        const delta_pair_item & pair,
        const dataset_request & anchor_request,
        const dataset_request & child_request,
        lora_runtime & anchor_lora,
        lora_runtime & child_lora,
        const std::string & mode,
        int repeat) {
    test5_row row;
    row.mode = mode;
    row.repeat = repeat;
    row.pair_name = pair.pair_name;
    row.group_name = pair.group_name;
    row.anchor_lora_id = pair.anchor_lora_id;
    row.child_lora_id = pair.child_lora_id;
    row.status = "started";

    experiment_options run_options = make_experiment_options(options, mode);
    llama_context * context = create_context(model, run_options, 4);
    if (context == nullptr) {
        row.status = "context_failed";
        return row;
    }
    llama_memory_t memory = llama_get_memory(context);

    tokenized_request anchor_tokens;
    tokenized_request child_tokens;
    if (!tokenize_request(vocab, anchor_request, anchor_tokens) ||
            !tokenize_request(vocab, child_request, child_tokens) ||
            anchor_tokens.prefix.empty() || child_tokens.prefix.empty() ||
            anchor_tokens.prefix != child_tokens.prefix) {
        row.status = "tokenize_or_prefix_failed";
        llama_free(context);
        return row;
    }
    row.prefix_tokens = (int) anchor_tokens.prefix.size();
    const std::vector<llama_token> suffix = make_suffix(child_tokens, options.suffix_tokens);
    row.suffix_tokens = (int) suffix.size();

    llama_memory_clear(memory, true);
    clear_lora(context);

    bind_lora(context, anchor_lora);
    double start = now_ms();
    if (!eval_tokens(context, anchor_tokens.prefix, 0, 0, options.prefix_chunk)) {
        row.status = "anchor_eval_failed";
        llama_free(context);
        return row;
    }
    row.anchor_eval_ms = now_ms() - start;

    bind_lora(context, child_lora);
    start = now_ms();
    if (!eval_tokens(context, child_tokens.prefix, 1, 0, options.prefix_chunk)) {
        row.status = "child_eval_failed";
        llama_free(context);
        return row;
    }
    row.child_eval_ms = now_ms() - start;

    // Keep the original probe as an explicit synchronization/validation point
    // for delta modes. The no-delta baseline omits it so its measured region
    // contains only child suffix inference.
    if (mode != "inference-only") {
        llama_kv_delta_probe_stats probe = {};
        start = now_ms();
        const bool probe_ok = llama_kv_seq_delta_probe(
                context, 0, 1, 0, row.prefix_tokens, &probe);
        row.probe_ms = now_ms() - start;
        if (!probe_ok) {
            row.status = "probe_failed";
            llama_free(context);
            return row;
        }
    }

    const llama_seq_id delta_seq = 2;
    const llama_seq_id recon_seq = 3;
    llama_kv_delta_branch_stats branch = {};
    const double measured_start = now_ms();
    uint64_t async_job = 0;
    bool build_ok = false;

    if (mode == "inference-only") {
        const double foreground_start = now_ms();
        if (!suffix.empty() && !eval_tokens(
                context, suffix, 1, row.prefix_tokens, options.prefix_chunk)) {
            row.status = "suffix_eval_failed";
            llama_free(context);
            return row;
        }
        if (!decode_one(context, suffix.empty() ? child_tokens.prefix.back() : suffix.back(),
                1, row.prefix_tokens + row.suffix_tokens)) {
            row.status = "decode_failed";
            llama_free(context);
            return row;
        }
        row.foreground_ms = now_ms() - foreground_start;
        row.ttft_ms = now_ms() - measured_start;
        row.total_ms = now_ms() - measured_start;
        row.status = "ok";

        // This baseline intentionally has no delta payload or reconstruction
        // result.  Zero values keep the CSV schema stable; consumers should
        // use the mode name to interpret them as not applicable.
        row.delta_build_ok = 0;
        row.reconstruction_ok = 0;
        llama_free(context);
        return row;
    }

    if (mode == "cuda-overlap") {
        const double submit_start = now_ms();
        build_ok = llama_kv_seq_delta_build_branch_async(
                context, 0, 1, delta_seq, 0, row.prefix_tokens,
                pair.anchor_request_id, pair.child_request_id, &async_job);
        row.delta_submit_ms = now_ms() - submit_start;
        if (!build_ok) {
            row.status = "async_submit_failed";
            llama_free(context);
            return row;
        }

        const double foreground_start = now_ms();
        if (!suffix.empty() && !eval_tokens(
                context, suffix, 1, row.prefix_tokens, options.prefix_chunk)) {
            llama_kv_seq_delta_build_branch_cancel(context, async_job);
            row.status = "suffix_eval_failed";
            llama_free(context);
            return row;
        }
        if (!decode_one(context, suffix.empty() ? child_tokens.prefix.back() : suffix.back(),
                1, row.prefix_tokens + row.suffix_tokens)) {
            llama_kv_seq_delta_build_branch_cancel(context, async_job);
            row.status = "decode_failed";
            llama_free(context);
            return row;
        }
        row.foreground_ms = now_ms() - foreground_start;
        row.ttft_ms = now_ms() - measured_start;

        const double finish_start = now_ms();
        build_ok = llama_kv_seq_delta_build_branch_finish(
                context, async_job, delta_seq, &branch);
        row.delta_finish_wait_ms = now_ms() - finish_start;
        row.overlap_window_ms = row.foreground_ms;
    } else {
        const double delta_start = now_ms();
        build_ok = mode == "cpu-sync"
                ? llama_kv_seq_delta_build_branch_cpu(
                        context, 0, 1, delta_seq, 0, row.prefix_tokens,
                        pair.anchor_request_id, pair.child_request_id, &branch)
                : llama_kv_seq_delta_build_branch(
                        context, 0, 1, delta_seq, 0, row.prefix_tokens,
                        pair.anchor_request_id, pair.child_request_id, &branch);
        row.delta_build_ms = now_ms() - delta_start;
        if (!build_ok) {
            row.status = "delta_build_failed";
            llama_free(context);
            return row;
        }

        const double foreground_start = now_ms();
        if (!suffix.empty() && !eval_tokens(
                context, suffix, 1, row.prefix_tokens, options.prefix_chunk)) {
            row.status = "suffix_eval_failed";
            llama_free(context);
            return row;
        }
        if (!decode_one(context, suffix.empty() ? child_tokens.prefix.back() : suffix.back(),
                1, row.prefix_tokens + row.suffix_tokens)) {
            row.status = "decode_failed";
            llama_free(context);
            return row;
        }
        row.foreground_ms = now_ms() - foreground_start;
        row.ttft_ms = now_ms() - measured_start;
    }

    row.total_ms = now_ms() - measured_start;
    row.delta_build_ok = build_ok ? 1 : 0;
    row.full_kv_bytes = branch.full_kv_bytes_equivalent;
    row.delta_bytes = branch.delta_q8_bytes + branch.delta_scale_bytes;
    row.saved_rate = branch.logical_saved_rate;
    row.status = build_ok ? "ok" : "delta_finish_failed";

    // Allocate an independent destination sequence for validation. This is
    // intentionally outside the measured interval because the materializer is
    // the old host-side path, not the scheduling path being compared.
    bind_lora(context, child_lora);
    if (eval_tokens(context, child_tokens.prefix, recon_seq, 0, options.prefix_chunk)) {
        finish_and_validate_delta(
                context, 0, 1, delta_seq, recon_seq, row.prefix_tokens, row);
    } else {
        row.status = "reconstruction_context_failed";
    }
    llama_kv_seq_delta_remove_branch(context, delta_seq);
    llama_free(context);
    return row;
}

static void write_test5_csv(const std::string & path, const std::vector<test5_row> & rows) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream output(path, std::ios::trunc);
    output << "mode,repeat,pair_name,group_name,anchor_lora_id,child_lora_id,prefix_tokens,suffix_tokens,"
              "anchor_eval_ms,child_eval_ms,probe_ms,delta_submit_ms,delta_build_ms,delta_finish_wait_ms,"
              "foreground_ms,ttft_ms,total_ms,overlap_window_ms,saved_rate,reconstruction_cos,reconstruction_l2,"
              "full_kv_bytes,delta_bytes,delta_build_ok,reconstruction_ok,status\n";
    output << std::setprecision(10);
    for (const auto & row : rows) {
        output << csv_escape(row.mode) << ',' << row.repeat << ',' << csv_escape(row.pair_name) << ','
               << csv_escape(row.group_name) << ',' << row.anchor_lora_id << ',' << row.child_lora_id << ','
               << row.prefix_tokens << ',' << row.suffix_tokens << ',' << row.anchor_eval_ms << ','
               << row.child_eval_ms << ',' << row.probe_ms << ',' << row.delta_submit_ms << ','
               << row.delta_build_ms << ',' << row.delta_finish_wait_ms << ',' << row.foreground_ms << ','
               << row.ttft_ms << ',' << row.total_ms << ',' << row.overlap_window_ms << ','
               << row.saved_rate << ',' << row.reconstruction_cos << ',' << row.reconstruction_l2 << ','
               << row.full_kv_bytes << ',' << row.delta_bytes << ',' << row.delta_build_ok << ','
               << row.reconstruction_ok << ',' << csv_escape(row.status) << '\n';
    }
}

static void write_test5_summary(const std::string & path, const std::vector<test5_row> & rows) {
    fs::create_directories(fs::path(path).parent_path());
    std::map<std::string, std::vector<const test5_row *>> grouped;
    for (const auto & row : rows) grouped[row.mode].push_back(&row);
    std::ofstream output(path, std::ios::trunc);
    output << "mode,count,ok_count,mean_delta_ms,mean_ttft_ms,mean_total_ms,mean_saved_rate,"
              "mean_reconstruction_cos,mean_reconstruction_l2\n";
    output << std::setprecision(10);
    for (const auto & [mode, items] : grouped) {
        int ok_count = 0;
        double delta_ms = 0.0;
        double ttft_ms = 0.0;
        double total_ms = 0.0;
        double saved_rate = 0.0;
        double recon_cos = 0.0;
        double recon_l2 = 0.0;
        for (const test5_row * row : items) {
            ok_count += row->status == "ok" ? 1 : 0;
            delta_ms += row->delta_build_ms + row->delta_submit_ms + row->delta_finish_wait_ms;
            ttft_ms += row->ttft_ms;
            total_ms += row->total_ms;
            saved_rate += row->saved_rate;
            recon_cos += row->reconstruction_cos;
            recon_l2 += row->reconstruction_l2;
        }
        const double count = std::max<size_t>(1, items.size());
        output << csv_escape(mode) << ',' << items.size() << ',' << ok_count << ','
               << delta_ms / count << ',' << ttft_ms / count << ',' << total_ms / count << ','
               << saved_rate / count << ',' << recon_cos / count << ',' << recon_l2 / count << '\n';
    }
}

int main(int argc, char ** argv) {
    test5_options options;
    if (!parse_test5_options(argc, argv, options)) return argc > 1 ? 1 : 0;

    const std::string pairs_path = options.workload_dir + "/delta/delta_pairs.jsonl";
    const std::string requests_path = options.workload_dir + "/delta/delta_requests.jsonl";
    const std::vector<delta_pair_item> all_pairs = load_delta_pairs(pairs_path);
    const std::vector<dataset_request> request_rows = load_requests(requests_path);
    std::unordered_map<int, dataset_request> requests;
    for (const auto & request : request_rows) requests[request.request_id] = request;

    std::vector<delta_pair_item> pairs;
    std::set<int> needed_loras;
    for (const auto & pair : all_pairs) {
        if ((int) pairs.size() >= options.max_pairs) break;
        const auto anchor = requests.find(pair.anchor_request_id);
        const auto child = requests.find(pair.child_request_id);
        if (anchor == requests.end() || child == requests.end() ||
                anchor->second.common_prefix_text != child->second.common_prefix_text) continue;
        pairs.push_back(pair);
        needed_loras.insert(pair.anchor_lora_id);
        needed_loras.insert(pair.child_lora_id);
    }
    if (pairs.empty()) {
        fprintf(stderr, "no valid delta pairs found in %s\n", pairs_path.c_str());
        return 1;
    }

    std::vector<lora_runtime> lora_list = load_lora_config(options.lora_config_path);
    std::unordered_map<int, lora_runtime *> loras;
    for (auto & lora : lora_list) {
        if (needed_loras.find(lora.lora_id) == needed_loras.end()) continue;
        if (lora.adapter_path.empty()) {
            fprintf(stderr, "missing adapter path for LoRA %d (%s)\n",
                    lora.lora_id, lora.logical_name.c_str());
            continue;
        }
        loras[lora.lora_id] = &lora;
    }

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(options.model_path.c_str(), model_params);
    if (model == nullptr) {
        fprintf(stderr, "failed to load model: %s\n", options.model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    for (auto & lora : lora_list) {
        if (needed_loras.find(lora.lora_id) == needed_loras.end() || lora.adapter_path.empty()) continue;
        const double load_start = now_ms();
        lora.adapter = llama_adapter_lora_init(model, lora.adapter_path.c_str());
        lora.initial_load_ms = now_ms() - load_start;
        if (lora.adapter == nullptr) {
            fprintf(stderr, "failed to load adapter %d: %s\n",
                    lora.lora_id, lora.adapter_path.c_str());
        }
    }

    std::vector<test5_row> rows;
    const std::vector<std::string> modes = test5_modes(options.mode);
    for (const std::string & mode : modes) {
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            for (const auto & pair : pairs) {
                const auto anchor_request = requests.find(pair.anchor_request_id);
                const auto child_request = requests.find(pair.child_request_id);
                const auto anchor_lora = loras.find(pair.anchor_lora_id);
                const auto child_lora = loras.find(pair.child_lora_id);
                if (anchor_request == requests.end() || child_request == requests.end() ||
                        anchor_lora == loras.end() || child_lora == loras.end() ||
                        anchor_lora->second->adapter == nullptr || child_lora->second->adapter == nullptr) {
                    continue;
                }
                test5_row row = run_test5_trial(
                        model, vocab, options, pair,
                        anchor_request->second, child_request->second,
                        *anchor_lora->second, *child_lora->second, mode, repeat);
                fprintf(stderr,
                        "test5 mode=%s repeat=%d pair=%s prefix=%d delta=%.3fms ttft=%.3fms total=%.3fms status=%s\n",
                        mode.c_str(), repeat, pair.pair_name.c_str(), row.prefix_tokens,
                        row.delta_build_ms + row.delta_submit_ms + row.delta_finish_wait_ms,
                        row.ttft_ms, row.total_ms, row.status.c_str());
                rows.push_back(std::move(row));
            }
        }
    }

    write_test5_csv(options.output_dir + "/fused_delta_results.csv", rows);
    write_test5_summary(options.output_dir + "/fused_delta_summary.csv", rows);
    for (auto & lora : lora_list) {
        if (lora.adapter != nullptr) llama_adapter_lora_free(lora.adapter);
    }
    llama_model_free(model);
    fprintf(stderr, "test5 complete: rows=%zu output=%s\n", rows.size(), options.output_dir.c_str());
    return rows.empty() ? 1 : 0;
}
