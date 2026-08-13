// Focused KV-delta scheduling experiment.
//
// The implementation reuses the data/model helpers from test4, but keeps the
// measured region starts at child-prefix processing and ends at the first
// token. "layer-wise" submits each layer's delta after that layer completes;
// "cuda-sync" and "cuda-overlap" submit one all-layer CUDA job after the full
// prefix. None of these modes is a single inference+delta CUDA kernel.

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
    int delta_jobs = 0;
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
                    "  [--mode all|traditional|layer-wise|cuda-sync|cuda-overlap]\n",
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
            options.mode != "traditional" &&
            options.mode != "layer-wise" && options.mode != "cuda-sync" &&
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
    if (mode == "all") return { "traditional", "layer-wise", "cuda-sync", "cuda-overlap" };
    return { mode };
}

struct layer_pipeline_state {
    llama_context * context = nullptr;
    bool active = false;
    bool submit_failed = false;
    llama_seq_id anchor_seq = 0;
    llama_seq_id child_seq = 1;
    llama_seq_id delta_seq = 2;
    llama_pos p0 = 0;
    llama_pos p1 = 0;
    int32_t parent_node_id = -1;
    int32_t child_node_id = -1;
    double submit_ms = 0.0;
    std::vector<uint64_t> jobs;
};

static bool layer_pipeline_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * state = static_cast<layer_pipeline_state *>(user_data);
    if (state == nullptr || !state->active) return false;

    int layer_id = -1;
    int parsed = 0;
    const bool is_layer_end = std::sscanf(
            tensor->name, "l_out-%d%n", &layer_id, &parsed) == 1 &&
            parsed == (int) std::strlen(tensor->name);
    if (ask) return is_layer_end;
    if (!is_layer_end || state->context == nullptr || state->submit_failed) return true;

    uint64_t job = 0;
    const double start = now_ms();
    const bool ok = llama_kv_seq_delta_build_layer_async(
            state->context, state->anchor_seq, state->child_seq, state->delta_seq,
            state->p0, state->p1, state->parent_node_id, state->child_node_id,
            layer_id, &job);
    state->submit_ms += now_ms() - start;
    if (!ok || job == 0) {
        state->submit_failed = true;
    } else {
        state->jobs.push_back(job);
    }
    return true;
}

static llama_context * create_test5_context(
        llama_model * model,
        const experiment_options & options,
        int sequence_count,
        layer_pipeline_state * pipeline) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = options.n_ctx;
    params.n_batch = options.n_batch;
    params.n_ubatch = options.n_ubatch;
    params.n_seq_max = (uint32_t) std::max(1, sequence_count);
    params.no_perf = true;
    params.kv_unified = true;
    if (pipeline != nullptr) {
        params.cb_eval = layer_pipeline_callback;
        params.cb_eval_user_data = pipeline;
    }
    llama_context * context = llama_init_from_model(model, params);
    if (pipeline != nullptr) pipeline->context = context;
    return context;
}

static bool eval_tokens_layer_pipeline(
        llama_context * context,
        const std::vector<llama_token> & tokens,
        llama_seq_id sequence,
        int start_position,
        int chunk_size,
        int n_ubatch,
        layer_pipeline_state & pipeline) {
    int offset = 0;
    const int physical_chunk = std::max(1, std::min(chunk_size, n_ubatch));
    while (offset < (int) tokens.size()) {
        const int count = std::min(physical_chunk, (int) tokens.size() - offset);
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (int i = 0; i < count; ++i) {
            batch_add(batch, tokens[offset + i], start_position + offset + i, sequence, false);
        }
        pipeline.p0 = start_position + offset;
        pipeline.p1 = pipeline.p0 + count;
        pipeline.active = true;
        const int result = llama_decode(context, batch);
        pipeline.active = false;
        llama_batch_free(batch);
        if (result != 0 || pipeline.submit_failed) return false;
        offset += count;
    }
    return true;
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
    layer_pipeline_state pipeline;
    pipeline.parent_node_id = pair.anchor_request_id;
    pipeline.child_node_id = pair.child_request_id;
    llama_context * context = create_test5_context(
            model, run_options, 4, mode == "layer-wise" ? &pipeline : nullptr);
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
    const double measured_start = now_ms();
    start = now_ms();
    const bool child_ok = mode == "layer-wise"
            ? eval_tokens_layer_pipeline(
                    context, child_tokens.prefix, 1, 0, options.prefix_chunk,
                    options.n_ubatch, pipeline)
            : eval_tokens(context, child_tokens.prefix, 1, 0, options.prefix_chunk);
    if (!child_ok) {
        row.status = "child_eval_failed";
        llama_free(context);
        return row;
    }
    // llama_decode() submits asynchronously. All modes use the same completed
    // prefix boundary, and the external delta stream must not read KV earlier.
    llama_synchronize(context);
    row.child_eval_ms = now_ms() - start;

    const llama_seq_id delta_seq = 2;
    const llama_seq_id recon_seq = 3;
    llama_kv_delta_branch_stats branch = {};
    uint64_t async_job = 0;
    bool build_ok = false;

    auto run_suffix_and_decode = [&]() {
        const double foreground_start = now_ms();
        if (!suffix.empty() && !eval_tokens(
                context, suffix, 1, row.prefix_tokens, options.prefix_chunk)) {
            row.status = "suffix_eval_failed";
            return false;
        }
        if (!decode_one(context, suffix.empty() ? child_tokens.prefix.back() : suffix.back(),
                1, row.prefix_tokens + row.suffix_tokens)) {
            row.status = "decode_failed";
            return false;
        }
        // TTFT ends only when the first-token computation is actually complete.
        llama_synchronize(context);
        row.foreground_ms = now_ms() - foreground_start;
        return true;
    };

    if (mode == "traditional") {
        if (!run_suffix_and_decode()) {
            llama_free(context);
            return row;
        }
        row.ttft_ms = now_ms() - measured_start;
        row.total_ms = row.ttft_ms;
        row.status = "ok";
        row.delta_build_ok = 0;
        row.reconstruction_ok = 0;
        llama_free(context);
        return row;
    }

    if (mode == "layer-wise") {
        row.delta_submit_ms = pipeline.submit_ms;
        row.delta_jobs = (int) pipeline.jobs.size();
        if (pipeline.submit_failed || pipeline.jobs.empty()) {
            row.status = "layer_submit_failed";
            llama_free(context);
            return row;
        }
        const double finish_start = now_ms();
        build_ok = true;
        for (size_t i = 0; i < pipeline.jobs.size(); ++i) {
            if (!llama_kv_seq_delta_build_branch_finish(
                    context, pipeline.jobs[i], delta_seq, &branch)) {
                build_ok = false;
                for (size_t j = i + 1; j < pipeline.jobs.size(); ++j) {
                    llama_kv_seq_delta_build_branch_cancel(context, pipeline.jobs[j]);
                }
                break;
            }
        }
        row.delta_finish_wait_ms = now_ms() - finish_start;
        if (!build_ok) {
            row.status = "layer_finish_failed";
            llama_free(context);
            return row;
        }
        if (!run_suffix_and_decode()) {
            llama_free(context);
            return row;
        }
        row.ttft_ms = now_ms() - measured_start;
        row.total_ms = row.ttft_ms;
    } else if (mode == "cuda-overlap") {
        const double submit_start = now_ms();
        build_ok = llama_kv_seq_delta_build_branch_async(
                context, 0, 1, delta_seq, 0, row.prefix_tokens,
                pair.anchor_request_id, pair.child_request_id, &async_job);
        row.delta_submit_ms = now_ms() - submit_start;
        row.delta_jobs = build_ok ? 1 : 0;
        if (!build_ok) {
            row.status = "async_submit_failed";
            llama_free(context);
            return row;
        }

        if (!run_suffix_and_decode()) {
            llama_kv_seq_delta_build_branch_cancel(context, async_job);
            llama_free(context);
            return row;
        }
        row.ttft_ms = now_ms() - measured_start;

        const double finish_start = now_ms();
        build_ok = llama_kv_seq_delta_build_branch_finish(
                context, async_job, delta_seq, &branch);
        row.delta_finish_wait_ms = now_ms() - finish_start;
        row.overlap_window_ms = row.foreground_ms;
    } else {
        const double delta_start = now_ms();
        build_ok = llama_kv_seq_delta_build_branch(
                context, 0, 1, delta_seq, 0, row.prefix_tokens,
                pair.anchor_request_id, pair.child_request_id, &branch);
        row.delta_build_ms = now_ms() - delta_start;
        row.delta_jobs = build_ok ? 1 : 0;
        if (!build_ok) {
            row.status = "delta_build_failed";
            llama_free(context);
            return row;
        }

        if (!run_suffix_and_decode()) {
            llama_free(context);
            return row;
        }
        row.ttft_ms = now_ms() - measured_start;
    }

    row.total_ms = now_ms() - measured_start;
    row.delta_build_ok = build_ok ? 1 : 0;
    row.full_kv_bytes = branch.full_kv_bytes_equivalent;
    row.delta_bytes = branch.delta_q8_bytes + branch.delta_scale_bytes;
    row.saved_rate = branch.logical_saved_rate;
    row.status = build_ok ? "ok" : "delta_finish_failed";

    // Similarity probing and reconstruction are validation-only and stay
    // outside the request timing window for every mode.
    llama_kv_delta_probe_stats source_probe = {};
    start = now_ms();
    const bool source_probe_ok = llama_kv_seq_delta_probe(
            context, 0, 1, 0, row.prefix_tokens, &source_probe);
    row.probe_ms = now_ms() - start;
    if (!source_probe_ok) row.status = "probe_failed";

    bind_lora(context, child_lora);
    if (source_probe_ok && eval_tokens(
            context, child_tokens.prefix, recon_seq, 0, options.prefix_chunk)) {
        finish_and_validate_delta(
                context, 0, 1, delta_seq, recon_seq, row.prefix_tokens, row);
    } else if (source_probe_ok) {
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
              "foreground_ms,ttft_ms,total_ms,overlap_window_ms,delta_jobs,saved_rate,reconstruction_cos,reconstruction_l2,"
              "full_kv_bytes,delta_bytes,delta_build_ok,reconstruction_ok,status\n";
    output << std::setprecision(10);
    for (const auto & row : rows) {
        output << csv_escape(row.mode) << ',' << row.repeat << ',' << csv_escape(row.pair_name) << ','
               << csv_escape(row.group_name) << ',' << row.anchor_lora_id << ',' << row.child_lora_id << ','
               << row.prefix_tokens << ',' << row.suffix_tokens << ',' << row.anchor_eval_ms << ','
               << row.child_eval_ms << ',' << row.probe_ms << ',' << row.delta_submit_ms << ','
               << row.delta_build_ms << ',' << row.delta_finish_wait_ms << ',' << row.foreground_ms << ','
               << row.ttft_ms << ',' << row.total_ms << ',' << row.overlap_window_ms << ',' << row.delta_jobs << ','
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
    output << "mode,count,ok_count,mean_child_eval_ms,mean_foreground_ms,mean_delta_ms,"
              "mean_ttft_ms,stddev_ttft_ms,p50_ttft_ms,p95_ttft_ms,mean_total_ms,mean_saved_rate,"
              "mean_reconstruction_cos,mean_reconstruction_l2\n";
    output << std::setprecision(10);
    for (const auto & [mode, items] : grouped) {
        int ok_count = 0;
        double child_eval_ms = 0.0;
        double foreground_ms = 0.0;
        double delta_ms = 0.0;
        double ttft_ms = 0.0;
        double total_ms = 0.0;
        double saved_rate = 0.0;
        double recon_cos = 0.0;
        double recon_l2 = 0.0;
        std::vector<double> ttft_values;
        for (const test5_row * row : items) {
            if (row->status != "ok") continue;
            ++ok_count;
            child_eval_ms += row->child_eval_ms;
            foreground_ms += row->foreground_ms;
            delta_ms += row->delta_build_ms + row->delta_submit_ms + row->delta_finish_wait_ms;
            ttft_ms += row->ttft_ms;
            ttft_values.push_back(row->ttft_ms);
            total_ms += row->total_ms;
            saved_rate += row->saved_rate;
            recon_cos += row->reconstruction_cos;
            recon_l2 += row->reconstruction_l2;
        }
        const double count = std::max(1, ok_count);
        const double mean_ttft = ttft_ms / count;
        double ttft_variance = 0.0;
        for (const double value : ttft_values) {
            const double difference = value - mean_ttft;
            ttft_variance += difference * difference;
        }
        const double ttft_stddev = std::sqrt(ttft_variance / count);
        std::sort(ttft_values.begin(), ttft_values.end());
        auto percentile = [&](double q) {
            if (ttft_values.empty()) return 0.0;
            const size_t index = std::min(
                    ttft_values.size() - 1,
                    (size_t) std::ceil(q * ttft_values.size()) - 1);
            return ttft_values[index];
        };
        output << csv_escape(mode) << ',' << items.size() << ',' << ok_count << ','
               << child_eval_ms / count << ',' << foreground_ms / count << ',' << delta_ms / count << ','
               << mean_ttft << ',' << ttft_stddev << ',' << percentile(0.50) << ',' << percentile(0.95) << ','
               << total_ms / count << ','
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
    const std::vector<std::string> base_modes = test5_modes(options.mode);
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        for (size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
            const auto & pair = pairs[pair_index];
            std::vector<std::string> modes = base_modes;
            if (modes.size() > 1) {
                const size_t offset = (pair_index + (size_t) repeat) % modes.size();
                std::rotate(modes.begin(), modes.begin() + offset, modes.end());
            }
            for (const std::string & mode : modes) {
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
                        "test5 mode=%s repeat=%d pair=%s prefix=%d jobs=%d delta=%.3fms ttft=%.3fms total=%.3fms status=%s\n",
                        mode.c_str(), repeat, pair.pair_name.c_str(), row.prefix_tokens,
                        row.delta_jobs,
                        row.delta_build_ms + row.delta_submit_ms + row.delta_finish_wait_ms,
                        row.ttft_ms, row.total_ms, row.status.c_str());
                rows.push_back(std::move(row));
            }
        }
    }

    write_test5_csv(options.output_dir + "/kv_delta_results.csv", rows);
    write_test5_summary(options.output_dir + "/kv_delta_summary.csv", rows);
    for (auto & lora : lora_list) {
        if (lora.adapter != nullptr) llama_adapter_lora_free(lora.adapter);
    }
    llama_model_free(model);
    fprintf(stderr, "test5 complete: rows=%zu output=%s\n", rows.size(), options.output_dir.c_str());
    return rows.empty() ? 1 : 0;
}
