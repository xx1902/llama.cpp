// MobiLoRA-style dataset experiment for llama.cpp.
//
// The program contains two independent experiments:
// 1. Delta probe: compare KV produced by different LoRAs for exactly the same
//    token prefix, then build a Q8 anchor + delta branch and record its size.
//    Delta 探测：比较不同 LoRA 在完全相同的 token prefix 上产生的 KV，
//    然后构建 Q8 anchor + delta 分支并记录其大小。
// 2. Online prefix cache: replay grouped_requests.jsonl in arrival order and
//    only reuse a cached prefix when both the prefix tokens and LoRA are exact.
//    在线 prefix 缓存：按到达顺序重放 grouped_requests.jsonl，
//    只有当 prefix token 和 LoRA 都完全匹配时才复用缓存。
//
// Important boundary:
// - Same text prefix + different LoRA cannot directly share dense KV.
//   相同文本 prefix + 不同 LoRA 不能直接共享稠密 KV。
// - Different suffix text is always evaluated normally.
//   不同的 suffix 文本总是正常计算。
// - No semantic-similarity or "similar word" KV reuse is performed here.
//   这里不进行语义相似性或"相似词"的 KV 复用。

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <clocale>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using json = nlohmann::json;

// =============================================================================
// 1. Runtime options  实验配置参数
// =============================================================================

struct experiment_options {
    std::string model_path = "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/Qwen2.5-1.5B-Instruct-f16.gguf";
    std::string workload_dir = "D:/ecnu_experiment/datasets/mobilora_workloads_original";
    std::string lora_config_path;  // LoRA 配置文件路径（默认从 workload_dir 推断）
    std::string output_dir = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base-test/output";

    int n_gpu_layers = 99;
    int n_ctx = 16384;
    int n_batch = 256;
    int n_ubatch = 64;
    int n_predict = 16;

    // Delta 实验参数
    int max_delta_pairs = 50;
    int delta_context_chunk = 16;
    int max_online_requests = 300;

    // These are physical full-KV limits used by the online exact-prefix test.
    int max_cache_nodes = 8;
    int max_cache_variants = 24;
    int max_cache_tokens = 8192;
    // off: do not probe/build cross-LoRA delta on the request path.
    // sync: probe and build delta before serving the request.
    // deferred: serve with full KV first, then build delta as background work.
    std::string cross_lora_policy = "off";

    bool system_v2 = false;
    int prefix_chunk_tokens = 128;
    int max_host_delta_mb = 1024;
    std::string prefetch_policy = "none";
    std::string prefetch_storage = "auto"; // auto | full | delta
    std::string prediction_file;
    int prediction_top_k = 1;
    std::string delta_store_dir;
    std::string delta_store_policy = "none"; // none | build | load | auto
    std::string background_policy = "arrival";
    double background_min_gap_ms = 15000.0;
};

// 打印使用说明
static void print_usage(const char * program) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --model PATH\n"
            "  --workload-dir DIR\n"
            "  --lora-config PATH\n"
            "  --output-dir DIR\n"
            "  --max-delta-pairs N\n"
            "  --max-online-requests N\n"
            "  --n-predict N\n"
            "  --n-ctx N\n"
            "  --max-cache-nodes N\n"
            "  --max-cache-variants N\n"
            "  --max-cache-tokens N\n"
            "  --cross-lora-policy off|sync|deferred\n"
            "  --system-v2 0|1\n"
            "  --prefix-chunk-tokens N\n"
            "  --max-host-delta-mb N\n"
            "  --prefetch-policy none|oracle|file\n"
            "  --prefetch-storage auto|full|delta\n"
            "  --prediction-file FILE\n"
            "  --prediction-top-k N\n"
            "  --delta-store-dir DIR\n"
            "  --delta-store-policy none|build|load|auto\n"
            "  --background-policy none|arrival|unlimited\n"
            "  --background-min-gap-ms N\n",
            program);
}

// 字符串转整数
static bool parse_int(const char * text, int & value) {
    try {
        value = std::stoi(text);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_double(const char * text, double & value) {
    try {
        value = std::stod(text);
        return true;
    } catch (...) {
        return false;
    }
}

// 解析命令行参数
static bool parse_options(
        int argc,
        char ** argv,
        experiment_options & options) {
    auto require_value = [&](int & index) -> const char * {
        if (index + 1 >= argc) {
            return nullptr;
        }
        return argv[++index];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const char * value = nullptr;

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        } else if (arg == "--model") {
            value = require_value(i);
            if (!value) return false;
            options.model_path = value;
        } else if (arg == "--workload-dir") {
            value = require_value(i);
            if (!value) return false;
            options.workload_dir = value;
        } else if (arg == "--lora-config") {
            value = require_value(i);
            if (!value) return false;
            options.lora_config_path = value;
        } else if (arg == "--output-dir") {
            value = require_value(i);
            if (!value) return false;
            options.output_dir = value;
        } else if (arg == "--max-delta-pairs") {
            value = require_value(i);
            if (!value || !parse_int(value, options.max_delta_pairs)) return false;
        } else if (arg == "--max-online-requests") {
            value = require_value(i);
            if (!value || !parse_int(value, options.max_online_requests)) return false;
        } else if (arg == "--n-predict") {
            value = require_value(i);
            if (!value || !parse_int(value, options.n_predict)) return false;
        } else if (arg == "--n-ctx") {
            value = require_value(i);
            if (!value || !parse_int(value, options.n_ctx)) return false;
        } else if (arg == "--max-cache-nodes") {
            value = require_value(i);
            if (!value || !parse_int(value, options.max_cache_nodes)) return false;
        } else if (arg == "--max-cache-variants") {
            value = require_value(i);
            if (!value || !parse_int(value, options.max_cache_variants)) return false;
        } else if (arg == "--max-cache-tokens") {
            value = require_value(i);
            if (!value || !parse_int(value, options.max_cache_tokens)) return false;
        } else if (arg == "--cross-lora-policy") {
            value = require_value(i);
            if (!value) return false;
            options.cross_lora_policy = value;
            if (options.cross_lora_policy != "off" &&
                    options.cross_lora_policy != "sync" &&
                    options.cross_lora_policy != "deferred") {
                fprintf(stderr, "invalid cross-LoRA policy: %s\n", value);
                return false;
            }
        } else if (arg == "--system-v2") {
            int enabled = 0;
            value = require_value(i);
            if (!value || !parse_int(value, enabled)) return false;
            options.system_v2 = enabled != 0;
        } else if (arg == "--prefix-chunk-tokens") {
            value = require_value(i);
            if (!value || !parse_int(value, options.prefix_chunk_tokens)) return false;
        } else if (arg == "--max-host-delta-mb") {
            value = require_value(i);
            if (!value || !parse_int(value, options.max_host_delta_mb)) return false;
        } else if (arg == "--prefetch-policy") {
            value = require_value(i);
            if (!value) return false;
            options.prefetch_policy = value;
            if (options.prefetch_policy != "none" && options.prefetch_policy != "oracle" &&
                    options.prefetch_policy != "file") {
                fprintf(stderr, "invalid prefetch policy: %s\n", value);
                return false;
            }
        } else if (arg == "--prefetch-storage") {
            value = require_value(i);
            if (!value) return false;
            options.prefetch_storage = value;
            if (options.prefetch_storage != "auto" &&
                    options.prefetch_storage != "full" &&
                    options.prefetch_storage != "delta") {
                fprintf(stderr, "invalid prefetch storage: %s\n", value);
                return false;
            }
        } else if (arg == "--prediction-file") {
            value = require_value(i);
            if (!value) return false;
            options.prediction_file = value;
        } else if (arg == "--prediction-top-k") {
            value = require_value(i);
            if (!value || !parse_int(value, options.prediction_top_k) ||
                    options.prediction_top_k <= 0) return false;
        } else if (arg == "--delta-store-dir") {
            value = require_value(i);
            if (!value) return false;
            options.delta_store_dir = value;
        } else if (arg == "--delta-store-policy") {
            value = require_value(i);
            if (!value) return false;
            options.delta_store_policy = value;
            if (options.delta_store_policy != "none" &&
                    options.delta_store_policy != "build" &&
                    options.delta_store_policy != "load" &&
                    options.delta_store_policy != "auto") {
                fprintf(stderr, "invalid delta store policy: %s\n", value);
                return false;
            }
        } else if (arg == "--background-policy") {
            value = require_value(i);
            if (!value) return false;
            options.background_policy = value;
            if (options.background_policy != "none" &&
                    options.background_policy != "arrival" &&
                    options.background_policy != "unlimited") {
                fprintf(stderr, "invalid background policy: %s\n", value);
                return false;
            }
        } else if (arg == "--background-min-gap-ms") {
            value = require_value(i);
            if (!value || !parse_double(value, options.background_min_gap_ms)) return false;
        } else {
            fprintf(stderr, "unknown option: %s\n", arg.c_str());
            return false;
        }
    }

    if (options.lora_config_path.empty()) {
        options.lora_config_path =
                options.workload_dir + "/lora_groups.json";
    }
    return true;
}

// =============================================================================
// 2. Dataset and runtime structures  数据结构定义
// =============================================================================

// LoRA 运行时信息
struct lora_runtime {
    int lora_id = -1;                    // LoRA ID
    std::string logical_name;            // 逻辑名称
    std::string group_name;              // 所属 group
    std::string adapter_path;            // 适配器文件路径
    bool is_anchor = false;              // 是否是 anchor LoRA
    llama_adapter_lora * adapter = nullptr;  // llama.cpp 适配器指针
};

// 数据集请求
struct dataset_request {
    int chunk_token_size = 0;
    std::vector<std::string> prefix_segment_types;
    struct prefix_segment {
        std::string type;
        std::string text;
    };
    std::vector<prefix_segment> prefix_segments;
    int request_id = -1;                 // 请求 ID
    std::string experiment;              // 实验类型
    std::string group_name;              // 所属 group
    std::string context_id;              // 上下文 ID
    std::string source_dataset;          // 源数据集
    int lora_id = -1;                    // 使用的 LoRA ID
    std::string lora_name;               // LoRA 名称
    std::string common_prefix_text;      // 公共 prefix 文本
    std::string common_prefix_hash;      // prefix 哈希（用于快速匹配）
    std::string prompt;                  // 完整 prompt
    long long arrival_ms = 0;            // 到达时间戳
    int user_id = -1;                    // 用户 ID
    int session_id = -1;                 // 会话 ID
    std::string app_name;                // 应用名称
};

// Delta pair 项（用于实验1）
struct delta_pair_item {
    std::string pair_name;               // pair 名称
    std::string group_name;              // group
    std::string context_id;              // 上下文
    std::string common_prefix_hash;      // prefix 哈希
    int anchor_request_id = -1;          // anchor 请求 ID
    int child_request_id = -1;           // child 请求 ID
    int anchor_lora_id = -1;             // anchor LoRA ID
    int child_lora_id = -1;              // child LoRA ID
};

// Tokenize 后的请求
struct tokenized_request {
    struct segment_range {
        int begin = 0;
        int end = 0;
        std::string type;
    };
    std::vector<llama_token> prefix;     // prefix tokens
    std::vector<llama_token> full;       // 完整 prompt tokens
    std::vector<llama_token> suffix;     // suffix tokens
    std::vector<segment_range> prefix_segments;
    bool exact_prefix_layout = false;    // prefix 是否精确匹配
};

// Delta 实验结果
struct delta_result {
    std::string pair_name;
    std::string group_name;
    std::string context_id;
    int anchor_request_id = -1;
    int child_request_id = -1;
    int anchor_lora_id = -1;
    int child_lora_id = -1;
    std::string anchor_lora_name;
    std::string child_lora_name;
    int prefix_tokens = 0;
    std::string status = "failed";
    std::string memory_kind = "unknown";
    int probed_layers = 0;
    int skipped_recurrent_layers = 0;
    int can_encode_delta = 0;
    double kv_cos = 0.0;
    double kv_l2 = 0.0;
    double anchor_eval_ms = 0.0;
    double child_eval_ms = 0.0;
    double probe_ms = 0.0;
    double delta_build_ms = 0.0;
    int delta_build_ok = 0;
    unsigned long long full_kv_bytes = 0;
    unsigned long long delta_q8_bytes = 0;
    unsigned long long delta_scale_bytes = 0;
    unsigned long long logical_saved_bytes = 0;
    double logical_saved_rate = 0.0;
};

// 每层 delta 结果
struct delta_layer_result {
    std::string pair_name;
    int layer_id = -1;
    double k_cos = 0.0;
    double v_cos = 0.0;
    double kv_cos = 0.0;
    double k_l2 = 0.0;
    double v_l2 = 0.0;
    double kv_l2 = 0.0;
};

// 在线实验结果
struct online_result {
    int chunk_hit_tokens = 0;
    int materialize_ok = 0;
    double materialize_ms = 0.0;
    double delta_validation_ms = 0.0;
    double reconstruction_cos = 0.0;
    double reconstruction_l2 = 0.0;
    int predicted_lora_id = -1;
    double idle_gap_ms = 0.0;
    double prefetch_ms = 0.0;
    int prefetch_built = 0;
    int prefetch_full_built = 0;
    int prefetch_delta_built = 0;
    int prefetch_skipped_no_anchor = 0;
    int prefetch_dropped_expired = 0;
    int delta_store_loaded = 0;
    int delta_store_saved = 0;
    int host_delta_variants = 0;
    unsigned long long host_delta_bytes = 0;
    int background_queue_length = 0;
    double background_overrun_ms = 0.0;
    std::string benchmark;
    int request_id = -1;
    long long arrival_ms = 0;
    int user_id = -1;
    int session_id = -1;
    std::string app_name;
    std::string group_name;
    std::string context_id;
    int lora_id = -1;
    std::string lora_name;
    std::string mode;
    int node_id = -1;
    int exact_prefix_hit = 0;
    int same_lora_variant_hit = 0;
    int cross_lora_prefix_match = 0;
    int prefix_tokens = 0;
    int suffix_tokens = 0;
    int prompt_tokens = 0;
    int delta_candidate = 0;
    double prefix_kv_cos = 0.0;
    double prefix_kv_l2 = 0.0;
    double delta_probe_ms = 0.0;
    double delta_build_ms = 0.0;
    double delta_background_ms = 0.0;
    int delta_build_ok = 0;
    double delta_saved_rate = 0.0;
    double lora_bind_ms = 0.0;
    double prefix_ms = 0.0;
    double suffix_ms = 0.0;
    double ttft_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    double tps = 0.0;
    int cache_nodes = 0;
    int cache_variants = 0;
    int physical_cache_tokens = 0;
};

enum class variant_residency {
    gpu_full,
    host_delta,
};

struct prefix_variant {
    int lora_id = -1;
    llama_seq_id cache_seq_id = -1;
    int hit_count = 0;
    int last_access_index = -1;
    variant_residency residency = variant_residency::gpu_full;
    unsigned long long delta_bytes = 0;
    double materialize_ms = 0.0;
    double predicted_probability = 0.0;
    double reconstruction_cos = 0.0;
    double reconstruction_l2 = 0.0;
};

struct prefix_node {
    int node_id = -1;
    std::string group_name;
    std::string context_id;
    std::string prefix_hash;
    std::vector<llama_token> prefix_tokens;
    int anchor_lora_id = -1;
    llama_seq_id anchor_seq_id = -1;
    int hit_count = 0;
    int last_access_index = -1;
    int parent_node_id = -1;
    int depth_tokens = 0;
    int chunk_begin = 0;
    int chunk_end = 0;
    std::string segment_kind = "user_context";
    std::vector<prefix_variant> variants;
};

// =============================================================================
// 3. JSON and CSV helpers JSON 和 CSV 辅助函数
// =============================================================================

static std::string csv_escape(const std::string & value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string result = "\"";
    for (char ch : value) {
        if (ch == '"') result += '"';
        result += ch;
    }
    result += '"';
    return result;
}

static bool read_json_file(const std::string & path, json & value) {
    std::ifstream input(path);
    if (!input) {
        fprintf(stderr, "failed to open JSON file: %s\n", path.c_str());
        return false;
    }
    try {
        input >> value;
        return true;
    } catch (const std::exception & error) {
        fprintf(stderr, "failed to parse JSON file %s: %s\n", path.c_str(), error.what());
        return false;
    }
}

static std::vector<json> read_jsonl(const std::string & path) {
    std::vector<json> rows;
    std::ifstream input(path);
    if (!input) {
        fprintf(stderr, "failed to open JSONL file: %s\n", path.c_str());
        return rows;
    }

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) continue;
        try {
            rows.push_back(json::parse(line));
        } catch (const std::exception & error) {
            fprintf(stderr,
                    "skip invalid JSONL row: file=%s line=%d error=%s\n",
                    path.c_str(),
                    line_number,
                    error.what());
        }
    }
    return rows;
}

static std::vector<lora_runtime> load_lora_config(const std::string & path) {
    json root;
    std::vector<lora_runtime> loras;
    if (!read_json_file(path, root) || !root.is_object()) {
        return loras;
    }

    for (auto group_it = root.begin(); group_it != root.end(); ++group_it) {
        if (!group_it.value().is_array()) continue;
        for (const auto & item : group_it.value()) {
            lora_runtime lora;
            lora.group_name = group_it.key();
            lora.lora_id = item.value("lora_id", -1);
            lora.logical_name = item.value("logical_name", "unknown");
            lora.is_anchor = item.value("is_anchor", false);
            if (item.contains("actual_adapter_path") &&
                    item["actual_adapter_path"].is_string()) {
                lora.adapter_path = item["actual_adapter_path"].get<std::string>();
            }
            loras.push_back(std::move(lora));
        }
    }
    return loras;
}

static dataset_request parse_request(const json & item) {
    dataset_request request;
    request.chunk_token_size = item.value("chunk_token_size", 0);
    if (item.contains("prefix_segment_types") && item["prefix_segment_types"].is_array()) {
        for (const auto & segment : item["prefix_segment_types"]) {
            if (segment.is_string()) {
                request.prefix_segment_types.push_back(segment.get<std::string>());
            }
        }
    }
    if (item.contains("prefix_segments") && item["prefix_segments"].is_array()) {
        for (const auto & segment : item["prefix_segments"]) {
            if (!segment.is_object()) continue;
            dataset_request::prefix_segment parsed;
            parsed.type = segment.value("type", "user_context");
            parsed.text = segment.value("text", "");
            if (!parsed.text.empty()) request.prefix_segments.push_back(std::move(parsed));
        }
    }
    request.request_id = item.value("request_id", -1);
    request.experiment = item.value("experiment", "unknown");
    request.group_name = item.value("group_name", "unknown");
    request.context_id = item.value("context_id", "unknown");
    request.source_dataset = item.value("source_dataset", "unknown");
    request.lora_id = item.value("lora_id", -1);
    request.lora_name = item.value("lora_name", "unknown");
    request.common_prefix_text = item.value("common_prefix_text", "");
    request.common_prefix_hash = item.value("common_prefix_hash", "");
    request.prompt = item.value("prompt", "");
    request.arrival_ms = item.value("arrival_ms", 0LL);
    request.user_id = item.value("user_id", -1);
    request.session_id = item.value("session_id", -1);
    request.app_name = item.value("app_name", "");
    return request;
}

static std::vector<dataset_request> load_requests(const std::string & path) {
    std::vector<dataset_request> requests;
    for (const auto & item : read_jsonl(path)) {
        requests.push_back(parse_request(item));
    }
    return requests;
}

static std::vector<delta_pair_item> load_delta_pairs(const std::string & path) {
    std::vector<delta_pair_item> pairs;
    for (const auto & item : read_jsonl(path)) {
        delta_pair_item pair;
        pair.pair_name = item.value("pair_name", "unknown");
        pair.group_name = item.value("group_name", "unknown");
        pair.context_id = item.value("context_id", "unknown");
        pair.common_prefix_hash = item.value("common_prefix_hash", "");
        pair.anchor_request_id = item.value("anchor_request_id", -1);
        pair.child_request_id = item.value("child_request_id", -1);
        pair.anchor_lora_id = item.value("anchor_lora_id", -1);
        pair.child_lora_id = item.value("child_lora_id", -1);
        pairs.push_back(std::move(pair));
    }
    return pairs;
}

struct prediction_candidate {
    int lora_id = -1;
    double probability = 0.0;
};

static std::unordered_map<int, std::vector<prediction_candidate>> load_prediction_file(
        const std::string & path) {
    std::unordered_map<int, std::vector<prediction_candidate>> predictions;
    if (path.empty()) return predictions;
    for (const auto & item : read_jsonl(path)) {
        const int request_id = item.value("request_id", -1);
        if (request_id < 0) continue;
        std::vector<prediction_candidate> candidates;
        if (item.contains("predictions") && item["predictions"].is_array()) {
            for (const auto & prediction : item["predictions"]) {
                if (!prediction.is_object()) continue;
                prediction_candidate candidate;
                candidate.lora_id = prediction.value("lora_id", -1);
                candidate.probability = prediction.value("probability", 0.0);
                if (candidate.lora_id >= 0) candidates.push_back(candidate);
            }
        } else {
            prediction_candidate candidate;
            candidate.lora_id = item.value("predicted_lora_id", -1);
            candidate.probability = item.value("probability", 1.0);
            if (candidate.lora_id >= 0) candidates.push_back(candidate);
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto & left, const auto & right) {
            return left.probability > right.probability;
        });
        if (!candidates.empty()) predictions[request_id] = std::move(candidates);
    }
    return predictions;
}

// =============================================================================
// 4. llama.cpp inference helpers 推理辅助函数
// =============================================================================

static double now_ms() {
    return ggml_time_us() / 1000.0;
}

static bool tokenize_text(
        const llama_vocab * vocab,
        const std::string & text,
        std::vector<llama_token> & tokens) {
    const int count = -llama_tokenize(
            vocab,
            text.c_str(),
            (int) text.size(),
            nullptr,
            0,
            true,
            true);
    if (count <= 0) return false;

    tokens.resize(count);
    const int result = llama_tokenize(
            vocab,
            text.c_str(),
            (int) text.size(),
            tokens.data(),
            count,
            true,
            true);
    return result >= 0;
}

static int common_prefix_length(
        const std::vector<llama_token> & a,
        const std::vector<llama_token> & b) {
    const int limit = std::min((int) a.size(), (int) b.size());
    int index = 0;
    while (index < limit && a[index] == b[index]) ++index;
    return index;
}

static bool tokenize_request(
        const llama_vocab * vocab,
        const dataset_request & request,
        tokenized_request & tokens) {
    if (!tokenize_text(vocab, request.common_prefix_text, tokens.prefix) ||
            !tokenize_text(vocab, request.prompt, tokens.full)) {
        return false;
    }

    const int matched = common_prefix_length(tokens.prefix, tokens.full);
    tokens.exact_prefix_layout = matched == (int) tokens.prefix.size();
    if (!tokens.exact_prefix_layout) {
        return false;
    }

    tokens.suffix.assign(
            tokens.full.begin() + tokens.prefix.size(),
            tokens.full.end());

    // Tokenize cumulative segment text so chunk boundaries follow the exact
    // prompt partition instead of relying on a first-chunk heuristic.
    tokens.prefix_segments.clear();
    if (!request.prefix_segments.empty()) {
        std::string cumulative;
        int previous_end = 0;
        for (const auto & segment : request.prefix_segments) {
            cumulative += segment.text;
            std::vector<llama_token> cumulative_tokens;
            if (!tokenize_text(vocab, cumulative, cumulative_tokens)) return false;
            int end = common_prefix_length(cumulative_tokens, tokens.prefix);
            end = std::max(previous_end, std::min(end, (int) tokens.prefix.size()));
            if (end > previous_end) {
                tokens.prefix_segments.push_back({ previous_end, end, segment.type });
                previous_end = end;
            }
        }
        if (previous_end < (int) tokens.prefix.size()) {
            const std::string type = request.prefix_segments.back().type;
            tokens.prefix_segments.push_back({ previous_end, (int) tokens.prefix.size(), type });
        }
    }
    if (tokens.prefix_segments.empty()) {
        tokens.prefix_segments.push_back({ 0, (int) tokens.prefix.size(), "user_context" });
    }
    return true;
}

static void batch_add(
        llama_batch & batch,
        llama_token token,
        llama_pos position,
        llama_seq_id sequence,
        bool logits) {
    const int index = batch.n_tokens;
    batch.token[index] = token;
    batch.pos[index] = position;
    batch.n_seq_id[index] = 1;
    batch.seq_id[index][0] = sequence;
    batch.logits[index] = logits ? 1 : 0;
    ++batch.n_tokens;
}

static bool eval_tokens(
        llama_context * context,
        const std::vector<llama_token> & tokens,
        llama_seq_id sequence,
        int start_position,
        int chunk_size) {
    int offset = 0;
    while (offset < (int) tokens.size()) {
        const int count = std::min(chunk_size, (int) tokens.size() - offset);
        llama_batch batch = llama_batch_init(count, 0, 1);
        for (int i = 0; i < count; ++i) {
            batch_add(
                    batch,
                    tokens[offset + i],
                    start_position + offset + i,
                    sequence,
                    false);
        }
        const int result = llama_decode(context, batch);
        llama_batch_free(batch);
        if (result != 0) {
            fprintf(stderr,
                    "eval failed: seq=%d start=%d offset=%d count=%d\n",
                    sequence,
                    start_position,
                    offset,
                    count);
            return false;
        }
        offset += count;
    }
    return true;
}

static bool decode_one(
        llama_context * context,
        llama_token token,
        llama_seq_id sequence,
        int position) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch_add(batch, token, position, sequence, true);
    const int result = llama_decode(context, batch);
    llama_batch_free(batch);
    return result == 0;
}

static void clear_lora(llama_context * context) {
    llama_set_adapters_lora(context, nullptr, 0, nullptr);
}

static double bind_lora(
        llama_context * context,
        const lora_runtime & lora) {
    llama_adapter_lora * adapters[] = { lora.adapter };
    float scales[] = { 1.0f };
    const double start = now_ms();
    const int result = llama_set_adapters_lora(context, adapters, 1, scales);
    const double end = now_ms();
    if (result != 0) {
        fprintf(stderr, "failed to bind LoRA %d (%s)\n",
                lora.lora_id,
                lora.logical_name.c_str());
    }
    return end - start;
}

static llama_context * create_context(
        llama_model * model,
        const experiment_options & options,
        int sequence_count) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = options.n_ctx;
    params.n_batch = options.n_batch;
    params.n_ubatch = options.n_ubatch;
    params.n_seq_max = (uint32_t) std::max(1, sequence_count);
    params.no_perf = true;
    params.kv_unified = true;
    return llama_init_from_model(model, params);
}

static bool run_decode(
        llama_context * context,
        llama_seq_id sequence,
        llama_token repeated_token,
        int prompt_tokens,
        int n_predict,
        double request_start_ms,
        online_result & result) {
    const double decode_start = now_ms();
    for (int i = 0; i < n_predict; ++i) {
        if (!decode_one(context, repeated_token, sequence, prompt_tokens + i)) {
            return false;
        }
        if (i == 0) {
            result.ttft_ms = now_ms() - request_start_ms;
        }
    }
    const double decode_end = now_ms();
    result.decode_ms = decode_end - decode_start;
    result.total_ms = decode_end - request_start_ms;
    result.tps = n_predict > 0
            ? n_predict / std::max(0.001, result.decode_ms / 1000.0)
            : 0.0;
    return true;
}

// Probe and encode one cross-LoRA prefix after both full KV variants exist.
// scratch_seq is temporary and is cleared before and after delta construction.
static void build_online_delta(
        llama_context * context,
        llama_memory_t memory,
        llama_seq_id anchor_seq,
        llama_seq_id child_seq,
        llama_seq_id scratch_seq,
        int prefix_tokens,
        int anchor_id,
        int child_id,
        online_result & result) {
    llama_memory_seq_rm(memory, scratch_seq, -1, -1);

    llama_kv_delta_probe_stats probe = {};
    double start = now_ms();
    const bool probe_ok = llama_kv_seq_delta_probe(
            context,
            anchor_seq,
            child_seq,
            0,
            prefix_tokens,
            &probe);
    result.delta_probe_ms = now_ms() - start;
    result.delta_candidate = probe_ok && probe.can_reuse_as_delta ? 1 : 0;
    result.prefix_kv_cos = probe.kv_cos_avg;
    result.prefix_kv_l2 = probe.kv_l2_avg;

    if (probe_ok && probe.can_reuse_as_delta) {
        llama_kv_delta_branch_stats branch = {};
        start = now_ms();
        const bool build_ok = llama_kv_seq_delta_build_branch(
                context,
                anchor_seq,
                child_seq,
                scratch_seq,
                0,
                prefix_tokens,
                anchor_id,
                child_id,
                &branch);
        result.delta_build_ms = now_ms() - start;
        result.delta_build_ok = build_ok ? 1 : 0;
        result.delta_saved_rate = branch.logical_saved_rate;
    }

    llama_memory_seq_rm(memory, scratch_seq, -1, -1);
}

// =============================================================================
// 5. Exact-prefix cross-LoRA delta experiment  Delta 实验（实验1）
// =============================================================================

static void run_delta_experiment(
        llama_model * model,
        const llama_vocab * vocab,
        const experiment_options & options,
        const std::unordered_map<int, dataset_request> & requests,
        const std::vector<delta_pair_item> & pairs,
        const std::unordered_map<int, lora_runtime *> & loras,
        std::vector<delta_result> & results,
        std::vector<delta_layer_result> & layer_results) {
    const int pair_limit = std::min(options.max_delta_pairs, (int) pairs.size());

    for (int chunk_begin = 0; chunk_begin < pair_limit;
            chunk_begin += options.delta_context_chunk) {
        const int chunk_end = std::min(
                pair_limit,
                chunk_begin + options.delta_context_chunk);
        llama_context * context = create_context(
                model,
                options,
                2 + (chunk_end - chunk_begin));
        if (!context) {
            fprintf(stderr, "failed to create delta context\n");
            return;
        }

        llama_memory_t memory = llama_get_memory(context);

        for (int pair_index = chunk_begin; pair_index < chunk_end; ++pair_index) {
            const delta_pair_item & pair = pairs[pair_index];
            delta_result result;
            result.pair_name = pair.pair_name;
            result.group_name = pair.group_name;
            result.context_id = pair.context_id;
            result.anchor_request_id = pair.anchor_request_id;
            result.child_request_id = pair.child_request_id;
            result.anchor_lora_id = pair.anchor_lora_id;
            result.child_lora_id = pair.child_lora_id;

            const auto anchor_request_it = requests.find(pair.anchor_request_id);
            const auto child_request_it = requests.find(pair.child_request_id);
            const auto anchor_lora_it = loras.find(pair.anchor_lora_id);
            const auto child_lora_it = loras.find(pair.child_lora_id);

            if (anchor_request_it == requests.end() ||
                    child_request_it == requests.end()) {
                result.status = "missing_request";
                results.push_back(result);
                continue;
            }
            if (anchor_lora_it == loras.end() || child_lora_it == loras.end()) {
                result.status = "missing_adapter";
                results.push_back(result);
                continue;
            }

            const dataset_request & anchor_request = anchor_request_it->second;
            const dataset_request & child_request = child_request_it->second;
            lora_runtime & anchor_lora = *anchor_lora_it->second;
            lora_runtime & child_lora = *child_lora_it->second;
            result.anchor_lora_name = anchor_lora.logical_name;
            result.child_lora_name = child_lora.logical_name;

            if (anchor_lora.group_name != pair.group_name ||
                    child_lora.group_name != pair.group_name) {
                result.status = "adapter_group_mismatch";
                results.push_back(result);
                continue;
            }

            // Delta is legal only when the compared text prefix is identical.
            if (anchor_request.common_prefix_hash != child_request.common_prefix_hash ||
                    anchor_request.common_prefix_text != child_request.common_prefix_text ||
                    pair.common_prefix_hash != anchor_request.common_prefix_hash) {
                result.status = "non_identical_prefix_rejected";
                results.push_back(result);
                continue;
            }

            std::vector<llama_token> prefix_tokens;
            if (!tokenize_text(vocab, anchor_request.common_prefix_text, prefix_tokens) ||
                    prefix_tokens.empty()) {
                result.status = "tokenize_failed";
                results.push_back(result);
                continue;
            }
            result.prefix_tokens = (int) prefix_tokens.size();

            llama_memory_clear(memory, true);
            clear_lora(context);

            const llama_seq_id anchor_seq = 0;
            const llama_seq_id child_seq = 1;
            const llama_seq_id delta_seq = 2 + (pair_index - chunk_begin);

            bind_lora(context, anchor_lora);
            double start = now_ms();
            const bool anchor_ok = eval_tokens(
                    context,
                    prefix_tokens,
                    anchor_seq,
                    0,
                    options.n_ubatch);
            result.anchor_eval_ms = now_ms() - start;
            if (!anchor_ok) {
                result.status = "anchor_eval_failed";
                results.push_back(result);
                continue;
            }

            bind_lora(context, child_lora);
            start = now_ms();
            const bool child_ok = eval_tokens(
                    context,
                    prefix_tokens,
                    child_seq,
                    0,
                    options.n_ubatch);
            result.child_eval_ms = now_ms() - start;
            if (!child_ok) {
                result.status = "child_eval_failed";
                results.push_back(result);
                continue;
            }

            llama_kv_delta_probe_stats probe = {};
            start = now_ms();
            const bool probe_ok = llama_kv_seq_delta_probe(
                    context,
                    anchor_seq,
                    child_seq,
                    0,
                    result.prefix_tokens,
                    &probe);
            result.probe_ms = now_ms() - start;
            result.status = probe.probe_status;
            result.memory_kind = probe.memory_kind;
            result.probed_layers = probe.probed_layers;
            result.skipped_recurrent_layers = probe.skipped_recurrent_layers;
            result.can_encode_delta = probe.can_reuse_as_delta;
            result.kv_cos = probe.kv_cos_avg;
            result.kv_l2 = probe.kv_l2_avg;

            if (probe_ok) {
                for (int layer_index = 0; layer_index < probe.n_layers; ++layer_index) {
                    const auto & source = probe.layers[layer_index];
                    delta_layer_result layer;
                    layer.pair_name = pair.pair_name;
                    layer.layer_id = source.layer_id;
                    layer.k_cos = source.k_cos_avg;
                    layer.v_cos = source.v_cos_avg;
                    layer.kv_cos = source.kv_cos_avg;
                    layer.k_l2 = source.k_l2_avg;
                    layer.v_l2 = source.v_l2_avg;
                    layer.kv_l2 = source.kv_l2_avg;
                    layer_results.push_back(layer);
                }
            }

            llama_kv_delta_branch_stats branch = {};
            start = now_ms();
            const bool branch_ok = llama_kv_seq_delta_build_branch(
                    context,
                    anchor_seq,
                    child_seq,
                    delta_seq,
                    0,
                    result.prefix_tokens,
                    pair.anchor_request_id,
                    pair.child_request_id,
                    &branch);
            result.delta_build_ms = now_ms() - start;
            result.delta_build_ok = branch_ok ? 1 : 0;
            result.full_kv_bytes = branch.full_kv_bytes_equivalent;
            result.delta_q8_bytes = branch.delta_q8_bytes;
            result.delta_scale_bytes = branch.delta_scale_bytes;
            result.logical_saved_bytes = branch.logical_saved_bytes;
            result.logical_saved_rate = branch.logical_saved_rate;

            if (!probe_ok && result.status.empty()) result.status = "probe_failed";
            results.push_back(result);

            fprintf(stderr,
                    "delta pair %d/%d: %s prefix=%d cos=%.6f l2=%.6f "
                    "q8_saved=%.2f%% status=%s\n",
                    pair_index + 1,
                    pair_limit,
                    pair.pair_name.c_str(),
                    result.prefix_tokens,
                    result.kv_cos,
                    result.kv_l2,
                    result.logical_saved_rate * 100.0,
                    result.status.c_str());
        }

        clear_lora(context);
        llama_free(context);
    }
}

// =============================================================================
// 6. Baseline and online exact-prefix experiment  在线实验（实验2）
// =============================================================================

static int count_variants(const std::vector<prefix_node> & nodes) {
    int count = 0;
    for (const auto & node : nodes) count += (int) node.variants.size();
    return count;
}

static int count_cache_tokens(const std::vector<prefix_node> & nodes) {
    int count = 0;
    for (const auto & node : nodes) {
        count += (int) node.prefix_tokens.size() * (int) node.variants.size();
    }
    return count;
}

static int find_node_index(
        const std::vector<prefix_node> & nodes,
        const dataset_request & request,
        const std::vector<llama_token> & prefix_tokens) {
    for (int index = 0; index < (int) nodes.size(); ++index) {
        const auto & node = nodes[index];
        if (node.group_name == request.group_name &&
                node.prefix_hash == request.common_prefix_hash &&
                node.prefix_tokens == prefix_tokens) {
            return index;
        }
    }
    return -1;
}

static prefix_variant * find_variant(prefix_node & node, int lora_id) {
    for (auto & variant : node.variants) {
        if (variant.lora_id == lora_id) return &variant;
    }
    return nullptr;
}

static void release_node(
        llama_memory_t memory,
        const prefix_node & node,
        std::vector<llama_seq_id> & free_sequences) {
    for (const auto & variant : node.variants) {
        llama_memory_seq_rm(memory, variant.cache_seq_id, -1, -1);
        free_sequences.push_back(variant.cache_seq_id);
    }
}

static bool ensure_cache_capacity(
        llama_memory_t memory,
        std::vector<prefix_node> & nodes,
        std::vector<llama_seq_id> & free_sequences,
        const experiment_options & options,
        int additional_tokens,
        int protected_node_id) {
    while (free_sequences.empty() ||
            count_variants(nodes) >= options.max_cache_variants ||
            count_cache_tokens(nodes) + additional_tokens > options.max_cache_tokens ||
            ((int) nodes.size() >= options.max_cache_nodes && protected_node_id < 0)) {
        int victim_index = -1;
        int oldest_access = std::numeric_limits<int>::max();
        for (int index = 0; index < (int) nodes.size(); ++index) {
            if (nodes[index].node_id == protected_node_id) continue;
            if (nodes[index].last_access_index < oldest_access) {
                oldest_access = nodes[index].last_access_index;
                victim_index = index;
            }
        }
        if (victim_index < 0) return false;

        fprintf(stderr,
                "evict prefix node=%d context=%s variants=%zu\n",
                nodes[victim_index].node_id,
                nodes[victim_index].context_id.c_str(),
                nodes[victim_index].variants.size());
        release_node(memory, nodes[victim_index], free_sequences);
        nodes.erase(nodes.begin() + victim_index);
    }
    return true;
}

static online_result run_uncached_request(
        llama_context * context,
        llama_memory_t memory,
        const experiment_options & options,
        const dataset_request & request,
        const tokenized_request & tokens,
        lora_runtime & lora,
        llama_seq_id request_seq,
        const std::string & benchmark,
        const std::string & mode) {
    online_result result;
    result.benchmark = benchmark;
    result.request_id = request.request_id;
    result.arrival_ms = request.arrival_ms;
    result.user_id = request.user_id;
    result.session_id = request.session_id;
    result.app_name = request.app_name;
    result.group_name = request.group_name;
    result.context_id = request.context_id;
    result.lora_id = request.lora_id;
    result.lora_name = lora.logical_name;
    result.mode = mode;
    result.prefix_tokens = (int) tokens.prefix.size();
    result.suffix_tokens = (int) tokens.suffix.size();
    result.prompt_tokens = (int) tokens.full.size();

    llama_memory_seq_rm(memory, request_seq, -1, -1);
    const double request_start = now_ms();
    result.lora_bind_ms = bind_lora(context, lora);

    const double eval_start = now_ms();
    if (!eval_tokens(context, tokens.full, request_seq, 0, options.n_ubatch)) {
        result.mode += "_eval_failed";
        return result;
    }
    result.suffix_ms = now_ms() - eval_start;

    const llama_token repeated = tokens.full.back();
    run_decode(
            context,
            request_seq,
            repeated,
            (int) tokens.full.size(),
            options.n_predict,
            request_start,
            result);
    llama_memory_seq_rm(memory, request_seq, -1, -1);
    return result;
}

static std::vector<online_result> run_baseline_experiment(
        llama_model * model,
        const llama_vocab * vocab,
        const experiment_options & options,
        const std::vector<dataset_request> & requests,
        const std::unordered_map<int, lora_runtime *> & loras) {
    std::vector<online_result> results;
    llama_context * context = create_context(model, options, 1);
    if (!context) return results;
    llama_memory_t memory = llama_get_memory(context);

    const int limit = std::min(options.max_online_requests, (int) requests.size());
    for (int index = 0; index < limit; ++index) {
        const dataset_request & request = requests[index];
        const auto lora_it = loras.find(request.lora_id);
        if (lora_it == loras.end()) continue;
        if (lora_it->second->group_name != request.group_name) continue;

        tokenized_request tokens;
        if (!tokenize_request(vocab, request, tokens)) {
            fprintf(stderr, "baseline skip invalid prefix layout: request=%d\n", request.request_id);
            continue;
        }
        online_result result = run_uncached_request(
                context,
                memory,
                options,
                request,
                tokens,
                *lora_it->second,
                0,
                "baseline",
                "full_prompt_eval");
        result.cache_nodes = 0;
        result.cache_variants = 0;
        result.physical_cache_tokens = 0;
        results.push_back(result);
    }

    clear_lora(context);
    llama_free(context);
    return results;
}

static std::vector<online_result> run_online_experiment(
        llama_model * model,
        const llama_vocab * vocab,
        const experiment_options & options,
        const std::vector<dataset_request> & requests,
        const std::unordered_map<int, lora_runtime *> & loras,
        std::vector<prefix_node> & final_nodes) {
    std::vector<online_result> results;
    const int cache_sequence_count = std::max(1, options.max_cache_variants);
    const llama_seq_id request_seq = cache_sequence_count;
    llama_context * context = create_context(
            model,
            options,
            cache_sequence_count + 1);
    if (!context) return results;
    llama_memory_t memory = llama_get_memory(context);

    std::vector<llama_seq_id> free_sequences;
    for (int sequence = cache_sequence_count - 1; sequence >= 0; --sequence) {
        free_sequences.push_back((llama_seq_id) sequence);
    }

    std::vector<prefix_node> nodes;
    int next_node_id = 0;
    const int limit = std::min(options.max_online_requests, (int) requests.size());

    for (int request_index = 0; request_index < limit; ++request_index) {
        const dataset_request & request = requests[request_index];
        const auto lora_it = loras.find(request.lora_id);
        if (lora_it == loras.end()) continue;
        lora_runtime & lora = *lora_it->second;
        if (lora.group_name != request.group_name) continue;

        tokenized_request tokens;
        if (!tokenize_request(vocab, request, tokens) || tokens.full.empty()) {
            fprintf(stderr, "online skip invalid prefix layout: request=%d\n", request.request_id);
            continue;
        }
        const double request_start = now_ms();

        int node_index = find_node_index(nodes, request, tokens.prefix);
        if (node_index < 0) {
            if ((int) tokens.prefix.size() > options.max_cache_tokens ||
                    !ensure_cache_capacity(
                            memory,
                            nodes,
                            free_sequences,
                            options,
                            (int) tokens.prefix.size(),
                            -1)) {
                online_result result = run_uncached_request(
                        context,
                        memory,
                        options,
                        request,
                        tokens,
                        lora,
                        request_seq,
                        "online",
                        "uncached_prefix_capacity_limit");
                result.cache_nodes = (int) nodes.size();
                result.cache_variants = count_variants(nodes);
                result.physical_cache_tokens = count_cache_tokens(nodes);
                results.push_back(result);
                continue;
            }

            prefix_node node;
            node.node_id = next_node_id++;
            node.group_name = request.group_name;
            node.context_id = request.context_id;
            node.prefix_hash = request.common_prefix_hash;
            node.prefix_tokens = tokens.prefix;
            node.anchor_lora_id = request.lora_id;
            node.last_access_index = request_index;

            const llama_seq_id cache_seq = free_sequences.back();
            free_sequences.pop_back();
            const double build_bind_ms = bind_lora(context, lora);
            const double prefix_start = now_ms();
            if (!eval_tokens(context, tokens.prefix, cache_seq, 0, options.n_ubatch)) {
                free_sequences.push_back(cache_seq);
                online_result result = run_uncached_request(
                        context,
                        memory,
                        options,
                        request,
                        tokens,
                        lora,
                        request_seq,
                        "online",
                        "anchor_build_failed_full_eval");
                results.push_back(result);
                continue;
            }
            const double prefix_ms = now_ms() - prefix_start;
            node.anchor_seq_id = cache_seq;
            node.variants.push_back({ request.lora_id, cache_seq, 0 });
            nodes.push_back(std::move(node));
            node_index = (int) nodes.size() - 1;

            online_result result;
            result.benchmark = "online";
            result.request_id = request.request_id;
            result.arrival_ms = request.arrival_ms;
            result.user_id = request.user_id;
            result.session_id = request.session_id;
            result.app_name = request.app_name;
            result.group_name = request.group_name;
            result.context_id = request.context_id;
            result.lora_id = request.lora_id;
            result.lora_name = lora.logical_name;
            result.mode = "build_exact_prefix_anchor";
            result.node_id = nodes[node_index].node_id;
            result.prefix_tokens = (int) tokens.prefix.size();
            result.suffix_tokens = (int) tokens.suffix.size();
            result.prompt_tokens = (int) tokens.full.size();
            result.prefix_ms = prefix_ms;
            result.lora_bind_ms = build_bind_ms;

            llama_memory_seq_rm(memory, request_seq, -1, -1);
            const double copy_start = now_ms();
            llama_memory_seq_cp(
                    memory,
                    cache_seq,
                    request_seq,
                    0,
                    (int) tokens.prefix.size());
            result.prefix_ms += now_ms() - copy_start;
            const double suffix_start = now_ms();
            if (eval_tokens(
                    context,
                    tokens.suffix,
                    request_seq,
                    (int) tokens.prefix.size(),
                    options.n_ubatch)) {
                result.suffix_ms = now_ms() - suffix_start;
                run_decode(
                        context,
                        request_seq,
                        tokens.full.back(),
                        (int) tokens.full.size(),
                        options.n_predict,
                        request_start,
                        result);
            }
            llama_memory_seq_rm(memory, request_seq, -1, -1);
            result.cache_nodes = (int) nodes.size();
            result.cache_variants = count_variants(nodes);
            result.physical_cache_tokens = count_cache_tokens(nodes);
            results.push_back(result);
            continue;
        }

        const int node_id = nodes[node_index].node_id;
        nodes[node_index].last_access_index = request_index;
        nodes[node_index].hit_count++;
        prefix_variant * variant = find_variant(nodes[node_index], request.lora_id);

        online_result result;
        result.benchmark = "online";
        result.request_id = request.request_id;
        result.arrival_ms = request.arrival_ms;
        result.user_id = request.user_id;
        result.session_id = request.session_id;
        result.app_name = request.app_name;
        result.group_name = request.group_name;
        result.context_id = request.context_id;
        result.lora_id = request.lora_id;
        result.lora_name = lora.logical_name;
        result.node_id = node_id;
        result.exact_prefix_hit = 1;
        result.prefix_tokens = (int) tokens.prefix.size();
        result.suffix_tokens = (int) tokens.suffix.size();
        result.prompt_tokens = (int) tokens.full.size();

        if (!variant) {
            result.cross_lora_prefix_match = 1;
            if (!ensure_cache_capacity(
                        memory,
                        nodes,
                        free_sequences,
                        options,
                        (int) tokens.prefix.size(),
                        node_id)) {
                result = run_uncached_request(
                        context,
                        memory,
                        options,
                        request,
                        tokens,
                        lora,
                        request_seq,
                        "online",
                        "cross_lora_prefix_match_no_variant_capacity");
                result.node_id = node_id;
                result.exact_prefix_hit = 1;
                result.cross_lora_prefix_match = 1;
                result.cache_nodes = (int) nodes.size();
                result.cache_variants = count_variants(nodes);
                result.physical_cache_tokens = count_cache_tokens(nodes);
                results.push_back(result);
                continue;
            }

            node_index = -1;
            for (int index = 0; index < (int) nodes.size(); ++index) {
                if (nodes[index].node_id == node_id) {
                    node_index = index;
                    break;
                }
            }
            if (node_index < 0) continue;

            const llama_seq_id child_cache_seq = free_sequences.back();
            free_sequences.pop_back();
            result.lora_bind_ms = bind_lora(context, lora);
            const double prefix_start = now_ms();
            if (!eval_tokens(
                        context,
                        tokens.prefix,
                        child_cache_seq,
                        0,
                        options.n_ubatch)) {
                free_sequences.push_back(child_cache_seq);
                continue;
            }
            result.prefix_ms = now_ms() - prefix_start;

            if (options.cross_lora_policy == "sync") {
                build_online_delta(
                        context,
                        memory,
                        nodes[node_index].anchor_seq_id,
                        child_cache_seq,
                        request_seq,
                        (int) tokens.prefix.size(),
                        nodes[node_index].anchor_lora_id,
                        request.lora_id,
                        result);
                result.mode = "build_cross_lora_sync_delta";
            } else if (options.cross_lora_policy == "deferred") {
                result.mode = "build_cross_lora_deferred_delta";
            } else {
                result.mode = "build_cross_lora_full_kv_only";
            }

            nodes[node_index].variants.push_back({ request.lora_id, child_cache_seq, 0 });
            variant = &nodes[node_index].variants.back();
        } else {
            result.same_lora_variant_hit = 1;
            result.mode = "reuse_exact_prefix_same_lora";
            variant->hit_count++;
            result.lora_bind_ms = bind_lora(context, lora);
        }

        llama_memory_seq_rm(memory, request_seq, -1, -1);
        const double copy_start = now_ms();
        llama_memory_seq_cp(
                memory,
                variant->cache_seq_id,
                request_seq,
                0,
                (int) tokens.prefix.size());
        result.prefix_ms += now_ms() - copy_start;

        const double suffix_start = now_ms();
        if (eval_tokens(
                    context,
                    tokens.suffix,
                    request_seq,
                    (int) tokens.prefix.size(),
                    options.n_ubatch)) {
            result.suffix_ms = now_ms() - suffix_start;
            run_decode(
                    context,
                    request_seq,
                    tokens.full.back(),
                    (int) tokens.full.size(),
                    options.n_predict,
                    request_start,
                    result);
        }
        llama_memory_seq_rm(memory, request_seq, -1, -1);
        if (result.cross_lora_prefix_match &&
                options.cross_lora_policy == "deferred" &&
                variant != nullptr) {
            const double background_start = now_ms();
            build_online_delta(
                    context,
                    memory,
                    nodes[node_index].anchor_seq_id,
                    variant->cache_seq_id,
                    request_seq,
                    (int) tokens.prefix.size(),
                    nodes[node_index].anchor_lora_id,
                    request.lora_id,
                    result);
            result.delta_background_ms = now_ms() - background_start;
        }
        result.cache_nodes = (int) nodes.size();
        result.cache_variants = count_variants(nodes);
        result.physical_cache_tokens = count_cache_tokens(nodes);
        results.push_back(result);
    }

    final_nodes = nodes;
    clear_lora(context);
    llama_free(context);
    return results;
}

// =============================================================================
// 7. Save results 保存结果
// =============================================================================

// System-v2 helpers are intentionally kept in this experiment binary so the
// legacy exact-prefix path remains available as a baseline.
static int count_gpu_cache_tokens_v2(const std::vector<prefix_node> & nodes) {
    int total = 0;
    for (const auto & node : nodes) {
        for (const auto & variant : node.variants) {
            if (variant.residency == variant_residency::gpu_full) total += node.depth_tokens;
        }
    }
    return total;
}

static int count_host_delta_variants(const std::vector<prefix_node> & nodes) {
    int total = 0;
    for (const auto & node : nodes) {
        for (const auto & variant : node.variants) {
            total += variant.residency == variant_residency::host_delta ? 1 : 0;
        }
    }
    return total;
}

static unsigned long long count_host_delta_bytes(const std::vector<prefix_node> & nodes) {
    unsigned long long total = 0;
    for (const auto & node : nodes) {
        for (const auto & variant : node.variants) {
            if (variant.residency == variant_residency::host_delta) total += variant.delta_bytes;
        }
    }
    return total;
}

static int find_node_by_id(const std::vector<prefix_node> & nodes, int node_id) {
    for (int index = 0; index < (int) nodes.size(); ++index) {
        if (nodes[index].node_id == node_id) return index;
    }
    return -1;
}

static bool node_is_leaf(const std::vector<prefix_node> & nodes, int node_id) {
    for (const auto & node : nodes) {
        if (node.parent_node_id == node_id) return false;
    }
    return true;
}

static double family_value(const prefix_node & node, int request_index) {
    double predicted = 0.0;
    double materialize = 0.0;
    unsigned long long bytes = 0;
    int variant_hits = 0;
    for (const auto & variant : node.variants) {
        predicted += variant.predicted_probability;
        materialize += variant.materialize_ms;
        bytes += variant.delta_bytes;
        variant_hits += variant.hit_count;
        if (variant.residency == variant_residency::gpu_full) {
            bytes += (unsigned long long) node.depth_tokens * 1024ULL;
        }
    }
    const int age = std::max(0, request_index - node.last_access_index);
    return 4.0 * std::log1p((double) node.hit_count + variant_hits)
            + 20.0 * predicted
            + 0.002 * node.depth_tokens
            + 4.0 / (1.0 + age)
            - 0.05 * ((double) bytes / 1024.0 / 1024.0)
            - 0.0005 * materialize;
}

static void release_node_v2(
        llama_context * context,
        llama_memory_t memory,
        const prefix_node & node,
        std::vector<llama_seq_id> & free_sequences) {
    for (const auto & variant : node.variants) {
        if (variant.residency == variant_residency::host_delta) {
            llama_kv_seq_delta_remove_branch(context, variant.cache_seq_id);
        } else {
            llama_memory_seq_rm(memory, variant.cache_seq_id, -1, -1);
        }
        free_sequences.push_back(variant.cache_seq_id);
    }
}

static bool ensure_cache_capacity_v2(
        llama_context * context,
        llama_memory_t memory,
        std::vector<prefix_node> & nodes,
        std::vector<llama_seq_id> & free_sequences,
        const experiment_options & options,
        int additional_gpu_tokens,
        unsigned long long additional_host_bytes,
        int protected_node_id,
        int request_index) {
    const unsigned long long host_limit =
            (unsigned long long) std::max(0, options.max_host_delta_mb) * 1024ULL * 1024ULL;
    while (free_sequences.empty() ||
            count_variants(nodes) >= options.max_cache_variants ||
            count_gpu_cache_tokens_v2(nodes) + additional_gpu_tokens > options.max_cache_tokens ||
            count_host_delta_bytes(nodes) + additional_host_bytes > host_limit ||
            (int) nodes.size() > options.max_cache_nodes) {
        int victim_index = -1;
        double victim_value = std::numeric_limits<double>::infinity();
        for (int index = 0; index < (int) nodes.size(); ++index) {
            if (nodes[index].node_id == protected_node_id ||
                    !node_is_leaf(nodes, nodes[index].node_id)) continue;
            const double value = family_value(nodes[index], request_index);
            if (value < victim_value) {
                victim_value = value;
                victim_index = index;
            }
        }
        if (victim_index < 0) return false;
        release_node_v2(context, memory, nodes[victim_index], free_sequences);
        nodes.erase(nodes.begin() + victim_index);
    }
    return true;
}

static int find_chunk_node(
        const std::vector<prefix_node> & nodes,
        const std::string & group_name,
        int parent_node_id,
        const std::vector<llama_token> & prefix,
        int depth_tokens) {
    for (int index = 0; index < (int) nodes.size(); ++index) {
        const auto & node = nodes[index];
        if (node.group_name != group_name || node.parent_node_id != parent_node_id ||
                node.depth_tokens != depth_tokens ||
                (int) node.prefix_tokens.size() != depth_tokens) continue;
        if (std::equal(node.prefix_tokens.begin(), node.prefix_tokens.end(), prefix.begin())) return index;
    }
    return -1;
}

static std::vector<int> ensure_chunk_path(
        std::vector<prefix_node> & nodes,
        const dataset_request & request,
        const tokenized_request & tokenized,
        int chunk_tokens,
        int & next_node_id) {
    std::vector<int> path;
    const std::vector<llama_token> & prefix = tokenized.prefix;
    const int size = (int) prefix.size();
    const int step = std::max(1, chunk_tokens);
    int parent_id = -1;
    for (const auto & segment : tokenized.prefix_segments) {
        const int segment_begin = std::max(0, std::min(segment.begin, size));
        const int segment_end = std::max(segment_begin, std::min(segment.end, size));
        for (int begin = segment_begin; begin < segment_end; begin += step) {
            const int end = std::min(segment_end, begin + step);
            int index = find_chunk_node(nodes, request.group_name, parent_id, prefix, end);
            if (index < 0) {
                prefix_node node;
                node.node_id = next_node_id++;
                node.parent_node_id = parent_id;
                node.group_name = request.group_name;
                node.context_id = request.context_id;
                node.prefix_hash = request.common_prefix_hash + ":" + std::to_string(end);
                node.prefix_tokens.assign(prefix.begin(), prefix.begin() + end);
                node.depth_tokens = end;
                node.chunk_begin = begin;
                node.chunk_end = end;
                node.segment_kind = segment.type;
                nodes.push_back(std::move(node));
                index = (int) nodes.size() - 1;
            }
            path.push_back(nodes[index].node_id);
            parent_id = nodes[index].node_id;
        }
    }
    return path;
}

static bool restore_variant_v2(
        llama_context * context,
        llama_memory_t memory,
        const prefix_node & node,
        prefix_variant & variant,
        llama_seq_id dst_seq,
        online_result * result) {
    llama_memory_seq_rm(memory, dst_seq, -1, -1);
    const double start = now_ms();
    if (variant.residency == variant_residency::gpu_full) {
        llama_memory_seq_cp(memory, variant.cache_seq_id, dst_seq, 0, node.depth_tokens);
        if (result) result->prefix_ms += now_ms() - start;
        return true;
    }
    llama_memory_seq_cp(memory, node.anchor_seq_id, dst_seq, 0, node.depth_tokens);
    llama_kv_delta_materialize_stats stats = {};
    const bool ok = llama_kv_seq_delta_materialize_branch(
            context, node.anchor_seq_id, variant.cache_seq_id, dst_seq,
            0, node.depth_tokens, &stats);
    const double elapsed = now_ms() - start;
    variant.materialize_ms = elapsed;
    if (result) {
        result->prefix_ms += elapsed;
        result->materialize_ms += elapsed;
        result->materialize_ok = ok ? 1 : 0;
        result->reconstruction_cos = variant.reconstruction_cos;
        result->reconstruction_l2 = variant.reconstruction_l2;
    }
    return ok;
}

static std::string delta_store_path(
        const experiment_options & options,
        const prefix_node & node,
        int child_lora_id) {
    if (options.delta_store_dir.empty()) return {};
    std::filesystem::path directory(options.delta_store_dir);
    std::string filename =
            node.group_name + "_" + node.prefix_hash +
            "_d" + std::to_string(node.depth_tokens) +
            "_a" + std::to_string(node.anchor_lora_id) +
            "_c" + std::to_string(child_lora_id) + ".kvdelta";
    for (char & ch : filename) {
        if (ch == '<' || ch == '>' || ch == ':' || ch == '"' || ch == '/' ||
                ch == '\\' || ch == '|' || ch == '?' || ch == '*') ch = '_';
    }
    return (directory / filename).string();
}

static bool should_load_delta_store(const experiment_options & options) {
    return options.delta_store_policy == "load" ||
            options.delta_store_policy == "auto";
}

static bool should_save_delta_store(const experiment_options & options) {
    return options.delta_store_policy == "build" ||
            options.delta_store_policy == "auto";
}

static bool convert_variant_to_host_delta(
        llama_context * context,
        llama_memory_t memory,
        prefix_node & node,
        prefix_variant & variant,
        llama_seq_id validation_seq,
        const experiment_options & options,
        const std::vector<prefix_node> & nodes,
        online_result * result) {
    if (variant.residency != variant_residency::gpu_full ||
            variant.lora_id == node.anchor_lora_id || node.anchor_seq_id < 0) return false;

    llama_kv_delta_probe_stats probe = {};
    double start = now_ms();
    const bool probe_ok = llama_kv_seq_delta_probe(
            context, node.anchor_seq_id, variant.cache_seq_id,
            0, node.depth_tokens, &probe);
    if (result) {
        result->delta_probe_ms += now_ms() - start;
        result->prefix_kv_cos = probe.kv_cos_avg;
        result->prefix_kv_l2 = probe.kv_l2_avg;
    }
    if (!probe_ok || !probe.can_reuse_as_delta ||
            probe.kv_cos_avg < 0.97 || probe.kv_l2_avg > 0.25) return false;

    llama_kv_delta_branch_stats branch = {};
    start = now_ms();
    const bool build_ok = llama_kv_seq_delta_build_branch(
            context, node.anchor_seq_id, variant.cache_seq_id, variant.cache_seq_id,
            0, node.depth_tokens, node.parent_node_id, node.node_id, &branch);
    if (result) {
        result->delta_build_ms += now_ms() - start;
        result->delta_build_ok = build_ok ? 1 : 0;
        result->delta_saved_rate = branch.logical_saved_rate;
    }
    if (!build_ok) return false;

    const unsigned long long delta_bytes = branch.delta_q8_bytes + branch.delta_scale_bytes;
    const unsigned long long host_limit =
            (unsigned long long) std::max(0, options.max_host_delta_mb) * 1024ULL * 1024ULL;
    if (count_host_delta_bytes(nodes) + delta_bytes > host_limit) {
        llama_kv_seq_delta_remove_branch(context, variant.cache_seq_id);
        return false;
    }

    // Reconstruct into a temporary sequence and compare it with the full child
    // before the dense child KV is removed.
    llama_memory_seq_rm(memory, validation_seq, -1, -1);
    llama_memory_seq_cp(memory, node.anchor_seq_id, validation_seq, 0, node.depth_tokens);
    llama_kv_delta_materialize_stats materialize = {};
    const double materialize_start = now_ms();
    const bool materialize_ok = llama_kv_seq_delta_materialize_branch(
            context, node.anchor_seq_id, variant.cache_seq_id, validation_seq,
            0, node.depth_tokens, &materialize);
    const double materialize_ms = now_ms() - materialize_start;
    llama_kv_delta_probe_stats reconstruction = {};
    const bool reconstruction_ok = materialize_ok && llama_kv_seq_delta_probe(
            context, variant.cache_seq_id, validation_seq,
            0, node.depth_tokens, &reconstruction);
    llama_memory_seq_rm(memory, validation_seq, -1, -1);
    if (!materialize_ok) {
        llama_kv_seq_delta_remove_branch(context, variant.cache_seq_id);
        return false;
    }
    if (result) {
        result->delta_validation_ms += materialize_ms;
        if (reconstruction_ok) {
            result->reconstruction_cos = reconstruction.kv_cos_avg;
            result->reconstruction_l2 = reconstruction.kv_l2_avg;
        }
    }
    llama_memory_seq_rm(memory, variant.cache_seq_id, -1, -1);
    variant.residency = variant_residency::host_delta;
    variant.delta_bytes = delta_bytes;
    variant.materialize_ms = materialize_ms;
    if (reconstruction_ok) {
        variant.reconstruction_cos = reconstruction.kv_cos_avg;
        variant.reconstruction_l2 = reconstruction.kv_l2_avg;
    }
    if (should_save_delta_store(options)) {
        const std::string path = delta_store_path(options, node, variant.lora_id);
        if (!path.empty()) {
            std::filesystem::create_directories(std::filesystem::path(path).parent_path());
            if (llama_kv_seq_delta_save_branch(
                        context, variant.cache_seq_id, path.c_str()) && result) {
                result->delta_store_saved++;
            }
        }
    }
    return true;
}

static int load_stored_deltas_for_path(
        llama_context * context,
        llama_memory_t memory,
        const experiment_options & options,
        std::vector<prefix_node> & nodes,
        const std::vector<int> & path,
        int lora_id,
        std::vector<llama_seq_id> & free_sequences,
        int request_index,
        online_result & result) {
    if (!should_load_delta_store(options)) return 0;
    int loaded = 0;
    for (int node_id : path) {
        int index = find_node_by_id(nodes, node_id);
        if (index < 0 || nodes[index].anchor_seq_id < 0 ||
                nodes[index].anchor_lora_id == lora_id ||
                find_variant(nodes[index], lora_id) != nullptr) continue;
        const std::string file = delta_store_path(options, nodes[index], lora_id);
        if (file.empty() || !std::filesystem::exists(file)) continue;
        const unsigned long long bytes = std::filesystem::file_size(file);
        if (!ensure_cache_capacity_v2(
                    context, memory, nodes, free_sequences, options,
                    0, bytes, node_id, request_index) || free_sequences.empty()) continue;
        index = find_node_by_id(nodes, node_id);
        if (index < 0) continue;
        const llama_seq_id delta_seq = free_sequences.back();
        free_sequences.pop_back();
        if (!llama_kv_seq_delta_load_branch(
                    context,
                    nodes[index].anchor_seq_id,
                    delta_seq,
                    file.c_str(),
                    nodes[index].parent_node_id,
                    nodes[index].node_id)) {
            free_sequences.push_back(delta_seq);
            continue;
        }
        prefix_variant variant;
        variant.lora_id = lora_id;
        variant.cache_seq_id = delta_seq;
        variant.last_access_index = request_index;
        variant.residency = variant_residency::host_delta;
        variant.delta_bytes = bytes;
        nodes[index].variants.push_back(variant);
        nodes[index].last_access_index = request_index;
        loaded++;
    }
    result.delta_store_loaded += loaded;
    return loaded;
}

static std::vector<int> cache_missing_chunks(
        llama_context * context,
        llama_memory_t memory,
        const experiment_options & options,
        std::vector<prefix_node> & nodes,
        const std::vector<int> & path,
        int lora_id,
        llama_seq_id request_seq,
        std::vector<llama_seq_id> & free_sequences,
        int request_index) {
    std::vector<int> added;
    for (int node_id : path) {
        int index = find_node_by_id(nodes, node_id);
        if (index < 0 || find_variant(nodes[index], lora_id) != nullptr) continue;
        const int depth = nodes[index].depth_tokens;
        if (!ensure_cache_capacity_v2(
                    context, memory, nodes, free_sequences, options,
                    depth, 0, node_id, request_index)) continue;
        index = find_node_by_id(nodes, node_id);
        if (index < 0 || free_sequences.empty()) continue;
        const llama_seq_id cache_seq = free_sequences.back();
        free_sequences.pop_back();
        llama_memory_seq_rm(memory, cache_seq, -1, -1);
        llama_memory_seq_cp(memory, request_seq, cache_seq, 0, depth);
        prefix_variant variant;
        variant.lora_id = lora_id;
        variant.cache_seq_id = cache_seq;
        variant.last_access_index = request_index;
        nodes[index].variants.push_back(variant);
        if (nodes[index].anchor_seq_id < 0) {
            nodes[index].anchor_seq_id = cache_seq;
            nodes[index].anchor_lora_id = lora_id;
        }
        nodes[index].last_access_index = request_index;
        added.push_back(node_id);
    }
    return added;
}

static bool background_allowed(const experiment_options & options, double idle_gap_ms) {
    if (options.background_policy == "none") return false;
    if (options.background_policy == "unlimited") return true;
    return idle_gap_ms >= options.background_min_gap_ms;
}

static int prefetch_oracle_chunks(
        llama_context * context,
        llama_memory_t memory,
        const llama_vocab * vocab,
        const experiment_options & options,
        const dataset_request & next_request,
        lora_runtime & next_lora,
        std::vector<prefix_node> & nodes,
        std::vector<llama_seq_id> & free_sequences,
        llama_seq_id validation_seq,
        int request_index,
        int & next_node_id,
        double predicted_probability,
        online_result & current_result,
        int & full_built,
        int & delta_built,
        int & skipped_no_anchor) {
    tokenized_request tokens;
    if (!tokenize_request(vocab, next_request, tokens) || tokens.prefix.empty()) return 0;
    const int chunk_size = options.prefix_chunk_tokens > 0
            ? options.prefix_chunk_tokens : std::max(1, next_request.chunk_token_size);
    const std::vector<int> path = ensure_chunk_path(
            nodes, next_request, tokens, chunk_size, next_node_id);
    int built = 0;
    bind_lora(context, next_lora);
    load_stored_deltas_for_path(
            context, memory, options, nodes, path, next_request.lora_id,
            free_sequences, request_index, current_result);

    for (int node_id : path) {
        int index = find_node_by_id(nodes, node_id);
        if (index < 0) continue;
        prefix_variant * existing = find_variant(nodes[index], next_request.lora_id);
        if (existing != nullptr) {
            existing->predicted_probability = std::max(
                    existing->predicted_probability, predicted_probability);
            continue;
        }
        if (options.prefetch_storage == "delta" && nodes[index].anchor_seq_id < 0) {
            skipped_no_anchor++;
            continue;
        }
        const int depth = nodes[index].depth_tokens;
        if (!ensure_cache_capacity_v2(
                    context, memory, nodes, free_sequences, options,
                    depth, 0, node_id, request_index) || free_sequences.empty()) break;
        index = find_node_by_id(nodes, node_id);
        if (index < 0) break;
        const llama_seq_id seq = free_sequences.back();
        free_sequences.pop_back();
        llama_memory_seq_rm(memory, seq, -1, -1);

        bool ready = false;
        if (nodes[index].parent_node_id >= 0) {
            const int parent_index = find_node_by_id(nodes, nodes[index].parent_node_id);
            if (parent_index >= 0) {
                prefix_variant * parent = find_variant(nodes[parent_index], next_request.lora_id);
                if (parent != nullptr && restore_variant_v2(
                            context, memory, nodes[parent_index], *parent, seq, nullptr)) {
                    const std::vector<llama_token> chunk(
                            tokens.prefix.begin() + nodes[index].chunk_begin,
                            tokens.prefix.begin() + nodes[index].chunk_end);
                    ready = eval_tokens(
                            context, chunk, seq, nodes[index].chunk_begin, options.n_ubatch);
                }
            }
        }
        if (!ready) {
            const std::vector<llama_token> cumulative(tokens.prefix.begin(), tokens.prefix.begin() + depth);
            ready = eval_tokens(context, cumulative, seq, 0, options.n_ubatch);
        }
        if (!ready) {
            free_sequences.push_back(seq);
            continue;
        }

        prefix_variant variant;
        variant.lora_id = next_request.lora_id;
        variant.cache_seq_id = seq;
        variant.last_access_index = request_index;
        variant.predicted_probability = predicted_probability;
        nodes[index].variants.push_back(variant);
        prefix_variant & stored = nodes[index].variants.back();
        if (nodes[index].anchor_seq_id < 0) {
            nodes[index].anchor_seq_id = seq;
            nodes[index].anchor_lora_id = next_request.lora_id;
            full_built++;
        } else if (nodes[index].anchor_lora_id != next_request.lora_id) {
            if (options.prefetch_storage == "full") {
                full_built++;
            } else {
                const bool converted = convert_variant_to_host_delta(
                        context, memory, nodes[index], stored, validation_seq,
                        options, nodes, &current_result);
                if (converted) {
                    delta_built++;
                } else if (options.prefetch_storage == "delta") {
                    llama_memory_seq_rm(memory, stored.cache_seq_id, -1, -1);
                    free_sequences.push_back(stored.cache_seq_id);
                    nodes[index].variants.pop_back();
                    continue;
                } else {
                    full_built++;
                }
            }
        } else {
            full_built++;
        }
        built++;
    }
    return built;
}

struct prefetch_job {
    dataset_request request;
    double probability = 1.0;
    int enqueue_index = -1;
    int target_request_id = -1;
    long long deadline_arrival_ms = 0;
};

static std::vector<online_result> run_online_system_v2(
        llama_model * model,
        const llama_vocab * vocab,
        const experiment_options & options,
        const std::vector<dataset_request> & requests,
        const std::unordered_map<int, lora_runtime *> & loras,
        std::vector<prefix_node> & final_nodes) {
    std::vector<online_result> results;
    const int cache_sequence_count = std::max(2, options.max_cache_variants);
    const llama_seq_id request_seq = cache_sequence_count;
    const llama_seq_id validation_seq = cache_sequence_count + 1;
    llama_context * context = create_context(model, options, cache_sequence_count + 2);
    if (!context) return results;
    llama_memory_t memory = llama_get_memory(context);

    std::vector<llama_seq_id> free_sequences;
    for (int sequence = cache_sequence_count - 1; sequence >= 0; --sequence) {
        free_sequences.push_back((llama_seq_id) sequence);
    }
    std::vector<prefix_node> nodes;
    std::deque<prefetch_job> prefetch_queue;
    const auto file_predictions = load_prediction_file(options.prediction_file);
    int next_node_id = 0;
    const int limit = std::min(options.max_online_requests, (int) requests.size());

    for (int request_index = 0; request_index < limit; ++request_index) {
        const dataset_request & request = requests[request_index];
        int dropped_expired = 0;
        while (!prefetch_queue.empty() &&
                prefetch_queue.front().target_request_id <= request.request_id) {
            prefetch_queue.pop_front();
            dropped_expired++;
        }
        const auto lora_it = loras.find(request.lora_id);
        if (lora_it == loras.end()) continue;
        lora_runtime & lora = *lora_it->second;
        tokenized_request tokens;
        if (!tokenize_request(vocab, request, tokens) || tokens.full.empty()) continue;

        const int chunk_size = options.prefix_chunk_tokens > 0
                ? options.prefix_chunk_tokens : std::max(1, request.chunk_token_size);
        const std::vector<int> path = ensure_chunk_path(
                nodes, request, tokens, chunk_size, next_node_id);

        online_result result;
        result.benchmark = "online";
        result.request_id = request.request_id;
        result.arrival_ms = request.arrival_ms;
        result.user_id = request.user_id;
        result.session_id = request.session_id;
        result.app_name = request.app_name;
        result.group_name = request.group_name;
        result.context_id = request.context_id;
        result.lora_id = request.lora_id;
        result.lora_name = lora.logical_name;
        result.prefix_tokens = (int) tokens.prefix.size();
        result.suffix_tokens = (int) tokens.suffix.size();
        result.prompt_tokens = (int) tokens.full.size();
        result.prefetch_dropped_expired = dropped_expired;
        const double request_start = now_ms();
        result.lora_bind_ms = bind_lora(context, lora);
        load_stored_deltas_for_path(
                context, memory, options, nodes, path, request.lora_id,
                free_sequences, request_index, result);

        int hit_index = -1;
        prefix_variant * hit_variant = nullptr;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            const int index = find_node_by_id(nodes, *it);
            if (index < 0) continue;
            prefix_variant * variant = find_variant(nodes[index], request.lora_id);
            if (variant != nullptr) {
                hit_index = index;
                hit_variant = variant;
                break;
            }
        }

        int hit_depth = 0;
        bool restored = false;
        if (hit_index >= 0 && hit_variant != nullptr) {
            hit_depth = nodes[hit_index].depth_tokens;
            restored = restore_variant_v2(
                    context, memory, nodes[hit_index], *hit_variant, request_seq, &result);
            if (restored) {
                hit_variant->hit_count++;
                hit_variant->last_access_index = request_index;
                nodes[hit_index].hit_count++;
                nodes[hit_index].last_access_index = request_index;
                result.node_id = nodes[hit_index].node_id;
                result.chunk_hit_tokens = hit_depth;
                result.exact_prefix_hit = hit_depth == (int) tokens.prefix.size() ? 1 : 0;
                result.same_lora_variant_hit = result.exact_prefix_hit;
                result.mode = hit_variant->residency == variant_residency::host_delta
                        ? "chunk_prefix_delta_materialize" : "chunk_prefix_gpu_reuse";
            }
        }
        if (!restored) {
            llama_memory_seq_rm(memory, request_seq, -1, -1);
            hit_depth = 0;
            result.mode = "chunk_prefix_build";
            for (int node_id : path) {
                const int index = find_node_by_id(nodes, node_id);
                if (index >= 0 && !nodes[index].variants.empty() &&
                        nodes[index].anchor_lora_id != request.lora_id) {
                    result.cross_lora_prefix_match = 1;
                    break;
                }
            }
        }

        const double prefix_start = now_ms();
        if (hit_depth < (int) tokens.prefix.size()) {
            const std::vector<llama_token> remaining(
                    tokens.prefix.begin() + hit_depth, tokens.prefix.end());
            if (!eval_tokens(context, remaining, request_seq, hit_depth, options.n_ubatch)) {
                result.mode += "_prefix_eval_failed";
                results.push_back(result);
                continue;
            }
        }
        result.prefix_ms += now_ms() - prefix_start;

        const std::vector<int> added = cache_missing_chunks(
                context, memory, options, nodes, path, request.lora_id,
                request_seq, free_sequences, request_index);

        if (options.cross_lora_policy == "sync") {
            for (int node_id : added) {
                const int index = find_node_by_id(nodes, node_id);
                if (index < 0) continue;
                prefix_variant * variant = find_variant(nodes[index], request.lora_id);
                if (variant != nullptr) {
                    convert_variant_to_host_delta(
                            context, memory, nodes[index], *variant, validation_seq,
                            options, nodes, &result);
                }
            }
        }

        const double suffix_start = now_ms();
        if (eval_tokens(context, tokens.suffix, request_seq,
                    (int) tokens.prefix.size(), options.n_ubatch)) {
            result.suffix_ms = now_ms() - suffix_start;
            run_decode(context, request_seq, tokens.full.back(),
                    (int) tokens.full.size(), options.n_predict, request_start, result);
        }
        llama_memory_seq_rm(memory, request_seq, -1, -1);

        const double next_arrival = request_index + 1 < limit
                ? (double) requests[request_index + 1].arrival_ms
                : (double) request.arrival_ms;
        result.idle_gap_ms = std::max(0.0, next_arrival - (double) request.arrival_ms);

        if (options.cross_lora_policy == "deferred" &&
                background_allowed(options, result.idle_gap_ms)) {
            const double background_start = now_ms();
            online_result background_metrics;
            for (int node_id : added) {
                const int index = find_node_by_id(nodes, node_id);
                if (index < 0) continue;
                prefix_variant * variant = find_variant(nodes[index], request.lora_id);
                if (variant != nullptr) {
                    convert_variant_to_host_delta(
                            context, memory, nodes[index], *variant, validation_seq,
                            options, nodes, &background_metrics);
                }
            }
            result.delta_background_ms = now_ms() - background_start;
        }

        if (options.prefetch_policy == "oracle" && request_index + 1 < limit) {
            const dataset_request & next_request = requests[request_index + 1];
            result.predicted_lora_id = next_request.lora_id;
            bool duplicate = false;
            for (const auto & job : prefetch_queue) {
                if (job.request.context_id == next_request.context_id &&
                        job.request.lora_id == next_request.lora_id) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                prefetch_queue.push_back({
                        next_request,
                        1.0,
                        request_index,
                        next_request.request_id,
                        next_request.arrival_ms });
            }
        } else if (options.prefetch_policy == "file" && request_index + 1 < limit) {
            const auto prediction_it = file_predictions.find(request.request_id);
            if (prediction_it != file_predictions.end()) {
                const int top_k = std::min(options.prediction_top_k,
                        (int) prediction_it->second.size());
                for (int prediction_index = 0; prediction_index < top_k; ++prediction_index) {
                    const prediction_candidate & candidate = prediction_it->second[prediction_index];
                    if (loras.find(candidate.lora_id) == loras.end()) continue;
                    dataset_request predicted_request = request;
                    predicted_request.lora_id = candidate.lora_id;
                    if (prediction_index == 0) result.predicted_lora_id = candidate.lora_id;
                    bool duplicate = false;
                    for (const auto & job : prefetch_queue) {
                        if (job.request.context_id == predicted_request.context_id &&
                                job.request.lora_id == predicted_request.lora_id) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        prefetch_queue.push_back({
                                predicted_request,
                                candidate.probability,
                                request_index,
                                requests[request_index + 1].request_id,
                                requests[request_index + 1].arrival_ms });
                    }
                }
            }
        }

        double remaining_gap = options.background_policy == "unlimited"
                ? std::numeric_limits<double>::infinity()
                : result.idle_gap_ms;
        while (!prefetch_queue.empty() &&
                background_allowed(options, remaining_gap)) {
            const prefetch_job job = prefetch_queue.front();
            const auto predicted_lora = loras.find(job.request.lora_id);
            if (predicted_lora == loras.end()) {
                prefetch_queue.pop_front();
                continue;
            }
            const double prefetch_start = now_ms();
            online_result prefetch_metrics;
            int full_built = 0;
            int delta_built = 0;
            int skipped_no_anchor = 0;
            result.prefetch_built += prefetch_oracle_chunks(
                    context, memory, vocab, options, job.request, *predicted_lora->second,
                     nodes, free_sequences, validation_seq, request_index,
                     next_node_id, job.probability, prefetch_metrics,
                    full_built, delta_built, skipped_no_anchor);
            result.prefetch_full_built += full_built;
            result.prefetch_delta_built += delta_built;
            result.prefetch_skipped_no_anchor += skipped_no_anchor;
            const double elapsed = now_ms() - prefetch_start;
            result.prefetch_ms += elapsed;
            prefetch_queue.pop_front();
            if (options.background_policy != "unlimited") {
                remaining_gap -= elapsed;
                if (remaining_gap < 0.0) {
                    result.background_overrun_ms += -remaining_gap;
                    break;
                }
            }
        }

        result.cache_nodes = (int) nodes.size();
        result.cache_variants = count_variants(nodes);
        result.physical_cache_tokens = count_gpu_cache_tokens_v2(nodes);
        result.host_delta_variants = count_host_delta_variants(nodes);
        result.host_delta_bytes = count_host_delta_bytes(nodes);
        result.background_queue_length = (int) prefetch_queue.size();
        results.push_back(result);
    }

    final_nodes = nodes;
    clear_lora(context);
    for (const auto & node : nodes) release_node_v2(context, memory, node, free_sequences);
    llama_free(context);
    return results;
}

static void save_delta_results(
        const std::string & output_dir,
        const std::vector<delta_result> & results,
        const std::vector<delta_layer_result> & layers) {
    std::filesystem::create_directories(output_dir);

    std::ofstream summary(output_dir + "/delta_prefix_probe.csv");
    summary << "pair_name,group_name,context_id,anchor_request_id,child_request_id,"
            << "anchor_lora_id,child_lora_id,anchor_lora_name,child_lora_name,"
            << "prefix_tokens,status,memory_kind,probed_layers,skipped_recurrent_layers,"
            << "can_encode_delta,kv_cos,kv_l2,anchor_eval_ms,child_eval_ms,probe_ms,"
            << "delta_build_ms,delta_build_ok,full_kv_bytes,delta_q8_bytes,"
            << "delta_scale_bytes,logical_saved_bytes,logical_saved_rate\n";
    for (const auto & row : results) {
        summary << csv_escape(row.pair_name) << ','
                << csv_escape(row.group_name) << ','
                << csv_escape(row.context_id) << ','
                << row.anchor_request_id << ',' << row.child_request_id << ','
                << row.anchor_lora_id << ',' << row.child_lora_id << ','
                << csv_escape(row.anchor_lora_name) << ','
                << csv_escape(row.child_lora_name) << ','
                << row.prefix_tokens << ',' << csv_escape(row.status) << ','
                << csv_escape(row.memory_kind) << ',' << row.probed_layers << ','
                << row.skipped_recurrent_layers << ',' << row.can_encode_delta << ','
                << row.kv_cos << ',' << row.kv_l2 << ','
                << row.anchor_eval_ms << ',' << row.child_eval_ms << ','
                << row.probe_ms << ',' << row.delta_build_ms << ','
                << row.delta_build_ok << ',' << row.full_kv_bytes << ','
                << row.delta_q8_bytes << ',' << row.delta_scale_bytes << ','
                << row.logical_saved_bytes << ',' << row.logical_saved_rate << '\n';
    }

    std::ofstream layer_file(output_dir + "/delta_prefix_layers.csv");
    layer_file << "pair_name,layer_id,k_cos,v_cos,kv_cos,k_l2,v_l2,kv_l2\n";
    for (const auto & row : layers) {
        layer_file << csv_escape(row.pair_name) << ',' << row.layer_id << ','
                   << row.k_cos << ',' << row.v_cos << ',' << row.kv_cos << ','
                   << row.k_l2 << ',' << row.v_l2 << ',' << row.kv_l2 << '\n';
    }
}

static void save_online_results(
        const std::string & output_dir,
        const std::vector<online_result> & results) {
    std::filesystem::create_directories(output_dir);
    std::ofstream output(output_dir + "/online_request_results.csv");
    output << "benchmark,request_id,arrival_ms,user_id,session_id,app_name,group_name,"
           << "context_id,lora_id,lora_name,mode,node_id,exact_prefix_hit,"
           << "same_lora_variant_hit,cross_lora_prefix_match,prefix_tokens,suffix_tokens,"
           << "prompt_tokens,delta_candidate,prefix_kv_cos,prefix_kv_l2,delta_probe_ms,"
           << "delta_build_ms,delta_background_ms,delta_build_ok,delta_saved_rate,lora_bind_ms,"
           << "prefix_ms,suffix_ms,ttft_ms,decode_ms,total_ms,tps,cache_nodes,"
           << "cache_variants,physical_cache_tokens,chunk_hit_tokens,materialize_ok,"
           << "materialize_ms,delta_validation_ms,reconstruction_cos,reconstruction_l2,predicted_lora_id,"
           << "idle_gap_ms,prefetch_ms,prefetch_built,prefetch_full_built,prefetch_delta_built,"
           << "prefetch_skipped_no_anchor,prefetch_dropped_expired,delta_store_loaded,delta_store_saved,host_delta_variants,"
           << "host_delta_bytes,background_queue_length,background_overrun_ms\n";
    for (const auto & row : results) {
        output << csv_escape(row.benchmark) << ',' << row.request_id << ','
               << row.arrival_ms << ',' << row.user_id << ',' << row.session_id << ','
               << csv_escape(row.app_name) << ',' << csv_escape(row.group_name) << ','
               << csv_escape(row.context_id) << ',' << row.lora_id << ','
               << csv_escape(row.lora_name) << ',' << csv_escape(row.mode) << ','
               << row.node_id << ',' << row.exact_prefix_hit << ','
               << row.same_lora_variant_hit << ',' << row.cross_lora_prefix_match << ','
               << row.prefix_tokens << ',' << row.suffix_tokens << ','
               << row.prompt_tokens << ',' << row.delta_candidate << ','
               << row.prefix_kv_cos << ',' << row.prefix_kv_l2 << ','
               << row.delta_probe_ms << ',' << row.delta_build_ms << ','
               << row.delta_background_ms << ',' << row.delta_build_ok << ','
               << row.delta_saved_rate << ','
               << row.lora_bind_ms << ',' << row.prefix_ms << ',' << row.suffix_ms << ','
               << row.ttft_ms << ',' << row.decode_ms << ',' << row.total_ms << ','
               << row.tps << ',' << row.cache_nodes << ',' << row.cache_variants << ','
               << row.physical_cache_tokens << ',' << row.chunk_hit_tokens << ','
               << row.materialize_ok << ',' << row.materialize_ms << ','
               << row.delta_validation_ms << ',' << row.reconstruction_cos << ','
               << row.reconstruction_l2 << ','
               << row.predicted_lora_id << ',' << row.idle_gap_ms << ','
               << row.prefetch_ms << ',' << row.prefetch_built << ','
               << row.prefetch_full_built << ',' << row.prefetch_delta_built << ','
               << row.prefetch_skipped_no_anchor << ','
               << row.prefetch_dropped_expired << ',' << row.delta_store_loaded << ','
               << row.delta_store_saved << ',' << row.host_delta_variants << ','
               << row.host_delta_bytes << ','
               << row.background_queue_length << ',' << row.background_overrun_ms << '\n';
    }
}

static void save_tree_summary(
        const std::string & output_dir,
        const std::vector<prefix_node> & nodes) {
    std::filesystem::create_directories(output_dir);
    std::ofstream output(output_dir + "/online_prefix_tree.csv");
    output << "node_id,parent_node_id,group_name,context_id,prefix_hash,prefix_tokens,"
           << "chunk_begin,chunk_end,segment_kind,anchor_lora_id,node_hit_count,"
           << "last_access_index,variant_lora_id,variant_seq_id,variant_hit_count,"
           << "variant_residency,delta_bytes,materialize_ms,predicted_probability,"
           << "reconstruction_cos,reconstruction_l2\n";
    for (const auto & node : nodes) {
        for (const auto & variant : node.variants) {
            output << node.node_id << ',' << node.parent_node_id << ','
                   << csv_escape(node.group_name) << ','
                   << csv_escape(node.context_id) << ',' << node.prefix_hash << ','
                   << node.prefix_tokens.size() << ',' << node.chunk_begin << ','
                   << node.chunk_end << ',' << csv_escape(node.segment_kind) << ','
                   << node.anchor_lora_id << ','
                   << node.hit_count << ',' << node.last_access_index << ','
                   << variant.lora_id << ',' << variant.cache_seq_id << ','
                   << variant.hit_count << ','
                   << (variant.residency == variant_residency::gpu_full ? "gpu_full" : "host_delta")
                   << ',' << variant.delta_bytes << ',' << variant.materialize_ms << ','
                   << variant.predicted_probability << ',' << variant.reconstruction_cos << ','
                   << variant.reconstruction_l2 << '\n';
        }
    }
}

// =============================================================================
// 8. Main 主函数
// =============================================================================

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    // 解析参数
    experiment_options options;
    if (!parse_options(argc, argv, options)) {
        return argc > 1 ? 1 : 0;
    }
    // 构造文件路径
    const std::string delta_request_path = options.workload_dir + "/delta/delta_requests.jsonl";
    const std::string delta_pair_path = options.workload_dir + "/delta/delta_pairs.jsonl";
    const std::string grouped_request_path = options.workload_dir + "/grouped/grouped_requests.jsonl";

    // 加载 LoRA 配置
    std::vector<lora_runtime> lora_list = load_lora_config(options.lora_config_path);
    if (lora_list.empty()) {
        fprintf(stderr, "no LoRA configuration loaded\n");
        return 1;
    }

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(
            options.model_path.c_str(),
            model_params);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", options.model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::unordered_map<int, lora_runtime *> loaded_loras;
    for (auto & lora : lora_list) {
        if (lora.adapter_path.empty()) {
            fprintf(stderr,
                    "skip LoRA %d (%s): actual_adapter_path is empty\n",
                    lora.lora_id,
                    lora.logical_name.c_str());
            continue;
        }
        lora.adapter = llama_adapter_lora_init(model, lora.adapter_path.c_str());
        if (!lora.adapter) {
            fprintf(stderr,
                    "skip LoRA %d (%s): failed to load %s\n",
                    lora.lora_id,
                    lora.logical_name.c_str(),
                    lora.adapter_path.c_str());
            continue;
        }
        loaded_loras[lora.lora_id] = &lora;
    }

    if (loaded_loras.size() < 2) {
        fprintf(stderr,
                "need at least two valid adapters. Fill actual_adapter_path in %s\n",
                options.lora_config_path.c_str());
        for (auto & lora : lora_list) {
            if (lora.adapter) llama_adapter_lora_free(lora.adapter);
        }
        llama_model_free(model);
        return 1;
    }

    const std::vector<dataset_request> delta_requests =
            load_requests(delta_request_path);
    const std::vector<delta_pair_item> delta_pairs =
            load_delta_pairs(delta_pair_path);
    const std::vector<dataset_request> grouped_requests =
            load_requests(grouped_request_path);

    std::unordered_map<int, dataset_request> delta_request_map;
    for (const auto & request : delta_requests) {
        delta_request_map[request.request_id] = request;
    }

    fprintf(stderr,
            "loaded workload: delta_requests=%zu delta_pairs=%zu grouped_requests=%zu adapters=%zu\n",
            delta_requests.size(),
            delta_pairs.size(),
            grouped_requests.size(),
            loaded_loras.size());

    std::vector<delta_result> delta_results;
    std::vector<delta_layer_result> delta_layers;
    run_delta_experiment(
            model,
            vocab,
            options,
            delta_request_map,
            delta_pairs,
            loaded_loras,
            delta_results,
            delta_layers);
    save_delta_results(options.output_dir, delta_results, delta_layers);

    std::vector<online_result> online_results = run_baseline_experiment(
            model,
            vocab,
            options,
            grouped_requests,
            loaded_loras);

    std::vector<prefix_node> final_nodes;
    std::vector<online_result> exact_prefix_results;
    if (options.system_v2) {
        exact_prefix_results = run_online_system_v2(
                model,
                vocab,
                options,
                grouped_requests,
                loaded_loras,
                final_nodes);
    } else {
        exact_prefix_results = run_online_experiment(
                model,
                vocab,
                options,
                grouped_requests,
                loaded_loras,
                final_nodes);
    }
    online_results.insert(
            online_results.end(),
            exact_prefix_results.begin(),
            exact_prefix_results.end());
    save_online_results(options.output_dir, online_results);
    save_tree_summary(options.output_dir, final_nodes);

    for (auto & lora : lora_list) {
        if (lora.adapter) llama_adapter_lora_free(lora.adapter);
    }
    llama_model_free(model);

    fprintf(stderr,
            "dataset experiment finished: delta_rows=%zu online_rows=%zu nodes=%zu output=%s\n",
            delta_results.size(),
            online_results.size(),
            final_nodes.size(),
            options.output_dir.c_str());
    return 0;
}
