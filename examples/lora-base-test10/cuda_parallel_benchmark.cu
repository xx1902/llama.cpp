// Scheduling microbenchmark for the lora-base-test10 personal experiment.
//
// This program compares two independent questions:
//   1. Does CPU TF-IDF overlap with a GPU inference-like kernel?
//   2. Does a low-priority CUDA keyword proxy overlap with a high-priority
//      inference-like kernel?
//
// The CUDA kernels are scheduling proxies, not an LLM or BGE-M3 implementation.

#include <cuda_runtime.h>

#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json = nlohmann::json;

#define CUDA_CHECK(call) check_cuda((call), #call, __FILE__, __LINE__)

void check_cuda(cudaError_t status, const char * expression, const char * file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    throw std::runtime_error(
        std::string("CUDA failure: ") + expression + " at " + file + ":" + std::to_string(line) +
        " (" + cudaGetErrorString(status) + ")");
}

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    int limit = 87;
    int trials = 3;
    int inference_iterations = 32000;
    int keyword_iterations = 22000;
};

void print_usage(const char * program) {
    std::cout
        << "Usage: " << program << " --input REQUESTS.jsonl --output RESULTS.csv [options]\n"
        << "Options:\n"
        << "  --limit N                  requests to benchmark (default: 87)\n"
        << "  --trials N                 trials per request/profile (default: 3)\n"
        << "  --inference-iterations N   CUDA inference proxy work (default: 32000)\n"
        << "  --keyword-iterations N     CUDA keyword proxy work (default: 22000)\n";
}

int parse_positive(const std::string & value, const char * name) {
    const int parsed = std::stoi(value);
    if (parsed <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
    return parsed;
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                throw std::runtime_error("missing value after " + arg);
            }
            return argv[i];
        };
        if (arg == "--input") {
            options.input = next();
        } else if (arg == "--output") {
            options.output = next();
        } else if (arg == "--limit") {
            options.limit = parse_positive(next(), "--limit");
        } else if (arg == "--trials") {
            options.trials = parse_positive(next(), "--trials");
        } else if (arg == "--inference-iterations") {
            options.inference_iterations = parse_positive(next(), "--inference-iterations");
        } else if (arg == "--keyword-iterations") {
            options.keyword_iterations = parse_positive(next(), "--keyword-iterations");
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (options.input.empty() || options.output.empty()) {
        throw std::runtime_error("--input and --output are required");
    }
    return options;
}

struct Request {
    int request_id = 0;
    int prefix_tokens = 0;
    int lora_id = 0;
    std::string context_id;
    std::string text;
};

std::vector<Request> load_requests(const Options & options) {
    std::ifstream input(options.input);
    if (!input) {
        throw std::runtime_error("cannot open dataset: " + options.input.string());
    }

    std::vector<Request> requests;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const json item = json::parse(line);
        Request request;
        request.request_id = item.value("request_id", static_cast<int>(requests.size()));
        request.prefix_tokens = item.value("estimated_common_prefix_tokens", 0);
        request.lora_id = item.value("lora_id", -1);
        request.context_id = item.value("context_id", std::string("unknown"));
        request.text = item.value("common_prefix_text", std::string());
        if (!request.text.empty()) {
            requests.push_back(std::move(request));
        }
    }
    if (requests.empty()) {
        throw std::runtime_error("dataset contained no usable common_prefix_text rows");
    }
    return requests;
}

std::vector<Request> select_by_lora(const std::vector<Request> & corpus, int limit) {
    if (limit >= static_cast<int>(corpus.size())) {
        return corpus;
    }
    std::map<int, std::vector<const Request *>> by_lora;
    for (const Request & request : corpus) {
        by_lora[request.lora_id].push_back(&request);
    }
    std::vector<Request> selected;
    selected.reserve(limit);
    for (std::size_t round = 0; static_cast<int>(selected.size()) < limit; ++round) {
        bool added = false;
        for (const auto & entry : by_lora) {
            if (round < entry.second.size()) {
                selected.push_back(*entry.second[round]);
                added = true;
                if (static_cast<int>(selected.size()) == limit) {
                    break;
                }
            }
        }
        if (!added) {
            break;
        }
    }
    return selected;
}

std::vector<std::string> tokenize_ascii_words(const std::string & text) {
    std::vector<std::string> tokens;
    std::string token;
    token.reserve(24);
    for (unsigned char character : text) {
        if (std::isalnum(character) || character == '_') {
            token.push_back(static_cast<char>(std::tolower(character)));
        } else if (token.size() >= 2) {
            tokens.push_back(std::move(token));
            token.clear();
        } else {
            token.clear();
        }
    }
    if (token.size() >= 2) {
        tokens.push_back(std::move(token));
    }
    return tokens;
}

class TfidfExtractor {
public:
    explicit TfidfExtractor(const std::vector<Request> & requests) {
        std::unordered_map<std::string, int> document_frequency;
        for (const Request & request : requests) {
            auto tokens = tokenize_ascii_words(request.text);
            std::sort(tokens.begin(), tokens.end());
            tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
            for (const std::string & token : tokens) {
                ++document_frequency[token];
            }
        }
        const double documents = static_cast<double>(requests.size());
        idf_.reserve(document_frequency.size());
        for (const auto & entry : document_frequency) {
            idf_[entry.first] = std::log((1.0 + documents) / (1.0 + entry.second)) + 1.0;
        }
    }

    struct Result {
        int keyword_count = 0;
        double checksum = 0.0;
    };

    Result extract(const std::string & text, int top_k = 8) const {
        const auto tokens = tokenize_ascii_words(text);
        std::unordered_map<std::string, int> counts;
        counts.reserve(tokens.size());
        for (const std::string & token : tokens) {
            ++counts[token];
        }

        std::vector<std::pair<double, std::string>> scored;
        scored.reserve(counts.size());
        const double denominator = std::max<std::size_t>(tokens.size(), 1);
        for (const auto & entry : counts) {
            const auto found = idf_.find(entry.first);
            const double idf = found == idf_.end() ? 1.0 : found->second;
            scored.emplace_back((entry.second / denominator) * idf, entry.first);
        }

        const int selected = std::min<int>(top_k, static_cast<int>(scored.size()));
        std::partial_sort(scored.begin(), scored.begin() + selected, scored.end(),
                          [](const auto & left, const auto & right) {
                              if (left.first != right.first) {
                                  return left.first > right.first;
                              }
                              return left.second < right.second;
                          });

        double checksum = 0.0;
        for (int i = 0; i < selected; ++i) {
            checksum += scored[i].first * static_cast<double>(scored[i].second.size() + i + 1);
        }
        return {selected, checksum};
    }

private:
    std::unordered_map<std::string, double> idf_;
};

class KeywordWorker {
public:
    explicit KeywordWorker(const TfidfExtractor & extractor) : extractor_(extractor), thread_(&KeywordWorker::loop, this) {}

    ~KeywordWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        work_ready_.notify_one();
        thread_.join();
    }

    void submit(const std::string * text) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_) {
            throw std::runtime_error("keyword worker already has pending work");
        }
        text_ = text;
        pending_ = true;
        complete_ = false;
        work_ready_.notify_one();
    }

    TfidfExtractor::Result wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        work_done_.wait(lock, [&] { return complete_; });
        complete_ = false;
        return result_;
    }

private:
    void loop() {
        for (;;) {
            const std::string * text = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_ready_.wait(lock, [&] { return stopping_ || pending_; });
                if (stopping_) {
                    return;
                }
                text = text_;
                pending_ = false;
            }
            const auto result = extractor_.extract(*text);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                result_ = result;
                complete_ = true;
            }
            work_done_.notify_one();
        }
    }

    const TfidfExtractor & extractor_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;
    const std::string * text_ = nullptr;
    TfidfExtractor::Result result_;
    bool pending_ = false;
    bool complete_ = false;
    bool stopping_ = false;
};

__global__ void compute_proxy_kernel(float * values, int count, int iterations, float phase) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = values[index] + phase + static_cast<float>(index % 17) * 0.0001f;
    for (int i = 0; i < iterations; ++i) {
        value = __sinf(value * 1.000031f + phase) + __cosf(value * 0.999971f - phase);
    }
    values[index] = value;
}

struct Profile {
    const char * name;
    int inference_blocks;
    int keyword_blocks;
};

struct GpuMeasurement {
    double total_ms = 0.0;
    float inference_ms = 0.0f;
    float keyword_ms = 0.0f;
};

struct BenchmarkRow {
    int inference_iterations = 0;
    int keyword_iterations = 0;
    double baseline_total_ms = 0.0;
    float baseline_inference_ms = 0.0f;
    double cpu_keyword_ms = 0.0;
    double cpu_serial_total_ms = 0.0;
    float cpu_serial_inference_ms = 0.0f;
    double cpu_parallel_total_ms = 0.0;
    float cpu_parallel_inference_ms = 0.0f;
    GpuMeasurement gpu_serial;
    GpuMeasurement gpu_overlap;
    int keyword_count = 0;
    double checksum = 0.0;
};

class CudaBenchmark {
public:
    CudaBenchmark(int max_blocks, int high_priority, int low_priority) {
        CUDA_CHECK(cudaStreamCreateWithPriority(&high_stream_, cudaStreamNonBlocking, high_priority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&low_stream_, cudaStreamNonBlocking, low_priority));
        const int count = max_blocks * threads_per_block_;
        CUDA_CHECK(cudaMalloc(&high_values_, static_cast<std::size_t>(count) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&low_values_, static_cast<std::size_t>(count) * sizeof(float)));
        CUDA_CHECK(cudaMemset(high_values_, 0, static_cast<std::size_t>(count) * sizeof(float)));
        CUDA_CHECK(cudaMemset(low_values_, 0, static_cast<std::size_t>(count) * sizeof(float)));
    }

    ~CudaBenchmark() {
        if (low_values_ != nullptr) {
            cudaFree(low_values_);
        }
        if (high_values_ != nullptr) {
            cudaFree(high_values_);
        }
        if (low_stream_ != nullptr) {
            cudaStreamDestroy(low_stream_);
        }
        if (high_stream_ != nullptr) {
            cudaStreamDestroy(high_stream_);
        }
    }

    void warm_up(const Profile & profile, int inference_iterations, int keyword_iterations) {
        launch(high_stream_, high_values_, profile.inference_blocks, inference_iterations, 0.013f);
        launch(low_stream_, low_values_, profile.keyword_blocks, keyword_iterations, 0.027f);
        CUDA_CHECK(cudaStreamSynchronize(high_stream_));
        CUDA_CHECK(cudaStreamSynchronize(low_stream_));
    }

    std::pair<double, float> inference_only(const Profile & profile, int iterations) {
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        const auto wall_start = Clock::now();
        CUDA_CHECK(cudaEventRecord(start, high_stream_));
        launch(high_stream_, high_values_, profile.inference_blocks, iterations, 0.011f);
        CUDA_CHECK(cudaEventRecord(end, high_stream_));
        CUDA_CHECK(cudaEventSynchronize(end));
        const auto wall_end = Clock::now();
        float gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, end));
        CUDA_CHECK(cudaEventDestroy(end));
        CUDA_CHECK(cudaEventDestroy(start));
        return {elapsed_ms(wall_start, wall_end), gpu_ms};
    }

    GpuMeasurement serial(const Profile & profile, int inference_iterations, int keyword_iterations) {
        cudaEvent_t all_start = nullptr;
        cudaEvent_t inference_end = nullptr;
        cudaEvent_t all_end = nullptr;
        CUDA_CHECK(cudaEventCreate(&all_start));
        CUDA_CHECK(cudaEventCreate(&inference_end));
        CUDA_CHECK(cudaEventCreate(&all_end));

        const auto wall_start = Clock::now();
        CUDA_CHECK(cudaEventRecord(all_start, high_stream_));
        launch(high_stream_, high_values_, profile.inference_blocks, inference_iterations, 0.017f);
        CUDA_CHECK(cudaEventRecord(inference_end, high_stream_));
        launch(high_stream_, low_values_, profile.keyword_blocks, keyword_iterations, 0.031f);
        CUDA_CHECK(cudaEventRecord(all_end, high_stream_));
        CUDA_CHECK(cudaEventSynchronize(all_end));
        const auto wall_end = Clock::now();

        GpuMeasurement result;
        result.total_ms = elapsed_ms(wall_start, wall_end);
        CUDA_CHECK(cudaEventElapsedTime(&result.inference_ms, all_start, inference_end));
        CUDA_CHECK(cudaEventElapsedTime(&result.keyword_ms, inference_end, all_end));
        CUDA_CHECK(cudaEventDestroy(all_end));
        CUDA_CHECK(cudaEventDestroy(inference_end));
        CUDA_CHECK(cudaEventDestroy(all_start));
        return result;
    }

    GpuMeasurement overlap(const Profile & profile, int inference_iterations, int keyword_iterations) {
        cudaEvent_t inference_start = nullptr;
        cudaEvent_t inference_end = nullptr;
        cudaEvent_t keyword_start = nullptr;
        cudaEvent_t keyword_end = nullptr;
        CUDA_CHECK(cudaEventCreate(&inference_start));
        CUDA_CHECK(cudaEventCreate(&inference_end));
        CUDA_CHECK(cudaEventCreate(&keyword_start));
        CUDA_CHECK(cudaEventCreate(&keyword_end));

        const auto wall_start = Clock::now();
        CUDA_CHECK(cudaEventRecord(inference_start, high_stream_));
        launch(high_stream_, high_values_, profile.inference_blocks, inference_iterations, 0.019f);
        CUDA_CHECK(cudaEventRecord(inference_end, high_stream_));
        CUDA_CHECK(cudaEventRecord(keyword_start, low_stream_));
        launch(low_stream_, low_values_, profile.keyword_blocks, keyword_iterations, 0.037f);
        CUDA_CHECK(cudaEventRecord(keyword_end, low_stream_));
        CUDA_CHECK(cudaEventSynchronize(inference_end));
        CUDA_CHECK(cudaEventSynchronize(keyword_end));
        const auto wall_end = Clock::now();

        GpuMeasurement result;
        result.total_ms = elapsed_ms(wall_start, wall_end);
        CUDA_CHECK(cudaEventElapsedTime(&result.inference_ms, inference_start, inference_end));
        CUDA_CHECK(cudaEventElapsedTime(&result.keyword_ms, keyword_start, keyword_end));
        CUDA_CHECK(cudaEventDestroy(keyword_end));
        CUDA_CHECK(cudaEventDestroy(keyword_start));
        CUDA_CHECK(cudaEventDestroy(inference_end));
        CUDA_CHECK(cudaEventDestroy(inference_start));
        return result;
    }

    cudaStream_t high_stream() const {
        return high_stream_;
    }

    void launch_inference(const Profile & profile, int iterations) {
        launch(high_stream_, high_values_, profile.inference_blocks, iterations, 0.023f);
    }

private:
    void launch(cudaStream_t stream, float * values, int blocks, int iterations, float phase) {
        const int count = blocks * threads_per_block_;
        compute_proxy_kernel<<<blocks, threads_per_block_, 0, stream>>>(values, count, iterations, phase);
        CUDA_CHECK(cudaGetLastError());
    }

    static constexpr int threads_per_block_ = 256;
    float * high_values_ = nullptr;
    float * low_values_ = nullptr;
    cudaStream_t high_stream_ = nullptr;
    cudaStream_t low_stream_ = nullptr;
};

BenchmarkRow benchmark_request(
        const Request & request,
        const Profile & profile,
        const Options & options,
        const TfidfExtractor & extractor,
        KeywordWorker & worker,
        CudaBenchmark & cuda) {
    BenchmarkRow row;
    const double token_scale = std::clamp(request.prefix_tokens / 640.0, 0.5, 2.5);
    row.inference_iterations = std::max(1, static_cast<int>(std::lround(options.inference_iterations * token_scale)));
    row.keyword_iterations = std::max(1, static_cast<int>(std::lround(options.keyword_iterations * token_scale)));

    const auto baseline = cuda.inference_only(profile, row.inference_iterations);
    row.baseline_total_ms = baseline.first;
    row.baseline_inference_ms = baseline.second;

    const auto cpu_start = Clock::now();
    const auto serial_keywords = extractor.extract(request.text);
    const auto cpu_end = Clock::now();
    row.cpu_keyword_ms = elapsed_ms(cpu_start, cpu_end);
    row.keyword_count = serial_keywords.keyword_count;
    row.checksum = serial_keywords.checksum;

    const auto serial_start = Clock::now();
    const auto serial_result = extractor.extract(request.text);
    cudaEvent_t serial_gpu_start = nullptr;
    cudaEvent_t serial_gpu_end = nullptr;
    CUDA_CHECK(cudaEventCreate(&serial_gpu_start));
    CUDA_CHECK(cudaEventCreate(&serial_gpu_end));
    CUDA_CHECK(cudaEventRecord(serial_gpu_start, cuda.high_stream()));
    cuda.launch_inference(profile, row.inference_iterations);
    CUDA_CHECK(cudaEventRecord(serial_gpu_end, cuda.high_stream()));
    CUDA_CHECK(cudaEventSynchronize(serial_gpu_end));
    const auto serial_end = Clock::now();
    row.cpu_serial_total_ms = elapsed_ms(serial_start, serial_end);
    CUDA_CHECK(cudaEventElapsedTime(&row.cpu_serial_inference_ms, serial_gpu_start, serial_gpu_end));
    CUDA_CHECK(cudaEventDestroy(serial_gpu_end));
    CUDA_CHECK(cudaEventDestroy(serial_gpu_start));
    row.checksum += serial_result.checksum;

    cudaEvent_t parallel_gpu_start = nullptr;
    cudaEvent_t parallel_gpu_end = nullptr;
    CUDA_CHECK(cudaEventCreate(&parallel_gpu_start));
    CUDA_CHECK(cudaEventCreate(&parallel_gpu_end));
    const auto parallel_start = Clock::now();
    worker.submit(&request.text);
    CUDA_CHECK(cudaEventRecord(parallel_gpu_start, cuda.high_stream()));
    cuda.launch_inference(profile, row.inference_iterations);
    CUDA_CHECK(cudaEventRecord(parallel_gpu_end, cuda.high_stream()));
    CUDA_CHECK(cudaEventSynchronize(parallel_gpu_end));
    const auto parallel_keywords = worker.wait();
    const auto parallel_end = Clock::now();
    row.cpu_parallel_total_ms = elapsed_ms(parallel_start, parallel_end);
    CUDA_CHECK(cudaEventElapsedTime(&row.cpu_parallel_inference_ms, parallel_gpu_start, parallel_gpu_end));
    CUDA_CHECK(cudaEventDestroy(parallel_gpu_end));
    CUDA_CHECK(cudaEventDestroy(parallel_gpu_start));
    row.checksum += parallel_keywords.checksum;

    row.gpu_serial = cuda.serial(profile, row.inference_iterations, row.keyword_iterations);
    row.gpu_overlap = cuda.overlap(profile, row.inference_iterations, row.keyword_iterations);
    return row;
}

void write_header(std::ofstream & output) {
    output
        << "profile,request_id,prefix_tokens,lora_id,context_id,trial,"
        << "baseline_total_ms,baseline_inference_ms,cpu_keyword_ms,"
        << "cpu_serial_total_ms,cpu_serial_inference_ms,"
        << "cpu_parallel_total_ms,cpu_parallel_inference_ms,"
        << "gpu_keyword_serial_ms,gpu_serial_total_ms,gpu_serial_inference_ms,"
        << "gpu_overlap_total_ms,gpu_overlap_inference_ms,gpu_overlap_keyword_ms,"
        << "high_priority,low_priority,inference_blocks,keyword_blocks,"
        << "inference_iterations,keyword_iterations,keyword_count,checksum\n";
}

void write_row(
        std::ofstream & output,
        const Profile & profile,
        const Request & request,
        int trial,
        int high_priority,
        int low_priority,
        const BenchmarkRow & row) {
    output << profile.name << ',' << request.request_id << ',' << request.prefix_tokens << ','
           << request.lora_id << ',' << std::quoted(request.context_id) << ',' << trial << ','
           << row.baseline_total_ms << ',' << row.baseline_inference_ms << ',' << row.cpu_keyword_ms << ','
           << row.cpu_serial_total_ms << ',' << row.cpu_serial_inference_ms << ','
           << row.cpu_parallel_total_ms << ',' << row.cpu_parallel_inference_ms << ','
           << row.gpu_serial.keyword_ms << ',' << row.gpu_serial.total_ms << ',' << row.gpu_serial.inference_ms << ','
           << row.gpu_overlap.total_ms << ',' << row.gpu_overlap.inference_ms << ',' << row.gpu_overlap.keyword_ms << ','
           << high_priority << ',' << low_priority << ',' << profile.inference_blocks << ','
           << profile.keyword_blocks << ',' << row.inference_iterations << ',' << row.keyword_iterations << ','
           << row.keyword_count << ',' << row.checksum << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto corpus = load_requests(options);
        const auto requests = select_by_lora(corpus, options.limit);
        const TfidfExtractor extractor(corpus);
        KeywordWorker worker(extractor);

        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        int low_priority = 0;
        int high_priority = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority));

        const int sms = std::max(properties.multiProcessorCount, 1);
        const std::vector<Profile> profiles = {
            {"headroom", std::max(8, sms / 2), std::max(4, sms / 4)},
            {"saturated", sms * 4, sms * 2},
        };
        const int max_blocks = std::max(profiles[0].inference_blocks, profiles[1].inference_blocks);
        CudaBenchmark cuda(max_blocks, high_priority, low_priority);
        for (const Profile & profile : profiles) {
            cuda.warm_up(profile, options.inference_iterations, options.keyword_iterations);
        }

        if (!options.output.parent_path().empty()) {
            std::filesystem::create_directories(options.output.parent_path());
        }
        std::ofstream output(options.output);
        if (!output) {
            throw std::runtime_error("cannot write output: " + options.output.string());
        }
        output << std::fixed << std::setprecision(6);
        write_header(output);

        for (const Profile & profile : profiles) {
            for (const Request & request : requests) {
                for (int trial = 0; trial < options.trials; ++trial) {
                    const BenchmarkRow row = benchmark_request(request, profile, options, extractor, worker, cuda);
                    write_row(output, profile, request, trial, high_priority, low_priority, row);
                }
            }
        }

        std::cout << "device=" << properties.name << "\n"
                  << "corpus_requests=" << corpus.size() << " sampled_requests=" << requests.size()
                  << " trials=" << options.trials << " profiles=" << profiles.size() << "\n"
                  << "stream_priority_high=" << high_priority << " stream_priority_low=" << low_priority << "\n"
                  << "output=" << options.output.string() << "\n";
        return 0;
    } catch (const std::exception & error) {               
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
