// simple-page-pool.cpp
//
// KV cache 与 LoRA adapter 统一页块池实验
//
// 本程序用于验证连续分配与页块池化分配在长时间运行下的差异。
// 它不执行真实 LLM 推理，而是实现真实的分配器逻辑，并用 KV cache / LoRA adapter
// 的申请释放事件来测试外部碎片、页面利用率和分配失败率。
//
// continuous:
// - 每个对象必须占用连续页块。
// - 总空闲容量足够但缺少连续空间时，分配失败。
//
// paged_pool:
// - 每个对象可以映射到多个离散页块。
// - 只要总空闲页数足够，就可以分配成功。
//
// 输出：
// D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output/page_pool_results.csv

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

enum class object_type {
    kv_cache,
    lora_adapter,
};

struct alloc_event {
    int step = 0;
    object_type type = object_type::kv_cache;
    int pages = 0;
    int ttl = 0;
};

struct memory_object {
    int id = -1;
    object_type type = object_type::kv_cache;
    int ttl = 0;
    std::vector<int> pages;
};

struct metric_row {
    std::string allocator;
    int step = 0;
    double memory_usage_gb = 0.0;
    double used_rate = 0.0;
    double external_fragmentation = 0.0;
    double allocation_fail_rate = 0.0;
    int live_kv = 0;
    int live_lora = 0;
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output";

// 每页 16 MiB，768 页约等于 12 GiB。
static constexpr int N_PAGES = 768;
static constexpr double PAGE_SIZE_GB = 16.0 / 1024.0;
static constexpr int N_STEPS = 800;

class continuous_allocator {
public:
    explicit continuous_allocator(int n_pages)
        : pages(n_pages, -1) {}

    bool allocate(memory_object & obj, int need_pages) {
        int start = -1;
        int run = 0;

        for (int i = 0; i < (int) pages.size(); i++) {
            if (pages[i] == -1) {
                if (run == 0) {
                    start = i;
                }

                run++;

                if (run >= need_pages) {
                    break;
                }
            } else {
                start = -1;
                run = 0;
            }
        }

        if (start < 0 || run < need_pages) {
            return false;
        }

        obj.pages.clear();

        for (int i = start; i < start + need_pages; i++) {
            pages[i] = obj.id;
            obj.pages.push_back(i);
        }

        return true;
    }

    void free_object(const memory_object & obj) {
        for (int p : obj.pages) {
            if (p >= 0 && p < (int) pages.size()) {
                pages[p] = -1;
            }
        }
    }

    metric_row make_metric(
            const std::string & name,
            int step,
            int alloc_count,
            int fail_count,
            int live_kv,
            int live_lora) const {
        int used = 0;
        int free_pages = 0;
        int current_free_run = 0;
        int largest_free_run = 0;

        for (int v : pages) {
            if (v == -1) {
                free_pages++;
                current_free_run++;
                largest_free_run = std::max(largest_free_run, current_free_run);
            } else {
                used++;
                current_free_run = 0;
            }
        }

        metric_row row;
        row.allocator = name;
        row.step = step;
        row.memory_usage_gb = used * PAGE_SIZE_GB;
        row.used_rate = (double) used / pages.size();
        row.external_fragmentation =
                free_pages > 0 ? 1.0 - (double) largest_free_run / free_pages : 0.0;
        row.allocation_fail_rate =
                alloc_count > 0 ? (double) fail_count / alloc_count : 0.0;
        row.live_kv = live_kv;
        row.live_lora = live_lora;

        return row;
    }

private:
    std::vector<int> pages;
};

class paged_pool_allocator {
public:
    explicit paged_pool_allocator(int n_pages)
        : pages(n_pages, -1) {}

    bool allocate(memory_object & obj, int need_pages) {
        std::vector<int> free_pages;

        for (int i = 0; i < (int) pages.size(); i++) {
            if (pages[i] == -1) {
                free_pages.push_back(i);
            }
        }

        if ((int) free_pages.size() < need_pages) {
            return false;
        }

        obj.pages.clear();

        for (int i = 0; i < need_pages; i++) {
            const int page_id = free_pages[i];
            pages[page_id] = obj.id;
            obj.pages.push_back(page_id);
        }

        return true;
    }

    void free_object(const memory_object & obj) {
        for (int p : obj.pages) {
            if (p >= 0 && p < (int) pages.size()) {
                pages[p] = -1;
            }
        }
    }

    metric_row make_metric(
            const std::string & name,
            int step,
            int alloc_count,
            int fail_count,
            int live_kv,
            int live_lora) const {
        int used = 0;

        for (int v : pages) {
            if (v != -1) {
                used++;
            }
        }

        metric_row row;
        row.allocator = name;
        row.step = step;
        row.memory_usage_gb = used * PAGE_SIZE_GB;
        row.used_rate = (double) used / pages.size();

        // 分页池不要求连续物理页，因此外部碎片不会阻止对象分配。
        row.external_fragmentation = 0.0;
        row.allocation_fail_rate =
                alloc_count > 0 ? (double) fail_count / alloc_count : 0.0;
        row.live_kv = live_kv;
        row.live_lora = live_lora;

        return row;
    }

private:
    std::vector<int> pages;
};

static std::vector<alloc_event> build_workload() {
    std::mt19937 rng(42);

    std::vector<alloc_event> events;

    std::uniform_int_distribution<int> kv_count_dist(2, 6);
    std::uniform_int_distribution<int> kv_pages_dist(2, 18);
    std::uniform_int_distribution<int> kv_ttl_dist(4, 28);

    std::uniform_int_distribution<int> lora_pages_dist(12, 48);
    std::uniform_int_distribution<int> lora_ttl_dist(120, 360);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    for (int step = 1; step <= N_STEPS; step++) {
        // 多请求场景下，KV cache 频繁动态产生。
        const int n_kv = kv_count_dist(rng);

        for (int i = 0; i < n_kv; i++) {
            alloc_event e;
            e.step = step;
            e.type = object_type::kv_cache;
            e.pages = kv_pages_dist(rng);
            e.ttl = kv_ttl_dist(rng);
            events.push_back(e);
        }

        // LoRA adapter 生命周期更长，频率更低，但占用页块更大。
        if (prob_dist(rng) < 0.22) {
            alloc_event e;
            e.step = step;
            e.type = object_type::lora_adapter;
            e.pages = lora_pages_dist(rng);
            e.ttl = lora_ttl_dist(rng);
            events.push_back(e);
        }
    }

    return events;
}

template <typename Allocator>
static std::vector<metric_row> run_allocator(
        const std::string & allocator_name,
        const std::vector<alloc_event> & events) {
    Allocator allocator(N_PAGES);

    std::unordered_map<int, memory_object> live_objects;
    std::vector<metric_row> rows;

    int next_id = 0;
    int alloc_count = 0;
    int fail_count = 0;
    size_t event_idx = 0;

    for (int step = 1; step <= N_STEPS; step++) {
        std::vector<int> expired;

        for (auto & it : live_objects) {
            it.second.ttl--;

            if (it.second.ttl <= 0) {
                expired.push_back(it.first);
            }
        }

        for (int id : expired) {
            allocator.free_object(live_objects[id]);
            live_objects.erase(id);
        }

        while (event_idx < events.size() && events[event_idx].step == step) {
            const alloc_event & e = events[event_idx++];

            memory_object obj;
            obj.id = next_id++;
            obj.type = e.type;
            obj.ttl = e.ttl;

            alloc_count++;

            if (allocator.allocate(obj, e.pages)) {
                live_objects[obj.id] = obj;
            } else {
                fail_count++;
            }
        }

        int live_kv = 0;
        int live_lora = 0;

        for (const auto & it : live_objects) {
            if (it.second.type == object_type::kv_cache) {
                live_kv++;
            } else {
                live_lora++;
            }
        }

        rows.push_back(
                allocator.make_metric(
                        allocator_name,
                        step,
                        alloc_count,
                        fail_count,
                        live_kv,
                        live_lora));
    }

    return rows;
}

static void save_results(const std::vector<metric_row> & rows) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/page_pool_results.csv";
    std::ofstream fout(path);

    fout << "allocator,step,memory_usage_gb,used_rate,external_fragmentation,"
         << "allocation_fail_rate,live_kv,live_lora\n";

    for (const auto & r : rows) {
        fout << r.allocator << ","
             << r.step << ","
             << r.memory_usage_gb << ","
             << r.used_rate << ","
             << r.external_fragmentation << ","
             << r.allocation_fail_rate << ","
             << r.live_kv << ","
             << r.live_lora << "\n";
    }

    fprintf(stderr, "saved results to %s\n", path.c_str());
}

int main() {
    const std::vector<alloc_event> workload = build_workload();

    std::vector<metric_row> all_rows;

    auto continuous_rows =
            run_allocator<continuous_allocator>("continuous", workload);

    auto paged_rows =
            run_allocator<paged_pool_allocator>("paged_pool", workload);

    all_rows.insert(all_rows.end(), continuous_rows.begin(), continuous_rows.end());
    all_rows.insert(all_rows.end(), paged_rows.begin(), paged_rows.end());

    save_results(all_rows);

    fprintf(stderr, "page pool experiment finished.\n");

    return 0;
}