// simple-page-fragment.cpp
//
// 连续分配与分页分配的实际占用页数对比实验
//
// 这个实验用于模拟论文中的内存池化效果：
//
// 1. 连续分配：
//    - 对象按真实大小连续申请。
//    - 前期对象紧凑排列，连续分配可能更省。
//    - 释放对象后会留下空洞。
//    - 后续对象如果不能放入已有空洞，需要向后扩展。
//    - 已经向系统申请过的内存不会立即收缩，所以实际占用可能持续增加。
//
// 2. 分页分配：
//    - 对象按固定页大小切分。
//    - 前期可能存在页内浪费，所以占用可能略高。
//    - 释放后的页可以被任意对象复用。
//    - 占用呈阶梯状增长，后期更稳定。
//
// 输出：
// output/page_pool_allocated_trace.csv
// output/page_pool_allocated_summary.csv

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <vector>

enum class object_type {
    kv,
    lora,
};

enum class allocator_type {
    continuous,
    paged,
};

struct object_record {
    int object_id = -1;
    object_type type = object_type::kv;
    int logical_units = 0;
    int allocated_pages = 0;
    std::vector<int> page_ids;
};

struct operation {
    std::string event;
    object_type type = object_type::kv;
    int object_id = -1;

    // 逻辑大小，单位不是页。
    // 连续分配可以按逻辑大小直接申请。
    // 分页分配需要转换成固定页块数。
    int logical_units = 0;
};

struct trace_sample {
    std::string allocator;
    int step = 0;
    std::string event;
    std::string object_type;
    int object_id = -1;
    int logical_units = 0;

    int live_logical_units = 0;
    int live_allocated_pages = 0;

    // 本实验最重要指标：
    // 连续分配：已经向系统申请并保留的页数。
    // 分页分配：页池中当前实际占用的页数。
    int allocated_pages = 0;

    int free_reusable_pages = 0;
    int internal_waste_units = 0;
    int external_hole_pages = 0;

    int active_objects = 0;
    int active_kv = 0;
    int active_lora = 0;
};

struct summary_sample {
    std::string allocator;
    int total_steps = 0;
    int peak_allocated_pages = 0;
    int final_allocated_pages = 0;
    int peak_live_allocated_pages = 0;
    int peak_internal_waste_units = 0;
    int peak_external_hole_pages = 0;
    double avg_allocated_pages = 0.0;
};

static const std::string output_dir =
        "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/simple-lora-memory/output";

static const int page_unit_size = 4;

static const char * allocator_name(allocator_type type) {
    switch (type) {
        case allocator_type::continuous:
            return "continuous";
        case allocator_type::paged:
            return "paged";
    }

    return "unknown";
}

static const char * object_type_name(object_type type) {
    switch (type) {
        case object_type::kv:
            return "kv";
        case object_type::lora:
            return "lora";
    }

    return "unknown";
}

static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

class allocated_memory_model {
public:
    explicit allocated_memory_model(allocator_type type) :
        type(type) {
    }

    void allocate(int object_id, object_type obj_type, int logical_units) {
        if (type == allocator_type::continuous) {
            allocate_continuous(object_id, obj_type, logical_units);
        } else {
            allocate_paged(object_id, obj_type, logical_units);
        }
    }

    void free_object(int object_id) {
        auto it = objects.find(object_id);

        if (it == objects.end()) {
            return;
        }

        for (const int page_id : it->second.page_ids) {
            if (page_id >= 0 && page_id < (int) page_used.size()) {
                page_used[page_id] = false;
            }
        }

        objects.erase(it);
    }

    int get_live_logical_units() const {
        int total = 0;

        for (const auto & item : objects) {
            total += item.second.logical_units;
        }

        return total;
    }

    int get_live_allocated_pages() const {
        int total = 0;

        for (const auto & item : objects) {
            total += item.second.allocated_pages;
        }

        return total;
    }

    int get_allocated_pages() const {
        if (type == allocator_type::continuous) {
            // 连续分配表示已经从系统申请过的高水位页数。
            // 即使中间对象释放，内存空洞仍然保留，不立即归还系统。
            return high_watermark;
        }

        // 分页分配表示当前真实被对象占用的页数。
        // 空闲页可以复用，因此释放后不算正在占用。
        int used = 0;

        for (const bool flag : page_used) {
            if (flag) {
                used++;
            }
        }

        return used;
    }

    int get_free_reusable_pages() const {
        int free_pages = 0;

        for (const bool flag : page_used) {
            if (!flag) {
                free_pages++;
            }
        }

        return free_pages;
    }

    int get_internal_waste_units() const {
        if (type == allocator_type::continuous) {
            return 0;
        }

        int waste = 0;

        for (const auto & item : objects) {
            const object_record & obj = item.second;
            waste += obj.allocated_pages * page_unit_size - obj.logical_units;
        }

        return waste;
    }

    int get_external_hole_pages() const {
        if (type != allocator_type::continuous) {
            return 0;
        }

        return std::max(0, high_watermark - get_live_allocated_pages());
    }

    int get_active_objects() const {
        return (int) objects.size();
    }

    int get_active_by_type(object_type type) const {
        int count = 0;

        for (const auto & item : objects) {
            if (item.second.type == type) {
                count++;
            }
        }

        return count;
    }

private:
    void ensure_capacity(int pages) {
        if ((int) page_used.size() < pages) {
            page_used.resize(pages, false);
        }
    }

    int find_continuous_hole(int required_pages) const {
        int run_start = -1;
        int run_len = 0;

        for (int i = 0; i < high_watermark; i++) {
            if (!page_used[i]) {
                if (run_len == 0) {
                    run_start = i;
                }

                run_len++;

                if (run_len >= required_pages) {
                    return run_start;
                }
            } else {
                run_start = -1;
                run_len = 0;
            }
        }

        return -1;
    }

    void allocate_continuous(int object_id, object_type obj_type, int logical_units) {
        // 连续分配按逻辑大小折算成页。
        // 这里不引入页内浪费，模拟“前期连续分配更省”的现象。
        const int required_pages = logical_units;

        int start = find_continuous_hole(required_pages);

        if (start < 0) {
            start = high_watermark;
            high_watermark += required_pages;
            ensure_capacity(high_watermark);
        }

        object_record obj;
        obj.object_id = object_id;
        obj.type = obj_type;
        obj.logical_units = logical_units;
        obj.allocated_pages = required_pages;

        for (int i = start; i < start + required_pages; i++) {
            page_used[i] = true;
            obj.page_ids.push_back(i);
        }

        objects[object_id] = obj;
    }

    void allocate_paged(int object_id, object_type obj_type, int logical_units) {
        // 分页分配必须按固定页大小向上取整。
        // 这会带来页内浪费，所以前期可能略高。
        const int required_pages = ceil_div(logical_units, page_unit_size);

        object_record obj;
        obj.object_id = object_id;
        obj.type = obj_type;
        obj.logical_units = logical_units;
        obj.allocated_pages = required_pages;

        for (int i = 0; i < (int) page_used.size() && (int) obj.page_ids.size() < required_pages; i++) {
            if (!page_used[i]) {
                page_used[i] = true;
                obj.page_ids.push_back(i);
            }
        }

        while ((int) obj.page_ids.size() < required_pages) {
            page_used.push_back(true);
            obj.page_ids.push_back((int) page_used.size() - 1);
        }

        high_watermark = std::max(high_watermark, (int) page_used.size());

        objects[object_id] = obj;
    }

private:
    allocator_type type;
    std::vector<bool> page_used;
    std::map<int, object_record> objects;
    int high_watermark = 0;
};

static trace_sample make_trace(
        allocator_type allocator,
        int step,
        const operation & op,
        const allocated_memory_model & model) {
    trace_sample s;

    s.allocator = allocator_name(allocator);
    s.step = step;
    s.event = op.event;
    s.object_type = object_type_name(op.type);
    s.object_id = op.object_id;
    s.logical_units = op.logical_units;
    s.live_logical_units = model.get_live_logical_units();
    s.live_allocated_pages = model.get_live_allocated_pages();
    s.allocated_pages = model.get_allocated_pages();
    s.free_reusable_pages = model.get_free_reusable_pages();
    s.internal_waste_units = model.get_internal_waste_units();
    s.external_hole_pages = model.get_external_hole_pages();
    s.active_objects = model.get_active_objects();
    s.active_kv = model.get_active_by_type(object_type::kv);
    s.active_lora = model.get_active_by_type(object_type::lora);

    return s;
}

static std::vector<operation> build_workload() {
    std::vector<operation> ops;

    std::mt19937 rng(2026);

    std::uniform_int_distribution<int> kv_small(2, 5);
    std::uniform_int_distribution<int> kv_mid(5, 9);
    std::uniform_int_distribution<int> lora_size(10, 16);

    std::vector<int> live_kv;
    std::vector<int> live_lora;

    int next_id = 1;

    // 前 15 步：主要是连续申请。
    // 连续分配无页内浪费，前期可能更低。
    for (int step = 0; step < 15; step++) {
        operation op;

        if (step == 5 || step == 11) {
            op.event = "alloc_lora";
            op.type = object_type::lora;
            op.object_id = next_id++;
            op.logical_units = lora_size(rng);
            live_lora.push_back(op.object_id);
        } else {
            op.event = "alloc_kv";
            op.type = object_type::kv;
            op.object_id = next_id++;
            op.logical_units = kv_small(rng);
            live_kv.push_back(op.object_id);
        }

        ops.push_back(op);
    }

    // 15-32 步：开始释放旧 KV，制造连续空洞。
    for (int step = 15; step < 32; step++) {
        operation op;

        if ((step % 3 == 0) && !live_kv.empty()) {
            op.event = "free_kv";
            op.type = object_type::kv;
            op.object_id = live_kv.front();
            op.logical_units = 0;
            live_kv.erase(live_kv.begin());
        } else if ((step % 8 == 0) && !live_lora.empty()) {
            op.event = "free_lora";
            op.type = object_type::lora;
            op.object_id = live_lora.front();
            op.logical_units = 0;
            live_lora.erase(live_lora.begin());
        } else {
            op.event = "alloc_kv";
            op.type = object_type::kv;
            op.object_id = next_id++;
            op.logical_units = kv_mid(rng);
            live_kv.push_back(op.object_id);
        }

        ops.push_back(op);
    }

    // 32-50 步：混合替换。
    // 连续分配开始因为外部空洞产生额外占用；
    // 分页分配释放页后能复用，增长更慢。
    for (int step = 32; step < 50; step++) {
        operation op;

        if ((step % 5 == 0) && !live_kv.empty()) {
            op.event = "free_kv";
            op.type = object_type::kv;
            op.object_id = live_kv.front();
            op.logical_units = 0;
            live_kv.erase(live_kv.begin());
        } else if ((step % 9 == 0) && !live_lora.empty()) {
            op.event = "free_lora";
            op.type = object_type::lora;
            op.object_id = live_lora.front();
            op.logical_units = 0;
            live_lora.erase(live_lora.begin());
        } else if (step % 7 == 0) {
            op.event = "alloc_lora";
            op.type = object_type::lora;
            op.object_id = next_id++;
            op.logical_units = lora_size(rng);
            live_lora.push_back(op.object_id);
        } else {
            op.event = "alloc_kv";
            op.type = object_type::kv;
            op.object_id = next_id++;
            op.logical_units = kv_mid(rng);
            live_kv.push_back(op.object_id);
        }

        ops.push_back(op);
    }

    return ops;
}

static std::vector<trace_sample> run_experiment(
        allocator_type allocator,
        const std::vector<operation> & ops) {
    allocated_memory_model model(allocator);

    std::vector<trace_sample> trace;

    operation start;
    start.event = "start";
    start.type = object_type::kv;
    start.object_id = -1;
    start.logical_units = 0;

    trace.push_back(make_trace(
            allocator,
            0,
            start,
            model));

    for (int i = 0; i < (int) ops.size(); i++) {
        const operation & op = ops[i];

        if (op.event == "alloc_kv" || op.event == "alloc_lora") {
            model.allocate(op.object_id, op.type, op.logical_units);
        } else if (op.event == "free_kv" || op.event == "free_lora") {
            model.free_object(op.object_id);
        }

        trace.push_back(make_trace(
                allocator,
                i + 1,
                op,
                model));
    }

    return trace;
}

static summary_sample summarize(
        allocator_type allocator,
        const std::vector<trace_sample> & trace) {
    summary_sample s;

    s.allocator = allocator_name(allocator);
    s.total_steps = (int) trace.size();

    double sum = 0.0;

    for (const auto & item : trace) {
        s.peak_allocated_pages = std::max(s.peak_allocated_pages, item.allocated_pages);
        s.final_allocated_pages = item.allocated_pages;
        s.peak_live_allocated_pages = std::max(s.peak_live_allocated_pages, item.live_allocated_pages);
        s.peak_internal_waste_units = std::max(s.peak_internal_waste_units, item.internal_waste_units);
        s.peak_external_hole_pages = std::max(s.peak_external_hole_pages, item.external_hole_pages);
        sum += item.allocated_pages;
    }

    if (!trace.empty()) {
        s.avg_allocated_pages = sum / (double) trace.size();
    }

    return s;
}

static void save_trace(const std::vector<trace_sample> & trace) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/page_pool_allocated_trace.csv";
    std::ofstream fout(path);

    fout << "allocator,step,event,object_type,object_id,logical_units,"
         << "live_logical_units,live_allocated_pages,allocated_pages,"
         << "free_reusable_pages,internal_waste_units,external_hole_pages,"
         << "active_objects,active_kv,active_lora\n";

    for (const auto & s : trace) {
        fout << s.allocator << ","
             << s.step << ","
             << s.event << ","
             << s.object_type << ","
             << s.object_id << ","
             << s.logical_units << ","
             << s.live_logical_units << ","
             << s.live_allocated_pages << ","
             << s.allocated_pages << ","
             << s.free_reusable_pages << ","
             << s.internal_waste_units << ","
             << s.external_hole_pages << ","
             << s.active_objects << ","
             << s.active_kv << ","
             << s.active_lora << "\n";
    }

    fprintf(stderr, "saved trace to %s\n", path.c_str());
}

static void save_summary(const std::vector<summary_sample> & summaries) {
    std::filesystem::create_directories(output_dir);

    const std::string path = output_dir + "/page_pool_allocated_summary.csv";
    std::ofstream fout(path);

    fout << "allocator,total_steps,peak_allocated_pages,final_allocated_pages,"
         << "peak_live_allocated_pages,peak_internal_waste_units,"
         << "peak_external_hole_pages,avg_allocated_pages\n";

    for (const auto & s : summaries) {
        fout << s.allocator << ","
             << s.total_steps << ","
             << s.peak_allocated_pages << ","
             << s.final_allocated_pages << ","
             << s.peak_live_allocated_pages << ","
             << s.peak_internal_waste_units << ","
             << s.peak_external_hole_pages << ","
             << s.avg_allocated_pages << "\n";
    }

    fprintf(stderr, "saved summary to %s\n", path.c_str());
}

int main() {
    const std::vector<operation> ops = build_workload();

    const auto continuous_trace = run_experiment(
            allocator_type::continuous,
            ops);

    const auto paged_trace = run_experiment(
            allocator_type::paged,
            ops);

    std::vector<trace_sample> all_trace;
    all_trace.insert(
            all_trace.end(),
            continuous_trace.begin(),
            continuous_trace.end());

    all_trace.insert(
            all_trace.end(),
            paged_trace.begin(),
            paged_trace.end());

    std::vector<summary_sample> summaries;
    summaries.push_back(summarize(
            allocator_type::continuous,
            continuous_trace));

    summaries.push_back(summarize(
            allocator_type::paged,
            paged_trace));

    save_trace(all_trace);
    save_summary(summaries);

    fprintf(stderr, "page pool allocated memory experiment finished.\n");

    return 0;
}