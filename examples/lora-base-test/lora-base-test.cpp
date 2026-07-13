// 多 LoRA 在线 Prefix KV 复用 + Suffix Delta Probe 实验代码。
// 数据结构 - 基础工具函数 - LoRA 和 prompt 构造 - KV delta - 在线 prefix tree - 实验运行函数 - 保存实验结果 - main 主流程

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "llama.h"
#include "generated_lora_tree_qwen2.5.hpp"

#include <windows.h>
#include <psapi.h>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#pragma comment(lib, "Psapi.lib")

// ===============================
// 1.全局输出目录
// ===============================
// 所有 CSV 结果统一写到这个 output 目录。
static const std::string output_dir = "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base-test/output";


// ===============================
// 2.数据结构
// ===============================
// ======================== 离线参数转换结构 ========================
// 运行时 LoRA 节点
// generated_lora_tree_qwen2.5.hpp 里通常保存静态树信息，这里把它转换成运行时结构，并额外保存 llama_adapter_lora 指针。
struct lora_node_runtime {
    int id = 0;                            // 当前 LoRA 节点 id
    int parent_id = -1;                    // 父 LoRA 节点 id，用于表示 LoRA tree
    std::string name;                      // 完整名称
    std::string short_name;                // 短名称，写 CSV / 图表时更方便
    std::string path;                      // LoRA 文件路径
    std::string group_name;                // 所属 group，例如 code / correction。
    bool is_anchor = false;                // 是否是该 group 的 anchor LoRA
    llama_adapter_lora * adapter = nullptr;// llama.cpp 加载后的 LoRA adapter 指针
};

// prompt 模板运行时结构
// 每个 group 可以有几个代表性 prompt，用于把请求路由到相似 group。
struct prompt_pattern_runtime {
    std::string text;                      // 模板原文
    std::vector<llama_token> tokens;       // 模板 token 序列
};

// LoRA group 运行时结构
// 一个 group 里有一个 anchor LoRA 和多个相似 LoRA。
struct lora_group_runtime {
    int group_id = 0;                                      // group id
    std::string group_name;                                // group 名称
    int anchor_lora_id = 0;                                // anchor LoRA 的 id
    std::vector<int> lora_ids;                             // group 下所有 LoRA id
    std::vector<prompt_pattern_runtime> prompt_patterns;   // 该 group 的路由模板
};



// ======================== 请求 ========================
// 单个实验请求
// 一个请求 = 使用哪个 LoRA + 输入 prompt + 上下文标签 + 第几次重复实验。
struct request_item {
    int leaf_lora_id = 0;                  // 当前请求使用的 LoRA id。
    std::string prompt;                    // 当前请求的完整 prompt。
    std::string context_tag = "short";     // 例如 ctx_50 / ctx_100 / ctx_200。
    int repeat_id = 0;                     // 第几轮重复实验，用于统计均值/方差。
};

// tokenized 后的请求
// prompt 在主流程里会先 tokenize，避免每次实验重复 tokenize。
struct request_tokens {
    std::vector<llama_token> full;         // 完整 prompt token。
    int routed_group_id = -1;              // 该 prompt 被路由到哪个 group。
};

// ======================== 前缀树节点 ========================
// 在线 prefix tree 节点
// 每个节点保存一个已经 materialize 的 prefix KV，后续请求命中后可以 seq_cp 复用。
// materialize：原来只是一个“可以计算出来的东西”，现在把它真正算出来并写进 KV cache。
// 比如：执行 KV_B = KV_A + delta
// seq：KV cache 里的“请求编号”
// llama.cpp 的 KV cache 可以同时保存多个请求的 KV，为了区分不同请求，会给每个请求一个：llama_seq_id
struct online_prefix_node {
    int node_id = -1;                      // 当前在线节点 id
    int group_id = -1;                     // 节点所属 group

    int parent_node_id = -1;               // prefix tree 父节点；当前简化版多数为 -1
    int delta_parent_node_id = -1;         // 如果 suffix delta 可用，指向 delta anchor 节点

    llama_seq_id cache_seq_id = -1;        // 当前 prefix KV 存放在哪个 seq 里

    int prefix_len = 0;                    // 当前节点保存的 prefix token 长度
    int hit_count = 0;                     // 后续请求命中该节点的次数

    int prefix_can_reuse = 0;              // probe 判断 prefix 是否可复用
    int suffix_can_delta = 0;              // probe 判断 suffix 是否可做 delta
    int should_open_new_branch = 1;        // 是否应该单独开新分支

    double suffix_kv_cos = 0.0;            // suffix KV cosine
    double suffix_kv_l2 = 0.0;             // suffix KV L2

    std::string reuse_decision = "new_branch";  // 系统决策
    std::string group_name;                     // group 名称
    std::vector<llama_token> prefix_tokens;     // 当前节点保存的 token 序列
};

// 在线路由结果
// route_online_prefix_node() 会返回当前请求是否命中已有 prefix node
struct online_route_result {
    int node_id = -1;                      // 命中的在线节点 id
    int group_id = -1;                     // 命中的 group id
    int exact_prefix_len = 0;              // 命中的公共 prefix 长度
    int suffix_len = 0;                    // 剩余 suffix 长度
    bool exact_prefix_hit = false;         // 是否命中已有 prefix
};

// ======================== 实验结果 ========================
// 单个请求的一条实验结果
// baseline、online prefix reuse、suffix delta materialize 都写成这个结构。
struct sample_result {
    std::string mode;                      // 实验模式，例如 baseline_no_group_kv_reuse
    std::string group_name;                // group 名称
    std::string lora_name;                 // LoRA 短名称

    int repeat_id = 0;                     // 第几轮重复实验
    std::string context_tag = "short";     // ctx_50 / ctx_100 / ctx_200 等
    int materialize_timed = 0;             // suffix delta materialize 是否完成真实计时

    int routed_group_id = -1;              // prompt 路由到的 group id
    std::string reuse_decision = "unknown";// 当前请求的复用决策

    int leaf_lora_id = 0;                  // 当前请求使用的 LoRA id
    int n_prompt_tokens = 0;               // prompt token 数
    int n_prefix_tokens = 0;               // 复用 prefix token 数
    int n_suffix_tokens = 0;               // 需要处理的 suffix token 数
    int n_predict = 0;                     // decode 生成 token 数

    int online_node_id = -1;               // 命中的 online prefix node id
    int exact_prefix_hit = 0;              // 是否命中 exact prefix

    int suffix_delta_materialized = 0;     // suffix delta 是否 materialize 成功
    double delta_materialize_ms = 0.0;     // delta materialize 耗时
    double delta_fp32_mb = 0.0;            // delta 原始 FP32 估算大小
    double materialized_kv_mb = 0.0;       // materialize 写回 KV 的大小

    double estimated_saved_kv_mb = 0.0;    // 根据 prefix token 估算节省的 KV 内存

    double route_ms = 0.0;                 // prompt 路由耗时
    double prefix_ms = 0.0;                // prefix copy / build 耗时
    double lora_bind_ms = 0.0;             // LoRA 绑定耗时
    double suffix_ms = 0.0;                // suffix eval 耗时
    double ttft_ms = 0.0;                  // time to first token
    double decode_ms = 0.0;                // decode 阶段耗时
    double total_ms = 0.0;                 // 总耗时
    double tps = 0.0;                      // decode tokens/s

    double gpu_start_mb = 0.0;             // 请求开始前 GPU 显存
    double gpu_peak_mb = 0.0;              // 请求期间 GPU 显存峰值
    double gpu_peak_delta_mb = 0.0;        // GPU 显存增加量

    double prefix_reuse_rate = 0.0;        // prefix token 占比
    double suffix_delta_rate = 0.0;        // suffix token 占比
};

// ======================== KV相似 ========================
// 描述某一个具体范围内的 KV 差异结果
struct kv_delta_range_result {
    // 当前比较的是哪类 memory，通常是 llama_kv_cache
    // 混合结构模型可能有 recurrent memory这种不一定能逐 token 比较
    std::string memory_kind = "unknown"; 
    std::string probe_status = "failed";    // ok-成功 failed-失败 empty_range-没token
    int probed_kv_modules = 0;              // 实际参与比较的 KV module 数量
    int probed_layers = 0;              // 实际参与比较的 attention KV 层数
    int skipped_recurrent_layers = 0;   // 被跳过的 recurrent 层数

    double kv_l2_avg = 0.0;             // 当前范围内 KV 的平均 L2 距离
    double kv_cos_avg = 0.0;            // 当前范围内 KV 的平均余弦相似度

    int reusable = 0;                   // 当前范围是否被底层 probe 判断为可复用
};

// 一个 probe case
// A 通常表示 anchor prompt，B 表示 child prompt
struct kv_delta_probe_case {
    std::string group_name;                // group 名称
    std::string pair_name;                 // pair 名称
    std::string prompt_a;                  // anchor prompt
    std::string prompt_b;                  // child prompt
    llama_seq_id seq_a = -1;               // A 的 seq id
    llama_seq_id seq_b = -1;               // B 的 seq id

    int anchor_request_index = -1;         // A 在 requests 里的 index
    int child_request_index = -1;          // B 在 requests 里的 index
};

// 一个 pair 的 KV 差异 probe 结果 -- 这里是指两个相似的提示词
struct kv_delta_probe_result {
    std::string pair_name;                 // pair 名称，例如 ctx_50_write_vs_optimize
    std::string group_name;                // group 名称，例如 code

    std::string memory_kind;               // KV memory 类型，例如 llama_kv_cache 如果是 Qwen3.5 可能是llama_memory_recurrent
    std::string probe_status;              // probe 状态，ok / failed / empty_range 等

    int seq_a = -1;                        // A 分支 seq id
    int seq_b = -1;                        // B 分支 seq id

    int tokens_a = 0;                      // prompt A token 数
    int tokens_b = 0;                      // prompt B token 数
    int common_prefix_tokens = 0;          // A/B 公共 prefix token 数
    int compared_tokens = 0;               // 实际比较 token 数，一般是 min(A, B)

    int probed_kv_modules = 0;             // probe 到的 KV module 数
    int probed_layers = 0;                 // probe 到的 attention KV 层数
    int skipped_recurrent_layers = 0;      // 跳过的 recurrent 层数

    double full_kv_l2_avg = 0.0;           // full 范围 KV L2
    double full_kv_cos_avg = 0.0;          // full 范围 KV cosine

    double prefix_kv_l2_avg = 0.0;         // prefix 范围 KV L2
    double prefix_kv_cos_avg = 0.0;        // prefix 范围 KV cosine

    double suffix_kv_l2_avg = 0.0;         // suffix 范围 KV L2
    double suffix_kv_cos_avg = 0.0;        // suffix 范围 KV cosine

    int prefix_can_reuse = 0;              // prefix 是否满足复用阈值
    int suffix_can_delta = 0;              // suffix 是否满足 delta 阈值
    int should_open_new_branch = 0;        // 是否应该单独开分支

    int delta_parent_node_id = -1;         // delta 父节点，当前简化为 0 / -1
    std::string reuse_decision = "unknown";// 复用决策
    std::string tree_action = "unknown";   // 映射到 tree 的动作
};


// ===============================
// 3.基础工具函数
// ===============================
// ======================== 基础功能 ========================
// 返回当前时间，单位 ms。
static double now_ms() {
    return ggml_time_us() / 1000.0;
}

// 通过 nvidia-smi 查询当前 GPU 已用显存，单位 MB。
// 注意：这个函数比较慢，只适合实验统计，不适合高频生产路径。
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

// ======================== 推理 ========================
// 把字符串 tokenize 成 llama token
// add_special=true 表示包含模型需要的特殊 token。
static bool tokenize_text(
        const llama_vocab * vocab,
        const std::string & text,
        std::vector<llama_token> & tokens) {
    const int n = -llama_tokenize(
            vocab,
            text.c_str(),
            (int) text.size(),
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
            text.c_str(),
            (int) text.size(),
            tokens.data(),
            (int) tokens.size(),
            true,
            true);

    return ret >= 0;
}

// 往 llama_batch 里添加一个 token。
// llama_batch 是 llama.cpp 一次送进模型的一小批 token
// 这里每调用一次 batch_add，就往 batch 里塞入一个 token 及其位置信息
static void batch_add(
        llama_batch & batch,       // 要写入的 batch
        llama_token token,         // 当前要送进模型的 token id
        llama_pos pos,             // 当前 token 在该 seq 中的位置
        llama_seq_id seq_id,       // 当前 token 属于哪个 seq / 请求分支
        bool logits) {             // 是否需要模型输出这个位置的 logits
    const int i = batch.n_tokens;  // 当前 batch 已有 token 数，也是新 token 写入的位置

    batch.token[i] = token;        // 写入 token id
    batch.pos[i] = pos;            // 写入 token 的 position
    batch.n_seq_id[i] = 1;         // 当前 token 只属于 1 个 seq
    batch.seq_id[i][0] = seq_id;   // 记录这个 token 属于哪个 seq
    batch.logits[i] = logits ? 1 : 0; // 是否为这个 token 计算输出 logits

    batch.n_tokens++;              // batch token 数加 1
}

// 对一段 token 做 prefill/eval
// 作用：把 prompt token 送进模型前向计算，让模型生成并写入 KV cache
// 例如 prompt 有 100 个 token，这个函数会把它们分 chunk 喂给 llama_decode
static bool eval_tokens(
        llama_context * ctx,                       // llama.cpp 上下文，里面有模型状态和 KV cache
        const std::vector<llama_token> & tokens,   // 要送进模型的一段 token
        llama_seq_id seq_id,                       // 这些 token 属于哪个 seq
        int start_pos,                             // 这些 token 在 seq 中从哪个 position 开始
        bool logits_last) {                        // 是否只在最后一个 token 位置输出 logits
    if (tokens.empty()) {                          // 如果没有 token，就不需要计算
        return true;
    }

    const int max_chunk = 64;                      // 每次最多送 64 个 token，避免 batch 太大
    int offset = 0;                                // 当前已经处理到 tokens 的哪个位置

    while (offset < (int) tokens.size()) {         // 循环直到所有 token 都处理完
        const int chunk =
                std::min(max_chunk, (int) tokens.size() - offset); // 本轮处理 token 数

        llama_batch batch = llama_batch_init(chunk, 0, 1); // 创建一个 batch

        for (int i = 0; i < chunk; i++) {          // 把本轮 chunk 的 token 填入 batch
            const int token_index = offset + i;    // 当前 token 在 tokens 中的全局下标

            const bool need_logits =
                    logits_last &&
                    token_index == (int) tokens.size() - 1; // 只有最后 token 需要 logits

            batch_add(
                    batch,                         // 当前 batch
                    tokens[token_index],           // 当前 token id
                    start_pos + token_index,       // token 在 seq 里的 position
                    seq_id,                        // 当前 seq id
                    need_logits);                  // 是否输出 logits
        }

        const int ret = llama_decode(ctx, batch);  // 真正执行模型前向，写入 KV cache
        llama_batch_free(batch);                   // 释放 batch 内存

        if (ret != 0) {                            // ret 非 0 表示 llama_decode 失败
            fprintf(stderr,
                    "eval_tokens failed: seq=%d start=%d offset=%d chunk=%d\n",
                    seq_id,
                    start_pos,
                    offset,
                    chunk);
            return false;
        }

        offset += chunk;                           // 进入下一段 chunk
    }

    return true;                                   // 所有 token 都成功 eval
}


// decode 一个 token。
// 作用：在已有 KV cache 的基础上，继续生成下一个 token。
// 这里为了做实验，通常重复喂一个 token，不做真实采样。
static bool decode_one(
        llama_context * ctx,       // llama.cpp 上下文
        llama_token token,         // 要送入模型的 token
        llama_seq_id seq_id,       // 这个 token 属于哪个 seq
        int pos) {                 // 这个 token 在 seq 中的位置
    llama_batch batch = llama_batch_init(1, 0, 1); // 创建只包含 1 个 token 的 batch

    batch_add(
            batch,                 // batch
            token,                 // 当前 token
            pos,                   // token position
            seq_id,                // seq id
            true);                 // decode 阶段需要 logits

    const int ret = llama_decode(ctx, batch);      // 执行一次前向，得到 logits 并写 KV
    llama_batch_free(batch);                       // 释放 batch

    return ret == 0;                               // ret==0 表示成功
}

// ======================== LoRA相关功能 ========================
// 清空当前上下文里绑定的 LoRA adapter
static void clear_lora(llama_context * ctx) {
    llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
}

// 计算两个 token 序列的公共前缀长度
static int common_prefix_len(
        const std::vector<llama_token> & a,
        const std::vector<llama_token> & b) {
    const int n = std::min((int) a.size(), (int) b.size());

    int i = 0;

    while (i < n && a[i] == b[i]) {
        i++;
    }

    return i;
}

// ======================== KV 内存估算 ========================
// 根据 ggml type 粗略返回每个元素占多少字节。
// 这里用于估算 KV cache 每 token 大概占多少 MB。
// 注意：量化类型真实大小可能更复杂，这里只是实验可视化估算。
static size_t ggml_type_size_simple(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return 4;                      // float32：4 字节。

        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return 2;                      // float16 / bfloat16：2 字节。

        case GGML_TYPE_Q8_0:
            return 1;                      // q8 粗略按 1 字节估算。

        default:
            return 2;                      // 默认按 f16 估算。
    }
}

// 估算当前模型每个 token 的 KV cache 大小，单位 MB
// 用途：画图时估算 Prefix KV 复用大概节省多少 KV 内存
static double estimate_kv_mb_per_token(
        const llama_model * model,         // 当前加载的模型
        enum ggml_type type_k,             // K cache 的数据类型
        enum ggml_type type_v) {           // V cache 的数据类型
    const int n_layer = llama_model_n_layer(model); // 模型层数
    const int n_embd = llama_model_n_embd(model);   // hidden size
    const int n_head = llama_model_n_head(model);   // attention head 数
    const int n_head_kv = llama_model_n_head_kv(model); // KV head 数，GQA/MQA 时小于 n_head

    const int n_embd_kv =
            n_embd * n_head_kv / std::max(1, n_head);
    // 计算每层 K/V 的实际 KV hidden 维度
    // 普通 MHA 时 n_head_kv == n_head，所以 n_embd_kv == n_embd
    // GQA/MQA 时 n_head_kv 更小，所以 KV cache 更省

    const size_t bytes_per_token =
            (size_t) n_layer *
            ((size_t) n_embd_kv * ggml_type_size_simple(type_k) +
             (size_t) n_embd_kv * ggml_type_size_simple(type_v));
    // 每个 token 在每层都要保存 K 和 V
    // 所以大小 = 层数 * (K 大小 + V 大小)

    return (double) bytes_per_token / 1024.0 / 1024.0;
    // 转成 MB
}


// ===============================
// 4. LoRA 和 prompt 构造
// ===============================
// ======================== 获取 LoRA 配置 ========================


// 从 generated_lora_tree_qwen2.5.hpp 读取静态 LoRA 配置，转成运行时 lora_node_runtime
// 运行时结构比静态配置多一个 adapter 指针，后面加载 LoRA 后会填进去

// LoRA 静态配置 -> 运行时结构
// 从 generated_lora_tree_qwen2.5.hpp 读取静态 LoRA 配置，
// 转成运行时 lora_node_runtime。
// 运行时结构比静态配置多一个 adapter 指针，后面加载 LoRA 后会填进去。
static std::vector<lora_node_runtime> make_runtime_lora_nodes() {
    std::vector<lora_node_runtime> nodes;  // 保存所有运行时 LoRA 节点。

    const std::vector<generated_lora_spec> specs =
            make_generated_lora_specs();
    // 读取自动生成的静态 LoRA 列表。
    // generated_lora_spec 里通常包含 id、parent_id、path、group_name 等。

    for (const auto & spec : specs) {      // 遍历每个静态 LoRA 配置。
        lora_node_runtime node;            // 创建一个运行时 LoRA 节点。

        node.id = spec.id;                 // 复制 LoRA id。
        node.parent_id = spec.parent_id;   // 复制父节点 id，用于 LoRA tree。
        node.name = spec.name;             // 复制完整名称。
        node.short_name = spec.short_name; // 复制短名称。
        node.path = spec.path;             // 复制 LoRA 文件路径。
        node.group_name = spec.group_name; // 复制所属 group 名称。
        node.is_anchor = spec.is_anchor;   // 复制是否是 anchor LoRA。

        nodes.push_back(node);             // 加入运行时节点列表。
    }

    return nodes;                          // 返回所有 LoRA runtime 节点。
}

// 从 generated_lora_tree_qwen2.5.hpp 读取 group 配置，并把 group prompt pattern tokenize 成 token 序列。
// 后续 route_prompt_to_group() 会用这些 token 做 prefix 匹配。
static std::vector<lora_group_runtime> make_runtime_groups(
        const llama_vocab * vocab) {       // vocab 用于 tokenize group prompt pattern。
    std::vector<lora_group_runtime> groups;// 保存所有运行时 group。

    const std::vector<generated_lora_group> generated_groups =
            make_generated_lora_groups();
    // 读取自动生成的 group 配置。

    for (const auto & src : generated_groups) { // 遍历每个静态 group。
        lora_group_runtime group;          // 创建运行时 group。

        group.group_id = src.group_id;     // 复制 group id。
        group.group_name = src.group_name; // 复制 group 名称。
        group.anchor_lora_id = src.anchor_lora_id; // 复制 anchor LoRA id。
        group.lora_ids = src.lora_ids;     // 复制该 group 下所有 LoRA id。

        for (const auto & pattern : src.prompt_patterns) {
            prompt_pattern_runtime item;   // 创建运行时 prompt pattern。

            item.text = pattern.text;      // 保存 pattern 原文。
            tokenize_text(vocab, item.text, item.tokens);
            // 把 pattern 转成 token。
            // 后面实际请求也会 tokenize，然后比较 token 级公共前缀。

            group.prompt_patterns.push_back(item);
            // 把这个 pattern 加入 group。
        }

        groups.push_back(group);           // 把 group 加入 groups。
    }

    return groups;                         // 返回所有 runtime groups。
}


// ======================== 生成提示词 ========================

// 构造一段公共文章正文。
// repeat_blocks 越大，文章越长，公共 prefix 越长。
// 当前实验用它模拟 MobileLoRA 里“同一篇文章，不同任务指令”的场景。
static std::string make_common_article_body(int repeat_blocks) {
    const std::string block =
            "Large language models are increasingly deployed on mobile and edge devices. "
            "In these scenarios, users often send several requests that share the same long context "
            "but differ only in the final task instruction. For example, the same document may be "
            "summarized, rewritten, translated, or optimized by different LoRA adapters. "
            "A context-aware KV cache reuse system can keep the shared prefix as an anchor and only "
            "process the task-specific suffix for each branch. This reduces repeated prefill work "
            "and may lower time to first token when the shared context is long enough. ";
    // 一个 block 是一段技术文章。
    // 重复多次后得到更长上下文。

    std::string text;                                  // 保存最终文章。

    for (int i = 0; i < repeat_blocks; ++i) {          // 重复拼接 block。
        text += block;
    }

    return text;                                       // 返回公共文章正文。
}

// 根据公共文章正文和任务指令构造完整 prompt。
// article 是大段公共 prefix。
// task 是最后不同的 suffix。
static std::string make_mobile_like_prompt(
        const std::string & article,                   // 公共文章正文。
        const std::string & task) {                    // 不同任务指令。
    return
            "You are a helpful coding assistant. "
            "Please read the following technical article carefully.\n\n"
            "Article:\n" +
            article +
            "\n\nTask:\n" +
            task;
    // 最终结构：
    // 公共部分：system-like instruction + Article + article
    // 差异部分：Task + task
}


// ======================== 实验请求 ========================
// 构造当前实验的基础 requests。
// 每个 context_tag 下有 3 个请求：
// 1. write：anchor LoRA 请求，用来建立 prefix node。
// 2. optimize：相似 LoRA 请求，用来测试 prefix reuse。
// 3. explain：相似 LoRA 请求，用来测试 prefix reuse。
static std::vector<request_item> make_long_context_requests() {
    const std::string article_50 =
            make_common_article_body(1);               // ctx_50：较短文章。

    const std::string article_100 =
            make_common_article_body(2);               // ctx_100：更长文章。

    const std::string article_200 =
            make_common_article_body(4);               // ctx_200：备用。

    const std::string article_500 =
            make_common_article_body(10);              // ctx_500：备用。

    const std::string article_1000 =
            make_common_article_body(20);              // ctx_1000：备用。

    std::vector<request_item> requests;                // 保存所有请求。

    auto add_case = [&](const std::string & tag,       // context tag，例如 ctx_50。
                        const std::string & article) { // 当前 context 的文章正文。
        requests.push_back({
            0,                                         // leaf_lora_id=0，anchor LoRA。
            make_mobile_like_prompt(
                    article,
                    "Write a simple Python implementation based on this article."),
            tag,                                       // context_tag。
        });

        requests.push_back({
            1,                                         // leaf_lora_id=1，相似 LoRA。
            make_mobile_like_prompt(
                    article,
                    "Optimize the Python implementation based on this article."),
            tag,                                       // context_tag。
        });

        requests.push_back({
            2,                                         // leaf_lora_id=2，相似 LoRA。
            make_mobile_like_prompt(
                    article,
                    "Explain the Python implementation based on this article."),
            tag,                                       // context_tag。
        });
    };

    add_case("ctx_50", article_50);                    // 添加 ctx_50 的 3 个请求。
    add_case("ctx_100", article_100);                  // 添加 ctx_100 的 3 个请求。

    // 如果要跑更长上下文，可以打开下面这些。
    // add_case("ctx_200", article_200);
    // add_case("ctx_500", article_500);
    // add_case("ctx_1000", article_1000);

    return requests;                                   // 返回基础请求列表。
}

// 把基础 requests 重复 n_repeats 次。
// 这样可以做多次实验，后续 Python 取均值和标准差。
static std::vector<request_item> make_repeated_long_context_requests(int n_repeats) {
    std::vector<request_item> base_requests =
            make_long_context_requests();
    // 先构造一轮基础请求。
    // 例如 ctx_50 write / optimize / explain + ctx_100 write / optimize / explain。

    std::vector<request_item> requests;                // 保存重复后的请求。

    for (int repeat_id = 0; repeat_id < n_repeats; ++repeat_id) {
        // 外层循环：第几轮重复实验。

        for (auto req : base_requests) {
            // 拷贝基础请求。
            // 注意这里用 auto req 是值拷贝，不会修改 base_requests。

            req.repeat_id = repeat_id;
            // 标记当前请求属于第几轮 repeat。

            requests.push_back(req);
            // 加入最终请求列表。
        }
    }

    return requests;                                   // 返回重复后的请求列表。
}


// ======================== LoRA 绑定 ========================

// 根据 leaf LoRA id 构建从 root/anchor 到 leaf 的 LoRA path。
// 例如 LoRA tree 是：anchor -> code_r16v2，
// 那 path 可能是 [anchor_id, code_r16v2_id]。
static std::vector<int> build_lora_path(
        const std::vector<lora_node_runtime> & nodes, // 所有 LoRA runtime 节点。
        int leaf_id) {                                // 目标 leaf LoRA id。
    std::vector<int> path;                            // 保存路径。
    int cur = leaf_id;                                // 从 leaf 开始向父节点回溯。

    while (cur >= 0 && cur < (int) nodes.size()) {    // 只要 cur 是合法 id。
        path.push_back(cur);                          // 把当前节点加入路径。
        cur = nodes[cur].parent_id;                   // 移动到父节点。
    }

    std::reverse(path.begin(), path.end());
    // 回溯得到的是 leaf -> root，所以需要反转成 root -> leaf。

    return path;                                      // 返回 LoRA 路径。
}

// 把一条 LoRA path 绑定到当前 llama_context。
// llama.cpp 允许同时绑定多个 LoRA adapter，并为每个 adapter 设置 scale。
// 当前实验里 scale 都设成 1.0。
static double bind_lora_path(
        llama_context * ctx,                          // 当前 llama context。
        const std::vector<lora_node_runtime> & nodes, // 所有 LoRA runtime 节点。
        int leaf_id) {                                // 目标 leaf LoRA id。
    const std::vector<int> path =
            build_lora_path(nodes, leaf_id);
    // 先构建 root/anchor -> leaf 的路径。

    std::vector<llama_adapter_lora *> adapters;       // 要绑定的 adapter 指针列表。
    std::vector<float> scales;                        // 每个 adapter 的 scale。

    for (int id : path) {                             // 遍历路径上的 LoRA id。
        if (id >= 0 && id < (int) nodes.size() &&
                nodes[id].adapter != nullptr) {       // 确保 id 合法且 adapter 已加载。
            adapters.push_back(nodes[id].adapter);    // 加入 adapter。
            scales.push_back(1.0f);                   // 当前 adapter scale 设为 1。
        }
    }

    const double t0 = now_ms();                       // 记录绑定开始时间。

    const int ret = llama_set_adapters_lora(
            ctx,                                      // 当前 context。
            adapters.data(),                          // adapter 数组。
            adapters.size(),                          // adapter 数量。
            scales.data());                           // scale 数组。

    const double t1 = now_ms();                       // 记录绑定结束时间。

    if (ret != 0) {                                   // ret 非 0 表示绑定失败。
        fprintf(stderr, "failed to bind LoRA path for leaf %d\n", leaf_id);
    }

    return t1 - t0;                                   // 返回绑定耗时 ms。
}



// ===============================
// 5. KV delta
// ===============================
// ======================== kv决策 ========================
// 根据 probe 的三个布尔量生成可读的复用决策 - 算法层面判断
static std::string make_reuse_decision(
        int prefix_can_reuse,
        int suffix_can_delta,
        int should_open_new_branch) {
    if (should_open_new_branch) {
        return "new_branch";
    }

    if (prefix_can_reuse && suffix_can_delta) {
        return "prefix_reuse_suffix_delta";
    }

    if (prefix_can_reuse && !suffix_can_delta) {
        return "prefix_reuse_suffix_recompute";
    }

    if (!prefix_can_reuse && suffix_can_delta) {
        return "suffix_delta_only";
    }

    return "unknown";
}

// 把复用决策进一步映射成 prefix tree 的动作 - 系统结构层面
static std::string make_tree_action(const std::string & reuse_decision) {
    if (reuse_decision == "new_branch") {
        return "create_independent_branch";
    }

    if (reuse_decision == "prefix_reuse_suffix_delta") {
        return "attach_delta_branch";
    }

    if (reuse_decision == "prefix_reuse_suffix_recompute") {
        return "reuse_prefix_recompute_suffix";
    }

    if (reuse_decision == "suffix_delta_only") {
        return "attach_suffix_delta_without_prefix";
    }

    return "unknown";
}


// ========================  kv比较 ========================

// 生成当前实验使用的 KV delta probe case
// 每个 case 表示一组要比较的 prompt pair
// A 通常是 anchor 请求，例如 write
// B 通常是 child 请求，例如 optimize / explain
static std::vector<kv_delta_probe_case> make_kv_delta_probe_cases() {
    const std::string article_50 =
            make_common_article_body(1);  // 构造 ctx_50 的公共文章正文

    const std::string article_100 =
            make_common_article_body(2);  // 构造 ctx_100 的公共文章正文

    std::vector<kv_delta_probe_case> cases; // 保存所有 probe case

    auto add_probe_case = [&](
            const std::string & tag,      // 上下文标签，例如 ctx_50
            const std::string & article,  // 当前 case 共享的文章正文
            int seq_base,                 // 给这个 case 分配的 seq 起始 id
            int anchor_request_index,     // anchor 请求在 requests 里的位置
            int optimize_request_index,   // optimize 请求在 requests 里的位置
            int explain_request_index) {  // explain 请求在 requests 里的位置
        const std::string prompt_a =
                make_mobile_like_prompt(
                        article,
                        "Write a simple Python implementation based on this article.");
        // prompt_a 是 anchor prompt：根据文章写代码

        const std::string prompt_b =
                make_mobile_like_prompt(
                        article,
                        "Optimize the Python implementation based on this article.");
        // prompt_b 是 child prompt：根据文章优化代码

        const std::string prompt_c =
                make_mobile_like_prompt(
                        article,
                        "Explain the Python implementation based on this article.");
        // prompt_c 是 child prompt：根据文章解释代码

        cases.push_back({
            "code",                       // group_name
            tag + "_write_vs_optimize",   // pair_name
            prompt_a,                     // A prompt
            prompt_b,                     // B prompt
            (llama_seq_id) seq_base,      // A 使用的 seq id
            (llama_seq_id) (seq_base + 1),// B 使用的 seq id
            anchor_request_index,         // A 在 requests 里的 index
            optimize_request_index,       // B 在 requests 里的 index
        });

        cases.push_back({
            "code",                       // group_name
            tag + "_write_vs_explain",    // pair_name
            prompt_a,                     // A prompt
            prompt_c,                     // B prompt
            (llama_seq_id) (seq_base + 2),// A 使用另一个 seq id，避免和上一个 case 冲突
            (llama_seq_id) (seq_base + 3),// B 使用另一个 seq id
            anchor_request_index,         // A 在 requests 里的 index
            explain_request_index,        // B 在 requests 里的 index
        });
    };

    add_probe_case(
            "ctx_50",                     // 当前上下文标签
            article_50,                   // 当前上下文文章
            10,                           // seq id 从 10 开始
            0,                            // requests[0] = ctx_50 write
            1,                            // requests[1] = ctx_50 optimize
            2);                           // requests[2] = ctx_50 explain

    add_probe_case(
            "ctx_100",                    // 当前上下文标签
            article_100,                  // 当前上下文文章
            20,                           // seq id 从 20 开始
            3,                            // requests[3] = ctx_100 write
            4,                            // requests[4] = ctx_100 optimize
            5);                           // requests[5] = ctx_100 explain

    return cases;                         // 返回所有 probe case
}


// 调用 llama.cpp 内部 API，比较 seq_a 和 seq_b 在 [p0, p1) 的 KV 差异。
// 返回 cosine / L2 / probed layers 等统计。
static kv_delta_range_result run_kv_delta_probe_range(
        llama_context * ctx,
        llama_seq_id seq_a,
        llama_seq_id seq_b,
        llama_pos p0,
        llama_pos p1) {
    kv_delta_range_result r;

    if (p1 <= p0) {
        r.probe_status = "empty_range";
        return r;
    }

    llama_kv_delta_probe_stats stats = {};

    const bool ok =
            llama_kv_seq_delta_probe(
                    ctx,
                    seq_a,
                    seq_b,
                    p0,
                    p1,
                    &stats);

    r.memory_kind = stats.memory_kind;
    r.probe_status = stats.probe_status;

    if (r.memory_kind.empty()) {
        r.memory_kind = "unknown";
    }

    if (r.probe_status.empty()) {
        r.probe_status = ok ? "ok" : "failed";
    }

    r.probed_kv_modules = stats.probed_kv_modules;
    r.probed_layers = stats.probed_layers;
    r.skipped_recurrent_layers = stats.skipped_recurrent_layers;

    r.kv_l2_avg = stats.kv_l2_avg;
    r.kv_cos_avg = stats.kv_cos_avg;
    r.reusable = stats.can_reuse_as_delta;

    return r;
}

// 对两个 prompt 做完整 KV delta probe
// 过程：
// 1. tokenize prompt A/B。
// 2. eval A，做 self-check。
// 3. seq_cp A -> B，验证 KV copy 路径是否正确。
// 4. 清空 memory，重新 eval A/B。
// 5. 分别比较 full / prefix / suffix。
// 6. 根据阈值生成系统决策。
static kv_delta_probe_result run_kv_delta_probe_pair(
        llama_context * ctx,              // 推理上下文
        const llama_vocab * vocab,        // 词表
        const std::string & group_name,   // 分类标签，如 "sanity", "same_prefix"
        const std::string & pair_name,    // case 名称，如 "same_prefix_monkey_tiger"
        const std::string & prompt_a,     // 第一个 prompt
        const std::string & prompt_b,     // 第二个 prompt
        llama_seq_id seq_a,               // prompt_a 使用的 seq ID
        llama_seq_id seq_b) {             // prompt_b 使用的 seq ID
            
    kv_delta_probe_result r;              // 结果结构体

    r.group_name = group_name;            // "sanity"
    r.pair_name = pair_name;              // "same_prompt_monkey"
    r.seq_a = seq_a;                      // 0
    r.seq_b = seq_b;                      // 1
    r.should_open_new_branch = 1;         // 默认: 需要开新分支（保守策略）

    std::vector<llama_token> tokens_a;    // prompt_a 的 token ID 列表
    std::vector<llama_token> tokens_b;    // prompt_b 的 token ID 列表

    // 将文本转为 token。如果任一失败，返回错误
    if (!tokenize_text(vocab, prompt_a, tokens_a) || !tokenize_text(vocab, prompt_b, tokens_b)) {
        r.probe_status = "tokenize_failed";
        fprintf(stderr, "kv delta probe tokenize failed: pair=%s\n", pair_name.c_str());
        return r;
    }

    r.tokens_a = (int) tokens_a.size();           // a的长度
    r.tokens_b = (int) tokens_b.size();           // b的长度
    r.common_prefix_tokens = common_prefix_len(tokens_a, tokens_b);  // 两个的前缀和的长度
    r.compared_tokens = std::min(r.tokens_a, r.tokens_b);            // a,b最短的长度域
    if (r.compared_tokens <= 0) {                 // 有一个提示词为空
        r.probe_status = "empty_tokens";
        fprintf(stderr, "kv delta probe empty tokens: pair=%s\n", pair_name.c_str());
        return r;
    }

    clear_lora(ctx);                             // 清除当前绑定的 LoRA
    llama_memory_t mem = llama_get_memory(ctx);  // 获取 KV memory 管理器

    // -----------------------------
    // 1. Copy-check 阶段：只验证跨 seq 读取是否可靠
    // -----------------------------
    llama_memory_clear(mem, true);               // 清空所有 KV Cache
    // 在 seq_a 上完整 eval prompt_a
    const bool ok_a_for_copy = eval_tokens(ctx, tokens_a, seq_a, 0, false);
    if (!ok_a_for_copy) {   // eval 失败则返回
        r.probe_status = "eval_a_for_copy_check_failed";
        fprintf(stderr,
                "kv delta probe eval A for copy-check failed: pair=%s seq_a=%d\n",
                pair_name.c_str(),
                seq_a);
        return r;
    }

    // 比较 seq_a 和 seq_a（自己和自己比）
    const kv_delta_range_result self_before_copy = run_kv_delta_probe_range(ctx, seq_a, seq_a, 0, r.compared_tokens);
    fprintf(stderr,
            "kv delta self-check-before-copy: pair=%s seq=%d status=%s kv_cos=%.6f kv_l2=%.6f\n",
            pair_name.c_str(),
            seq_a,
            self_before_copy.probe_status.c_str(),
            self_before_copy.kv_cos_avg,
            self_before_copy.kv_l2_avg);

    if (self_before_copy.probe_status != "ok" ||
            self_before_copy.kv_cos_avg < 0.999 ||
            self_before_copy.kv_l2_avg > 1e-2) {
        r.memory_kind = self_before_copy.memory_kind;
        r.probe_status = "self_check_before_copy_failed";
        return r;
    }

    llama_memory_seq_cp(mem, seq_a, seq_b, 0, r.compared_tokens);

    const kv_delta_range_result copy_check =
            run_kv_delta_probe_range(ctx, seq_a, seq_b, 0, r.compared_tokens);

    fprintf(stderr,
            "kv delta copy-check: pair=%s seq_a=%d seq_b=%d status=%s kv_cos=%.6f kv_l2=%.6f\n",
            pair_name.c_str(),
            seq_a,
            seq_b,
            copy_check.probe_status.c_str(),
            copy_check.kv_cos_avg,
            copy_check.kv_l2_avg);

    if (copy_check.probe_status != "ok" ||
            copy_check.kv_cos_avg < 0.998 ||
            copy_check.kv_l2_avg > 1e-3) {
        r.memory_kind = copy_check.memory_kind;
        r.probe_status = "copy_check_failed";

        fprintf(stderr,
                "kv delta probe stopped by copy-check: pair=%s\n",
                pair_name.c_str());

        return r;
    }

    // -----------------------------
    // 2. 正式 probe 阶段：必须重新清空并重新 eval A/B
    // -----------------------------
    llama_memory_clear(mem, true);

    const bool ok_a =
            eval_tokens(ctx, tokens_a, seq_a, 0, false);

    if (!ok_a) {
        r.probe_status = "eval_a_failed";
        fprintf(stderr,
                "kv delta probe eval A failed: pair=%s seq_a=%d tokens_a=%d\n",
                pair_name.c_str(),
                seq_a,
                r.tokens_a);
        return r;
    }

    const bool ok_b =
            eval_tokens(ctx, tokens_b, seq_b, 0, false);

    if (!ok_b) {
        r.probe_status = "eval_b_failed";
        fprintf(stderr,
                "kv delta probe eval B failed: pair=%s seq_b=%d tokens_b=%d\n",
                pair_name.c_str(),
                seq_b,
                r.tokens_b);
        return r;
    }

    // 正式 eval 后再检查一次，防止 memory 被清空后没有重新写入 KV。
    const kv_delta_range_result self_a =
            run_kv_delta_probe_range(ctx, seq_a, seq_a, 0, r.compared_tokens);

    fprintf(stderr,
            "kv delta self-check-A: pair=%s seq=%d status=%s kv_cos=%.6f kv_l2=%.6f\n",
            pair_name.c_str(),
            seq_a,
            self_a.probe_status.c_str(),
            self_a.kv_cos_avg,
            self_a.kv_l2_avg);

    if (self_a.probe_status != "ok" ||
            self_a.kv_cos_avg < 0.999 ||
            self_a.kv_l2_avg > 1e-2) {
        r.memory_kind = self_a.memory_kind;
        r.probe_status = "self_check_a_failed";
        return r;
    }

    const kv_delta_range_result self_b =
            run_kv_delta_probe_range(ctx, seq_b, seq_b, 0, r.compared_tokens);

    fprintf(stderr,
            "kv delta self-check-B: pair=%s seq=%d status=%s kv_cos=%.6f kv_l2=%.6f\n",
            pair_name.c_str(),
            seq_b,
            self_b.probe_status.c_str(),
            self_b.kv_cos_avg,
            self_b.kv_l2_avg);

    if (self_b.probe_status != "ok" ||
            self_b.kv_cos_avg < 0.999 ||
            self_b.kv_l2_avg > 1e-2) {
        r.memory_kind = self_b.memory_kind;
        r.probe_status = "self_check_b_failed";
        return r;
    }

    // -----------------------------
    // 3. 正式计算 full / prefix / suffix
    // -----------------------------
    const kv_delta_range_result full =
            run_kv_delta_probe_range(
                    ctx,
                    seq_a,
                    seq_b,
                    0,
                    r.compared_tokens);

    kv_delta_range_result prefix;

    if (r.common_prefix_tokens == r.compared_tokens) {
        prefix = full;
    } else {
        prefix =
                run_kv_delta_probe_range(
                        ctx,
                        seq_a,
                        seq_b,
                        0,
                        r.common_prefix_tokens);
    }

    kv_delta_range_result suffix;

    if (r.common_prefix_tokens >= r.compared_tokens) {
        suffix.memory_kind = full.memory_kind;
        suffix.probe_status = "empty_range";
        suffix.probed_kv_modules = full.probed_kv_modules;
        suffix.probed_layers = full.probed_layers;
        suffix.skipped_recurrent_layers = full.skipped_recurrent_layers;
        suffix.kv_l2_avg = 0.0;
        suffix.kv_cos_avg = 1.0;
        suffix.reusable = 1;
    } else {
        suffix =
                run_kv_delta_probe_range(
                        ctx,
                        seq_a,
                        seq_b,
                        r.common_prefix_tokens,
                        r.compared_tokens);
    }

    r.memory_kind = full.memory_kind;
    r.probe_status = full.probe_status;

    r.probed_kv_modules = full.probed_kv_modules;
    r.probed_layers = full.probed_layers;
    r.skipped_recurrent_layers = full.skipped_recurrent_layers;

    r.full_kv_l2_avg = full.kv_l2_avg;
    r.full_kv_cos_avg = full.kv_cos_avg;

    r.prefix_kv_l2_avg = prefix.kv_l2_avg;
    r.prefix_kv_cos_avg = prefix.kv_cos_avg;

    r.suffix_kv_l2_avg = suffix.kv_l2_avg;
    r.suffix_kv_cos_avg = suffix.kv_cos_avg;

    if (full.probe_status != "ok") {
        r.probe_status = full.probe_status;
        r.should_open_new_branch = 1;
        return r;
    }

    r.prefix_can_reuse =
            r.common_prefix_tokens > 0 &&
            prefix.probe_status == "ok" &&
            prefix.kv_cos_avg > 0.80;

    const bool suffix_empty =
            r.common_prefix_tokens >= r.compared_tokens;

    r.suffix_can_delta =
            suffix_empty ||
            (suffix.probe_status == "ok" &&
             (suffix.kv_cos_avg > 0.80 || suffix.kv_l2_avg < 0.05));

    r.should_open_new_branch =
            (r.prefix_can_reuse || r.suffix_can_delta) ? 0 : 1;

    r.reuse_decision =
            make_reuse_decision(
                    r.prefix_can_reuse,
                    r.suffix_can_delta,
                    r.should_open_new_branch);

    r.tree_action =
            make_tree_action(r.reuse_decision);

    r.delta_parent_node_id =
            r.suffix_can_delta ? 0 : -1;

    fprintf(stderr,
            "kv delta probe pair=%s memory=%s status=%s "
            "tokens_a=%d tokens_b=%d common_prefix=%d compared=%d "
            "full_cos=%.6f prefix_cos=%.6f suffix_cos=%.6f "
            "full_l2=%.6f prefix_l2=%.6f suffix_l2=%.6f "
            "prefix_reuse=%d suffix_delta=%d new_branch=%d decision=%s tree_action=%s\n",
            pair_name.c_str(),
            r.memory_kind.c_str(),
            r.probe_status.c_str(),
            r.tokens_a,
            r.tokens_b,
            r.common_prefix_tokens,
            r.compared_tokens,
            r.full_kv_cos_avg,
            r.prefix_kv_cos_avg,
            r.suffix_kv_cos_avg,
            r.full_kv_l2_avg,
            r.prefix_kv_l2_avg,
            r.suffix_kv_l2_avg,
            r.prefix_can_reuse,
            r.suffix_can_delta,
            r.should_open_new_branch,
            r.reuse_decision.c_str(),
            r.tree_action.c_str());

    fprintf(stderr,
            "probe ranges: pair=%s full=[0,%d), prefix=[0,%d), suffix=[%d,%d)\n",
            pair_name.c_str(),
            r.compared_tokens,
            r.common_prefix_tokens,
            r.common_prefix_tokens,
            r.compared_tokens);

    return r;
}



// ======================== KV delta ========================

// 创建一个独立 context，专门跑 KV delta probe。
// 注意：n_seq_max 不要太大，否则 n_ctx 较大时可能创建 context 失败。
static std::vector<kv_delta_probe_result> run_kv_delta_probe_suite(
        llama_model * model,
        const llama_vocab * vocab,
        int n_ctx,
        const std::vector<kv_delta_probe_case> & cases) {
    std::vector<kv_delta_probe_result> results;

    llama_context_params ctx_params =
            llama_context_default_params();

    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = 256;
    ctx_params.n_ubatch = 64;
    ctx_params.n_seq_max = 64;
    ctx_params.no_perf = true;
    ctx_params.kv_unified = true;

    llama_context * ctx =
            llama_init_from_model(model, ctx_params);

    if (ctx == nullptr) {
        fprintf(stderr, "failed to create kv-delta-probe context\n");
        return results;
    }

    for (const auto & c : cases) {
        results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        c.group_name,
                        c.pair_name,
                        c.prompt_a,
                        c.prompt_b,
                        c.seq_a,
                        c.seq_b));
    }

    clear_lora(ctx);
    llama_free(ctx);

    fprintf(stderr,
        "kv delta probe suite: cases=%zu n_ctx=%d n_seq_max=%u\n",
        cases.size(),
        n_ctx,
        ctx_params.n_seq_max);

    return results;
}


// Probe 结果映射到 Online Node
// 根据 child_request_index 找到对应的 probe 结果。
// repeat 实验时，外面会把 i 归一化成 base_request_index 后再传进来。
static const kv_delta_probe_result * find_probe_for_child_request(
        const std::vector<kv_delta_probe_result> & probe_results,
        const std::vector<kv_delta_probe_case> & probe_cases,
        int child_request_index) {
    for (const auto & c : probe_cases) {
        if (c.child_request_index != child_request_index) {
            continue;
        }

        for (const auto & r : probe_results) {
            if (r.pair_name == c.pair_name) {
                return &r;
            }
        }
    }

    return nullptr;
}


// 把 probe 结果写入 online prefix node。
// 这样后续可视化可以看到每个 node 的 suffix cosine / L2 / 决策。
static void apply_probe_to_online_node(
        online_prefix_node & node,
        const kv_delta_probe_result & probe,
        int anchor_node_id) {
    node.parent_node_id = anchor_node_id;
    node.delta_parent_node_id =
            probe.suffix_can_delta ? anchor_node_id : -1;

    node.prefix_can_reuse = probe.prefix_can_reuse;
    node.suffix_can_delta = probe.suffix_can_delta;
    node.should_open_new_branch = probe.should_open_new_branch;

    node.suffix_kv_cos = probe.suffix_kv_cos_avg;
    node.suffix_kv_l2 = probe.suffix_kv_l2_avg;
    node.reuse_decision = probe.reuse_decision;
}


// 保存 KV delta probe 结果到 CSV
// 后续 Python 可视化脚本会读取这个文件
static void save_kv_delta_probe_results(
        const std::vector<kv_delta_probe_result> & results) { // 所有 pair 的 probe 结果
    std::filesystem::create_directories(output_dir);          // 如果 output 目录不存在，就创建

    const std::string path =
            output_dir + "/kv_delta_probe_summary.csv";       // CSV 文件路径

    std::ofstream fout(path);                                 // 打开输出文件

    fout << "pair_name,group_name,memory_kind,probe_status,seq_a,seq_b,"
         << "tokens_a,tokens_b,common_prefix_tokens,compared_tokens,"
         << "probed_kv_modules,probed_layers,skipped_recurrent_layers,"
         << "full_kv_l2_avg,full_kv_cos_avg,"
         << "prefix_kv_l2_avg,prefix_kv_cos_avg,"
         << "suffix_kv_l2_avg,suffix_kv_cos_avg,"
         << "prefix_can_reuse,suffix_can_delta,should_open_new_branch,"
         << "delta_parent_node_id,reuse_decision,tree_action\n"; // 写 CSV 表头

    for (const auto & r : results) {                          // 遍历每一个 pair 的结果
        fout << r.pair_name << ","                            // pair 名称
             << r.group_name << ","                           // group 名称
             << r.memory_kind << ","                          // KV memory 类型
             << r.probe_status << ","                         // probe 是否成功
             << r.seq_a << ","                                // A seq id
             << r.seq_b << ","                                // B seq id
             << r.tokens_a << ","                             // A token 数
             << r.tokens_b << ","                             // B token 数
             << r.common_prefix_tokens << ","                 // 公共 prefix token 数
             << r.compared_tokens << ","                      // 实际比较 token 数
             << r.probed_kv_modules << ","                    // probe 的 KV module 数
             << r.probed_layers << ","                        // probe 的层数
             << r.skipped_recurrent_layers << ","             // 跳过 recurrent 层数
             << r.full_kv_l2_avg << ","                       // full KV L2
             << r.full_kv_cos_avg << ","                      // full KV cosine
             << r.prefix_kv_l2_avg << ","                     // prefix KV L2
             << r.prefix_kv_cos_avg << ","                    // prefix KV cosine
             << r.suffix_kv_l2_avg << ","                     // suffix KV L2
             << r.suffix_kv_cos_avg << ","                    // suffix KV cosine
             << r.prefix_can_reuse << ","                     // prefix 是否可复用
             << r.suffix_can_delta << ","                     // suffix 是否可做 delta
             << r.should_open_new_branch << ","               // 是否新建分支
             << r.delta_parent_node_id << ","                 // delta 父节点
             << r.reuse_decision << ","                       // 复用决策
             << r.tree_action << "\n";                        // tree 操作
    }

    fprintf(stderr,
            "saved kv delta probe results to %s, rows=%zu\n",
            path.c_str(),
            results.size());                                  // 打印保存位置和行数
}










// ===============================
// 6. 在线 prefix tree
// ===============================
// ======================== group定位 ========================
// 根据 prompt token 和每个 group 的 prompt pattern 做公共前缀匹配，
// 选择公共前缀最长的 group。
static int route_prompt_to_group(
        const std::vector<lora_group_runtime> & groups,      // 所有 LoRA group
        const std::vector<llama_token> & prompt_tokens) {    // 当前请求的 prompt token
    int best_group = groups.empty() ? -1 : groups[0].group_id;
    // 默认 group：如果 groups 为空则 -1，否则先用第一个 group

    int best_score = -1;
    // best_score 表示当前找到的最长公共 prefix 长度

    for (const auto & group : groups) {     // 遍历每个 group
        for (const auto & pattern : group.prompt_patterns) {
            // 遍历该 group 下的每个 prompt pattern

            const int score =
                    common_prefix_len(prompt_tokens, pattern.tokens);
            // 计算当前请求 prompt 和该 pattern 的公共 token 前缀长度

            if (score > best_score) {       // 如果这个 group/pattern 更匹配
                best_score = score;         // 更新最长匹配长度
                best_group = group.group_id;// 更新最佳 group
            }
        }
    }

    return best_group;                      // 返回路由到的 group id
}

// 根据 group_id 找到对应 group。
// 找不到时返回 nullptr。
static const lora_group_runtime * find_group(
        const std::vector<lora_group_runtime> & groups, // 所有 group。
        int group_id) {                                 // 要找的 group id。
    for (const auto & group : groups) {                 // 遍历 group。
        if (group.group_id == group_id) {               // id 匹配。
            return &group;                              // 返回该 group 地址。
        }
    }

    return nullptr;                                     // 没找到。
}

// ======================== prefix命中 ========================
// 在已有 online prefix nodes 中查找当前 prompt 是否能命中某个 prefix。
// 命中规则：
// 1. 必须在同一个 routed_group_id 下。
// 2. 选择和当前 prompt 公共前缀最长的 node。
static online_route_result route_online_prefix_node(
        const std::vector<online_prefix_node> & nodes,  // 当前已经建立的 online prefix nodes。
        int routed_group_id,                            // 当前请求路由到的 group id。
        const std::vector<llama_token> & prompt_tokens) {// 当前请求的 prompt token。
    online_route_result result;                         // 返回结果。

    result.group_id = routed_group_id;                  // 记录当前 group。
    result.suffix_len = (int) prompt_tokens.size();     // 默认 suffix 是完整 prompt。

    int best_node_id = -1;                              // 当前最佳命中 node id。
    int best_prefix_len = 0;                            // 当前最长公共 prefix 长度。

    for (const auto & node : nodes) {                   // 遍历已有 prefix node。
        if (node.group_id != routed_group_id) {         // 只在同 group 内查找。
            continue;
        }

        const int prefix_len =
                common_prefix_len(prompt_tokens, node.prefix_tokens);
        // 计算当前 prompt 和该 node 保存的 prefix_tokens 的公共前缀长度。

        if (prefix_len > best_prefix_len) {             // 如果这个 node 匹配更长。
            best_prefix_len = prefix_len;               // 更新最佳 prefix 长度。
            best_node_id = node.node_id;                // 更新最佳 node id。
        }
    }

    result.node_id = best_node_id;                      // 记录命中的 node。
    result.exact_prefix_len = best_prefix_len;          // 记录复用 prefix 长度。
    result.suffix_len = (int) prompt_tokens.size() - best_prefix_len;
    // 当前请求剩下需要单独处理的 suffix 长度。

    result.exact_prefix_hit = best_node_id >= 0 && best_prefix_len > 0;
    // 只要找到 node 且公共 prefix > 0，就认为命中。

    return result;                                      // 返回路由结果。
}

// 根据 node_id 找到 online prefix node。
// 找不到返回 nullptr。
static const online_prefix_node * find_online_node(
        const std::vector<online_prefix_node> & nodes, // 所有 online prefix node。
        int node_id) {                                 // 要找的 node id。
    for (const auto & node : nodes) {                  // 遍历 nodes。
        if (node.node_id == node_id) {                 // id 匹配。
            return &node;                              // 返回 node 地址。
        }
    }

    return nullptr;                                    // 没找到。
}

// ========================  结果 ========================
// 保存 online prefix tree 的状态。
// 这个 CSV 用来画在线节点、命中次数、suffix 决策。
static void save_online_prefix_delta_tree(
        const std::vector<online_prefix_node> & nodes) {
    std::filesystem::create_directories(output_dir);

    const std::string path =
            output_dir + "/online_prefix_delta_tree_summary.csv";

    std::ofstream fout(path);

    fout << "node_id,group_id,group_name,parent_node_id,delta_parent_node_id,"
         << "cache_seq_id,prefix_len,hit_count,"
         << "prefix_can_reuse,suffix_can_delta,should_open_new_branch,"
         << "suffix_kv_cos,suffix_kv_l2,reuse_decision\n";

    for (const auto & node : nodes) {
        fout << node.node_id << ","
             << node.group_id << ","
             << node.group_name << ","
             << node.parent_node_id << ","
             << node.delta_parent_node_id << ","
             << node.cache_seq_id << ","
             << node.prefix_len << ","
             << node.hit_count << ","
             << node.prefix_can_reuse << ","
             << node.suffix_can_delta << ","
             << node.should_open_new_branch << ","
             << node.suffix_kv_cos << ","
             << node.suffix_kv_l2 << ","
             << node.reuse_decision << "\n";
    }

    fprintf(stderr, "saved online prefix delta tree to %s\n", path.c_str());
}

// ===============================
// 7. 实验运行函数
// ===============================
// ======================== baseline ========================
// baseline 路径：不复用任何 Prefix KV。
// 每个请求都绑定对应 LoRA，然后把完整 prompt 从头到尾 eval 一遍。
// 这条路径用于和 online prefix reuse 对比 TTFT / total / TPS。
static sample_result run_baseline_request(
        llama_context * ctx,                          // llama.cpp context，包含模型状态和 KV cache。
        const std::vector<lora_node_runtime> & lora_nodes, // 所有 LoRA runtime 节点。
        const std::vector<lora_group_runtime> & groups,    // 所有 LoRA group。
        const request_item & req,                    // 当前请求信息。
        const request_tokens & toks,                 // 当前请求 tokenize 后的 token。
        int request_index,                           // 当前请求编号，用作 seq id。
        int n_predict) {                             // decode 生成 token 数。
    sample_result r;                                 // 保存本次实验结果。

    r.repeat_id = req.repeat_id;                     // 记录第几轮重复实验。
    r.context_tag = req.context_tag;                 // 记录 ctx_50 / ctx_100 等上下文标签。

    const lora_group_runtime * group =
            find_group(groups, toks.routed_group_id);
    // 根据 token 路由结果找到当前请求所属 group。

    r.mode = "baseline_no_group_kv_reuse";           // 记录实验模式：不复用 KV。
    r.group_name = group ? group->group_name : "unknown"; // 记录 group 名称。
    r.leaf_lora_id = req.leaf_lora_id;               // 记录当前请求使用的 LoRA id。
    r.lora_name = lora_nodes[req.leaf_lora_id].short_name; // 记录 LoRA 短名称。
    r.n_prompt_tokens = (int) toks.full.size();      // 记录 prompt token 数。
    r.n_prefix_tokens = 0;                           // baseline 不复用 prefix，所以为 0。
    r.n_suffix_tokens = (int) toks.full.size();      // baseline 中整段 prompt 都算作 suffix。
    r.n_predict = n_predict;                         // 记录 decode token 数。

    if (r.n_prompt_tokens > 0) {                     // 避免除以 0。
        r.prefix_reuse_rate =
                (double) r.n_prefix_tokens / (double) r.n_prompt_tokens;
        // baseline prefix 复用比例 = 0。

        r.suffix_delta_rate =
                (double) r.n_suffix_tokens / (double) r.n_prompt_tokens;
        // baseline suffix 比例 = 1。
    }

    const llama_seq_id seq_id = request_index;       // baseline 中每个 request 用自己的 seq。
    const double gpu_start = get_gpu_used_mb();      // 记录请求开始前 GPU 显存。
    double gpu_peak = gpu_start;                     // 初始化显存峰值。

    const double t0 = now_ms();                      // 记录整个请求开始时间。

    r.lora_bind_ms =
            bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);
    // 绑定当前请求对应 LoRA，并记录耗时。

    const double suffix0 = now_ms();                 // 记录完整 prompt eval 开始时间。

    eval_tokens(
            ctx,                                     // 当前 context。
            toks.full,                               // 完整 prompt token。
            seq_id,                                  // 写入当前 seq。
            0,                                       // 从 position 0 开始。
            true);                                   // 最后一个 token 输出 logits。

    const double suffix1 = now_ms();                 // 记录完整 prompt eval 结束时间。

    r.suffix_ms = suffix1 - suffix0;                 // baseline 中 suffix_ms 实际等于完整 prefill 时间。

    const llama_token repeated = toks.full.back();   // 为了做固定 decode，用 prompt 最后一个 token 反复 decode。

    const double decode0 = now_ms();                 // decode 阶段开始时间。

    for (int i = 0; i < n_predict; i++) {            // 循环生成 n_predict 个 token。
        const int pos = (int) toks.full.size() + i;  // 当前 decode token 的 position。

        decode_one(ctx, repeated, seq_id, pos);      // 执行一个 token 的 decode。

        if (i == 0) {                                // 第一个 decode 完成时就是 TTFT。
            r.ttft_ms = now_ms() - t0;               // TTFT = 从请求开始到第一个 token 完成。
        }

        gpu_peak = std::max(gpu_peak, get_gpu_used_mb()); // 更新 GPU 显存峰值。
    }

    const double decode1 = now_ms();                 // decode 阶段结束时间。

    r.decode_ms = decode1 - decode0;                 // decode 总耗时。
    r.total_ms = decode1 - t0;                       // 请求总耗时。
    r.tps = n_predict / std::max(0.001, r.decode_ms / 1000.0);
    // TPS = 生成 token 数 / decode 秒数。

    r.gpu_start_mb = gpu_start;                      // 保存开始显存。
    r.gpu_peak_mb = gpu_peak;                        // 保存峰值显存。
    r.gpu_peak_delta_mb = gpu_peak - gpu_start;      // 保存显存增量。

    return r;                                        // 返回实验结果。
}


// online build 路径：当前请求没有命中已有 prefix node，
// 所以需要把完整 prompt eval 一遍，并把它作为新的 prefix node 存进 cache_seq_id。
// 后续相似请求命中这个 node 后，就可以复制它的 Prefix KV。
static sample_result run_online_build_prefix_request(
        llama_context * ctx,                          // llama.cpp context。
        llama_memory_t mem,                           // llama.cpp memory/KV 管理对象。
        const std::vector<lora_node_runtime> & lora_nodes, // LoRA runtime 节点。
        const std::vector<lora_group_runtime> & groups,    // LoRA group。
        const request_item & req,                     // 当前请求。
        const request_tokens & toks,                  // 当前请求 token。
        llama_seq_id cache_seq_id,                    // 用来保存 prefix KV 的 seq。
        llama_seq_id request_seq_id,                  // 当前请求实际 decode 使用的 seq。
        int node_id,                                  // 新建 online prefix node id。
        int n_predict) {                              // decode token 数。
    sample_result r;                                  // 保存实验结果。

    r.repeat_id = req.repeat_id;                      // 记录 repeat id。
    r.context_tag = req.context_tag;                  // 记录上下文标签。

    const lora_group_runtime * group =
            find_group(groups, toks.routed_group_id); // 找到当前请求所属 group。

    r.mode = "online_prefix_build";                   // 记录模式：在线建 prefix node。
    r.group_name = group ? group->group_name : "unknown"; // group 名称。
    r.lora_name = lora_nodes[req.leaf_lora_id].short_name; // LoRA 名称。
    r.leaf_lora_id = req.leaf_lora_id;                // LoRA id。
    r.online_node_id = node_id;                       // 当前新建 node id。
    r.exact_prefix_hit = 0;                           // 建节点时不算命中已有 prefix。

    r.routed_group_id = toks.routed_group_id;         // 记录路由 group。
    r.reuse_decision = "create_prefix_node";          // 决策：创建 prefix node。

    r.n_prompt_tokens = (int) toks.full.size();       // prompt token 数。
    r.n_prefix_tokens = 0;                            // 建节点时没有复用 prefix。
    r.n_suffix_tokens = (int) toks.full.size();       // 整段 prompt 都需要 eval。
    r.n_predict = n_predict;                          // decode token 数。

    if (r.n_prompt_tokens > 0) {                      // 避免除 0。
        r.prefix_reuse_rate = 0.0;                    // 没有复用 prefix。
        r.suffix_delta_rate = 1.0;                    // 整段都算作需要计算。
    }

    const double gpu_start = get_gpu_used_mb();       // 记录开始显存。
    double gpu_peak = gpu_start;                      // 初始化峰值显存。

    const double t0 = now_ms();                       // 请求开始时间。

    clear_lora(ctx);                                  // 建 prefix anchor 时先清空 LoRA。
    // 这里设计含义：prefix anchor 先按 base model 写入 cache_seq_id。
    // 后面请求命中后，再绑定对应 LoRA 处理 suffix / decode。

    const double prefix0 = now_ms();                  // prefix build 开始时间。

    eval_tokens(
            ctx,                                      // 当前 context。
            toks.full,                                // 完整 prompt token。
            cache_seq_id,                             // 写入 prefix cache seq。
            0,                                        // 从 position 0 开始。
            false);                                   // 建 prefix 时不需要 logits。

    const double prefix1 = now_ms();                  // prefix build 结束时间。

    r.prefix_ms = prefix1 - prefix0;                  // 记录建 prefix KV 耗时。

    llama_memory_seq_cp(
            mem,                                      // memory/KV 管理对象。
            cache_seq_id,                             // 源 seq：prefix cache。
            request_seq_id,                           // 目标 seq：当前请求。
            0,                                        // 从 position 0 开始复制。
            (int) toks.full.size());                  // 复制完整 prompt 长度。
    // 建节点请求本身也要 decode，所以把 cache_seq_id 的 KV 复制到 request_seq_id。

    r.lora_bind_ms =
            bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);
    // 绑定当前请求 LoRA。

    const llama_token repeated = toks.full.back();    // 固定 decode token。

    const double decode0 = now_ms();                  // decode 开始时间。

    for (int i = 0; i < n_predict; i++) {             // decode n_predict 个 token。
        const int pos = (int) toks.full.size() + i;   // 当前 decode position。

        decode_one(ctx, repeated, request_seq_id, pos);// decode 一个 token。

        if (i == 0) {                                 // 第一个 token 完成。
            r.ttft_ms = now_ms() - t0;                // 记录 TTFT。
        }

        gpu_peak = std::max(gpu_peak, get_gpu_used_mb()); // 更新显存峰值。
    }

    const double decode1 = now_ms();                  // decode 结束时间。

    r.decode_ms = decode1 - decode0;                  // decode 耗时。
    r.total_ms = decode1 - t0;                        // 总耗时。
    r.tps = n_predict / std::max(0.001, r.decode_ms / 1000.0); // decode TPS。
    r.gpu_start_mb = gpu_start;                       // 开始显存。
    r.gpu_peak_mb = gpu_peak;                         // 峰值显存。
    r.gpu_peak_delta_mb = gpu_peak - gpu_start;       // 显存增量。

    return r;                                         // 返回结果。
}



// online prefix reuse 路径：
// 1. 从已有 online prefix node 复制 Prefix KV 到当前 request seq。
// 2. 绑定当前 LoRA。
// 3. 对不同的 suffix token 重新 eval，写入 suffix KV。
// 4. decode n_predict 个 token。
//
// 当前实验主要收益来自这条路径。
static sample_result run_online_prefix_reuse_request(
        llama_context * ctx,                          // llama.cpp context。
        llama_memory_t mem,                           // memory/KV 管理对象。
        const std::vector<lora_node_runtime> & lora_nodes, // LoRA runtime 节点。
        const std::vector<lora_group_runtime> & groups,    // LoRA group。
        const online_prefix_node & node,              // 命中的 online prefix node。
        const request_item & req,                     // 当前请求。
        const request_tokens & toks,                  // 当前请求 token。
        llama_seq_id request_seq_id,                  // 当前请求 seq。
        int matched_prefix_len,                       // 命中的 prefix token 长度。
        int n_predict,                                // decode token 数。
        double kv_mb_per_token) {                     // 每 token KV 估算大小。
    sample_result r;                                  // 保存实验结果。

    r.repeat_id = req.repeat_id;                      // repeat id。
    r.context_tag = req.context_tag;                  // 上下文标签。

    const lora_group_runtime * group =
            find_group(groups, toks.routed_group_id); // 找到所属 group。

    r.mode = "online_prefix_reuse";                   // 记录模式：在线 prefix 复用。
    r.group_name = group ? group->group_name : "unknown"; // group 名称。
    r.lora_name = lora_nodes[req.leaf_lora_id].short_name; // LoRA 名称。
    r.leaf_lora_id = req.leaf_lora_id;                // LoRA id。

    r.routed_group_id = toks.routed_group_id;         // 路由 group id。
    r.reuse_decision = "exact_prefix_reuse_suffix_recompute";
    // 当前路径含义：prefix 直接复用，suffix 重新计算。

    r.online_node_id = node.node_id;                  // 命中的 node id。
    r.exact_prefix_hit = 1;                           // 表示命中 prefix。

    r.n_prompt_tokens = (int) toks.full.size();       // prompt token 数。
    r.n_prefix_tokens = matched_prefix_len;           // 复用 prefix token 数。
    r.n_suffix_tokens = (int) toks.full.size() - matched_prefix_len;
    // suffix token 数 = 总长度 - prefix 长度。
    r.n_predict = n_predict;                          // decode token 数。

    if (r.n_prompt_tokens > 0) {                      // 避免除 0。
        r.prefix_reuse_rate =
                (double) r.n_prefix_tokens / (double) r.n_prompt_tokens;
        // prefix 复用比例。

        r.suffix_delta_rate =
                (double) r.n_suffix_tokens / (double) r.n_prompt_tokens;
        // suffix 占比。
    }

    r.estimated_saved_kv_mb =
            (double) matched_prefix_len * kv_mb_per_token;
    // 估算节省的 KV = 复用 prefix token 数 * 每 token KV 大小。

    const double gpu_start = get_gpu_used_mb();       // 开始显存。
    double gpu_peak = gpu_start;                      // 峰值显存。

    const double t0 = now_ms();                       // 请求开始时间。

    const double prefix0 = now_ms();                  // prefix copy 开始。

    llama_memory_seq_cp(
            mem,                                      // memory/KV 管理对象。
            node.cache_seq_id,                        // 源 seq：prefix node 的 cache seq。
            request_seq_id,                           // 目标 seq：当前请求 seq。
            0,                                        // 从 position 0 开始。
            matched_prefix_len);                      // 复制 matched_prefix_len 个 token 的 KV。

    const double prefix1 = now_ms();                  // prefix copy 结束。

    r.prefix_ms = prefix1 - prefix0;                  // prefix copy 耗时。

    r.lora_bind_ms =
            bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);
    // 绑定当前请求的 LoRA。

    std::vector<llama_token> suffix(
            toks.full.begin() + matched_prefix_len,
            toks.full.end());
    // 取出不同的 suffix token。
    // 这些 token 没有被 prefix KV 覆盖，需要重新 eval。

    const double suffix0 = now_ms();                  // suffix eval 开始。

    eval_tokens(
            ctx,                                      // 当前 context。
            suffix,                                   // suffix token。
            request_seq_id,                           // 写入当前请求 seq。
            matched_prefix_len,                       // suffix 的 position 从 prefix_len 开始。
            true);                                    // 最后一个 suffix token 输出 logits。

    const double suffix1 = now_ms();                  // suffix eval 结束。

    r.suffix_ms = suffix1 - suffix0;                  // suffix eval 耗时。

    const llama_token repeated =
            suffix.empty() ? toks.full.back() : suffix.back();
    // decode 时重复使用最后一个 token。
    // 如果 suffix 为空，就用完整 prompt 最后一个 token。

    const double decode0 = now_ms();                  // decode 开始。

    for (int i = 0; i < n_predict; i++) {             // decode n_predict 个 token。
        const int pos = (int) toks.full.size() + i;   // 当前 decode position。

        decode_one(ctx, repeated, request_seq_id, pos);// decode 一个 token。

        if (i == 0) {                                 // 第一个 token 完成。
            r.ttft_ms = now_ms() - t0;                // TTFT。
        }

        gpu_peak = std::max(gpu_peak, get_gpu_used_mb()); // 更新显存峰值。
    }

    const double decode1 = now_ms();                  // decode 结束。

    r.decode_ms = decode1 - decode0;                  // decode 耗时。
    r.total_ms = decode1 - t0;                        // 请求总耗时。
    r.tps = n_predict / std::max(0.001, r.decode_ms / 1000.0); // TPS。
    r.gpu_start_mb = gpu_start;                       // 开始显存。
    r.gpu_peak_mb = gpu_peak;                         // 峰值显存。
    r.gpu_peak_delta_mb = gpu_peak - gpu_start;       // 显存增量。

    return r;                                         // 返回结果。
}


// suffix delta materialize 路径：
// 目标是尝试跳过 suffix eval，改用 anchor KV + delta 恢复 suffix KV。
// 当前这条路径仍是实验性路径。
// 如果 llama_kv_seq_delta_materialize() 失败，则提前返回，materialize_timed=0。
// 只有 materialize 成功并且后续 decode 完成，materialize_timed 才会置为 1。
static sample_result run_online_suffix_delta_materialize_request(
        llama_context * ctx,                          // llama.cpp context。
        llama_memory_t mem,                           // memory/KV 管理对象。
        const std::vector<lora_node_runtime> & lora_nodes, // LoRA runtime 节点。
        const std::vector<lora_group_runtime> & groups,    // LoRA group。
        const online_prefix_node & anchor_node,       // 命中的 anchor prefix node。
        const request_item & req,                     // 当前请求。
        const request_tokens & toks,                  // 当前请求 token。
        llama_seq_id child_full_seq_id,               // 已经完整算过 suffix 的 child seq。
        llama_seq_id request_seq_id,                  // materialize 目标 seq。
        int matched_prefix_len,                       // prefix 长度。
        int n_predict,                                // decode token 数。
        double kv_mb_per_token) {                     // 每 token KV 估算大小。
    sample_result r;                                  // 保存结果。

    r.repeat_id = req.repeat_id;                      // repeat id。
    r.context_tag = req.context_tag;                  // 上下文标签。
    r.materialize_timed = 0;                          // 默认 materialize 未完成真实计时。

    const lora_group_runtime * group =
            find_group(groups, toks.routed_group_id); // 找到所属 group。

    r.mode = "online_suffix_delta_materialize";       // 记录模式。
    r.group_name = group ? group->group_name : "unknown"; // group 名称。
    r.lora_name = lora_nodes[req.leaf_lora_id].short_name; // LoRA 名称。
    r.leaf_lora_id = req.leaf_lora_id;                // LoRA id。

    r.routed_group_id = toks.routed_group_id;         // group id。
    r.reuse_decision = "prefix_reuse_suffix_delta_materialize";
    // 当前路径目标：prefix 复用 + suffix delta materialize。

    r.online_node_id = anchor_node.node_id;           // anchor node id。
    r.exact_prefix_hit = 1;                           // 命中 prefix。

    r.n_prompt_tokens = (int) toks.full.size();       // prompt token 数。
    r.n_prefix_tokens = matched_prefix_len;           // prefix token 数。
    r.n_suffix_tokens = (int) toks.full.size() - matched_prefix_len;
    // suffix token 数。
    r.n_predict = n_predict;                          // decode token 数。

    if (r.n_prompt_tokens > 0) {                      // 避免除 0。
        r.prefix_reuse_rate =
                (double) r.n_prefix_tokens / (double) r.n_prompt_tokens;
        // prefix 复用比例。

        r.suffix_delta_rate =
                (double) r.n_suffix_tokens / (double) r.n_prompt_tokens;
        // suffix 占比。
    }

    r.estimated_saved_kv_mb =
            (double) matched_prefix_len * kv_mb_per_token;
    // 估算 prefix 复用节省的 KV 大小。

    const double gpu_start = get_gpu_used_mb();       // 记录开始显存。
    double gpu_peak = gpu_start;                      // 初始化显存峰值。

    const double t0 = now_ms();                       // 请求开始时间。

    const double prefix0 = now_ms();                  // prefix copy 开始。

    llama_memory_seq_cp(
            mem,                                      // memory/KV 管理对象。
            anchor_node.cache_seq_id,                 // 源 seq：anchor prefix cache。
            request_seq_id,                           // 目标 seq：当前 materialize 请求。
            0,                                        // 从 position 0 开始。
            matched_prefix_len);                      // 复制 prefix 范围。

    const double prefix1 = now_ms();                  // prefix copy 结束。

    r.prefix_ms = prefix1 - prefix0;                  // prefix copy 耗时。

    r.lora_bind_ms =
            bind_lora_path(ctx, lora_nodes, req.leaf_lora_id);
    // 绑定当前请求 LoRA。

    llama_kv_delta_materialize_stats delta_stats = {};
    // materialize API 返回的统计信息，包括 delta 大小、写回 KV 大小、失败状态等。

    const double delta0 = now_ms();                   // delta materialize 开始。

    const bool ok =
            llama_kv_seq_delta_materialize(
                    ctx,                              // 当前 context。
                    anchor_node.cache_seq_id,         // anchor seq，提供 KV_A。
                    child_full_seq_id,                // child full seq，提供已算好的 KV_B。
                    request_seq_id,                   // 目标 seq，把恢复出的 KV 写到这里。
                    matched_prefix_len,               // suffix 起点。
                    (int) toks.full.size(),           // suffix 终点，也就是 prompt 总长度。
                    &delta_stats);                    // 输出统计信息。

    const double delta1 = now_ms();                   // delta materialize 结束。

    r.delta_materialize_ms = delta1 - delta0;         // 记录 materialize 耗时。
    r.suffix_delta_materialized = ok ? 1 : 0;         // 是否 materialize 成功。
    r.materialize_timed = 0;                          // 还没 decode 完，不算完整计时成功。

    r.delta_fp32_mb =
            (double) delta_stats.delta_fp32_bytes / 1024.0 / 1024.0;
    // 记录 delta 的 FP32 等价大小。

    r.materialized_kv_mb =
            (double) delta_stats.materialized_kv_bytes / 1024.0 / 1024.0;
    // 记录写回目标 seq 的 KV 大小。

    if (!ok) {                                       // 如果 materialize 失败。
        fprintf(stderr,
                "suffix delta materialize failed: request_seq=%d child_full_seq=%d status=%s\n",
                request_seq_id,
                child_full_seq_id,
                delta_stats.status);
        return r;                                    // 提前返回，materialize_timed 仍为 0。
    }

    r.suffix_ms = 0.0;
    // 如果 materialize 成功，suffix KV 已经由 delta 恢复并写回，
    // 所以这里不再 eval suffix，suffix_ms 记为 0。

    const llama_token repeated = toks.full.back();   // 固定 decode token。

    const double decode0 = now_ms();                 // decode 开始。

    for (int i = 0; i < n_predict; i++) {            // decode n_predict 个 token。
        const int pos = (int) toks.full.size() + i;  // 当前 decode position。

        decode_one(ctx, repeated, request_seq_id, pos);// decode 一个 token。

        if (i == 0) {                                // 第一个 token 完成。
            r.ttft_ms = now_ms() - t0;               // TTFT。
        }

        gpu_peak = std::max(gpu_peak, get_gpu_used_mb()); // 更新显存峰值。
    }

    const double decode1 = now_ms();                 // decode 结束。

    r.decode_ms = decode1 - decode0;                 // decode 耗时。
    r.total_ms = decode1 - t0;                       // 总耗时。
    r.tps = n_predict / std::max(0.001, r.decode_ms / 1000.0); // TPS。
    r.materialize_timed = 1;                         // materialize + decode 完整跑通。
    r.gpu_start_mb = gpu_start;                      // 开始显存。
    r.gpu_peak_mb = gpu_peak;                        // 峰值显存。
    r.gpu_peak_delta_mb = gpu_peak - gpu_start;      // 显存增量。

    return r;                                        // 返回结果。
}

// ===============================
// 8. 保存实验结果
// ===============================
static void save_results(
        const std::vector<sample_result> & results) {
    std::filesystem::create_directories(output_dir);

    const std::string path =
            output_dir + "/group_node_kv_lora_tree_summary.csv";

    std::ofstream fout(path);

    fout << "repeat_id,context_tag,materialize_timed,"
        << "suffix_delta_materialized,delta_materialize_ms,delta_fp32_mb,materialized_kv_mb,"
        << "mode,group_name,lora_name,leaf_lora_id,"
        << "routed_group_id,reuse_decision,"
        << "online_node_id,exact_prefix_hit,"
        << "n_prompt_tokens,n_prefix_tokens,n_suffix_tokens,"
        << "prefix_reuse_rate,suffix_delta_rate,n_predict,"
        << "route_ms,prefix_ms,lora_bind_ms,suffix_ms,"
        << "ttft_ms,decode_ms,total_ms,tps,"
        << "gpu_start_mb,gpu_peak_mb,gpu_peak_delta_mb,"
        << "estimated_saved_kv_mb\n";

    for (const auto & r : results) {
        fout << r.repeat_id << ","
            << r.context_tag << ","
            << r.materialize_timed << ","
            << r.suffix_delta_materialized << ","
            << r.delta_materialize_ms << ","
            << r.delta_fp32_mb << ","
            << r.materialized_kv_mb << ","
             << r.mode << ","
             << r.group_name << ","
             << r.lora_name << ","
             << r.leaf_lora_id << ","
             << r.routed_group_id << ","
             << r.reuse_decision << ","
             << r.online_node_id << ","
             << r.exact_prefix_hit << ","
             << r.n_prompt_tokens << ","
             << r.n_prefix_tokens << ","
             << r.n_suffix_tokens << ","
             << r.prefix_reuse_rate << ","
             << r.suffix_delta_rate << ","
             << r.n_predict << ","
             << r.route_ms << ","
             << r.prefix_ms << ","
             << r.lora_bind_ms << ","
             << r.suffix_ms << ","
             << r.ttft_ms << ","
             << r.decode_ms << ","
             << r.total_ms << ","
             << r.tps << ","
             << r.gpu_start_mb << ","
             << r.gpu_peak_mb << ","
             << r.gpu_peak_delta_mb << ","
             << r.estimated_saved_kv_mb << "\n";
    }

    fprintf(stderr, "saved results to %s\n", path.c_str());
}



// ==========================================
// 第 4 部分：main 主流程
// 作用：
// 1. 加载 Qwen2.5 base model。
// 2. 加载多个 LoRA adapter。
// 3. 构造长上下文多 LoRA 请求。
// 4. 先跑 KV delta probe，判断哪些 pair 适合 delta。
// 5. 跑 baseline，也就是完全不复用 KV。
// 6. 跑 online prefix tree，也就是 prefix KV 复用主线。
// 7. 尝试 suffix delta materialize 实验路径。
// 8. 保存 CSV，交给 Python 画图。
// ==========================================


int main() {
    std::setlocale(LC_NUMERIC, "C");  // 设置数字格式区域为 C，保证小数点使用 "."
    
    // Qwen3.5-4B
    // const std::string model_path = "D:/ecnu_experiment/Model/Qwen3.5-4B-gguf/Qwen3.5-4B-BF16.gguf";
    // Qwen2.5-1.5B
    const std::string model_path = "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/Qwen2.5-1.5B-Instruct-f16.gguf";
    const int ngl = 99;          // 模型多少层放到 GPU 上
    const int n_ctx = 16384;     // llama.cpp context 的最大上下文长度，决定 KV cache 可容纳的 token 数
    const int n_predict = 32;    // 每条请求后续 decode 生成多少个 token，用来测 TTFT / TPS。

    ggml_backend_load_all();     // 加载所有可用 ggml backend，例如 CPU / CUDA。
    llama_model_params model_params = llama_model_default_params();  // 获取默认模型加载参数。
    model_params.n_gpu_layers = ngl;    // 设置 GPU offload 层数。

    // 从 GGUF 文件加载 base model
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params); 
    if (model == nullptr) {      // 如果模型加载失败，打印错误并退出。
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    // 从模型中获取 vocab，后续 tokenize prompt 时需要用
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 是否只跑纯 Transformer（不加 LoRA） true是，false不是
    const bool pure_transformer_probe_only = false;
    if (pure_transformer_probe_only) { // 如果只做纯模型 KV probe
        fprintf(stderr, "\n===== pure transformer KV delta probe only =====\n");  // 打印当前模式
        std::vector<kv_delta_probe_result> probe_results; // 保存 probe 结果

        // 创建 context 参数
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx; // 设置上下文长度。
        ctx_params.n_batch = 256; // 设置 batch token 数。
        ctx_params.n_ubatch = 64; // 设置 micro-batch token 数。
        ctx_params.n_seq_max = 1024; // 设置最多同时使用的 seq 数量。
        ctx_params.no_perf = true;// 关闭性能统计，减少额外输出。
        ctx_params.kv_unified = true; // 使用 unified KV buffer，保证 seq_cp 等 full KV 操作可用。

        // 根据模型和 context 参数创建 llama context
        llama_context * ctx = llama_init_from_model(model, ctx_params);
        if (ctx == nullptr) { // 如果 context 创建失败，释放 model 并退出。
            fprintf(stderr, "failed to create pure transformer probe context\n");
            llama_model_free(model);
            return 1;
        }

        // sanity case：两个 prompt 完全相同，理论上 KV 应该高度一致。
        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "sanity",
                        "same_prompt_monkey",
                        "You are an animal introduction expert. Please introduce the characteristics of monkey.",
                        "You are an animal introduction expert. Please introduce the characteristics of monkey.",
                        0,
                        1));

        // same prefix case：prompt 前缀相同，但最后实体 monkey / tiger 不同。
        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "same_prefix_no_lora",
                        "same_prefix_monkey_tiger_no_lora",
                        "You are an animal introduction expert. Please introduce the characteristics of monkey.",
                        "You are an animal introduction expert. Please introduce the characteristics of tiger.",
                        2,
                        3));

        // mobile-like case：共享同一篇文章，只是最后任务 summarize / rewrite 不同。
        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "mobile_like",
                        "same_article_summary_vs_rewrite",
                        "You are a document assistant. Article: Tigers are large cats native to Asia. They have orange fur with black stripes. Tigers are solitary hunters and usually live in forests, grasslands, and wetlands. They are strong swimmers and use stealth to approach prey. Please summarize this article.",
                        "You are a document assistant. Article: Tigers are large cats native to Asia. They have orange fur with black stripes. Tigers are solitary hunters and usually live in forests, grasslands, and wetlands. They are strong swimmers and use stealth to approach prey. Please rewrite this article.",
                        4,
                        5));

        // product case：共享同一段商品描述，只是最后 summarize / optimize 不同。
        probe_results.push_back(
                run_kv_delta_probe_pair(
                        ctx,
                        vocab,
                        "mobile_like",
                        "same_product_summary_vs_optimize",
                        "You are a product assistant. Article: The product is a portable smart camera with night vision, long battery life, local storage, and wireless connection. It is designed for home security and travel monitoring. Please summarize this article.",
                        "You are a product assistant. Article: The product is a portable smart camera with night vision, long battery life, local storage, and wireless connection. It is designed for home security and travel monitoring. Please optimize this article.",
                        6,
                        7));

        // 保存纯模型 probe 结果到 CSV。
        save_kv_delta_probe_results(probe_results);

        // 清空当前 context 上绑定的 LoRA，虽然此分支没有 LoRA，但保持清理习惯。
        clear_lora(ctx);

        // 释放 context。
        llama_free(ctx);

        // 释放模型。
        llama_model_free(model);

        // 打印结束信息。
        fprintf(stderr, "pure transformer KV delta probe finished.\n");

        // 正常结束程序。
        return 0;
    }

    // 从 generated_lora_tree_qwen2.5.hpp 中读取 LoRA 静态配置，转换成运行时结构。
    std::vector<lora_node_runtime> lora_nodes =
            make_runtime_lora_nodes();

    // 遍历每个 LoRA 节点，加载对应 adapter。
    for (auto & node : lora_nodes) {
        // 从 node.path 加载 LoRA adapter。
        node.adapter =
                llama_adapter_lora_init(
                        model,
                        node.path.c_str());

        // 如果某个 LoRA 加载失败，释放已经加载成功的 LoRA，并退出。
        if (node.adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", node.path.c_str());

            // 释放之前已经成功加载的 LoRA adapter。
            for (auto & old : lora_nodes) {
                if (old.adapter != nullptr) {
                    llama_adapter_lora_free(old.adapter);
                }
            }

            // 释放模型。
            llama_model_free(model);

            // 返回错误码。
            return 1;
        }
    }

    // 根据 generated group 配置构造运行时 group，并 tokenize 每个 group 的 prompt pattern。
    std::vector<lora_group_runtime> groups =
            make_runtime_groups(vocab);

    // 每个基础 case 重复 3 次，用于后续统计平均值。
    const int n_repeats = 3;

    // 构造长上下文请求列表，例如 ctx_50 / ctx_100 下的 write / optimize / explain。
    std::vector<request_item> requests =
            make_repeated_long_context_requests(n_repeats);

    // 保存 tokenize 后的请求。
    std::vector<request_tokens> tokenized;

    // 遍历所有请求，逐条 tokenize 并路由到 group。
    for (const auto & req : requests) {
        // 创建当前请求的 token 结构。
        request_tokens toks;

        // 把 prompt 文本转成 token 序列，保存到 toks.full。
        tokenize_text(vocab, req.prompt, toks.full);

        // 记录路由开始时间。
        const double route0 = now_ms();

        // 根据 prompt token 和 group pattern，选择最匹配的 group。
        toks.routed_group_id =
                route_prompt_to_group(groups, toks.full);

        // 记录路由结束时间。
        const double route1 = now_ms();

        // 目前 route0 / route1 没有写入结果，只是保留变量避免 warning。
        (void) route0;
        (void) route1;

        // 保存当前 tokenize 后的请求。
        tokenized.push_back(toks);
    }

    // 打印请求路由结果。
    fprintf(stderr, "request routing result:\n");

    // 遍历所有请求，输出它被分到哪个 group。
    for (int i = 0; i < (int) requests.size(); i++) {
        // 根据 routed_group_id 找到 group。
        const lora_group_runtime * group =
                find_group(groups, tokenized[i].routed_group_id);

        // 打印 request -> group 的映射。
        fprintf(stderr,
                "  request %d -> group %s\n",
                i,
                group ? group->group_name.c_str() : "unknown");
    }

    // 保存所有实验结果，包括 baseline、online prefix reuse、suffix delta materialize。
    std::vector<sample_result> results;

    // 构造 KV delta probe 的 pair cases。
    std::vector<kv_delta_probe_case> probe_cases =
            make_kv_delta_probe_cases();

    // 独立创建 context 并执行 KV delta probe。
    std::vector<kv_delta_probe_result> probe_results =
            run_kv_delta_probe_suite(
                    model,
                    vocab,
                    n_ctx,
                    probe_cases);

    // 保存 KV delta probe 结果。
    save_kv_delta_probe_results(probe_results);

    // baseline 实验作用域：使用独立 context，避免污染 online prefix 实验。
    {
        // 创建 baseline context 参数。
        llama_context_params ctx_params =
                llama_context_default_params();

        // 设置上下文长度。
        ctx_params.n_ctx = n_ctx;

        // 设置 batch 大小。
        ctx_params.n_batch = 256;

        // 设置 micro-batch 大小。
        ctx_params.n_ubatch = 64;

        // baseline 中每个 request 使用一个 seq，所以 n_seq_max 设置成 requests.size()。
        ctx_params.n_seq_max = (uint32_t) requests.size();

        // 关闭性能统计。
        ctx_params.no_perf = true;

        // 使用 unified KV buffer。
        ctx_params.kv_unified = true;

        // 创建 baseline context。
        llama_context * ctx =
                llama_init_from_model(model, ctx_params);

        // 如果创建失败，打印错误并退出。
        if (ctx == nullptr) {
            fprintf(stderr, "failed to create baseline context\n");
            return 1;
        }

        // 遍历所有请求，逐条完整 prefill，不做 KV 复用。
        for (int i = 0; i < (int) requests.size(); i++) {
            // 运行 baseline 请求。
            sample_result r =
                    run_baseline_request(
                            ctx,
                            lora_nodes,
                            groups,
                            requests[i],
                            tokenized[i],
                            i,
                            n_predict);

            // 保存 baseline 结果。
            results.push_back(r);
        }

        // 清空当前 context 的 LoRA 绑定。
        clear_lora(ctx);

        // 释放 baseline context。
        llama_free(ctx);
    }

    // online prefix tree 实验作用域。
    {
        // 保存在线 prefix tree 节点。
        std::vector<online_prefix_node> online_nodes;

        // 最少复用 prefix token 数，小于这个阈值就不走 prefix reuse。
        const int min_reuse_prefix_tokens = 4;

        // prefix cache seq 从 0 开始分配。
        const int cache_seq_base = 0;

        // request seq 从 128 开始，避免和 cache seq 冲突。
        const int request_seq_base = 128;

        // delta materialize 目标 seq 从 request_seq_base + requests.size() 开始。
        const int delta_request_seq_base = request_seq_base + (int) requests.size();

        // 创建 online prefix context 参数。
        llama_context_params ctx_params =
                llama_context_default_params();

        // 设置上下文长度。
        ctx_params.n_ctx = n_ctx;

        // 设置 batch 大小。
        ctx_params.n_batch = 256;

        // 设置 micro-batch 大小。
        ctx_params.n_ubatch = 64;

        // 设置最大 seq 数量，要容纳 cache seq、request seq、delta seq。
        ctx_params.n_seq_max =
                (uint32_t) (request_seq_base + (int) requests.size() * 3 + 32);

        // 关闭性能统计。
        ctx_params.no_perf = true;

        // 使用 unified KV buffer，保证 seq_cp / delta materialize 能工作。
        ctx_params.kv_unified = true;

        // 创建 online prefix context。
        llama_context * ctx =
                llama_init_from_model(model, ctx_params);

        // 如果 context 创建失败，退出。
        if (ctx == nullptr) {
            fprintf(stderr, "failed to create online-prefix context\n");
            return 1;
        }

        // 获取 llama.cpp 的 memory/KV 管理对象。
        llama_memory_t mem =
                llama_get_memory(ctx);

        // 估算每个 token 的 KV cache 占用，单位 MB。
        const double kv_mb_per_token =
                estimate_kv_mb_per_token(
                        model,
                        ctx_params.type_k,
                        ctx_params.type_v);

        // 打印每 token KV 内存估算值。
        fprintf(stderr,
                "estimated kv memory per token: %.6f MB\n",
                kv_mb_per_token);

        // 遍历所有请求，执行 online prefix build / reuse / delta materialize。
        for (int i = 0; i < (int) requests.size(); i++) {
            // 当前原始请求。
            const request_item & req = requests[i];

            // 当前请求的 token。
            const request_tokens & toks = tokenized[i];

            // 找到当前请求对应 group。
            const lora_group_runtime * group =
                    find_group(groups, toks.routed_group_id);

            // 如果没有路由到 group，就跳过。
            if (group == nullptr) {
                fprintf(stderr, "request %d has no routed group\n", i);
                continue;
            }

            // 在已有 online prefix nodes 中查找是否命中公共 prefix。
            online_route_result route =
                    route_online_prefix_node(
                            online_nodes,
                            toks.routed_group_id,
                            toks.full);

            // 当前请求实际使用的 seq id。
            const llama_seq_id request_seq_id =
                    (llama_seq_id) (request_seq_base + i);

            // 判断当前请求是不是该 group 的 anchor LoRA 请求。
            const bool is_anchor_request =
                    group != nullptr && req.leaf_lora_id == group->anchor_lora_id;

            // 如果不是 anchor 请求，并且命中了足够长的 prefix，就走 online prefix reuse。
            if (!is_anchor_request &&
                    route.exact_prefix_hit &&
                    route.exact_prefix_len >= min_reuse_prefix_tokens) {
                // 根据命中的 node_id 找到 prefix node。
                const online_prefix_node * hit =
                        find_online_node(online_nodes, route.node_id);

                // 如果没找到，说明 tree 状态异常，跳过。
                if (hit == nullptr) {
                    fprintf(stderr,
                            "online prefix node not found: %d\n",
                            route.node_id);
                    continue;
                }

                // 运行 prefix reuse：复制 hit 节点的 prefix KV，再重新计算 suffix。
                sample_result r =
                        run_online_prefix_reuse_request(
                                ctx,
                                mem,
                                lora_nodes,
                                groups,
                                *hit,
                                req,
                                toks,
                                request_seq_id,
                                route.exact_prefix_len,
                                n_predict,
                                kv_mb_per_token);

                // 保存 prefix reuse 结果。
                results.push_back(r);

                // 如果当前请求有 suffix，就尝试 suffix delta materialize。
                // 这里 request_seq_id 是刚刚完整算过 suffix 的 child KV。
                if (route.suffix_len > 0) {
                    // 给 delta materialize 结果分配新的目标 seq。
                    const llama_seq_id delta_request_seq_id =
                            (llama_seq_id) (request_seq_base + (int) requests.size() + i);

                    // 尝试用 anchor KV + child_full KV 构造 delta，并 materialize 到新 seq。
                    sample_result r_delta =
                            run_online_suffix_delta_materialize_request(
                                    ctx,
                                    mem,
                                    lora_nodes,
                                    groups,
                                    *hit,
                                    req,
                                    toks,
                                    request_seq_id,
                                    delta_request_seq_id,
                                    route.exact_prefix_len,
                                    n_predict,
                                    kv_mb_per_token);

                    // 保存 delta materialize 结果。
                    results.push_back(r_delta);

                    // 打印 suffix delta materialize 的详细统计。
                    fprintf(stderr,
                            "suffix delta materialize: request=%d anchor_node=%d child_full_seq=%d dst_seq=%d ok=%d delta_ms=%.4f suffix_saved_ms=%.4f delta_fp32_mb=%.4f materialized_kv_mb=%.4f\n",
                            i,
                            hit->node_id,
                            request_seq_id,
                            delta_request_seq_id,
                            r_delta.suffix_delta_materialized,
                            r_delta.delta_materialize_ms,
                            r.suffix_ms,
                            r_delta.delta_fp32_mb,
                            r_delta.materialized_kv_mb);
                }

                // 基础请求数量为 6：ctx_50 三条 + ctx_100 三条。
                const int base_request_count = 6;

                // 多 repeat 时，把请求 index 归一化回 0~5，方便匹配 probe case。
                const int normalized_request_index = i % base_request_count;

                // 根据当前 child request index 找到对应 probe 结果。
                const kv_delta_probe_result * probe =
                        find_probe_for_child_request(
                                probe_results,
                                probe_cases,
                                normalized_request_index);

                // 遍历 online nodes，更新命中的 node 状态。
                for (auto & node : online_nodes) {
                    // 只更新当前命中的 node。
                    if (node.node_id == route.node_id) {
                        // 命中次数 +1。
                        node.hit_count++;

                        // 标记 prefix 可以复用。
                        node.prefix_can_reuse = 1;

                        // 当前不是新分支，而是复用已有 prefix。
                        node.should_open_new_branch = 0;

                        // 如果 probe 成功，就把 probe 的 cosine / L2 / 决策写入 node。
                        if (probe != nullptr && probe->probe_status == "ok") {
                            apply_probe_to_online_node(
                                    node,
                                    *probe,
                                    route.node_id);
                        } else {
                            // 如果没有 probe 或 probe 失败，就保守设置为 suffix 重新计算。
                            node.suffix_can_delta = 0;
                            node.delta_parent_node_id = -1;
                            node.reuse_decision = "exact_prefix_reuse_suffix_recompute";
                        }

                        // 找到目标 node 后退出循环。
                        break;
                    }
                }

                // 打印 online prefix 命中情况。
                fprintf(stderr,
                        "online prefix hit: request=%d group=%s node=%d prefix=%d suffix=%d saved_kv=%.4f MB\n",
                        i,
                        group->group_name.c_str(),
                        route.node_id,
                        route.exact_prefix_len,
                        route.suffix_len,
                        r.estimated_saved_kv_mb);

                // 当前请求已经处理完，进入下一个请求。
                continue;
            }

            // 如果没有命中可复用 prefix，或者当前请求是 anchor 请求，就创建新的 prefix node。
            const int node_id = (int) online_nodes.size();

            // 为这个新 prefix node 分配 cache seq。
            const llama_seq_id cache_seq_id =
                    (llama_seq_id) (cache_seq_base + node_id);

            // 运行 online build：完整 eval 当前 prompt，把 KV 写入 cache_seq_id。
            sample_result r =
                    run_online_build_prefix_request(
                            ctx,
                            mem,
                            lora_nodes,
                            groups,
                            req,
                            toks,
                            cache_seq_id,
                            request_seq_id,
                            node_id,
                            n_predict);

            // 保存 build prefix 结果。
            results.push_back(r);

            // 创建新的 online prefix node。
            online_prefix_node node;

            // 节点 id。
            node.node_id = node_id;

            // 节点所属 group。
            node.group_id = toks.routed_group_id;

            // group 名称。
            node.group_name = group->group_name;

            // 当前节点的 KV 保存在哪个 cache seq。
            node.cache_seq_id = cache_seq_id;

            // 当前节点保存的 prefix 长度，这里是完整 prompt 长度。
            node.prefix_len = (int) toks.full.size();

            // 保存当前 prompt token，后续用于 common prefix 匹配。
            node.prefix_tokens = toks.full;

            // 初始命中次数为 0。
            node.hit_count = 0;

            // 新节点暂时没有父节点。
            node.parent_node_id = -1;

            // 新节点暂时没有 delta parent。
            node.delta_parent_node_id = -1;

            // 新节点还没有被 probe 判断为 prefix 可复用。
            node.prefix_can_reuse = 0;

            // 新节点还没有 suffix delta。
            node.suffix_can_delta = 0;

            // 新节点默认表示需要开新分支。
            node.should_open_new_branch = 1;

            // suffix cosine 初始化为 0。
            node.suffix_kv_cos = 0.0;

            // suffix L2 初始化为 0。
            node.suffix_kv_l2 = 0.0;

            // 节点决策标签为创建 prefix node。
            node.reuse_decision = "create_prefix_node";

            // 把新节点加入 online prefix tree。
            online_nodes.push_back(node);

            // 打印新节点创建信息。
            fprintf(stderr,
                    "online prefix node created: request=%d group=%s node=%d prefix_len=%d\n",
                    i,
                    group->group_name.c_str(),
                    node.node_id,
                    node.prefix_len);
        }

        // 打印 online prefix tree 总结。
        fprintf(stderr, "\nonline prefix tree summary:\n");

        // 遍历所有 online nodes，输出节点状态。
        for (const auto & node : online_nodes) {
            fprintf(stderr,
                    "  node=%d group=%s parent=%d delta_parent=%d prefix_len=%d hit_count=%d decision=%s suffix_cos=%.6f suffix_l2=%.6f\n",
                    node.node_id,
                    node.group_name.c_str(),
                    node.parent_node_id,
                    node.delta_parent_node_id,
                    node.prefix_len,
                    node.hit_count,
                    node.reuse_decision.c_str(),
                    node.suffix_kv_cos,
                    node.suffix_kv_l2);
        }

        // 保存 online prefix tree 到 CSV。
        save_online_prefix_delta_tree(online_nodes);

        // 清空 LoRA 绑定。
        clear_lora(ctx);

        // 释放 online prefix context。
        llama_free(ctx);
    }

    // 保存所有实验结果到 CSV。
    save_results(results);

    // 下面这大段是旧的手写 probe 实验代码，现在已经注释掉。
    // 它原来用于手动构造 same_prompt / code / mobile_like / cross_task 等 probe pair。
    // 当前主流程已经使用 make_kv_delta_probe_cases() 自动生成 probe cases。

    // 释放所有 LoRA adapter。
    for (auto & node : lora_nodes) {
        // 如果 adapter 已加载，就释放。
        if (node.adapter != nullptr) {
            llama_adapter_lora_free(node.adapter);
        }
    }

    // 释放 base model。
    llama_model_free(model);

    // 打印实验完成信息。
    fprintf(stderr, "group node KV reuse + LoRA tree experiment finished.\n");

    // 正常退出。
    return 0;
}


// int main(int argc, char ** argv) {
//     // ------------------------------------------------------------
//     // 1. 初始化 llama.cpp backend
//     // ------------------------------------------------------------

//     // llama_backend_init() 会初始化 llama.cpp 的后端环境。
//     // 如果启用了 CUDA，这里会准备 GPU backend。
//     llama_backend_init();

//     // llama_numa_init() 用于 NUMA 机器的内存优化。
//     // 普通单机实验里用 GGML_NUMA_STRATEGY_DISABLED 即可。
//     llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

//     // ------------------------------------------------------------
//     // 2. 设置路径参数
//     // ------------------------------------------------------------

//     // base model 路径。
//     // 这里建议使用纯 Transformer 的 Qwen2.5-1.5B GGUF。
//     // 注意：这是 base model，不是 LoRA。
//     const std::string model_path =
//         "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/Qwen2.5-1.5B-Instruct-f16.gguf";

//     // LoRA 根目录。
//     // 后面每个 adapter 都从这里拼路径。
//     const std::string lora_root =
//         "D:/ecnu_experiment/Model/LoRA/Qwen2.5_1.5B_gguf";

//     // 输出目录。
//     // 所有 CSV 都保存到这里，Python 脚本会读取这个目录。
//     const std::string output_dir =
//         "D:/ecnu_experiment/LLama.cpp/llama.cpp/examples/lora-base-test/output";

//     // ------------------------------------------------------------
//     // 3. 设置模型运行参数
//     // ------------------------------------------------------------

//     // llama_model_params 是加载模型时使用的参数。
//     llama_model_params model_params = llama_model_default_params();

//     // n_gpu_layers 表示有多少层放到 GPU 上。
//     // 999 通常表示尽量全放 GPU。
//     model_params.n_gpu_layers = 999;

//     // llama_context_params 是创建推理上下文时使用的参数。
//     llama_context_params ctx_params = llama_context_default_params();

//     // n_ctx 是 KV cache 的最大 token 容量。
//     // 长上下文实验需要设大一些。
//     ctx_params.n_ctx = 4096;

//     // n_batch 是一次 decode 可以处理的 token 数。
//     // 太大可能显存压力大，太小吞吐低。
//     ctx_params.n_batch = 512;

//     // n_ubatch 是底层 micro-batch 大小。
//     // CUDA 下过大可能引发显存或 graph reserve 压力。
//     ctx_params.n_ubatch = 64;

//     // 多 seq KV copy / seq_cp 路径需要 full KV buffer。
//     // 如果这里不是 true，之前会遇到：
//     // GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers") failed
//     ctx_params.kv_unified = true;

//     // 是否启用 flash attention。
//     // Qwen2.5 + CUDA 通常可以打开。
//     ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

//     // ------------------------------------------------------------
//     // 4. 加载 base model
//     // ------------------------------------------------------------

//     // 从 GGUF 文件加载模型。
//     llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

//     // 如果模型加载失败，直接退出。
//     if (model == nullptr) {
//         fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
//         llama_backend_free();
//         return 1;
//     }

//     // 从模型里拿 vocab。
//     // tokenize 时需要 vocab。
//     const llama_vocab * vocab = llama_model_get_vocab(model);

//     // ------------------------------------------------------------
//     // 5. 创建主 context
//     // ------------------------------------------------------------

//     // llama_context 是一次推理会话的上下文。
//     // KV cache 也挂在 context 里面。
//     llama_context * ctx = llama_init_from_model(model, ctx_params);

//     // 如果 context 创建失败，释放 model 后退出。
//     if (ctx == nullptr) {
//         fprintf(stderr, "failed to create llama context\n");
//         llama_model_free(model);
//         llama_backend_free();
//         return 1;
//     }

//     // ------------------------------------------------------------
//     // 6. 构造 LoRA 节点配置
//     // ------------------------------------------------------------

//     // runtime_loras 保存所有 LoRA adapter 的运行时信息。
//     // 每个节点包括：
//     // 1. LoRA 名字。
//     // 2. LoRA 文件路径。
//     // 3. llama_adapter_lora 指针。
//     // 4. 它属于哪个 group。
//     std::vector<lora_node_runtime> runtime_loras = make_runtime_lora_nodes();

//     // ------------------------------------------------------------
//     // 7. 加载 LoRA adapter
//     // ------------------------------------------------------------

//     for (auto & node : runtime_loras) {
//         // llama_adapter_lora_init() 会把 LoRA adapter 从 GGUF 文件加载进来。
//         node.adapter = llama_adapter_lora_init(model, node.path.c_str());

//         // 如果某个 LoRA 加载失败，记录错误。
//         if (node.adapter == nullptr) {
//             fprintf(stderr, "failed to load lora adapter: %s\n", node.path.c_str());
//             continue;
//         }

//         // 打印加载成功的信息，方便 debug。
//         fprintf(stderr, "loaded lora: name=%s path=%s\n",
//                 node.name.c_str(),
//                 node.path.c_str());
//     }

//     // ------------------------------------------------------------
//     // 8. 构造 LoRA group
//     // ------------------------------------------------------------

//     // group 表示一类任务或一组相似 LoRA。
//     // 例如：
//     // code group 里面可以有 code_r16 / code_r16v2 / code_r16v3。
//     // 后面 prefix tree 会按 group 做 anchor。
//     std::vector<lora_group_runtime> groups = make_runtime_groups(vocab);

//     // ------------------------------------------------------------
//     // 9. 构造长上下文实验请求
//     // ------------------------------------------------------------

//     // requests 是本次实验的所有请求。
//     // 每个 request 包含：
//     // 1. prompt 文本。
//     // 2. 使用哪个 LoRA。
//     // 3. 属于哪个 group。
//     // 4. context_tag，例如 ctx_50 / ctx_100。
//     // 5. repeat_id，用于多次重复实验取平均。
//     const int n_repeats = 3;

//     std::vector<request_item> requests =
//             make_repeated_long_context_requests(n_repeats);

//     // ------------------------------------------------------------
//     // 10. tokenize 所有请求
//     // ------------------------------------------------------------

//     std::vector<request_tokens> tokenized_requests;

//     // 预留空间，避免 vector 反复扩容。
//     tokenized_requests.reserve(requests.size());

//     for (int i = 0; i < (int) requests.size(); ++i) {
//         const request_item & req = requests[i];

//         request_tokens rt;
//         tokenize_text(vocab, req.prompt, rt.full);
//         rt.routed_group_id = route_prompt_to_group(groups, rt.full);

//         tokenized_requests.push_back(std::move(rt));
//     }

//     // ------------------------------------------------------------
//     // 11. 先跑 KV delta probe
//     // ------------------------------------------------------------

//     // probe 的目的：
//     // 比较两个相似请求 A/B 的 KV 是否接近。
//     //
//     // 例如：
//     // A = 长文章 + "请总结"
//     // B = 长文章 + "请优化"
//     //
//     // 如果公共 prefix 的 KV 一致，suffix 的 KV 差异也比较小，
//     // 后面才考虑保存 delta = KV_B - KV_A。
//     std::vector<kv_delta_probe_case> probe_cases =
//         make_kv_delta_probe_cases();

//     // 执行 probe。
//     // 这里会真正 eval 两个 prompt，并读取 KV cache 做 cosine / L2。
//     std::vector<kv_delta_probe_result> probe_results =
//             run_kv_delta_probe_suite(
//                     model,
//                     vocab,
//                     (int) ctx_params.n_ctx,
//                     probe_cases);

//     // 保存 probe 结果。
//     // Python 的 plot_kv_delta_probe.py 会读取这个 CSV。
//     save_kv_delta_probe_results(probe_results);

//     // ------------------------------------------------------------
//     // 12. 准备保存最终实验结果
//     // ------------------------------------------------------------

//     // sample_results 保存 baseline / online prefix / suffix delta 的计时结果。
//     std::vector<sample_result> sample_results;

//     // 预留空间。
//     sample_results.reserve(requests.size() * 4);

//     // estimated_kv_mb_per_token 用于估算每复用一个 token 可以节省多少 KV 显存。
//     const double kv_mb_per_token =
//         estimate_kv_mb_per_token(model, GGML_TYPE_F16, GGML_TYPE_F16);

//     // ------------------------------------------------------------
//     // 13. 跑 baseline：完全不复用 KV
//     // ------------------------------------------------------------

//     // baseline 是对照组。
//     // 每个请求都从 0 开始完整 prefill prompt。
//     // 不复制 prefix KV，也不使用 suffix delta。
//     const int n_predict = 32;

//     for (int i = 0; i < (int) tokenized_requests.size(); ++i) {
//         const request_item & req = requests[i];
//         const request_tokens & toks = tokenized_requests[i];

//         sample_result r = run_baseline_request(
//                 ctx,
//                 runtime_loras,
//                 groups,
//                 req,
//                 toks,
//                 i,
//                 n_predict);

//         sample_results.push_back(std::move(r));
//     }

//     // ------------------------------------------------------------
//     // 14. 清理 KV cache，准备跑 online prefix tree
//     // ------------------------------------------------------------

//     // baseline 跑完后，KV cache 里已经有很多 seq。
//     // 为了避免污染 online 实验，这里清掉全部 KV。
//     llama_memory_clear(llama_get_memory(ctx), true);

//     // 清空 LoRA。
//     clear_lora(ctx);

//     // ------------------------------------------------------------
//     // 15. 初始化 online prefix tree
//     // ------------------------------------------------------------

//     // online_nodes 是在线 prefix tree 的节点列表。
//     //
//     // 每个 node 表示一个已经缓存过的 prefix。
//     // 新请求进来后，会去找有没有公共 prefix 节点可以复用。
//     std::vector<online_prefix_node> online_nodes;

//     // next_seq_id 用于给每个 KV 分支分配 seq。
//     //
//     // seq 可以理解成 KV cache 里的“请求编号”或“分支编号”。
//     // 例如：
//     // seq=0 保存 anchor prompt 的 KV。
//     // seq=1 保存另一个 LoRA 请求的 KV。
//     llama_seq_id next_seq_id = 0;

//     // ------------------------------------------------------------
//     // 16. 跑 online prefix tree 主流程
//     // ------------------------------------------------------------

//     for (int i = 0; i < (int) tokenized_requests.size(); ++i) {
//         // 当前请求。
//         const request_tokens & rt = tokenized_requests[i];

//         // 当前请求配置。
//         const request_item & req = requests[i];

//         // --------------------------------------------------------
//         // 16.1 查找当前请求是否能命中已有 prefix node
//         // --------------------------------------------------------

//         // route_online_prefix_node() 会做几件事：
//         // 1. 在 online_nodes 里面找同 group 的节点。
//         // 2. 计算当前请求和节点 prefix 的公共前缀长度。
//         // 3. 如果公共前缀足够长，就认为 prefix hit。
//         // 4. 返回命中的 node id、prefix_len、suffix_len。
//         route_online_prefix_node(online_nodes, rt.routed_group_id, rt.full);

//         // --------------------------------------------------------
//         // 16.2 如果没有命中 prefix node，就创建新节点
//         // --------------------------------------------------------

//         if (!route.exact_prefix_hit) {
//             // 没命中说明当前请求没有可复用的 prefix。
//             // 所以需要完整 prefill 当前 prompt，并把它作为新的 anchor node。
//             sample_result r = run_online_build_prefix_request(
//                 ctx,
//                 rt,
//                 runtime_loras,
//                 next_seq_id,
//                 kv_mb_per_token);

//             // 保存当前实验结果。
//             sample_results.push_back(r);

//             // 构造新的 online prefix node。
//             online_prefix_node node;

//             // node_id 是节点编号。
//             node.node_id = (int) online_nodes.size();

//             // group_name 表示这个节点属于哪个任务组。
//             node.group_name = req.group_name;

//             // parent_node_id 表示 prefix tree 的父节点。
//             // 当前简化版本没有真正做多层树，所以先设为 -1。
//             node.parent_node_id = -1;

//             // delta_parent_node_id 表示如果后面 suffix delta 成功，
//             // 当前节点的 delta 是基于哪个 anchor 节点算出来的。
//             // 新建节点没有 delta parent。
//             node.delta_parent_node_id = -1;

//             // prefix_len 表示这个节点缓存了多少 prompt token。
//             node.prefix_len = (int) rt.full.size();

//             // prefix_tokens 保存这个节点对应的 token。
//             node.prefix_tokens = rt.full;

//             // seq_id 表示这个节点的 KV 存在哪个 seq 里。
//             node.cache_seq_id = r.seq_id;

//             // hit_count 表示后面有多少请求命中过这个节点。
//             node.hit_count = 0;

//             // prefix_can_reuse 表示这个节点能否被后续请求复用。
//             node.prefix_can_reuse = true;

//             // suffix_can_delta 表示这个节点是否有可用 suffix delta。
//             // 新建节点还没有 delta。
//             node.suffix_can_delta = false;

//             // suffix_delta_materialized 表示是否已经把 delta 恢复成真实 KV。
//             node.suffix_delta_materialized = false;

//             // 决策标签。
//             node.node.reuse_decision = "create_prefix_node";

//             // 保存节点。
//             online_nodes.push_back(std::move(node));

//             // 继续处理下一个请求。
//             continue;
//         }

//         // --------------------------------------------------------
//         // 16.3 如果命中 prefix node，先走 prefix KV 复用
//         // --------------------------------------------------------

//         // 找到被命中的 anchor node。
//         online_prefix_node * anchor_node =
//             find_online_node(online_nodes, route.node_id);

//         // 如果理论上命中了，但节点找不到，说明代码状态异常。
//         if (anchor_node == nullptr) {
//             fprintf(stderr, "route hit but anchor node not found: node_id=%d\n",
//                     route.node_id);
//             continue;
//         }

//         // 命中次数 +1。
//         anchor_node->hit_count += 1;

//         // 跑 online prefix reuse。
//         //
//         // 这一步主要做：
//         // 1. 把 anchor node 的 prefix KV copy 到新的 seq。
//         // 2. 对 suffix token 重新 eval。
//         // 3. 统计 TTFT / suffix eval 耗时。
//         sample_result reuse_result = run_online_prefix_reuse_request(
//             ctx,
//             rt,
//             runtime_loras,
//             *anchor_node,
//             route,
//             next_seq_id,
//             kv_mb_per_token);

//         // 保存 prefix reuse 结果。
//         sample_results.push_back(reuse_result);

//         // --------------------------------------------------------
//         // 16.4 把 probe 结果映射到 online node
//         // --------------------------------------------------------

//         // 根据当前请求找到对应的 probe 结果。
//         //
//         // 例如：
//         // 当前请求是 ctx_100 optimize，
//         // anchor 是 ctx_100 write，
//         // 那么这里会找 ctx_100_write_vs_optimize 的 probe 结果。
//         const kv_delta_probe_result * probe =
//             find_probe_for_child_request(probe_results, i);

//         // 如果找到了 probe 结果，就把它写回 node。
//         if (probe != nullptr) {
//             apply_probe_to_online_node(*anchor_node, *probe);
//         }

//         // --------------------------------------------------------
//         // 16.5 如果 probe 判断 suffix delta 可用，就尝试 materialize
//         // --------------------------------------------------------

//         // suffix_can_delta 表示 probe 指标满足阈值。
//         // 但它只说明“理论上可以保存 delta”，不代表 materialize 一定成功。
//         if (anchor_node->suffix_can_delta) {
//             // 尝试 suffix delta materialize。
//             //
//             // 这一步实验性比较强：
//             // 1. 根据 anchor KV 和 delta 恢复 child KV。
//             // 2. 把恢复出来的 KV 写入新的 seq。
//             // 3. 如果成功，可以避免重新 eval suffix。
//             sample_result delta_result =
//                 run_online_suffix_delta_materialize_request(
//                     ctx,
//                     rt,
//                     runtime_loras,
//                     *anchor_node,
//                     route,
//                     next_seq_id,
//                     kv_mb_per_token);

//             // 保存 suffix delta 结果。
//             sample_results.push_back(delta_result);

//             // 如果 materialize 成功，更新 node 状态。
//             if (delta_result.suffix_delta_materialized) {
//                 anchor_node->suffix_delta_materialized = true;
//                 anchor_node->delta_parent_node_id = anchor_node->node_id;
//                 anchor_node->decision_label = "prefix_reuse_suffix_delta_materialized";
//             } else {
//                 // 如果失败，保留 prefix reuse + suffix recompute 路径。
//                 anchor_node->decision_label = "prefix_reuse_suffix_recompute";
//             }
//         } else {
//             // 如果 probe 不支持 suffix delta，就只做 prefix reuse。
//             anchor_node->decision_label = "prefix_reuse_suffix_recompute";
//         }
//     }

//     // ------------------------------------------------------------
//     // 17. 保存 online prefix tree 结构
//     // ------------------------------------------------------------

//     // 保存 online tree 节点信息。
//     // Python 的 plot_online_prefix_delta_tree.py 会读取这个 CSV。
//     save_online_prefix_delta_tree(
//         output_dir + "/online_prefix_delta_tree_summary.csv",
//         online_nodes);

//     // ------------------------------------------------------------
//     // 18. 保存 sample_results
//     // ------------------------------------------------------------

//     // group_node_kv_lora_tree_summary.csv 是主要实验结果。
//     // Python 的 plot_online_prefix_reuse.py 会读取它画：
//     // 1. TTFT
//     // 2. 总耗时
//     // 3. TPS
//     // 4. 估算 KV 节省
//     save_results(
//         output_dir + "/group_node_kv_lora_tree_summary.csv",
//         sample_results);

//     // ------------------------------------------------------------
//     // 19. 打印 online prefix tree 简要信息
//     // ------------------------------------------------------------

//     fprintf(stderr, "\nonline prefix tree summary:\n");

//     for (const auto & node : online_nodes) {
//         fprintf(stderr,
//                 "  node=%d group=%s parent=%d delta_parent=%d "
//                 "prefix_len=%d hit_count=%d decision=%s "
//                 "suffix_cos=%.6f suffix_l2=%.6f\n",
//                 node.node_id,
//                 node.group_name.c_str(),
//                 node.parent_node_id,
//                 node.delta_parent_node_id,
//                 node.prefix_len,
//                 node.hit_count,
//                 node.decision_label.c_str(),
//                 node.suffix_kv_cos,
//                 node.suffix_kv_l2);
//     }

//     // ------------------------------------------------------------
//     // 20. 清理 LoRA adapter
//     // ------------------------------------------------------------

//     // 先从 context 里清空当前绑定的 LoRA。
//     clear_lora(ctx);

//     // 再释放每个 adapter。
//     for (auto & node : runtime_loras) {
//         if (node.adapter != nullptr) {
//             llama_adapter_lora_free(node.adapter);
//             node.adapter = nullptr;
//         }
//     }

//     // ------------------------------------------------------------
//     // 21. 清理 context 和 model
//     // ------------------------------------------------------------

//     // 释放 llama context。
//     llama_free(ctx);

//     // 释放 model。
//     llama_model_free(model);

//     // 释放 llama backend。
//     llama_backend_free();

//     // ------------------------------------------------------------
//     // 22. 正常结束
//     // ------------------------------------------------------------

//     fprintf(stderr, "group node KV reuse + LoRA tree experiment finished.\n");

//     return 0;
// }