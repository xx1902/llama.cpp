#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// 打印命令行使用方式
static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C"); // 设置数字格式，避免不同地区设置影响浮点数解析
    // 基座模型 GGUF 文件路径
    std::string model_path = "D:/ecnu_experiment/Model/Qwen3.5-4B-BF16.gguf";

    // LoRA 适配器 GGUF 文件路径
    // std::string lora_path = "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf";
    std::vector<std::string> lora_paths = {
        "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
        "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
        "D:/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
        "D:/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
    };
    float lora_scale = 1.0f; // LoRA 缩放系数，1.0 表示使用默认适配强度

    // 输入提示词
    std::string prompt = "Hello my name is";
    int ngl = 99; // GPU 卸载层数，99 表示尽可能将模型层放到 GPU 上
    int n_predict = 32; // 生成 token 数量

    // 命令行传入模型路径
    // {
    //     int i = 1;
    //     for (; i < argc; i++) {
    //         if (strcmp(argv[i], "-m") == 0) {
    //             if (i + 1 < argc) {
    //                 model_path = argv[++i];
    //             } else {
    //                 print_usage(argc, argv);
    //                 return 1;
    //             }
    //         } else if (strcmp(argv[i], "-n") == 0) {
    //             if (i + 1 < argc) {
    //                 try {
    //                     n_predict = std::stoi(argv[++i]);
    //                 } catch (...) {
    //                     print_usage(argc, argv);
    //                     return 1;
    //                 }
    //             } else {
    //                 print_usage(argc, argv);
    //                 return 1;
    //             }
    //         } else if (strcmp(argv[i], "-ngl") == 0) {
    //             if (i + 1 < argc) {
    //                 try {
    //                     ngl = std::stoi(argv[++i]);
    //                 } catch (...) {
    //                     print_usage(argc, argv);
    //                     return 1;
    //                 }
    //             } else {
    //                 print_usage(argc, argv);
    //                 return 1;
    //             }
    //         } else {
    //             // prompt starts here
    //             break;
    //         }
    //     }
    //     if (model_path.empty()) {
    //         print_usage(argc, argv);
    //         return 1;
    //     }
    //     if (i < argc) {
    //         prompt = argv[i++];
    //         for (; i < argc; i++) {
    //             prompt += " ";
    //             prompt += argv[i];
    //         }
    //     }
    // }


    // 加载 llama.cpp 支持的后端，例如 CUDA、CPU 等
    ggml_backend_load_all();

    // 初始化模型加载参数
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;  // 设置 GPU 卸载层数
    // 加载基座模型
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // ==============  加载 LoRA 适配器  ==============
    // LoRA adapter 指针
    // llama_adapter_lora * lora_adapter = nullptr;

    // 如果 LoRA 路径不为空，则加载 LoRA 适配器
    // llama_adapter_lora_init 会读取 LoRA GGUF 文件，并将 LoRA 权重挂到基座模型上
    // if (!lora_path.empty()) {
    //     fprintf(stderr, "\n========== LoRA CHECK ==========\n");
    //     fprintf(stderr, "LoRA path: %s\n", lora_path.c_str());
    //     fprintf(stderr, "%s: loading LoRA adapter: %s\n", __func__, lora_path.c_str());
        
    //     const auto t_lora_load_start = ggml_time_us();
    //     lora_adapter = llama_adapter_lora_init(model, lora_path.c_str());
    //     const auto t_lora_load_end = ggml_time_us();

    //     fprintf(stderr, "[LoRA] load time = %.2f ms\n", (t_lora_load_end - t_lora_load_start) / 1000.0);

    //     if (lora_adapter == nullptr) {
    //         fprintf(stderr, "%s: error: unable to load LoRA adapter\n", __func__);
    //         return 1;
    //     }
    // }

    // ==============  加载多 LoRA 适配器  ==============
    // LoRA adapter 指针
    std::vector<llama_adapter_lora *> lora_adapters;

    for (const auto & path : lora_paths) {

        fprintf(stderr, "\n========== LoRA loaded ==========\n");
        
        const auto t0 = ggml_time_us();
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, path.c_str());
        const auto t1 = ggml_time_us();

        if (adapter == nullptr) {
            fprintf(stderr, "failed to load LoRA: %s\n", path.c_str());
            return 1;
        }

        fprintf(stderr, "[LoRA] loaded %s in %.3f ms\n",
                path.c_str(), (t1 - t0) / 1000.0);

        lora_adapters.push_back(adapter);
    }

    // 获取模型词表，用于分词和 token 转文本
    const llama_vocab * vocab = llama_model_get_vocab(model);


    // ==============  对 prompt 进行分词  ==============
    // 第一次调用 llama_tokenize，传入 NULL 获取 prompt 分词后的 token 数量
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // 根据 token 数量分配空间
    std::vector<llama_token> prompt_tokens(n_prompt);

    // 第二次调用 llama_tokenize，真正把 prompt 转换成 token 序列
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // ==============  初始化推理上下文 context  ==============
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict - 1;   // 上下文长度，这里设置为 prompt token 数 + 生成 token 数
    ctx_params.n_batch = n_prompt;                 // batch 大小，这里 prompt 阶段一次处理所有 prompt token
    ctx_params.no_perf = false;                    // 开启性能统计，后面可以打印 eval time、prompt eval time 等信息


    // ==============  将 LoRA 适配器绑定到当前 context  ==============
    // if (lora_adapter != nullptr) {
    if (!lora_adapters.empty()) {
        // llama.cpp 支持同时设置多个 LoRA adapter
        // 这里先只使用一个 LoRA，因此 vector 里只有一个元素
        // std::vector<llama_adapter_lora *> adapters = { lora_adapter };
        // 同时设置多个 LoRA adapter
        for (size_t active_id = 0; active_id < lora_adapters.size(); active_id++) {
            fprintf(stderr, "\n========== RUN LoRA %zu ==========\n", active_id);
            fprintf(stderr, "LoRA path: %s\n", lora_paths[active_id].c_str());

            // 每个 LoRA 单独创建一个 context，保证 KV cache 和推理状态互不影响
            llama_context * ctx = llama_init_from_model(model, ctx_params);
            if (ctx == NULL) {
                fprintf(stderr, "%s: error: failed to create context\n", __func__);
                break;
            }

            std::vector<llama_adapter_lora *> adapters = { lora_adapters[active_id] };
            // 每个 LoRA 对应一个缩放系数
            std::vector<float> scales = { lora_scale };
            
            const auto t_lora_set_start = ggml_time_us();
            // 将 LoRA adapter 设置到当前推理上下文中
            // 设置成功后，后续 llama_decode 会自动计算：基座输出 + LoRA 增量
            int ret = llama_set_adapters_lora(ctx, adapters.data(), adapters.size(), scales.data());
            const auto t_lora_set_end = ggml_time_us();
            
            fprintf(stderr, "[LoRA] set adapter time = %.2f ms\n", (t_lora_set_end - t_lora_set_start) / 1000.0);
            fprintf(stderr, "%s: LoRA adapter enabled, scale = %.3f\n", __func__, lora_scale);

            if (ret != 0) {
                fprintf(stderr, "%s: error: failed to set LoRA adapter on context\n", __func__);
                llama_free(ctx);
                continue;
            }
        
            // ==============  初始化采样器  ==============
            auto sparams = llama_sampler_chain_default_params();
            sparams.no_perf = false;
            llama_sampler * smpl = llama_sampler_chain_init(sparams);
            // 使用 greedy 采样，每一步选择概率最高的 token
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

            printf("\n[LoRA %zu output] ", active_id);


            // ==============  打印 prompt 原文  ==============
            for (auto id : prompt_tokens) {
                char buf[128];
                int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                    llama_sampler_free(smpl);
                    llama_free(ctx);
                    continue;
                }
                std::string s(buf, n);
                printf("%s", s.c_str());
            }

            // ==============  准备 prompt batch  ==============
            // 将 prompt token 封装成 llama_batch
            // 后续 llama_decode 会处理这个 batch
            llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

            // 如果是 encoder-decoder 模型，需要先执行 encoder
            // Qwen 这类自回归模型一般不会走这个分支
            if (llama_model_has_encoder(model)) {
                if (llama_encode(ctx, batch)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    llama_sampler_free(smpl);
                    llama_free(ctx);
                    continue;
                }
                // 获取 decoder 起始 token
                llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
                if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                    decoder_start_token_id = llama_vocab_bos(vocab);
                }
                // decoder 的第一个 batch
                batch = llama_batch_get_one(&decoder_start_token_id, 1);
            }

            // ==============  主推理循环  ==============
            const auto t_main_start = ggml_time_us();
            int n_decode = 0;
            llama_token new_token_id;

            // n_pos 表示当前已经处理到的位置
            // 每轮 llama_decode 处理当前 batch，然后采样一个新 token
            for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
                // 执行当前 batch 的模型前向计算
                // 如果已经设置 LoRA，这里实际执行的是：基座模型计算 + LoRA 增量计算
                if (llama_decode(ctx, batch)) {
                    fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                    return 1;
                }

                // 更新当前位置
                n_pos += batch.n_tokens;

                // 从当前 logits 中采样下一个 token
                new_token_id = llama_sampler_sample(smpl, ctx, -1);

                // 如果采样到结束 token，则停止生成
                if (llama_vocab_is_eog(vocab, new_token_id)) {
                    break;
                }
                // 将新 token 转换成文本并输出
                char buf[128];
                int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
                if (n < 0) {
                    fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                    break;
                }

                std::string s(buf, n);
                printf("%s", s.c_str());
                fflush(stdout);

                // 将本轮生成的新 token 作为下一轮输入
                batch = llama_batch_get_one(&new_token_id, 1);
                n_decode += 1;
            }

            printf("\n");

            const auto t_main_end = ggml_time_us();

            // 打印整体生成速度
            fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
                    __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

            fprintf(stderr, "\n");
            // 打印采样器性能统计
            llama_perf_sampler_print(smpl);
            // 打印 context 性能统计，包括 prompt eval time、eval time、total time 等
            llama_perf_context_print(ctx);
            fprintf(stderr, "\n");

            // ==============  释放资源  ==============
            llama_sampler_free(smpl);
            llama_free(ctx);
        }
        // if (lora_adapter) {
        //     llama_adapter_lora_free(lora_adapter);
        // }
        for (auto * adapter : lora_adapters) {
            llama_adapter_lora_free(adapter);
        }
        llama_model_free(model);
        return 0;
    }
    else{
        // 基于基座模型创建推理上下文
        llama_context * ctx = llama_init_from_model(model, ctx_params); 
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
            return 1;
        }

        // ==============  初始化采样器  ==============
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = false;
        llama_sampler * smpl = llama_sampler_chain_init(sparams);

        // 使用 greedy 采样，每一步选择概率最高的 token
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());


        // ==============  打印 prompt 原文  ==============
        for (auto id : prompt_tokens) {
            char buf[128];
            int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
        }

        // ==============  准备 prompt batch  ==============
        // 将 prompt token 封装成 llama_batch
        // 后续 llama_decode 会处理这个 batch
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        // 如果是 encoder-decoder 模型，需要先执行 encoder
        // Qwen 这类自回归模型一般不会走这个分支
        if (llama_model_has_encoder(model)) {
            if (llama_encode(ctx, batch)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
            // 获取 decoder 起始 token
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
                decoder_start_token_id = llama_vocab_bos(vocab);
            }
            // decoder 的第一个 batch
            batch = llama_batch_get_one(&decoder_start_token_id, 1);
        }

        // ==============  主推理循环  ==============
        const auto t_main_start = ggml_time_us();
        int n_decode = 0;
        llama_token new_token_id;

        // n_pos 表示当前已经处理到的位置
        // 每轮 llama_decode 处理当前 batch，然后采样一个新 token
        for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
            // 执行当前 batch 的模型前向计算
            // 如果已经设置 LoRA，这里实际执行的是：基座模型计算 + LoRA 增量计算
            if (llama_decode(ctx, batch)) {
                fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }

            // 更新当前位置
            n_pos += batch.n_tokens;

            // 从当前 logits 中采样下一个 token
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // 如果采样到结束 token，则停止生成
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }
            // 将新 token 转换成文本并输出
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }

            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // 将本轮生成的新 token 作为下一轮输入
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }

        printf("\n");

        const auto t_main_end = ggml_time_us();

        // 打印整体生成速度
        fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
                __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

        fprintf(stderr, "\n");
        // 打印采样器性能统计
        llama_perf_sampler_print(smpl);
        // 打印 context 性能统计，包括 prompt eval time、eval time、total time 等
        llama_perf_context_print(ctx);
        fprintf(stderr, "\n");

        // ==============  释放资源  ==============
        llama_sampler_free(smpl);
        llama_free(ctx);

        // if (lora_adapter) {
        //     llama_adapter_lora_free(lora_adapter);
        // }
        for (auto * adapter : lora_adapters) {
            llama_adapter_lora_free(adapter);
        }
        
        llama_model_free(model);

        return 0;
    }
}
