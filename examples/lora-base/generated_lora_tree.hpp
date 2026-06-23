#pragma once

#include <string>
#include <vector>

struct generated_lora_spec {
    int id = 0;
    int parent_id = -1;
    std::string name;
    std::string short_name;
    std::string path;
    std::string group_name;
    bool is_anchor = false;
};

struct generated_prompt_pattern {
    std::string text;
};

struct generated_lora_group {
    int group_id = 0;
    std::string group_name;
    int anchor_lora_id = 0;
    std::vector<int> lora_ids;
    std::vector<generated_prompt_pattern> prompt_patterns;
};

static std::vector<generated_lora_spec> make_generated_lora_specs() {
    return {
        {
            0,
            -1,
            "subliminal_monkey",
            "monkey",
            "D:/ecnu_experiment/Model/subliminal-monkey.gguf",
            "animal",
            true,
        },
        {
            1,
            0,
            "subliminal_tiger",
            "tiger",
            "D:/ecnu_experiment/Model/subliminal-qwen35-4b-tiger.gguf",
            "animal",
            false,
        },
        {
            2,
            0,
            "subliminal_wolf",
            "wolf",
            "D:/ecnu_experiment/Model/subliminal-qwen35-4b-wolf.gguf",
            "animal",
            false,
        },
        {
            3,
            -1,
            "marketing_adapter",
            "marketing",
            "D:/ecnu_experiment/Model/qwen35-marketing-adapter.gguf",
            "marketing",
            true,
        },
    };
}

static std::vector<generated_lora_group> make_generated_lora_groups() {
    return {
        {
            0,
            "animal",
            0,
            std::vector<int>{ 0, 1, 2 },
            std::vector<generated_prompt_pattern>{
                { "You are an animal introduction expert. Please introduce the characteristics of monkey." },
                { "You are an animal introduction expert. Please introduce the characteristics of tiger." },
                { "You are an animal introduction expert. Please introduce the characteristics of wolf." },
            },
        },
        {
            1,
            "marketing",
            3,
            std::vector<int>{ 3 },
            std::vector<generated_prompt_pattern>{
                { "You are a marketing copywriting expert. Please write a short product slogan." },
                { "You are a marketing copywriting expert. Please write a product advertisement." },
            },
        },
    };
}
