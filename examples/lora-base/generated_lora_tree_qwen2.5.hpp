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
            "qwen25_code_r16",
            "code_r16",
            "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/qwen25-1.5b-code-r16.gguf",
            "code",
            true,
        },
        {
            1,
            0,
            "qwen25_code_r16v2",
            "code_r16v2",
            "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/qwen25-1.5b-code-r16v2.gguf",
            "code",
            false,
        },
        {
            2,
            0,
            "qwen25_code_r16v3",
            "code_r16v3",
            "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/qwen25-1.5b-code-r16v3.gguf",
            "code",
            false,
        },
        {
            3,
            -1,
            "qwen25_chinese_correction",
            "correction",
            "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/qwen25-1.5b-chinese-correction.gguf",
            "correction",
            true,
        },
        {
            4,
            -1,
            "qwen25_song_lyrics",
            "lyrics",
            "D:/ecnu_experiment/Model/Qwen2.5-1.5B-gguf/qwen25-1.5b-song-lyrics.gguf",
            "lyrics",
            true,
        },
    };
}

static std::vector<generated_lora_group> make_generated_lora_groups() {
    return {
        {
            0,
            "code",
            0,
            std::vector<int>{ 0, 1, 2 },
            std::vector<generated_prompt_pattern>{
                { "You are a helpful coding assistant. Please write a Python function." },
                { "You are a helpful coding assistant. Please optimize this Python function." },
                { "You are a helpful coding assistant. Please explain this Python function." },
            },
        },
        {
            1,
            "correction",
            3,
            std::vector<int>{ 3 },
            std::vector<generated_prompt_pattern>{
                { "You are a Chinese text correction assistant. Please correct this sentence." },
            },
        },
        {
            2,
            "lyrics",
            4,
            std::vector<int>{ 4 },
            std::vector<generated_prompt_pattern>{
                { "You are a song lyrics writing assistant. Please write a short lyric." },
            },
        },
    };
}