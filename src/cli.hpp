#pragma once

#include "common.hpp"

#include <string>

struct CliArgs {
    std::string model;
    int         n_ctx       = 0;
    int         ngl         = -1;
    int         n_threads   = 0;
    float       temp        = -1.0f;
    int         top_k       = -1;   // -1 = not set
    float       top_p       = -1.0f;
    float       repeat_penalty = -1.0f;
    bool        flash_attn  = false;
    bool        no_flash_attn = false;
    bool        mtp         = false;
    bool        help        = false;
    bool        invalid     = false;
    bool        version     = false;
    bool        setup       = false;
    std::string type_k;
    std::string type_v;
    std::string system_prompt;
    std::string prompt;
    std::string grammar;
    int         max_tokens  = -1;
};

// Bounds applied to CLI numeric options (also used by config validation).
inline constexpr int   MAX_CTX      = 1 << 22;   // 4M tokens sanity cap
inline constexpr int   MAX_THREADS  = 1024;
inline constexpr float MAX_TEMP     = 5.0f;
inline constexpr int   MAX_TOP_K    = 100000;

void print_usage();
CliArgs parse_args(int argc, char ** argv);
