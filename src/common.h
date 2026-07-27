#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── Version & Globals ──

static const char * ANVIL_LOGO = R"(
   ░███                          ░██░██ 
  ░██░██                            ░██ 
 ░██  ░██  ░████████  ░██    ░██ ░██░██ 
░█████████ ░██    ░██ ░██    ░██ ░██░██ 
░██    ░██ ░██    ░██  ░██  ░██  ░██░██ 
░██    ░██ ░██    ░██   ░██░██   ░██░██ 
░██    ░██ ░██    ░██    ░███    ░██░██
)";
static const char * ANVIL_VERSION = "0.2.0";
static const int  CONFIG_VERSION  = 2;

extern std::atomic<bool> g_interrupted;
static void signal_handler(int) {
    g_interrupted.store(true);
}

// ── Safe numeric parsing helpers ──

static inline bool parse_int(const std::string& s, int& out) {
    try {
        if (s.empty()) return false;
        size_t idx = 0;
        out = std::stoi(s, &idx);
        if (idx != s.size()) return false;
    } catch (...) {
        return false;
    }
    return true;
}

static inline bool parse_float(const std::string& s, float& out) {
    try {
        if (s.empty()) return false;
        size_t idx = 0;
        out = std::stof(s, &idx);
        if (idx != s.size()) return false;
    } catch (...) {
        return false;
    }
    return true;
}

static inline bool parse_uint64(const std::string& s, uint64_t& out) {
    try {
        if (s.empty()) return false;
        size_t idx = 0;
        out = std::stoull(s, &idx);
        if (idx != s.size()) return false;
    } catch (...) {
        return false;
    }
    return true;
}

// ── KV Cache Type Options ──

#include "ggml.h"

struct KVTypeOption {
    const char * label;
    ggml_type    type;
    const char * short_name;
};

static const KVTypeOption KV_OPTIONS[] = {
    { "f16    (no compression)",   GGML_TYPE_F16,       "f16"    },
    { "q8_0   (8-bit, lossless)",  GGML_TYPE_Q8_0,      "q8_0"   },
    { "turbo4 (TurboQuant 4-bit)", GGML_TYPE_TURBO4_0,  "turbo4" },
    { "turbo3 (TurboQuant 3-bit)", GGML_TYPE_TURBO3_0,  "turbo3" },
    { "turbo2 (TurboQuant 2-bit)", GGML_TYPE_TURBO2_0,  "turbo2" },
};
static const int KV_OPTIONS_COUNT = 5;

struct KVPreset {
    const char * label;
    int k_idx;
    int v_idx;
};

static const KVPreset KV_PRESETS[] = {
    { "Recommended   (K=turbo4, V=turbo3)  4.2x  ~1% loss",   2, 3 },
    { "Quality+      (K=q8_0,   V=turbo3)  ~3x   <1% loss",   1, 3 },
    { "Max Compress  (K=turbo4, V=turbo2)  6.1x  ~3% loss",   2, 4 },
    { "No Compress   (K=f16,    V=f16)     1x    baseline",    0, 0 },
    { "Custom...",                                              -1, -1 },
};
static const int KV_PRESET_COUNT = 5;

static inline ggml_type kv_type_from_name(const std::string & name) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (name == KV_OPTIONS[i].short_name) return KV_OPTIONS[i].type;
    }
    return GGML_TYPE_F16;
}

static inline const char * kv_type_short(ggml_type t) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (KV_OPTIONS[i].type == t) return KV_OPTIONS[i].short_name;
    }
    return "f16";
}

static inline int kv_type_index(ggml_type t) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (KV_OPTIONS[i].type == t) return i;
    }
    return 0;
}

// ── Hardware Info Structs ──

struct GPUInfo {
    std::string name;
    std::string vendor;
    uint64_t vram_mb = 0;
    bool is_discrete = false;
};

struct HWInfo {
    std::string os;
    std::string arch;
    std::string cpu;
    uint64_t ram_bytes = 0;
    int cpu_threads = 0;
    std::vector<GPUInfo> gpus;
    bool apple_silicon = false;
};

// ── Configuration ──

struct AnvilConfig {
    int         version    = CONFIG_VERSION;
    int         ngl        = 99;
    int         n_ctx      = 8192;
    int         n_threads  = 0;
    float       temp       = 0.8f;
    bool        flash_attn = true;
    bool        mtp        = false;
    ggml_type   type_k     = GGML_TYPE_Q8_0;
    ggml_type   type_v     = GGML_TYPE_TURBO3_0;
    bool        triattn    = false;
    std::string model;
};

// ── CLI Args ──

struct CliArgs {
    std::string model;
    int   n_ctx         = 0;
    int   ngl           = -1;
    int   n_threads     = 0;
    float temp          = -1;
    bool  flash_attn    = false;
    bool  no_flash_attn = false;
    bool  mtp           = false;
    bool  triattn       = false;
    bool  help          = false;
    bool  invalid       = false;
    bool  version       = false;
    std::string type_k;
    std::string type_v;
    std::string system_prompt;
    std::string prompt;
    std::string grammar;
    int   max_tokens    = -1;
};

// ── UTF-8 Buffer ──

struct Utf8Buffer {
    std::string pending;
    std::string feed(const std::string & chunk) {
        pending += chunk;
        size_t safe = pending.size();
        if (safe == 0) return "";
        size_t i = safe - 1;
        while (i > 0 && (pending[i] & 0xC0) == 0x80) i--;
        unsigned char lead = (unsigned char)pending[i];
        int expected = 1;
        if      ((lead & 0x80) == 0x00) expected = 1;
        else if ((lead & 0xE0) == 0xC0) expected = 2;
        else if ((lead & 0xF0) == 0xE0) expected = 3;
        else if ((lead & 0xF8) == 0xF0) expected = 4;
        int actual = (int)(safe - i);
        if (actual < expected) {
            safe = i;
        }
        std::string out = pending.substr(0, safe);
        pending = pending.substr(safe);
        return out;
    }
    std::string flush() {
        std::string out = pending;
        pending.clear();
        return out;
    }
};

// ── Chat Types ──

struct ChatMessage {
    std::string role;
    std::string content;
};

// ── Generation Stats ──

struct GenStats {
    int    tokens_generated = 0;
    double elapsed_sec      = 0.0;

    double tps() const {
        return elapsed_sec > 0.0 ? tokens_generated / elapsed_sec : 0.0;
    }
};
