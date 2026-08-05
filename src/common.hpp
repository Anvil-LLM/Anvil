#pragma once

#include "llama.h"
#include "ggml.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <clocale>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// ─── Version & branding ────────────────────────────────────────────────────

inline const char * ANVIL_LOGO = R"(
   ░███                          ░██░██ 
  ░██░██                            ░██ 
 ░██  ░██  ░████████  ░██    ░██ ░██░██ 
░█████████ ░██    ░██ ░██    ░██ ░██░██ 
░██    ░██ ░██    ░██  ░██  ░██  ░██░██ 
░██    ░██ ░██    ░██   ░██░██   ░██░██ 
░██    ░██ ░██    ░██    ░███    ░██░██
)";
inline const char * ANVIL_VERSION = "0.4.0";
inline const int    CONFIG_VERSION = 2;

// ─── Global state ──────────────────────────────────────────────────────────

inline std::atomic<bool> g_interrupted{false};
inline void anvil_signal_handler(int) {
    g_interrupted.store(true, std::memory_order_relaxed);
}

// ─── RAII wrappers around llama objects ────────────────────────────────────

struct LlamaModel {
    llama_model * p = nullptr;
    explicit LlamaModel(llama_model * p_ = nullptr) : p(p_) {}
    ~LlamaModel() { if (p) llama_model_free(p); }
    LlamaModel(const LlamaModel &) = delete;
    LlamaModel & operator=(const LlamaModel &) = delete;
    LlamaModel(LlamaModel && o) noexcept : p(o.p) { o.p = nullptr; }
    LlamaModel & operator=(LlamaModel && o) noexcept {
        if (this != &o) { if (p) llama_model_free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    llama_model * get() const { return p; }
    operator llama_model * () const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

struct LlamaContext {
    llama_context * p = nullptr;
    explicit LlamaContext(llama_context * p_ = nullptr) : p(p_) {}
    ~LlamaContext() { if (p) llama_free(p); }
    LlamaContext(const LlamaContext &) = delete;
    LlamaContext & operator=(const LlamaContext &) = delete;
    LlamaContext(LlamaContext && o) noexcept : p(o.p) { o.p = nullptr; }
    LlamaContext & operator=(LlamaContext && o) noexcept {
        if (this != &o) { if (p) llama_free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    llama_context * get() const { return p; }
    operator llama_context * () const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

struct LlamaSampler {
    llama_sampler * p = nullptr;
    explicit LlamaSampler(llama_sampler * p_ = nullptr) : p(p_) {}
    ~LlamaSampler() { if (p) llama_sampler_free(p); }
    LlamaSampler(const LlamaSampler &) = delete;
    LlamaSampler & operator=(const LlamaSampler &) = delete;
    LlamaSampler(LlamaSampler && o) noexcept : p(o.p) { o.p = nullptr; }
    LlamaSampler & operator=(LlamaSampler && o) noexcept {
        if (this != &o) { if (p) llama_sampler_free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    // Replace the owned sampler (frees the old one, if any).
    void reset(llama_sampler * p_) {
        if (p_ != p) { if (p) llama_sampler_free(p); p = p_; }
    }
    llama_sampler * get() const { return p; }
    operator llama_sampler * () const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

struct LlamaBackend {
    LlamaBackend() { llama_backend_init(); }
    ~LlamaBackend() { llama_backend_free(); }
};

// ─── Safe number parsing ───────────────────────────────────────────────────
// Full-string validation: rejects trailing garbage, ERANGE, and NaN/Inf.
// These replace the previous stoi/stof helpers which silently truncated.

inline bool parse_int(const std::string & s, int & out) {
    if (s.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (errno == ERANGE || end == s.c_str() || *end != '\0') return false;
    if (v < INT_MIN || v > INT_MAX) return false;
    out = static_cast<int>(v);
    return true;
}

inline bool parse_float(const std::string & s, float & out) {
    if (s.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const float v = std::strtof(s.c_str(), &end);
    if (errno == ERANGE || end == s.c_str() || *end != '\0') return false;
    if (!std::isfinite(v)) return false;
    out = v;
    return true;
}

inline bool parse_uint64(const std::string & s, uint64_t & out) {
    if (s.empty() || s[0] == '-') return false;
    errno = 0;
    char * end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno == ERANGE || end == s.c_str() || *end != '\0') return false;
    out = static_cast<uint64_t>(v);
    return true;
}

// ─── KV cache type options ─────────────────────────────────────────────────

struct KVTypeOption {
    const char * label;
    ggml_type    type;
    const char * short_name;
};

inline const KVTypeOption KV_OPTIONS[] = {
    { "f16    (no compression)",   GGML_TYPE_F16,       "f16"    },
    { "q8_0   (8-bit, lossless)",  GGML_TYPE_Q8_0,      "q8_0"   },
    { "turbo4 (TurboQuant 4-bit)", GGML_TYPE_TURBO4_0,  "turbo4" },
    { "turbo3 (TurboQuant 3-bit)", GGML_TYPE_TURBO3_0,  "turbo3" },
    { "turbo2 (TurboQuant 2-bit)", GGML_TYPE_TURBO2_0,  "turbo2" },
};

inline constexpr int KV_OPTIONS_COUNT = static_cast<int>(sizeof(KV_OPTIONS) / sizeof(KV_OPTIONS[0]));

struct KVPreset {
    const char * label;
    int k_idx;
    int v_idx;
};

inline const KVPreset KV_PRESETS[] = {
    { "Recommended   (K=turbo4, V=turbo3)  4.2x  ~1% loss",   2, 3 },
    { "Quality+      (K=q8_0,   V=turbo3)  ~3x   <1% loss",   1, 3 },
    { "Max Compress  (K=turbo4, V=turbo2)  6.1x  ~3% loss",   2, 4 },
    { "No Compress   (K=f16,    V=f16)     1x    baseline",    0, 0 },
    { "Custom...",                                              -1, -1 },
};

inline constexpr int KV_PRESETS_COUNT = static_cast<int>(sizeof(KV_PRESETS) / sizeof(KV_PRESETS[0]));

// ─── Hardware info ─────────────────────────────────────────────────────────

struct GPUInfo {
    std::string name;
    std::string vendor;
    uint64_t    vram_mb = 0;
    bool        is_discrete = false;
};

struct HWInfo {
    std::string os;
    std::string arch;
    std::string cpu;
    uint64_t    ram_bytes = 0;
    int         cpu_threads = 0;
    std::vector<GPUInfo> gpus;
    bool        apple_silicon = false;
};

// ─── Config ────────────────────────────────────────────────────────────────

struct AnvilConfig {
    int       version    = CONFIG_VERSION;
    int       ngl        = -1;
    int       n_ctx      = 0;
    int       n_threads  = 0;
    float     temp       = 0.8f;
    int       top_k      = 40;
    float     top_p      = 0.95f;
    float     repeat_penalty = 1.1f;
    bool      flash_attn = true;
    bool      mtp        = false;
    ggml_type type_k     = GGML_TYPE_Q8_0;
    ggml_type type_v     = GGML_TYPE_TURBO3_0;
    std::string model;
};

inline std::string config_dir() {
    const char * home = std::getenv("HOME");
#ifdef _WIN32
    if (!home || !home[0]) home = std::getenv("USERPROFILE");
#endif
    if (!home || !home[0]) home = ".";
    return std::string(home) + "/.anvil";
}

inline std::string config_path() {
    return config_dir() + "/config.json";
}

inline std::string sessions_dir() {
    return config_dir() + "/sessions";
}

// ─── Text helpers ──────────────────────────────────────────────────────────

// Buffers partial UTF-8 sequences so multi-byte characters split across
// token pieces are never printed half-rendered.
struct Utf8Buffer {
    std::string pending;

    std::string feed(const std::string & chunk) {
        pending += chunk;
        size_t safe = pending.size();
        if (safe == 0) return "";
        size_t i = safe - 1;
        while (i > 0 && (pending[i] & 0xC0) == 0x80) i--;
        const unsigned char lead = static_cast<unsigned char>(pending[i]);
        int expected = 1;
        if      ((lead & 0x80) == 0x00) expected = 1;
        else if ((lead & 0xE0) == 0xC0) expected = 2;
        else if ((lead & 0xF0) == 0xE0) expected = 3;
        else if ((lead & 0xF8) == 0xF0) expected = 4;
        const int actual = static_cast<int>(safe - i);
        if (actual < expected) safe = i;
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

struct ChatMessage {
    std::string role;
    std::string content;
};

struct GenStats {
    int    tokens_generated = 0;
    double elapsed_sec      = 0.0;

    double tps() const {
        return elapsed_sec > 0.0 ? tokens_generated / elapsed_sec : 0.0;
    }
};
