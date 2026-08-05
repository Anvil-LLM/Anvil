// anvil — single-binary local LLM runtime (llama-turbo backend: TurboQuant + MTP + NextN).
// Merged monolith — source history preserved in the modular commit (392c155).

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
#include <algorithm>
#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#include <sys/sysctl.h>
#endif
#include <unistd.h>
#include <glob.h>
#ifdef _WIN32
#include <windows.h>
#include <dxgi.h>
#endif
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <iostream>
#include <ctime>
#include <cinttypes>
#include <map>
#include <nlohmann/json.hpp>

// ──── src/common.hpp ────



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
    std::string system_prompt;   // per-run system prompt (from CLI or profile)
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
// ──── src/cli.hpp ────



struct CliArgs {
    std::string sub;                    // "run" (default) | models | profile | rm | pull
    std::vector<std::string> sub_args;  // raw args for subcommands
    std::string model;
    std::string friendly;               // registry display name, when resolved
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
    bool        save_profile = false;   // persist CLI overrides into the model profile
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
// ──── src/cli.cpp ────


void print_usage() {
    printf("anvil %s — Forge anything.\n\n", ANVIL_VERSION);
    printf("Usage:\n");
    printf("  anvil run <model> [options]     Run a model with chat REPL\n");
    printf("  anvil run <model> -p \"prompt\"   Single-shot generation\n");
    printf("  anvil <model> [options]         Run (model name or file path)\n");
    printf("  anvil models                    List registered models\n");
    printf("  anvil models import <file.gguf> [--name <name>]\n");
    printf("  anvil profile <name>            Show a model's persistent profile\n");
    printf("  anvil profile <name> set k=v .. Save profile settings\n");
    printf("  anvil rm <name> [--yes]         Unregister a model\n");
    printf("  anvil pull ollama:<name>[:tag]  Pull from the Ollama registry\n");
    printf("  anvil pull hf:<repo>[:file]     Pull from HuggingFace\n");
    printf("  anvil --help                    Show this help\n");
    printf("  anvil --version                 Show version\n");
    printf("  anvil --setup                   Re-run hardware setup TUI\n\n");
    printf("Options:\n");
    printf("  -c, --ctx <n>            Context size (default: auto from model)\n");
    printf("  -ngl, --n-gpu-layers <n> GPU layers to offload (default: auto)\n");
    printf("  -t, --temp <f>           Sampling temperature (default: 0.8)\n");
    printf("  --top-k <n>              Top-k sampling (default: 40, 0 = off)\n");
    printf("  --top-p <f>              Top-p (nucleus) sampling (default: 0.95, 1 = off)\n");
    printf("  --repeat-penalty <f>     Token repeat penalty (default: 1.1, 1 = off)\n");
    printf("  --threads <n>            CPU threads (default: auto)\n");
    printf("  --flash-attn             Enable flash attention (default: on)\n");
    printf("  --no-flash-attn          Disable flash attention\n");
    printf("  --type-k <type>          K cache type: f16|q8_0|turbo4|turbo3|turbo2\n");
    printf("  --type-v <type>          V cache type: f16|q8_0|turbo4|turbo3|turbo2\n");
    printf("  --mtp                    Enable MTP speculative decoding\n");
    printf("  --grammar <file>         GBNF grammar file for constrained output\n");
    printf("  -s, --system <text>      System prompt\n");
    printf("  -p, --prompt <text>      User prompt (non-interactive mode)\n");
    printf("  -n, --max-tokens <n>     Max tokens to generate (default: unlimited)\n");
    printf("      --save                Persist CLI overrides into the model's profile\n\n");
    printf("Model registry:\n");
    printf("  <model> may be a friendly name (registered via pull/import) or a file path.\n");
    printf("  Per-model settings live in ~/.anvil/models.json (profiles).\n");
    printf("  Precedence: CLI flags > model profile > global config.\n\n");
    printf("TurboQuant KV presets (from setup TUI):\n");
    printf("  Recommended:  K=turbo4 V=turbo3  (4.2x, ~1%% quality loss)\n");
    printf("  Quality+:     K=q8_0   V=turbo3  (~3x,  <1%% quality loss)\n");
    printf("  High Compression: K=turbo4 V=turbo2  (6.1x, ~3%% quality loss)\n\n");
    printf("Examples:\n");
    printf("  anvil run model.gguf\n");
    printf("  anvil run model.gguf --ctx 131072 --type-k q8_0 --type-v turbo3\n");
    printf("  anvil run model.gguf -p \"Explain quantum computing\" -n 200\n");
    printf("  anvil run model.gguf --grammar json.gbnf -p \"List 3 colors\"\n");
}

// Parse an integer option with full-string validation and range checking.
// On failure prints an error, marks the args invalid and returns false.
static bool set_int(const std::string & arg_name, const std::string & value,
                    int min, int max, int & out, CliArgs & a) {
    int v = 0;
    if (!parse_int(value, v) || v < min || v > max) {
        fprintf(stderr, "error: invalid value for %s: '%s' (expected %d..%d)\n",
                arg_name.c_str(), value.c_str(), min, max);
        a.invalid = true;
        a.help = true;
        return false;
    }
    out = v;
    return true;
}

static bool set_float(const std::string & arg_name, const std::string & value,
                      float min, float max, float & out, CliArgs & a) {
    float v = 0.0f;
    if (!parse_float(value, v) || v < min || v > max) {
        fprintf(stderr, "error: invalid value for %s: '%s' (expected %.2f..%.2f)\n",
                arg_name.c_str(), value.c_str(), min, max);
        a.invalid = true;
        a.help = true;
        return false;
    }
    out = v;
    return true;
}

CliArgs parse_args(int argc, char ** argv) {
    CliArgs a;
    // Bare invocation (no args) is a usage error (exit 1); an explicit --help
    // still exits 0. Standard CLI convention (git, docker, clang, ...).
    if (argc < 2) { a.invalid = true; a.help = true; return a; }
    // Subcommands keep their raw arguments for the dispatcher in main().
    const std::string first = argv[1];
    if (first == "models" || first == "profile" || first == "rm" || first == "pull") {
        a.sub = first;
        for (int i = 2; i < argc; i++) a.sub_args.emplace_back(argv[i]);
        return a;
    }
    int i = 1;
    if (i < argc && std::string(argv[i]) == "run") i++;
    for (; i < argc; i++) {
        const std::string arg = argv[i];
        if      (arg == "--help" || arg == "-h")                          { a.help = true; }
        else if (arg == "--version")                                      { a.version = true; }
        else if (arg == "--setup")                                        { a.setup = true; }
        else if ((arg == "-c" || arg == "--ctx") && i + 1 < argc) {
            set_int(arg, argv[++i], 1, MAX_CTX, a.n_ctx, a);
        }
        else if ((arg == "-ngl" || arg == "--ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            // -1 = auto/all, so allow -1
            set_int(arg, argv[++i], -1, 10000, a.ngl, a);
        }
        else if ((arg == "-t" || arg == "--temp") && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, MAX_TEMP, a.temp, a);
        }
        else if (arg == "--top-k" && i + 1 < argc) {
            set_int(arg, argv[++i], 0, MAX_TOP_K, a.top_k, a);
        }
        else if (arg == "--top-p" && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, 1.0f, a.top_p, a);
        }
        else if (arg == "--repeat-penalty" && i + 1 < argc) {
            set_float(arg, argv[++i], 1.0f, 100.0f, a.repeat_penalty, a);
        }
        else if (arg == "--threads" && i + 1 < argc) {
            set_int(arg, argv[++i], 1, MAX_THREADS, a.n_threads, a);
        }
        else if (arg == "--flash-attn")                                   { a.flash_attn = true; }
        else if (arg == "--no-flash-attn")                                { a.no_flash_attn = true; a.flash_attn = false; }
        else if (arg == "--type-k" && i + 1 < argc)                       { a.type_k = argv[++i]; }
        else if (arg == "--type-v" && i + 1 < argc)                       { a.type_v = argv[++i]; }
        else if (arg == "--mtp")                                          { a.mtp = true; }
        else if (arg == "--save")                                         { a.save_profile = true; }
        else if (arg == "--grammar" && i + 1 < argc)                      { a.grammar = argv[++i]; }
        else if ((arg == "-s" || arg == "--system") && i + 1 < argc)      { a.system_prompt = argv[++i]; }
        else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc)      { a.prompt = argv[++i]; }
        else if ((arg == "-n" || arg == "--max-tokens") && i + 1 < argc) {
            set_int(arg, argv[++i], -1, INT32_MAX, a.max_tokens, a);
        }
        else if (arg[0] != '-')                                           { a.model = arg; }
        else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            a.invalid = true;
            a.help = true;
        }
    }
    return a;
}
// ──── src/config.hpp ────


ggml_type kv_type_from_name(const std::string & name);
const char * kv_type_short(ggml_type type);

void write_config(const AnvilConfig & cfg);
AnvilConfig load_config();
bool config_exists();
// ──── src/config.cpp ────


ggml_type kv_type_from_name(const std::string & name) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (name == KV_OPTIONS[i].short_name) return KV_OPTIONS[i].type;
    }
    fprintf(stderr, "\033[33mwarning: unknown kv type '%s', using f16\033[0m\n", name.c_str());
    return GGML_TYPE_F16;
}

const char * kv_type_short(ggml_type type) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (KV_OPTIONS[i].type == type) return KV_OPTIONS[i].short_name;
    }
    return "f16";
}

// ─── Minimal JSON string getter ────────────────────────────────────────────
// Extracts the string/number value of a top-level key. Intentionally small and
// dependency-free; the config schema is flat and controlled by us.

static std::string json_get(const std::string & json, const std::string & key) {
    const std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        pos++;
        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                switch (json[pos]) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    case 'u': {
                        if (pos + 4 < json.size()) {
                            int cp = 0;
                            for (int j = 0; j < 4; j++) {
                                pos++;
                                cp <<= 4;
                                const char h = json[pos];
                                if (h >= '0' && h <= '9')      cp |= h - '0';
                                else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                                else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                            }
                            if (cp <= 0x7F) {
                                result += static_cast<char>(cp);
                            } else if (cp <= 0x7FF) {
                                result += static_cast<char>(0xC0 | (cp >> 6));
                                result += static_cast<char>(0x80 | (cp & 0x3F));
                            } else {
                                result += static_cast<char>(0xE0 | (cp >> 12));
                                result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                result += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                        }
                        break;
                    }
                    default: result += json[pos]; break;
                }
            } else {
                result += json[pos];
            }
            pos++;
        }
        return result;
    }
    auto end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != '\n' && json[end] != '\r') end++;
    std::string val = json.substr(pos, end - pos);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
    return val;
}

// ─── Config file I/O ───────────────────────────────────────────────────────

static std::string json_escape(const std::string & s) {
    std::string r;
    for (const char c : s) {
        switch (c) {
            case '\\': r += "\\\\"; break;
            case '"':  r += "\\\""; break;
            case '\n': r += "\\n";  break;
            case '\r': r += "\\r";  break;
            case '\t': r += "\\t";  break;
            default:   r += c;      break;
        }
    }
    return r;
}

void write_config(const AnvilConfig & cfg) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(config_dir(), ec);
    if (ec) {
        fprintf(stderr, "\033[33mwarning: could not create config dir %s: %s\033[0m\n",
                config_dir().c_str(), ec.message().c_str());
    }

    const std::string tmp = config_path() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) {
            fprintf(stderr, "\033[33mwarning: could not write config to %s\033[0m\n", tmp.c_str());
            return;
        }
        f << "{\n";
        f << "  \"version\": " << cfg.version << ",\n";
        f << "  \"ngl\": " << cfg.ngl << ",\n";
        f << "  \"n_ctx\": " << cfg.n_ctx << ",\n";
        f << "  \"n_threads\": " << cfg.n_threads << ",\n";
        f << "  \"temp\": " << cfg.temp << ",\n";
        f << "  \"top_k\": " << cfg.top_k << ",\n";
        f << "  \"top_p\": " << cfg.top_p << ",\n";
        f << "  \"repeat_penalty\": " << cfg.repeat_penalty << ",\n";
        f << "  \"flash_attn\": " << (cfg.flash_attn ? "true" : "false") << ",\n";
        f << "  \"mtp\": " << (cfg.mtp ? "true" : "false") << ",\n";
        f << "  \"type_k\": \"" << kv_type_short(cfg.type_k) << "\",\n";
        f << "  \"type_v\": \"" << kv_type_short(cfg.type_v) << "\",\n";
        f << "  \"model\": \"" << json_escape(cfg.model) << "\"\n";
        f << "}\n";
        f.flush();
        if (!f) {
            fprintf(stderr, "\033[33mwarning: failed writing config to %s\033[0m\n", tmp.c_str());
            return;
        }
    }
    // Atomic replace: a crash mid-write never corrupts the real config.
    fs::rename(tmp, config_path(), ec);
    if (ec) {
        fprintf(stderr, "\033[33mwarning: could not rename config into place at %s: %s\033[0m\n",
                config_path().c_str(), ec.message().c_str());
    }
}

static bool json_get_bool(const std::string & json, const std::string & key, bool def) {
    const std::string s = json_get(json, key);
    return s.empty() ? def : (s == "true");
}

AnvilConfig load_config() {
    AnvilConfig cfg;
    std::ifstream f(config_path());
    if (!f) return cfg;
    const std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    std::string s;
    int tmp_int = 0;
    float tmp_float = 0.0f;

    s = json_get(json, "version");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.version = tmp_int;
    s = json_get(json, "ngl");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.ngl = tmp_int;
    s = json_get(json, "n_ctx");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.n_ctx = tmp_int;
    s = json_get(json, "n_threads");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.n_threads = tmp_int;
    s = json_get(json, "temp");
    if (!s.empty() && parse_float(s, tmp_float)) cfg.temp = tmp_float;
    s = json_get(json, "top_k");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.top_k = tmp_int;
    s = json_get(json, "top_p");
    if (!s.empty() && parse_float(s, tmp_float)) cfg.top_p = tmp_float;
    s = json_get(json, "repeat_penalty");
    if (!s.empty() && parse_float(s, tmp_float)) cfg.repeat_penalty = tmp_float;

    cfg.flash_attn = json_get_bool(json, "flash_attn", cfg.flash_attn);
    cfg.mtp        = json_get_bool(json, "mtp", cfg.mtp);

    s = json_get(json, "type_k");
    if (!s.empty()) cfg.type_k = kv_type_from_name(s);
    s = json_get(json, "type_v");
    if (!s.empty()) cfg.type_v = kv_type_from_name(s);
    s = json_get(json, "model");
    if (!s.empty()) cfg.model = s;

    // v1 -> v2 migration: honor the old "no_turbo" key.
    if (cfg.version < 2) {
        const std::string no_turbo = json_get(json, "no_turbo");
        if (no_turbo == "true") {
            cfg.type_k = GGML_TYPE_F16;
            cfg.type_v = GGML_TYPE_F16;
        } else {
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO3_0;
        }
        cfg.version = CONFIG_VERSION;
    }
    return cfg;
}

bool config_exists() {
    std::ifstream f(config_path());
    return f.good();
}

// ──── src/models.hpp ────
// Friendly-name model registry (~/.anvil/models.json) with persistent
// per-model profiles. Zero new dependencies: nlohmann/json is vendored in
// the backend; writes are atomic (tmp + rename), matching config.json.

inline std::string models_json_path() { return config_dir() + "/models.json"; }
inline std::string models_dir()       { return config_dir() + "/models"; }

// Declared here (defined in hardware.cpp below) because import validates GGUF.
bool validate_gguf(const std::string & path);

struct ModelProfile {
    // Explicitly-set keys only; anything absent inherits the global config.
    std::map<std::string, nlohmann::json> settings;

    nlohmann::json to_json() const { return nlohmann::json(settings); }

    static ModelProfile from_json(const nlohmann::json & j) {
        ModelProfile p;
        if (j.is_object()) {
            for (auto it = j.begin(); it != j.end(); ++it) p.settings[it.key()] = it.value();
        }
        return p;
    }

    bool empty() const { return settings.empty(); }
    void clear()       { settings.clear(); }
    void set(const std::string & k, const nlohmann::json & v) { settings[k] = v; }
    bool unset(const std::string & k) { return settings.erase(k) > 0; }

    // Apply explicitly-set values onto a config; returns the number applied.
    // Tolerant of hand-edited JSON: a wrong-typed value warns and is skipped
    // rather than crashing the run (nlohmann throws json::type_error).
    int apply_to(AnvilConfig & cfg) const {
        int n = 0;
        for (const auto & [k, v] : settings) {
            try {
                if      (k == "n_ctx")          { cfg.n_ctx = v.get<int>();        n++; }
                else if (k == "ngl")            { cfg.ngl = v.get<int>();          n++; }
                else if (k == "n_threads")      { cfg.n_threads = v.get<int>();    n++; }
                else if (k == "temp")           { cfg.temp = v.get<float>();       n++; }
                else if (k == "top_k")          { cfg.top_k = v.get<int>();        n++; }
                else if (k == "top_p")          { cfg.top_p = v.get<float>();      n++; }
                else if (k == "repeat_penalty") { cfg.repeat_penalty = v.get<float>(); n++; }
                else if (k == "flash_attn")     { cfg.flash_attn = v.get<bool>();  n++; }
                else if (k == "mtp")            { cfg.mtp = v.get<bool>();         n++; }
                else if (k == "type_k")         { cfg.type_k = kv_type_from_name(v.get<std::string>()); n++; }
                else if (k == "type_v")         { cfg.type_v = kv_type_from_name(v.get<std::string>()); n++; }
                else if (k == "system_prompt")  { cfg.system_prompt = v.get<std::string>(); n++; }
            } catch (const nlohmann::json::exception &) {
                fprintf(stderr, "\033[33mwarning: skipping invalid profile key '%s' (wrong type)\033[0m\n", k.c_str());
            }
        }
        return n;
    }

    static bool valid_key(const std::string & k) {
        static const char * keys[] = {
            "n_ctx", "ngl", "n_threads", "temp", "top_k", "top_p",
            "repeat_penalty", "flash_attn", "mtp", "type_k", "type_v", "system_prompt"
        };
        for (const char * kk : keys) if (k == kk) return true;
        return false;
    }
};

struct ModelEntry {
    std::string name;
    std::string path;
    std::string source = "local";   // local | ollama | ollama-local | hf
    std::string source_id;          // e.g. library/llama3.2:3b or HF repo[:file]
    uint64_t    size_bytes = 0;
    std::string added;
    std::string desc;               // GGUF architecture description
    int64_t     trained_ctx = 0;
    std::string chat_template;      // from Ollama metadata layers (if any)
    std::string license;
    std::string params_json;        // raw Ollama params blob (if any)
    nlohmann::json gguf_meta;       // verbose GGUF header metadata
    ModelProfile profile;

    nlohmann::json to_json() const {
        return {
            {"name", name}, {"path", path}, {"source", source},
            {"source_id", source_id}, {"size_bytes", size_bytes}, {"added", added},
            {"desc", desc}, {"trained_ctx", trained_ctx},
            {"chat_template", chat_template}, {"license", license},
            {"params_json", params_json}, {"gguf_meta", gguf_meta},
            {"settings", profile.to_json()},
        };
    }

    static ModelEntry from_json(const nlohmann::json & j) {
        ModelEntry e;
        auto get = [&](const char * k, auto & out) {
            if (j.contains(k)) {
                try { out = j[k].get<std::decay_t<decltype(out)>>(); }
                catch (const nlohmann::json::exception &) {}
            }
        };
        get("name", e.name); get("path", e.path); get("source", e.source);
        get("source_id", e.source_id); get("size_bytes", e.size_bytes);
        get("added", e.added); get("desc", e.desc); get("trained_ctx", e.trained_ctx);
        get("chat_template", e.chat_template); get("license", e.license);
        get("params_json", e.params_json);
        if (j.contains("gguf_meta") && j["gguf_meta"].is_object()) e.gguf_meta = j["gguf_meta"];
        if (j.contains("settings")) e.profile = ModelProfile::from_json(j["settings"]);
        return e;
    }
};

// ──── src/models.cpp ────

static std::string now_iso() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    if (!std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t))) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(t));
    }
    return buf;
}

static std::string expand_home(const std::string & p) {
    if (p.size() > 1 && p[0] == '~' && (p[1] == '/' || p[1] == '\\')) {
        const char * home = std::getenv("HOME");
        if (home && home[0]) return std::string(home) + p.substr(1);
    }
    return p;
}

static std::string format_size(uint64_t bytes) {
    char buf[32];
    if (bytes >= 1073741824ULL)      std::snprintf(buf, sizeof(buf), "%.1f GB", static_cast<double>(bytes) / 1073741824.0);
    else if (bytes >= 1048576ULL)    std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / 1048576.0);
    else if (bytes >= 1024ULL)       std::snprintf(buf, sizeof(buf), "%.1f KB", static_cast<double>(bytes) / 1024.0);
    else                             std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
    return buf;
}

// Safe filename from a friendly name (ollama:llama3.2:3b -> llama3.2-3b).
static std::string slugify(const std::string & s) {
    std::string out;
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (std::isalnum(c) || c == '.' || c == '-' || c == '_') {
            out += static_cast<char>(std::tolower(c));
        } else if (c == ':' || c == '/' || c == ' ' || c == '@') {
            out += '-';
        }
    }
    while (out.size() > 1 && out.back() == '-') out.pop_back();
    if (out.empty()) out = "model";
    return out;
}

std::vector<ModelEntry> load_models() {
    std::vector<ModelEntry> models;
    std::ifstream f(models_json_path());
    if (!f) return models;
    nlohmann::json root;
    try {
        f >> root;
        if (root.contains("models") && root["models"].is_array()) {
            for (const auto & j : root["models"]) {
                try { models.push_back(ModelEntry::from_json(j)); }
                catch (const nlohmann::json::exception &) { /* skip malformed entry */ }
            }
        }
    } catch (const nlohmann::json::exception & e) {
        fprintf(stderr, "\033[33mwarning: %s is unreadable (%s); starting empty\033[0m\n",
                models_json_path().c_str(), e.what());
    }
    return models;
}

bool save_models(const std::vector<ModelEntry> & models) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(config_dir(), ec);
    if (ec) {
        fprintf(stderr, "\033[33mwarning: could not create config dir %s\033[0m\n", config_dir().c_str());
        return false;
    }
    nlohmann::json root;
    root["version"] = 1;
    root["models"] = nlohmann::json::array();
    for (const auto & e : models) root["models"].push_back(e.to_json());

    const std::string tmp = models_json_path() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) {
            fprintf(stderr, "\033[33mwarning: could not write %s\033[0m\n", tmp.c_str());
            return false;
        }
        f << root.dump(2) << "\n";
        f.flush();
        if (!f) {
            fprintf(stderr, "\033[33mwarning: failed writing %s\033[0m\n", tmp.c_str());
            return false;
        }
    }
    fs::rename(tmp, models_json_path(), ec);
    if (ec) {
        fprintf(stderr, "\033[33mwarning: could not rename %s into place: %s\033[0m\n",
                models_json_path().c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

ModelEntry * find_model(std::vector<ModelEntry> & models, const std::string & name) {
    for (auto & m : models) if (m.name == name) return &m;
    return nullptr;
}
const ModelEntry * find_model(const std::vector<ModelEntry> & models, const std::string & name) {
    for (const auto & m : models) if (m.name == name) return &m;
    return nullptr;
}

// Verbose GGUF header metadata (vocab-only load: fast, no weights).
struct ModelMeta {
    std::string desc;
    int64_t     trained_ctx = 0;
    nlohmann::json gguf_meta;   // every GGUF header key/value (strings)
};

static ModelMeta read_model_meta(const std::string & path) {
    ModelMeta meta;
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.vocab_only = true;
    LlamaModel m(llama_model_load_from_file(path.c_str(), mparams));
    if (!m) return meta;

    // NOTE: this fork's loader returns from the hparams phase on vocab_only
    // (llama-model.cpp: "if (hparams.vocab_only) return;"), so
    // llama_model_desc()/llama_model_n_ctx_train() are empty for vocab-only
    // models. The gguf_kv map is still fully populated, so we derive the
    // description and trained context from the captured header instead. This
    // also fixes the auto-ctx-from-model path, which was silently always 0.
    // Verbose metadata: capture every GGUF header key/value (strings only).
    // The *_by_index functions are snprintf-style: query the size with a null
    // buffer, then fill. Values can exceed 4 KB (chat templates), so a fixed
    // stack buffer would truncate; the two-call pattern is exact and safe.
    nlohmann::json meta_map = nlohmann::json::object();
    const int32_t count = llama_model_meta_count(m);
    if (count > 0 && count <= 4096) {
        for (int32_t i = 0; i < count; i++) {
            const int32_t klen = llama_model_meta_key_by_index(m, i, nullptr, 0);
            if (klen <= 0 || klen > (1 << 20)) continue;
            std::string key(static_cast<size_t>(klen) + 1, '\0');
            if (llama_model_meta_key_by_index(m, i, key.data(), key.size()) < 0) continue;
            key.resize(static_cast<size_t>(klen));

            const int32_t vlen = llama_model_meta_val_str_by_index(m, i, nullptr, 0);
            if (vlen < 0 || vlen > (1 << 20)) continue;
            std::string val(static_cast<size_t>(vlen) + 1, '\0');
            if (llama_model_meta_val_str_by_index(m, i, val.data(), val.size()) < 0) continue;
            val.resize(static_cast<size_t>(vlen));
            meta_map[key] = val;
        }
    }

    auto meta_str = [&](const std::string & key) -> std::string {
        if (meta_map.contains(key) && meta_map[key].is_string()) {
            return meta_map[key].get<std::string>();
        }
        return "";
    };
    const std::string arch = meta_str("general.architecture");
    std::string name = meta_str("general.name");
    if (name.empty()) name = meta_str("general.basename");

    // Trained context: general.context_length (modern) or <arch>.context_length.
    std::string ctx = meta_str("general.context_length");
    if (ctx.empty() && !arch.empty()) ctx = meta_str(arch + ".context_length");
    if (!ctx.empty()) {
        // Values may look like "8192" or "[8192]"; grab the first integer.
        size_t i = 0;
        while (i < ctx.size() && !std::isdigit(static_cast<unsigned char>(ctx[i]))) i++;
        if (i < ctx.size()) {
            errno = 0;
            char * end = nullptr;
            const long long v = std::strtoll(ctx.c_str() + i, &end, 10);
            if (errno != ERANGE && end != ctx.c_str() + i && v > 0) meta.trained_ctx = v;
        }
    }

    if (!name.empty()) meta.desc = arch.empty() ? name : name + " (arch: " + arch + ")";
    else if (!arch.empty()) meta.desc = arch;

    meta.gguf_meta = std::move(meta_map);
    return meta;
}

static bool valid_kv_name(const std::string & n) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (n == KV_OPTIONS[i].short_name) return true;
    }
    return false;
}

// Profile values reuse the same bounds as the CLI (set_int/set_float), so
// `anvil profile x set temp=99` fails exactly like `--temp 99`.
static bool parse_profile_value(const std::string & key, const std::string & val, ModelProfile & p) {
    int iv = 0; float fv = 0.0f;
    if (key == "n_ctx")      { if (!parse_int(val, iv) || iv < 1 || iv > MAX_CTX) return false; p.set(key, iv); return true; }
    if (key == "ngl")        { if (!parse_int(val, iv) || iv < -1 || iv > 10000) return false; p.set(key, iv); return true; }
    if (key == "n_threads")  { if (!parse_int(val, iv) || iv < 1 || iv > MAX_THREADS) return false; p.set(key, iv); return true; }
    if (key == "top_k")      { if (!parse_int(val, iv) || iv < 0 || iv > MAX_TOP_K) return false; p.set(key, iv); return true; }
    if (key == "temp")       { if (!parse_float(val, fv) || fv < 0.0f || fv > MAX_TEMP) return false; p.set(key, fv); return true; }
    if (key == "top_p")      { if (!parse_float(val, fv) || fv < 0.0f || fv > 1.0f) return false; p.set(key, fv); return true; }
    if (key == "repeat_penalty") { if (!parse_float(val, fv) || fv < 1.0f || fv > 100.0f) return false; p.set(key, fv); return true; }
    if (key == "flash_attn" || key == "mtp") {
        if (val != "true" && val != "false" && val != "1" && val != "0" && val != "on" && val != "off") return false;
        p.set(key, (val == "true" || val == "1" || val == "on")); return true;
    }
    if (key == "type_k" || key == "type_v") {
        if (!valid_kv_name(val)) return false;
        p.set(key, val); return true;
    }
    if (key == "system_prompt") { p.set(key, val); return true; }
    return false;
}

// ─── Name resolution ───────────────────────────────────────────────────────

// True when the argument names a file (path-ish) rather than a registry name.
static bool looks_like_file(const std::string & arg) {
    if (arg.find('/') != std::string::npos || arg.find('\\') != std::string::npos) return true;
    if (arg.size() > 5 && arg.compare(arg.size() - 5, 5, ".gguf") == 0) return true;
    if (std::filesystem::exists(expand_home(arg))) return true;
    return false;
}

// Resolve a `run` argument: registry name (friendly_out set) or a local file
// (friendly_out left empty). Returns false when neither matches.
bool resolve_model_arg(const std::string & arg, std::string & path_out, std::string & friendly_out) {
    const std::vector<ModelEntry> models = load_models();
    if (const ModelEntry * e = find_model(models, arg)) {
        path_out = e->path;
        friendly_out = e->name;
        return true;
    }
    if (looks_like_file(arg)) {
        path_out = expand_home(arg);
        friendly_out.clear();
        return true;
    }
    return false;
}

// ─── Commands ──────────────────────────────────────────────────────────────

// Register a local file under a friendly name. Reuses an existing entry that
// already points at the same file (canonical-path match) so running a file by
// path never duplicates its registry entry under a second name.
ModelEntry * auto_register(std::vector<ModelEntry> & models,
                           const std::string & path, const std::string & friendly,
                           const ModelMeta & meta) {
    if (ModelEntry * existing = find_model(models, friendly)) return existing;
    std::error_code ec;
    const std::filesystem::path canon = std::filesystem::canonical(path, ec);
    if (!ec) {
        for (auto & m : models) {
            std::error_code ec2;
            const std::filesystem::path mcanon = std::filesystem::canonical(m.path, ec2);
            if (!ec2 && mcanon == canon) return &m;
        }
    }
    ModelEntry e;
    e.name = friendly;
    e.path = path;
    e.source = "local";
    e.added = now_iso();
    const auto sz = std::filesystem::file_size(path, ec);
    e.size_bytes = ec ? 0 : sz;
    e.desc = meta.desc;
    e.trained_ctx = meta.trained_ctx;
    e.gguf_meta = meta.gguf_meta;
    models.push_back(std::move(e));
    return &models.back();
}

int cmd_rm(const std::vector<std::string> & args) {
    if (args.empty()) {
        fprintf(stderr, "usage: anvil rm <name> [--yes]\n");
        return 1;
    }
    const std::string name = args[0];
    bool delete_file = false;
    for (size_t i = 1; i < args.size(); i++) {
        if (args[i] == "--yes") delete_file = true;
        else {
            fprintf(stderr, "error: unknown option '%s'\n", args[i].c_str());
            return 1;
        }
    }
    std::vector<ModelEntry> models = load_models();
    auto it = std::find_if(models.begin(), models.end(),
                           [&](const ModelEntry & e) { return e.name == name; });
    if (it == models.end()) {
        fprintf(stderr, "error: no model named '%s'\n", name.c_str());
        return 1;
    }
    const std::string path = it->path;
    if (delete_file) {
        std::error_code ec;
        if (!std::filesystem::remove(path, ec)) {
            fprintf(stderr, "warning: could not delete %s (%s)\n", path.c_str(), ec.message().c_str());
        } else {
            fprintf(stderr, "Deleted %s\n", path.c_str());
        }
    }
    models.erase(it);
    if (!save_models(models)) return 1;
    printf("Removed '%s' from the registry.\n", name.c_str());
    return 0;
}

int cmd_models(const std::vector<std::string> & args) {
    if (args.empty()) {
        const std::vector<ModelEntry> models = load_models();
        if (models.empty()) {
            printf("No models registered.\n");
            printf("  anvil pull ollama:<name>[:tag]      pull from the Ollama registry\n");
            printf("  anvil pull hf:<repo>[:file]         pull from HuggingFace\n");
            printf("  anvil models import <file.gguf> [--name <name>]\n");
            return 0;
        }
        printf("%-26s %-12s %-10s %-10s %s\n", "NAME", "SOURCE", "SIZE", "TRAINED", "PATH");
        for (const auto & m : models) {
            char trained[32];
            if (m.trained_ctx > 0) std::snprintf(trained, sizeof(trained), "%lld", static_cast<long long>(m.trained_ctx));
            else std::snprintf(trained, sizeof(trained), "-");
            printf("%-26s %-12s %-10s %-10s %s\n", m.name.c_str(), m.source.c_str(),
                   format_size(m.size_bytes).c_str(), trained, m.path.c_str());
        }
        return 0;
    }

    if (args[0] == "import") {
        if (args.size() < 2) {
            fprintf(stderr, "usage: anvil models import <file.gguf> [--name <name>]\n");
            return 1;
        }
        std::string path = expand_home(args[1]);
        std::string name;
        for (size_t i = 2; i < args.size(); i++) {
            if (args[i] == "--name" && i + 1 < args.size()) name = args[++i];
            else {
                fprintf(stderr, "error: unknown import option '%s'\n", args[i].c_str());
                return 1;
            }
        }
        if (!validate_gguf(path)) {
            fprintf(stderr, "\033[31merror: '%s' is not a valid GGUF file\033[0m\n", path.c_str());
            return 1;
        }
        std::error_code ec;
        const std::filesystem::path canon = std::filesystem::canonical(path, ec);
        if (!ec) path = canon.string();

        std::vector<ModelEntry> models = load_models();
        if (name.empty()) name = slugify(std::filesystem::path(path).stem().string());
        if (find_model(models, name)) {
            fprintf(stderr, "error: a model named '%s' is already registered\n", name.c_str());
            return 1;
        }
        ModelEntry e;
        e.name = name;
        e.path = path;
        e.source = "local";
        e.added = now_iso();
        e.size_bytes = std::filesystem::file_size(path, ec);
        if (ec) e.size_bytes = 0;
        const ModelMeta meta = read_model_meta(path);
        e.desc = meta.desc;
        e.trained_ctx = meta.trained_ctx;
        e.gguf_meta = meta.gguf_meta;
        models.push_back(std::move(e));
        if (!save_models(models)) return 1;
        printf("Registered '%s' -> %s\n", name.c_str(), path.c_str());
        printf("  run with: anvil run %s\n", name.c_str());
        return 0;
    }

    if (args[0] == "rm" || args[0] == "remove") {
        return cmd_rm(std::vector<std::string>(args.begin() + 1, args.end()));
    }

    fprintf(stderr, "unknown 'models' subcommand '%s' (try: list, import, rm)\n", args[0].c_str());
    return 1;
}

int cmd_profile(const std::vector<std::string> & args) {
    if (args.empty()) {
        fprintf(stderr, "usage: anvil profile <name> [set k=v ...] [unset k ...] [reset]\n");
        return 1;
    }
    const std::string name = args[0];
    std::vector<ModelEntry> models = load_models();
    ModelEntry * e = find_model(models, name);
    if (!e) {
        fprintf(stderr, "error: no model named '%s'\n", name.c_str());
        return 1;
    }

    auto is_cmd = [](const std::string & s) { return s == "set" || s == "unset" || s == "reset"; };
    bool changed = false;
    bool show = true;
    for (size_t i = 1; i < args.size(); i++) {
        const std::string & a = args[i];
        if (a == "set") {
            show = false;
            const size_t before = i;
            while (i + 1 < args.size() && !is_cmd(args[i + 1])) {
                const std::string kv = args[++i];
                const size_t eq = kv.find('=');
                if (eq == std::string::npos) {
                    fprintf(stderr, "error: expected key=value, got '%s'\n", kv.c_str());
                    return 1;
                }
                const std::string key = kv.substr(0, eq);
                const std::string val = kv.substr(eq + 1);
                if (!ModelProfile::valid_key(key)) {
                    fprintf(stderr, "error: unknown profile key '%s' (valid: n_ctx ngl n_threads temp top_k top_p repeat_penalty flash_attn mtp type_k type_v system_prompt)\n", key.c_str());
                    return 1;
                }
                if (!parse_profile_value(key, val, e->profile)) {
                    fprintf(stderr, "error: invalid value '%s' for '%s'\n", val.c_str(), key.c_str());
                    return 1;
                }
                changed = true;
            }
            if (i == before) {
                fprintf(stderr, "usage: anvil profile %s set key=value ...\n", name.c_str());
                return 1;
            }
        } else if (a == "unset") {
            show = false;
            const size_t before = i;
            while (i + 1 < args.size() && !is_cmd(args[i + 1])) {
                const std::string key = args[++i];
                if (!ModelProfile::valid_key(key)) {
                    fprintf(stderr, "error: unknown profile key '%s'\n", key.c_str());
                    return 1;
                }
                if (e->profile.unset(key)) changed = true;
            }
            if (i == before) {
                fprintf(stderr, "usage: anvil profile %s unset key ...\n", name.c_str());
                return 1;
            }
        } else if (a == "reset") {
            show = false;
            e->profile.clear();
            changed = true;
        } else {
            fprintf(stderr, "error: unknown profile command '%s' (try: set, unset, reset)\n", a.c_str());
            return 1;
        }
    }

    if (show) {
        printf("Profile: %s  (path: %s)\n", e->name.c_str(), e->path.c_str());
        if (e->profile.empty()) {
            printf("  (empty — inherits global config defaults)\n");
        } else {
            for (const auto & [k, v] : e->profile.settings) {
                std::string s;
                if (v.is_string())            s = v.get<std::string>();
                else if (v.is_number_float()) { char b[32]; std::snprintf(b, sizeof(b), "%.4g", v.get<double>()); s = b; }
                else if (v.is_boolean())       s = v.get<bool>() ? "true" : "false";
                else                          s = v.dump();
                printf("  %-18s = %s\n", k.c_str(), s.c_str());
            }
        }
        return 0;
    }
    if (changed) {
        if (!save_models(models)) return 1;
        printf("Profile updated for '%s'.\n", e->name.c_str());
    }
    return 0;
}

// ──── src/pull.hpp ────
int cmd_pull(const std::vector<std::string> & args);

// ──── src/pull.cpp ────
// Model downloader: Ollama registry (OCI-style) + import from a local ollama
// install. HTTP goes through curl (wget fallback) exactly like install.sh — no
// new link-time dependencies, so the single binary stays dependency-free.
//
// Ollama models ARE GGUF: the `application/vnd.ollama.image.model` layer is
// the weights file. "Converting" from ollama therefore means extracting the
// GGUF blob plus its metadata layers (chat template, params, license) into a
// persistent anvil profile.

namespace {

constexpr const char * OLLAMA_BASE = "https://registry.ollama.ai/v2";

// Registry components are strictly [A-Za-z0-9._-]; validating before building
// shell commands keeps the curl/system() invocation injection-safe on every OS.
bool valid_registry_component(const std::string & s) {
    if (s.empty() || s.size() > 128) return false;
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (!(std::isalnum(c) || c == '.' || c == '-' || c == '_')) return false;
    }
    return true;
}

// Unique temp path for capturing command output (portable, no mkstemp).
std::string pull_tmp_path() {
    static std::atomic<unsigned> counter{0};
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = config_dir();
    return (dir / ("anvil-cmd-" + std::to_string(counter.fetch_add(1)) +
                   "-" + std::to_string(std::chrono::steady_clock::now()
                                            .time_since_epoch().count()) + ".tmp")).string();
}

// Runs a command and returns its stdout (empty on failure).
std::string capture(const std::string & cmd) {
    const std::string tmp = pull_tmp_path();
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    const std::string full = cmd + " > \"" + tmp + "\" 2>/dev/null";
    std::string out;
    if (system(full.c_str()) == 0) {
        std::ifstream f(tmp, std::ios::binary);
        out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    std::filesystem::remove(tmp, ec);
    return out;
}

// curl with a wget fallback, returning the response body. --max-time bounds
// metadata fetches so a stalled endpoint fails fast instead of hanging.
std::string http_get(const std::string & url, const std::string & extra_flags = "") {
    std::string out = capture("curl -fsSL --max-time 60 " + extra_flags + " \"" + url + "\"");
    if (out.empty()) {
        out = capture("wget -qO- --timeout=60 \"" + url + "\"");
    }
    return out;
}

// Downloads url to out_path atomically: writes <out>.part (resumable with
// --continue-at), verifies the caller's checksum later, then renames. Returns
// 0 on success; the .part is kept on failure so retries resume.
// expected_size > 0 guards against a stale .part from an aborted run of a
// *different* asset (e.g. a tag re-uploaded): a partial larger than the
// target is corrupt, so it is discarded and the download restarts.
int http_download(const std::string & url, const std::string & out_path,
                  uint64_t expected_size = 0) {
    const std::string part = out_path + ".part";
    if (expected_size > 0) {
        std::error_code st;
        const uint64_t cur = std::filesystem::file_size(part, st);
        if (!st && cur > expected_size) {
            fprintf(stderr, "  stale partial (%llu B > expected %llu B); restarting\n",
                    static_cast<unsigned long long>(cur),
                    static_cast<unsigned long long>(expected_size));
            std::filesystem::remove(part, st);
        }
    }
    // --retry/--connect-timeout keep flaky or stalled connections from hanging
    // a multi-GB download indefinitely.
    std::string cmd = "curl --fail --location --progress-bar --retry 3 --retry-delay 2 "
                      "--connect-timeout 20 --continue-at - -o \"" +
                      part + "\" \"" + url + "\"";
    const int rc = system(cmd.c_str());
    if (rc != 0) {
        fprintf(stderr, "\033[31mpull failed (curl exit %d). Partial download kept at %s — retry to resume.\033[0m\n",
                rc, part.c_str());
        return -1;
    }
    std::error_code ec;
    std::filesystem::rename(part, out_path, ec);
    if (ec) {
        fprintf(stderr, "\033[31merror: could not finalize %s (%s)\033[0m\n",
                out_path.c_str(), ec.message().c_str());
        return -1;
    }
    return 0;
}

// sha256 hex of a file, via sha256sum or shasum (install.sh conventions).
// Empty string when no tool is available.
std::string sha256_of(const std::string & path) {
    std::string hex = capture("sha256sum \"" + path + "\" | awk '{print $1}'");
    while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) hex.pop_back();
    if (!hex.empty()) return hex;
    hex = capture("shasum -a 256 \"" + path + "\" | awk '{print $1}'");
    while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) hex.pop_back();
    return hex;
}

struct OllamaLayer {
    std::string media_type;
    std::string digest;
    uint64_t    size = 0;
};

struct OllamaManifest {
    std::vector<OllamaLayer> layers;
    uint64_t total_size = 0;
};

bool parse_ollama_manifest(const std::string & json, OllamaManifest & out) {
    try {
        const nlohmann::json j = nlohmann::json::parse(json);
        if (!j.contains("layers") || !j["layers"].is_array()) return false;
        for (const auto & l : j["layers"]) {
            OllamaLayer layer;
            if (l.contains("mediaType")) layer.media_type = l["mediaType"].get<std::string>();
            if (l.contains("digest"))    layer.digest    = l["digest"].get<std::string>();
            if (l.contains("size"))      layer.size      = l["size"].get<uint64_t>();
            out.layers.push_back(std::move(layer));
        }
        return !out.layers.empty();
    } catch (const nlohmann::json::exception &) {
        return false;
    }
}

// Maps Ollama params blob keys onto the model profile (in-range only, same
// bounds as the CLI). The rest of the blob is kept raw in params_json.
void apply_ollama_params(const nlohmann::json & params, ModelProfile & p) {
    if (!params.is_object()) return;
    auto num = [&](const char * k, float & out) -> bool {
        if (!params.contains(k) || !params[k].is_number()) return false;
        out = params[k].get<float>();
        return true;
    };
    float f = 0.0f;
    if (num("temperature", f) && f >= 0.0f && f <= MAX_TEMP)     p.set("temp", f);
    if (num("top_p", f) && f >= 0.0f && f <= 1.0f)               p.set("top_p", f);
    if (num("repeat_penalty", f) && f >= 1.0f && f <= 100.0f)    p.set("repeat_penalty", f);
    if (num("top_k", f)) {
        const int iv = static_cast<int>(f);
        if (iv >= 0 && iv <= MAX_TOP_K) p.set("top_k", iv);
    }
    if (num("num_ctx", f)) {
        const int iv = static_cast<int>(f);
        if (iv >= 1 && iv <= MAX_CTX) p.set("n_ctx", iv);
    }
}

// Splits `[ns/]name[:tag]` (ollama/OCI convention), validating every part.
// Returns false on invalid input.
bool parse_ollama_spec(const std::string & spec,
                       std::string & ns, std::string & name, std::string & tag) {
    ns = "library";
    name = spec;
    tag = "latest";
    const size_t slash = spec.find('/');
    if (slash != std::string::npos) {
        ns = spec.substr(0, slash);
        name = spec.substr(slash + 1);
    }
    const size_t colon = name.find(':');
    if (colon != std::string::npos) {
        tag = name.substr(colon + 1);
        name = name.substr(0, colon);
    }
    return valid_registry_component(ns) && valid_registry_component(name) &&
           valid_registry_component(tag);
}

// Registers a pulled/imported model, refreshing GGUF metadata from disk.
int register_pulled(const std::string & friendly, const std::string & path,
                    const std::string & source, const std::string & source_id,
                    const std::string & template_text, const std::string & license,
                    const std::string & params_json) {
    std::vector<ModelEntry> models = load_models();
    if (find_model(models, friendly)) {
        fprintf(stderr, "\033[31merror: a model named '%s' is already registered (anvil rm %s to replace)\033[0m\n",
                friendly.c_str(), friendly.c_str());
        return 1;
    }
    const ModelMeta meta = read_model_meta(path);
    if (meta.desc.empty() && meta.trained_ctx <= 0) {
        fprintf(stderr, "\033[33mwarning: could not read GGUF metadata from %s (file may be incomplete)\033[0m\n",
                path.c_str());
    }
    ModelEntry e;
    e.name = friendly;
    e.path = path;
    e.source = source;
    e.source_id = source_id;
    e.added = now_iso();
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    e.size_bytes = ec ? 0 : sz;
    e.desc = meta.desc;
    e.trained_ctx = meta.trained_ctx;
    e.gguf_meta = meta.gguf_meta;
    e.chat_template = template_text;
    e.license = license;
    e.params_json = params_json;
    // Ollama params blob -> persistent profile defaults (same CLI bounds).
    try {
        if (!params_json.empty()) apply_ollama_params(nlohmann::json::parse(params_json), e.profile);
    } catch (const nlohmann::json::exception &) {}
    // Clamp a manifest-claimed num_ctx to the model's trained context: a
    // Modelfile can request 131072 on a 4k-trained model, which would just
    // waste KV memory (or fail to allocate). Also bounded by the global cap.
    if (e.trained_ctx > 0) {
        auto it = e.profile.settings.find("n_ctx");
        if (it != e.profile.settings.end() && it->second.is_number_integer()) {
            const int64_t nctx = it->second.get<int64_t>();
            const int64_t cap  = std::min<int64_t>(e.trained_ctx, MAX_CTX);
            if (nctx > cap) {
                fprintf(stderr, "\033[33mwarning: requested n_ctx=%lld exceeds trained context %lld; clamping to %lld\033[0m\n",
                        static_cast<long long>(nctx), static_cast<long long>(e.trained_ctx),
                        static_cast<long long>(cap));
                it->second = cap;
            }
        }
    }
    models.push_back(std::move(e));
    if (!save_models(models)) return 1;
    printf("\033[32mRegistered '%s' (%s) -> %s\033[0m\n", friendly.c_str(),
           format_size(e.size_bytes).c_str(), path.c_str());
    if (!e.desc.empty())      printf("  arch       : %s\n", e.desc.c_str());
    if (e.trained_ctx > 0)    printf("  trained ctx: %lld\n", static_cast<long long>(e.trained_ctx));
    if (!e.chat_template.empty()) printf("  template   : %zu bytes (from ollama metadata)\n", e.chat_template.size());
    if (!e.profile.empty()) {
        printf("  profile    : ");
        bool first = true;
        for (const auto & [k, v] : e.profile.settings) {
            std::string s;
            if (v.is_string())            s = v.get<std::string>();
            else if (v.is_number_float()) { char b[32]; std::snprintf(b, sizeof(b), "%.4g", v.get<double>()); s = b; }
            else if (v.is_boolean())       s = v.get<bool>() ? "true" : "false";
            else                          s = v.dump();
            printf("%s%s=%s", first ? "" : " ", k.c_str(), s.c_str());
            first = false;
        }
        printf(" (from ollama params)\n");
    }
    printf("  run        : anvil run %s\n", friendly.c_str());
    return 0;
}

// Verifies a downloaded blob against its registry digest (sha256:<hex>).
bool verify_digest(const std::string & path, const std::string & digest) {
    if (digest.compare(0, 7, "sha256:") != 0) return true;   // unknown scheme: skip
    const std::string expected = digest.substr(7);
    const std::string actual = sha256_of(path);
    if (actual.empty()) {
        fprintf(stderr, "\033[33mwarning: no sha256 tool available; skipping verification\033[0m\n");
        return true;
    }
    if (actual != expected) {
        fprintf(stderr, "\033[31mchecksum mismatch for %s (got %s, expected %s)\033[0m\n",
                path.c_str(), actual.c_str(), expected.c_str());
        return false;
    }
    printf("  verified sha256:%s\n", expected.c_str());
    return true;
}

// Pulls one blob from the registry into models_dir. Returns 0 on success.
// A checksum mismatch discards the corrupt file (final + .part) and retries
// once from scratch — a resumed .part can be corrupt even when curl reports
// success, and resuming from the bad bytes would loop forever.
int pull_blob(const std::string & ns, const std::string & model, const OllamaLayer & layer,
              const std::string & dest) {
    const std::string url = std::string(OLLAMA_BASE) + "/" + ns + "/" + model + "/blobs/" + layer.digest;
    printf("  %-28s %s\n", layer.media_type.c_str(), format_size(layer.size).c_str());
    for (int attempt = 0; attempt < 2; attempt++) {
        if (http_download(url, dest, layer.size) != 0) return -1;
        if (verify_digest(dest, layer.digest)) return 0;
        std::error_code ec;
        std::filesystem::remove(dest, ec);
        std::filesystem::remove(dest + ".part", ec);
        if (attempt == 0) fprintf(stderr, "  checksum mismatch; retrying from scratch...\n");
    }
    fprintf(stderr, "\033[31merror: checksum still mismatched after retry; giving up\033[0m\n");
    return -1;
}

} // namespace

// anvil pull ollama:[ns/]name[:tag]  |  ollama-local:[ns/]name[:tag]
int cmd_pull(const std::vector<std::string> & args) {
    if (args.empty()) {
        fprintf(stderr, "usage: anvil pull ollama:<name>[:tag] | ollama-local:<name>[:tag] | hf:<repo>[:file]\n");
        return 1;
    }
    if (args.size() > 1) {
        fprintf(stderr, "error: unexpected argument '%s'\n", args[1].c_str());
        return 1;
    }

    // Source prefix: ollama (registry) / ollama-local (~/.ollama) / hf (phase 3).
    std::string source = "ollama";
    std::string rest = args[0];
    const size_t colon = rest.find(':');
    if (colon != std::string::npos) {
        const std::string prefix = rest.substr(0, colon);
        if (prefix == "ollama" || prefix == "ollama-local") {
            source = prefix;
            rest = rest.substr(colon + 1);
        }
    }
    if (source == "hf") {
        fprintf(stderr, "error: HuggingFace pull is not built yet (coming in the next update).\n");
        return 1;
    }

    std::string ns, name, tag;
    if (!parse_ollama_spec(rest, ns, name, tag)) {
        fprintf(stderr, "error: invalid model spec '%s' (expected [ns/]name[:tag])\n", rest.c_str());
        return 1;
    }
    const std::string friendly = slugify(name + "-" + tag);
    const std::string source_id = ns + "/" + name + ":" + tag;

    OllamaManifest manifest;
    if (source == "ollama") {
        const std::string url = std::string(OLLAMA_BASE) + "/" + ns + "/" + name + "/manifests/" + tag;
        printf("Pulling from Ollama registry: %s\n", source_id.c_str());
        const std::string body = http_get(url, "-H \"Accept: application/vnd.docker.distribution.manifest.v2+json\"");
        if (body.empty() || !parse_ollama_manifest(body, manifest)) {
            fprintf(stderr, "\033[31merror: model '%s' not found in the Ollama registry (or network issue)\033[0m\n",
                    source_id.c_str());
            return 1;
        }
    } else {  // ollama-local: import an already-pulled ollama install.
        // Respect OLLAMA_MODELS (ollama's own store override); fall back to
        // the default ~/.ollama/models location.
        std::string base;
        const char * om = std::getenv("OLLAMA_MODELS");
        if (om && om[0]) {
            base = om;
            while (base.size() > 1 && base.back() == '/') base.pop_back();
        } else {
            const char * home = std::getenv("HOME");
#ifdef _WIN32
            if (!home || !home[0]) home = std::getenv("USERPROFILE");
#endif
            if (!home || !home[0]) home = ".";
            base = std::string(home) + "/.ollama/models";
        }
        const std::string mp = base + "/manifests/registry.ollama.ai/" + ns + "/" + name + "/" + tag;
        printf("Importing from local ollama install: %s\n", source_id.c_str());
        std::ifstream mf(mp);
        if (!mf) {
            fprintf(stderr, "\033[31merror: no local ollama model at %s (did you 'ollama pull %s' first?)\033[0m\n",
                    mp.c_str(), source_id.c_str());
            return 1;
        }
        const std::string body((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
        if (!parse_ollama_manifest(body, manifest)) {
            fprintf(stderr, "\033[31merror: unreadable manifest at %s\033[0m\n", mp.c_str());
            return 1;
        }
        // Resolve layer paths to blob files (blobs are named sha256-<hex>).
        for (auto & layer : manifest.layers) {
            std::string fname = layer.digest;
            std::replace(fname.begin(), fname.end(), ':', '-');
            layer.digest = base + "/blobs/" + fname;   // digest field now holds the path
        }
    }

    // Separate layers by media type.
    const OllamaLayer * model_layer = nullptr;
    std::vector<const OllamaLayer *> template_layers, params_layers, license_layers;
    for (const auto & layer : manifest.layers) {
        if (layer.media_type == "application/vnd.ollama.image.model") {
            if (model_layer) {
                fprintf(stderr, "\033[31merror: sharded models (multiple weight layers) are not supported yet\033[0m\n");
                return 1;
            }
            model_layer = &layer;
        } else if (layer.media_type == "application/vnd.ollama.image.template") template_layers.push_back(&layer);
        else if (layer.media_type == "application/vnd.ollama.image.params")  params_layers.push_back(&layer);
        else if (layer.media_type == "application/vnd.ollama.image.license") license_layers.push_back(&layer);
    }
    if (!model_layer) {
        fprintf(stderr, "\033[31merror: manifest has no GGUF model layer\033[0m\n");
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(models_dir(), ec);
    if (ec) {
        fprintf(stderr, "\033[31merror: could not create %s\033[0m\n", models_dir().c_str());
        return 1;
    }
    std::string model_path;
    if (source == "ollama") {
        model_path = models_dir() + "/" + friendly + ".gguf";
        if (std::filesystem::exists(model_path)) {
            fprintf(stderr, "\033[31merror: model file already exists at %s (anvil rm %s --yes to replace)\033[0m\n",
                    model_path.c_str(), friendly.c_str());
            return 1;
        }
        printf("Downloading %s (%s)...\n", friendly.c_str(), format_size(model_layer->size).c_str());
        if (pull_blob(ns, name, *model_layer, model_path) != 0) return 1;
    } else {
        // Prefer a hardlink into anvil's own store: zero extra disk (same
        // filesystem) and the model survives an ollama uninstall. Fall back to
        // referencing the blob directly when the stores span filesystems.
        const std::string linked = models_dir() + "/" + friendly + ".gguf";
        std::error_code lk;
        std::filesystem::remove(linked, lk);   // stale target from an aborted import
        lk.clear();
        std::filesystem::create_hard_link(model_layer->digest, linked, lk);
        if (!lk) {
            model_path = linked;
            printf("Linked local blob into anvil store: %s\n", linked.c_str());
        } else {
            model_path = model_layer->digest;   // direct blob reference, zero disk cost
            fprintf(stderr, "\033[33mwarning: could not hardlink blob into anvil store (%s); "
                            "referencing directly — deleting the ollama store will break this model\033[0m\n",
                    lk.message().c_str());
        }
    }

    // Metadata layers.
    std::string template_text, license_text, params_json;
    auto read_layer_text = [&](const OllamaLayer & layer) -> std::string {
        if (source == "ollama") {
            const std::string tmp = pull_tmp_path();
            std::error_code e2;
            if (http_download(std::string(OLLAMA_BASE) + "/" + ns + "/" + name + "/blobs/" + layer.digest, tmp) != 0)
                return "";
            std::ifstream f(tmp, std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::filesystem::remove(tmp, e2);
            return s;
        }
        std::ifstream f(layer.digest, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };
    if (!template_layers.empty()) template_text = read_layer_text(*template_layers[0]);
    if (!params_layers.empty())   params_json  = read_layer_text(*params_layers[0]);
    if (!license_layers.empty())  license_text = read_layer_text(*license_layers[0]);

    return register_pulled(friendly, model_path, source == "ollama" ? "ollama" : "ollama-local",
                           source_id, template_text, license_text, params_json);
}
// ──── src/hardware.hpp ────


HWInfo probe_hw();
int derive_ngl(const HWInfo & hw);
bool validate_gguf(const std::string & path);
// ──── src/hardware.cpp ────


#ifdef __APPLE__
#endif
#ifdef __linux__
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef _MSC_VER
#pragma comment(lib, "dxgi.lib")
#endif
#endif

#ifdef __APPLE__
static void detect_gpus_macos(HWInfo & hw) {
    CFMutableDictionaryRef matching = IOServiceMatching("IOGPU");
    if (!matching) return;

    mach_port_t mp = MACH_PORT_NULL;
    const kern_return_t main_ret = IOMainPort(MACH_PORT_NULL, &mp);
    if (main_ret != KERN_SUCCESS) { CFRelease(matching); return; }

    io_iterator_t iter = 0;
    const kern_return_t kr = IOServiceGetMatchingServices(mp, matching, &iter);
    if (kr != KERN_SUCCESS) return;  // matching consumed by the call

    io_object_t device;
    while ((device = IOIteratorNext(iter)) != 0) {
        GPUInfo gpu;
        gpu.vendor = "Apple";
        gpu.is_discrete = false;
        gpu.vram_mb = 0;
        gpu.name = "Apple GPU (unified " +
                   std::to_string(hw.ram_bytes / (1024ULL * 1024 * 1024)) + " GB)";
        hw.gpus.push_back(gpu);
        IOObjectRelease(device);
    }
    IOObjectRelease(iter);
}
#endif

#ifdef __linux__
static std::string run_cmd(const std::string & cmd) {
    std::string out;
    FILE * pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    char buf[4096];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        out.append(buf, n);
    }
    pclose(pipe);
    return out;
}

static void detect_gpus_linux(HWInfo & hw) {
    // nvidia-smi (when the proprietary driver is present)
    const std::string out = run_cmd(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null");
    if (!out.empty()) {
        std::istringstream ss(out);
        std::string line;
        while (std::getline(ss, line)) {
            while (!line.empty() && line.back() <= ' ') line.pop_back();
            if (line.empty()) continue;
            GPUInfo gpu;
            gpu.vendor = "NVIDIA";
            gpu.is_discrete = true;

            const auto comma = line.rfind(',');
            if (comma != std::string::npos) {
                std::string vram_str = line.substr(comma + 1);
                while (!vram_str.empty() && vram_str.back() <= ' ') vram_str.pop_back();
                while (!vram_str.empty() && vram_str.front() == ' ') vram_str.erase(vram_str.begin());
                uint64_t vram = 0;
                if (parse_uint64(vram_str, vram)) gpu.vram_mb = vram;
                gpu.name = line.substr(0, comma);
                while (!gpu.name.empty() && gpu.name.back() == ' ') gpu.name.pop_back();
            } else {
                gpu.name = line;
            }
            hw.gpus.push_back(gpu);
        }
    }

    // sysfs PCI enumeration (works with any driver, incl. nouveau/amdgpu)
    glob_t globbuf;
    if (glob("/sys/class/drm/card*/device/vendor", 0, nullptr, &globbuf) == 0) {
        for (size_t i = 0; i < globbuf.gl_pathc; i++) {
            std::ifstream vf(globbuf.gl_pathv[i]);
            std::string vendor_id;
            std::getline(vf, vendor_id);
            while (!vendor_id.empty() && vendor_id.back() <= ' ') vendor_id.pop_back();

            std::string base(globbuf.gl_pathv[i]);
            const auto pos = base.rfind("/device/vendor");
            if (pos == std::string::npos) continue;
            base = base.substr(0, pos);

            GPUInfo gpu;
            if (vendor_id == "0x1002") {
                gpu.vendor = "AMD";
                gpu.is_discrete = true;
            } else if (vendor_id == "0x8086") {
                gpu.vendor = "Intel";
                gpu.is_discrete = false;
            } else {
                continue;  // vendor card* entries may duplicate; skip non-GPU vendors
            }
            std::ifstream df(base + "/device/device");
            std::string did;
            std::getline(df, did);
            gpu.name = gpu.vendor + " GPU (" + did + ")";

            std::ifstream vf2(base + "/device/mem_info_vram_total");
            if (vf2.good()) {
                uint64_t bytes = 0;
                vf2 >> bytes;
                gpu.vram_mb = bytes / (1024 * 1024);
            }
            hw.gpus.push_back(gpu);
        }
        globfree(&globbuf);
    }
}
#endif

#ifdef _WIN32
static void detect_gpus_windows(HWInfo & hw) {
    IDXGIFactory * factory = nullptr;
    if (CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void **>(&factory)) != S_OK) return;
    IDXGIAdapter * adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC desc;
        if (adapter->GetDesc(&desc) == S_OK) {
            GPUInfo gpu;
            const int needed = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, nullptr, 0, nullptr, nullptr);
            if (needed > 1) {
                std::string name_buf(static_cast<size_t>(needed), '\0');
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf.data(), needed, nullptr, nullptr);
                if (!name_buf.empty() && name_buf.back() == '\0') name_buf.pop_back();
                gpu.name = std::move(name_buf);
            } else {
                gpu.name = "Unknown";
            }
            gpu.vram_mb = std::max(desc.DedicatedVideoMemory, desc.SharedSystemMemory) / (1024 * 1024);
            gpu.is_discrete = (desc.VendorId == 0x10DE || desc.VendorId == 0x1002);
            if (desc.VendorId == 0x10DE)      gpu.vendor = "NVIDIA";
            else if (desc.VendorId == 0x1002) gpu.vendor = "AMD";
            else if (desc.VendorId == 0x8086) gpu.vendor = "Intel";
            else                              gpu.vendor = "Other";
            hw.gpus.push_back(gpu);
        }
        adapter->Release();
    }
    factory->Release();
}
#endif

HWInfo probe_hw() {
    HWInfo hw;

#if defined(__APPLE__)
    hw.os = "macos";
#elif defined(__linux__)
    hw.os = "linux";
#elif defined(_WIN32)
    hw.os = "windows";
#else
    hw.os = "unknown";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    hw.arch = "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    hw.arch = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    hw.arch = "i386";
#else
    hw.arch = "unknown";
#endif

    hw.cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hw.cpu_threads <= 0) hw.cpu_threads = 0;

#ifdef __APPLE__
    {
        std::string buf;
        size_t len = 0;
        if (sysctlbyname("machdep.cpu.brand_string", nullptr, &len, nullptr, 0) == 0 && len > 0) {
            buf.resize(len);
            if (sysctlbyname("machdep.cpu.brand_string", &buf[0], &len, nullptr, 0) == 0) {
                while (!buf.empty() && buf.back() == '\0') buf.pop_back();
                hw.cpu = buf;
                hw.apple_silicon = (hw.cpu.find("Apple") != std::string::npos);
            }
        }
    }
    {
        uint64_t ram = 0;
        size_t len = sizeof(ram);
        if (sysctlbyname("hw.memsize", &ram, &len, nullptr, 0) == 0) {
            hw.ram_bytes = ram;
        }
    }
    detect_gpus_macos(hw);
#elif defined(__linux__)
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("model name") != std::string::npos) {
                const auto pos = line.find(':');
                if (pos != std::string::npos) {
                    hw.cpu = line.substr(pos + 2);
                    while (!hw.cpu.empty() && hw.cpu.back() <= ' ') hw.cpu.pop_back();
                }
                break;
            }
        }
    }
    {
        std::ifstream f("/proc/meminfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("MemTotal") != std::string::npos) {
                std::istringstream iss(line);
                std::string key;
                uint64_t kb = 0;
                std::string unit;
                if ((iss >> key >> kb >> unit) && key == "MemTotal" && unit == "kB") {
                    hw.ram_bytes = kb * 1024;
                }
                break;
            }
        }
    }
    detect_gpus_linux(hw);
#elif defined(_WIN32)
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        hw.arch = (si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64) ? "aarch64" : "x86_64";
        hw.cpu_threads = static_cast<int>(si.dwNumberOfProcessors);
        hw.cpu = "Windows CPU";
    }
    {
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            hw.ram_bytes = ms.ullTotalPhys;
        }
    }
    detect_gpus_windows(hw);
#endif
    return hw;
}

int derive_ngl(const HWInfo & hw) {
    // Apple Silicon and any discrete GPU: offload everything (-1 = auto).
    if (hw.apple_silicon) return -1;
    if (!hw.gpus.empty()) return -1;
    return 0;
}

bool validate_gguf(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4] = {0, 0, 0, 0};
    f.read(magic, 4);
    return f.gcount() == 4 && memcmp(magic, "GGUF", 4) == 0;
}
// ──── src/setup.hpp ────


AnvilConfig run_setup_tui(const HWInfo & hw, int max_ctx);
// ──── src/setup.cpp ────



AnvilConfig run_setup_tui(const HWInfo & hw, int max_ctx) {
    AnvilConfig cfg;
    cfg.ngl = derive_ngl(hw);
    cfg.n_ctx = max_ctx > 0 ? max_ctx : 0;

    // Context options: powers of two up to the model's trained context.
    std::vector<int> ctx_options;
    for (int c = 1; c > 0 && c <= max_ctx && c <= INT32_MAX / 2; c *= 2) {
        ctx_options.push_back(c);
    }
    ctx_options.push_back(0);  // "Custom..."
    const int custom_idx = static_cast<int>(ctx_options.size()) - 1;
    int ctx_sel = 0;
    for (size_t i = 0; i < ctx_options.size(); i++) {
        if (ctx_options[i] == cfg.n_ctx) { ctx_sel = static_cast<int>(i); break; }
    }

    std::vector<std::string> ctx_labels;
    for (const int c : ctx_options) {
        if (c == 0) ctx_labels.push_back("Custom...");
        else if (c >= 1024) ctx_labels.push_back(std::to_string(c / 1024) + "K tokens");
        else ctx_labels.push_back(std::to_string(c) + " tokens");
    }

    std::vector<std::string> kv_preset_labels;
    for (int i = 0; i < KV_PRESETS_COUNT; i++) kv_preset_labels.push_back(KV_PRESETS[i].label);

    std::vector<std::string> kv_type_labels;
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) kv_type_labels.push_back(KV_OPTIONS[i].label);

    const std::vector<std::string> fa_labels   = {"on (recommended)", "off"};
    const std::vector<std::string> temp_labels = {"0.7 (focused)", "0.8 (balanced)", "0.9 (creative)", "1.0 (wild)"};

    int kv_preset_sel = 0;
    int kv_k_sel      = 1;
    int kv_v_sel      = 3;
    int fa_sel        = 0;
    int temp_sel      = 1;
    bool custom_kv    = false;
    int menu_idx      = 0;

    std::vector<std::string> hw_lines;
    hw_lines.push_back("CPU  : " + hw.cpu);
    hw_lines.push_back("RAM  : " + std::to_string(hw.ram_bytes / (1024ULL * 1024 * 1024)) + " GB");
    hw_lines.push_back("Threads: " + std::to_string(hw.cpu_threads));
    for (const auto & gpu : hw.gpus) {
        hw_lines.push_back("GPU  : " + gpu.name + " (" + std::to_string(gpu.vram_mb) + " MB)");
    }

    auto screen = ftxui::ScreenInteractive::FitComponent();
    auto component = ftxui::Renderer([&]() {
        ftxui::Elements rows;
        rows.push_back(ftxui::text(ANVIL_LOGO) | ftxui::bold | ftxui::color(ftxui::Color::Yellow));
        rows.push_back(ftxui::text("  First-time setup  v" + std::string(ANVIL_VERSION)) | ftxui::bold | ftxui::color(ftxui::Color::Cyan));
        rows.push_back(ftxui::text(" "));
        rows.push_back(ftxui::text("  Detected hardware:") | ftxui::bold);
        for (const auto & line : hw_lines) rows.push_back(ftxui::text("    " + line));
        rows.push_back(ftxui::text(" "));

        auto render_setting = [&](const std::string & label, const std::vector<std::string> & opts,
                                  int sel, int idx, int indent = 2) {
            const std::string pad(static_cast<size_t>(indent), ' ');
            const bool active = (idx == menu_idx);
            auto lbl = ftxui::text(pad + label);
            if (active) lbl = lbl | ftxui::bold | ftxui::color(ftxui::Color::Cyan);
            else        lbl = lbl | ftxui::bold;
            rows.push_back(lbl);
            for (int i = 0; i < static_cast<int>(opts.size()); i++) {
                const bool selected = (i == sel);
                auto prefix = selected
                    ? ftxui::text(pad + "  ▸ ") | ftxui::bold | ftxui::color(ftxui::Color::Green)
                    : ftxui::text(pad + "  · ");
                auto label_el = ftxui::text(opts[static_cast<size_t>(i)]);
                if (selected && active) label_el = label_el | ftxui::bold | ftxui::color(ftxui::Color::Green);
                else if (selected)      label_el = label_el | ftxui::color(ftxui::Color::Green);
                rows.push_back(ftxui::hbox({prefix, label_el}));
            }
            rows.push_back(ftxui::text(" "));
        };

        render_setting("Context size", ctx_labels, ctx_sel, 0);
        render_setting("KV cache compression", kv_preset_labels, kv_preset_sel, 1);
        if (custom_kv) {
            render_setting("  K cache type", kv_type_labels, kv_k_sel, 4, 4);
            render_setting("  V cache type", kv_type_labels, kv_v_sel, 5, 4);
        }
        render_setting("Flash attention", fa_labels, fa_sel, 2);
        render_setting("Temperature", temp_labels, temp_sel, 3);

        std::string kv_summary;
        if (!custom_kv) {
            const int ki = KV_PRESETS[kv_preset_sel].k_idx;
            const int vi = KV_PRESETS[kv_preset_sel].v_idx;
            if (ki >= 0 && vi >= 0)
                kv_summary = "  Active: K=" + std::string(KV_OPTIONS[ki].short_name) +
                             " V=" + std::string(KV_OPTIONS[vi].short_name);
        } else {
            kv_summary = "  Active: K=" + std::string(KV_OPTIONS[kv_k_sel].short_name) +
                         " V=" + std::string(KV_OPTIONS[kv_v_sel].short_name);
        }
        rows.push_back(ftxui::text(kv_summary) | ftxui::dim);
        rows.push_back(ftxui::text(" "));
        rows.push_back(ftxui::text("  Tab switch  ↑/↓ change  Enter confirm  q cancel") | ftxui::dim);

        return ftxui::vbox(std::move(rows));
    });

    auto wrapped = component | ftxui::CatchEvent([&](ftxui::Event e) {
        if (e == ftxui::Event::Character('q')) { screen.Exit(); return true; }
        if (e == ftxui::Event::Return) { screen.Exit(); return true; }

        const int menu_count = custom_kv ? 6 : 4;
        if (e == ftxui::Event::Tab) {
            menu_idx = (menu_idx + 1) % menu_count;
            if (!custom_kv && menu_idx >= 4) menu_idx = 0;
            return true;
        }
        if (e == ftxui::Event::TabReverse) {
            menu_idx = (menu_idx - 1 + menu_count) % menu_count;
            if (!custom_kv && menu_idx >= 4) menu_idx = 3;
            return true;
        }
        auto cycle = [](int & sel, int count, int dir) {
            sel = (sel + dir + count) % count;
        };
        if (e == ftxui::Event::ArrowUp) {
            switch (menu_idx) {
                case 0: cycle(ctx_sel, static_cast<int>(ctx_labels.size()), -1); break;
                case 1:
                    cycle(kv_preset_sel, KV_PRESETS_COUNT, -1);
                    custom_kv = (kv_preset_sel == KV_PRESETS_COUNT - 1);
                    break;
                case 2: cycle(fa_sel, static_cast<int>(fa_labels.size()), -1); break;
                case 3: cycle(temp_sel, static_cast<int>(temp_labels.size()), -1); break;
                case 4: cycle(kv_k_sel, KV_OPTIONS_COUNT, -1); break;
                case 5: cycle(kv_v_sel, KV_OPTIONS_COUNT, -1); break;
                default: break;
            }
            return true;
        }
        if (e == ftxui::Event::ArrowDown) {
            switch (menu_idx) {
                case 0: cycle(ctx_sel, static_cast<int>(ctx_labels.size()), 1); break;
                case 1:
                    cycle(kv_preset_sel, KV_PRESETS_COUNT, 1);
                    custom_kv = (kv_preset_sel == KV_PRESETS_COUNT - 1);
                    break;
                case 2: cycle(fa_sel, static_cast<int>(fa_labels.size()), 1); break;
                case 3: cycle(temp_sel, static_cast<int>(temp_labels.size()), 1); break;
                case 4: cycle(kv_k_sel, KV_OPTIONS_COUNT, 1); break;
                case 5: cycle(kv_v_sel, KV_OPTIONS_COUNT, 1); break;
                default: break;
            }
            return true;
        }
        return false;
    });

    screen.Loop(wrapped);

    if (ctx_sel == custom_idx) {
        printf("\033[?25h");
        printf("Enter context size (tokens): ");
        fflush(stdout);
        std::string input;
        std::getline(std::cin, input);
        if (!parse_int(input, cfg.n_ctx) || cfg.n_ctx <= 0 || cfg.n_ctx > MAX_CTX) {
            fprintf(stderr, "\033[33mwarning: invalid context size '%s', using auto\033[0m\n", input.c_str());
            cfg.n_ctx = max_ctx > 0 ? max_ctx : 0;
        }
    } else {
        cfg.n_ctx = ctx_options[static_cast<size_t>(ctx_sel)];
    }

    if (custom_kv) {
        cfg.type_k = KV_OPTIONS[kv_k_sel].type;
        cfg.type_v = KV_OPTIONS[kv_v_sel].type;
    } else {
        const int ki = KV_PRESETS[kv_preset_sel].k_idx;
        const int vi = KV_PRESETS[kv_preset_sel].v_idx;
        cfg.type_k = KV_OPTIONS[ki].type;
        cfg.type_v = KV_OPTIONS[vi].type;
    }
    cfg.flash_attn = (fa_sel == 0);
    const float temps[] = {0.7f, 0.8f, 0.9f, 1.0f};
    cfg.temp = temps[temp_sel];

    return cfg;
}
// ──── src/chat.hpp ────


int run_chat(const CliArgs & cli, AnvilConfig cfg, const HWInfo & hw);
// ──── src/chat.cpp ────


// ─── Session export ────────────────────────────────────────────────────────

static std::string session_path() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(sessions_dir(), ec);
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    if (!std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t))) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(t));
    }
    return sessions_dir() + "/" + std::string(buf) + ".md";
}

static void export_session(const std::vector<ChatMessage> & msgs, const std::string & path) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "\033[31mcould not write session to %s\033[0m\n", path.c_str());
        return;
    }
    f << "# Anvil Chat Session\n\n";
    for (const auto & m : msgs) {
        if (m.role == "system")    f << "**[System]** " << m.content << "\n\n";
        else if (m.role == "user") f << "**[You]** " << m.content << "\n\n";
        else                       f << "**[Assistant]** " << m.content << "\n\n";
    }
    fprintf(stderr, "\033[32mSession exported to %s\033[0m\n", path.c_str());
}

// ─── Text helpers ──────────────────────────────────────────────────────────

// Token -> piece with a correctly sized buffer (previous code declared n+1
// bytes of capacity for an n-byte buffer).
static std::string token_to_str(const llama_vocab * vocab, llama_token token) {
    const int n = llama_token_to_piece(vocab, token, nullptr, 0, 0, true);
    if (n <= 0) return "";
    std::string s(static_cast<size_t>(n) + 1, '\0');
    const int written = llama_token_to_piece(vocab, token, s.data(), static_cast<int32_t>(s.size()), 0, true);
    if (written <= 0) return "";
    s.resize(static_cast<size_t>(std::min(written, n)));
    return s;
}

static void print_ctx_bar(int used, int total) {
    if (total <= 0) return;
    const float pct = static_cast<float>(used) / static_cast<float>(total);

    int cols = 0;
    const char * cols_env = getenv("COLUMNS");
    if (cols_env) parse_int(cols_env, cols);
    const int bar_width = cols > 0 ? cols / 2 : 0;
    if (bar_width <= 0) return;

    int filled = static_cast<int>(pct * static_cast<float>(bar_width));
    if (filled > bar_width) filled = bar_width;
    if (filled < 0) filled = 0;

    const char * color;
    if (pct < 0.5f)      color = "\033[32m";
    else if (pct < 0.8f) color = "\033[33m";
    else                 color = "\033[31m";

    fprintf(stderr, "  %sctx [", color);
    for (int i = 0; i < bar_width; i++) fprintf(stderr, i < filled ? "█" : "░");
    fprintf(stderr, "] %d%% (%d/%d)\033[0m\n",
            static_cast<int>(pct * 100.0f), used, total);
}

// ─── Sampler chain ─────────────────────────────────────────────────────────

static llama_sampler * build_sampler_chain(
        const llama_vocab * vocab, const AnvilConfig & cfg,
        bool grammar_active, const std::string & grammar_src) {
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // Order follows upstream llama.cpp: filtering samplers, then penalties,
    // then the token selector, then grammar (last, as it overrides selection).
    if (cfg.top_k > 0) llama_sampler_chain_add(smpl, llama_sampler_init_top_k(cfg.top_k));
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(cfg.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    if (cfg.temp > 0.0f) llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, cfg.repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    if (grammar_active) {
        llama_sampler * g = llama_sampler_init_grammar(vocab, grammar_src.c_str(), "root");
        if (g) llama_sampler_chain_add(smpl, g);
    }
    return smpl;
}

// ─── Tokenize helpers ──────────────────────────────────────────────────────
// Returns a negative count on overflow (INT32_MIN), guarded before negation.

static std::vector<llama_token> tokenize_render(
        const llama_vocab * vocab, const std::string & text) {
    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               nullptr, 0, false, true);
    if (n == INT32_MIN) return {};
    if (n < 0) n = -n;
    if (n <= 0) return {};
    std::vector<llama_token> toks(static_cast<size_t>(n));
    const int32_t m = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                     toks.data(), n, false, true);
    if (m < 0) return {};
    toks.resize(static_cast<size_t>(m));
    return toks;
}

// ─── Chat REPL ─────────────────────────────────────────────────────────────

int run_chat(const CliArgs & cli, AnvilConfig cfg, const HWInfo & hw) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.ngl;
    mparams.use_mmap = true;
    fprintf(stderr, "Loading model: %s ...\n", cli.model.c_str());
    const auto load_start = std::chrono::steady_clock::now();
    LlamaModel model(llama_model_load_from_file(cli.model.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "\033[31merror: failed to load model '%s'\033[0m\n", cli.model.c_str());
        return 1;
    }
    const double load_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_start).count();

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);
    const bool has_encoder = llama_model_has_encoder(model);
    const bool has_decoder = llama_model_has_decoder(model);

    fprintf(stderr, "\n\033[1;36mModel Info:\033[0m\n");
    fprintf(stderr, "  trained ctx : %d tokens\n", n_ctx_train);
    fprintf(stderr, "  requested   : %d tokens\n", cfg.n_ctx);
    fprintf(stderr, "  encoder     : %s\n", has_encoder ? "yes" : "no");
    fprintf(stderr, "  decoder     : %s\n", has_decoder ? "yes" : "no");
    fprintf(stderr, "  load time   : %.2fs\n", load_sec);
    if (cfg.n_ctx > n_ctx_train && n_ctx_train > 0) {
        fprintf(stderr, "\033[33m  ⚠ WARNING: requested ctx (%d) exceeds trained ctx (%d).\033[0m\n",
                cfg.n_ctx, n_ctx_train);
        fprintf(stderr, "  Quality may degrade beyond the trained context length.\n");
    }
    fprintf(stderr, "  flash attn  : %s\n",
            cfg.flash_attn ? "\033[32menabled\033[0m" : "\033[33mdisabled\033[0m (perf will suffer)");
    fprintf(stderr, "  KV cache    : K=\033[32m%s\033[0m V=\033[32m%s\033[0m\n",
            kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));
    if (cfg.mtp) fprintf(stderr, "  MTP         : \033[33menabled\033[0m\n");
    fprintf(stderr, "  sampling    : top_k=%d top_p=%.2f repeat=%.2f temp=%.2f\n",
            cfg.top_k, cfg.top_p, cfg.repeat_penalty, cfg.temp);
    fprintf(stderr, "\n");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = static_cast<uint32_t>(cfg.n_ctx);
    cparams.n_batch   = cparams.n_ctx;
    cparams.n_threads = cfg.n_threads > 0 ? cfg.n_threads : hw.cpu_threads;
    cparams.flash_attn_type = cfg.flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k = cfg.type_k;
    cparams.type_v = cfg.type_v;
    if (cfg.mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    LlamaContext ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        fprintf(stderr, "\033[31merror: failed to create context\033[0m\n");
        return 1;
    }

    // Grammar (read once, kept alive for the whole session).
    bool grammar_active = false;
    std::string grammar_src;
    if (!cli.grammar.empty()) {
        std::ifstream gf(cli.grammar);
        if (!gf) {
            fprintf(stderr, "\033[31merror: cannot open grammar file '%s'\033[0m\n", cli.grammar.c_str());
        } else {
            grammar_src.assign(std::istreambuf_iterator<char>(gf), std::istreambuf_iterator<char>());
            // Validate the grammar parses before building the real chain.
            LlamaSampler probe(llama_sampler_init_grammar(vocab, grammar_src.c_str(), "root"));
            grammar_active = static_cast<bool>(probe);
            if (!grammar_active) {
                fprintf(stderr, "\033[31merror: failed to parse grammar '%s'\033[0m\n", cli.grammar.c_str());
            }
        }
    }

    LlamaSampler smpl(build_sampler_chain(vocab, cfg, grammar_active, grammar_src));
    if (!smpl) {
        fprintf(stderr, "\033[31merror: failed to initialize sampler chain\033[0m\n");
        return 1;
    }
    if (grammar_active) fprintf(stderr, "  grammar     : \033[32m%s\033[0m\n", cli.grammar.c_str());

    printf("\033[1;33m%s\033[0m", ANVIL_LOGO);
    if (cli.friendly.empty()) printf("  model   : %s\n", cli.model.c_str());
    else                      printf("  model   : %s  (%s)\n", cli.friendly.c_str(), cli.model.c_str());
    printf("  backend : GPU layers=%d | flash=%s | threads=%d\n",
           cfg.ngl, cfg.flash_attn ? "on" : "off", cparams.n_threads);
    printf("  ctx     : %u tokens\n", cparams.n_ctx);
    printf("  KV      : K=%s V=%s\n", kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));
    printf("  temp    : %.2f\n", cfg.temp);
    if (cfg.mtp) printf("  spec    : MTP\n");
    if (grammar_active) printf("  grammar : %s\n", cli.grammar.c_str());
    printf("  commands: /exit /clear /stats /undo /export /model /temp <f> /ctx\n\n");

    // Conversation state. history owns every string; the llama_chat_message
    // view is rebuilt fresh for each template call, so no c_str() pointer can
    // ever dangle (fixes the previous role-pointer use-after-free).
    std::vector<ChatMessage> history;
    std::vector<llama_token> prev_tokens;  // tokens currently decoded in the KV
    Utf8Buffer utf8_buf;
    int total_tokens_generated = 0;
    double total_gen_time = 0.0;
    const char * tmpl = llama_model_chat_template(model, nullptr);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const llama_token bos_id = llama_vocab_bos(vocab);
    const std::string sys_prompt = !cli.system_prompt.empty() ? cli.system_prompt : cfg.system_prompt;

    if (!sys_prompt.empty()) {
        history.push_back({"system", sys_prompt});
    }

    // Render the full conversation (add_ass=false stops before the assistant
    // turn's header). Returns false when the model has no usable template.
    auto render_conversation = [&](bool add_ass, std::string & out) -> bool {
        if (!tmpl || !tmpl[0]) return false;
        std::vector<llama_chat_message> msgs;
        msgs.reserve(history.size());
        for (const auto & m : history) {
            msgs.push_back({m.role.c_str(), m.content.c_str()});
        }
        const int32_t n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                                    add_ass, nullptr, 0);
        if (n < 0) return false;
        out.resize(static_cast<size_t>(n) + 1);
        const int32_t m = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                                    add_ass, out.data(), static_cast<int32_t>(out.size()));
        if (m < 0) return false;
        out.resize(static_cast<size_t>(m));
        return true;
    };

    // Decode a batch of tokens, splitting into n_batch-sized chunks and
    // respecting the context limit. Positions are auto-tracked by the fork.
    llama_memory_t mem = llama_get_memory(ctx);
    auto decode_tokens = [&](const std::vector<llama_token> & toks) -> bool {
        const int32_t chunk = static_cast<int32_t>(llama_n_batch(ctx));
        for (size_t i = 0; i < toks.size(); i += static_cast<size_t>(chunk)) {
            const size_t n = std::min<size_t>(static_cast<size_t>(chunk), toks.size() - i);
            const int32_t n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
            if (n_ctx_used + static_cast<int32_t>(n) > static_cast<int32_t>(llama_n_ctx(ctx))) {
                fprintf(stderr, "\n\033[33m⚠ context window full (%d/%d)\033[0m\n",
                        n_ctx_used, static_cast<int32_t>(llama_n_ctx(ctx)));
                return false;
            }
            llama_batch batch = llama_batch_get_one(
                const_cast<llama_token *>(toks.data() + i), static_cast<int32_t>(n));
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "\033[31mdecode error\033[0m\n");
                return false;
            }
        }
        return true;
    };

    // Generation loop over the currently decoded state.
    auto generate = [&](std::string & response, GenStats & stats) -> bool {
        llama_sampler_reset(smpl.get());
        const auto gen_start = std::chrono::steady_clock::now();
        bool decode_ok = true;
        while (true) {
            const int32_t n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
            if (n_ctx_used >= static_cast<int32_t>(llama_n_ctx(ctx))) {
                fprintf(stderr, "\n\033[33m⚠ context window full (%d/%d)\033[0m\n",
                        n_ctx_used, static_cast<int32_t>(llama_n_ctx(ctx)));
                break;
            }
            if (cli.max_tokens > 0 && stats.tokens_generated >= cli.max_tokens) {
                fprintf(stderr, "\n\033[33m⚠ max tokens reached (%d)\033[0m\n", cli.max_tokens);
                break;
            }
            if (g_interrupted.load()) break;

            const llama_token id = llama_sampler_sample(smpl.get(), ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;

            const std::string piece = token_to_str(vocab, id);
            const std::string printable = utf8_buf.feed(piece);
            if (!printable.empty()) {
                printf("%s", printable.c_str());
                fflush(stdout);
            }
            response += piece;
            stats.tokens_generated++;
            prev_tokens.push_back(id);   // keep KV tracking in sync
            llama_sampler_accept(smpl.get(), id);

            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&id), 1);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "\033[31mdecode error\033[0m\n");
                decode_ok = false;
                break;
            }
        }
        const std::string tail = utf8_buf.flush();
        if (!tail.empty()) {
            printf("%s", tail.c_str());
            fflush(stdout);
        }
        stats.elapsed_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - gen_start).count();
        return decode_ok;
    };

    // Prepare a new user turn: renders the full conversation, re-tokenizes,
    // and decodes only the delta (or falls back to a full re-decode when the
    // previous state is no longer a prefix — e.g. after /undo, /clear, or a
    // template that re-tokenizes differently).
    auto start_turn = [&](std::string & formatted) -> bool {
        std::vector<llama_token> all = tokenize_render(vocab, formatted);
        if (all.empty()) {
            fprintf(stderr, "\033[31mtokenization failed\033[0m\n");
            return false;
        }
        if (!prev_tokens.empty()) {
            // Incremental: only decode tokens beyond what is already in the KV.
            if (prev_tokens.size() <= all.size() &&
                std::equal(prev_tokens.begin(), prev_tokens.end(), all.begin())) {
                std::vector<llama_token> delta(all.begin() + static_cast<long>(prev_tokens.size()), all.end());
                if (delta.empty()) return true;   // nothing new to decode
                if (!decode_tokens(delta)) return false;
                prev_tokens = std::move(all);
                return true;
            }
            // Prefix mismatch: full re-decode.
            llama_memory_clear(mem, true);
            prev_tokens.clear();
        }
        // First turn or full re-decode: add BOS manually (templates don't emit
        // it, and the fork's add_special path can double it on some templates).
        std::vector<llama_token> full;
        full.reserve(all.size() + 1);
        if (add_bos && (all.empty() || all[0] != bos_id)) full.push_back(bos_id);
        full.insert(full.end(), all.begin(), all.end());
        if (!decode_tokens(full)) return false;
        prev_tokens = std::move(all);   // track render tokens (BOS is implicit)
        return true;
    };

    auto finish_turn = [&](const std::string & resp) {
        if (!resp.empty()) {
            history.push_back({"assistant", resp});
        }
    };

    if (cli.prompt.empty()) {
        // ─── Interactive REPL ──────────────────────────────────────────────
        while (true) {
            g_interrupted.store(false);
            const int32_t n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
            if (n_ctx_used > 0) print_ctx_bar(n_ctx_used, static_cast<int>(llama_n_ctx(ctx)));
            printf("\033[32m> \033[0m");
            fflush(stdout);

            std::string user_input;
            if (!std::getline(std::cin, user_input)) break;
            while (!user_input.empty() && (user_input.back() == '\n' || user_input.back() == '\r'))
                user_input.pop_back();
            if (g_interrupted.load()) break;
            if (user_input.empty()) continue;

            if (user_input == "/exit" || user_input == "/quit") break;

            if (user_input == "/clear") {
                history.clear();
                llama_memory_clear(mem, true);
                prev_tokens.clear();
                if (!sys_prompt.empty()) {
                    history.push_back({"system", sys_prompt});
                }
                total_tokens_generated = 0;
                total_gen_time = 0.0;
                printf("Chat cleared.\n\n");
                continue;
            }

            if (user_input == "/stats") {
                printf("\n\033[1;36m── Session Stats ──\033[0m\n");
                printf("  turns           : %zu\n", history.size());
                printf("  tokens generated: %d\n", total_tokens_generated);
                printf("  total gen time  : %.2fs\n", total_gen_time);
                if (total_gen_time > 0)
                    printf("  avg speed       : %.1f t/s\n", total_tokens_generated / total_gen_time);
                const int32_t n_used = llama_memory_seq_pos_max(mem, 0) + 1;
                printf("  ctx used        : %d / %u (%.1f%%)\n", n_used,
                       cparams.n_ctx, 100.0 * n_used / cparams.n_ctx);
                printf("  KV cache        : K=%s V=%s\n", kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));
                printf("  temp            : %.2f\n", cfg.temp);
                printf("\n");
                continue;
            }

            if (user_input == "/undo") {
                if (history.size() >= 2 &&
                    history.back().role == "assistant" &&
                    history[history.size() - 2].role == "user") {
                    history.pop_back();
                    history.pop_back();
                    llama_memory_clear(mem, true);
                    prev_tokens.clear();
                    printf("Undid last turn. Next message will re-decode the conversation.\n\n");
                } else {
                    printf("Nothing to undo.\n\n");
                }
                continue;
            }

            if (user_input == "/export") {
                const std::string path = session_path();
                export_session(history, path);
                continue;
            }

            if (user_input == "/model") {
                printf("\n\033[1;36m── Model Info ──\033[0m\n");
                if (!cli.friendly.empty()) printf("  name       : %s\n", cli.friendly.c_str());
                printf("  file       : %s\n", cli.model.c_str());
                const int32_t alen = llama_model_desc(model, nullptr, 0);
                if (alen > 0) {
                    std::string arch_buf(static_cast<size_t>(alen) + 1, '\0');
                    llama_model_desc(model, arch_buf.data(), arch_buf.size());
                    printf("  arch       : %s\n", arch_buf.c_str());
                } else {
                    printf("  arch       : unknown\n");
                }
                printf("  trained ctx: %d\n", n_ctx_train);
                printf("  encoder    : %s\n", has_encoder ? "yes" : "no");
                printf("  decoder    : %s\n", has_decoder ? "yes" : "no");
                printf("  ngl        : %d\n", cfg.ngl);
                printf("\n");
                continue;
            }

            if (user_input.rfind("/temp ", 0) == 0) {
                float new_temp = 0.0f;
                if (!parse_float(user_input.substr(6), new_temp) || new_temp < 0.0f || new_temp > MAX_TEMP) {
                    printf("Invalid temperature.\n\n");
                    continue;
                }
                cfg.temp = new_temp;
                smpl.reset(build_sampler_chain(vocab, cfg, grammar_active, grammar_src));
                printf("Temperature set to %.2f\n\n", new_temp);
                continue;
            }

            if (user_input == "/ctx") {
                const int32_t n_used = llama_memory_seq_pos_max(mem, 0) + 1;
                print_ctx_bar(n_used, static_cast<int>(llama_n_ctx(ctx)));
                printf("\n");
                continue;
            }

            if (user_input[0] == '/') {
                printf("Unknown command: %s\n", user_input.c_str());
                printf("Available: /exit /clear /stats /undo /export /model /temp <f> /ctx\n\n");
                continue;
            }

            history.push_back({"user", user_input});
            std::string formatted;
            if (!render_conversation(true, formatted)) {
                // No chat template: fall back to raw prompt.
                history.pop_back();
                llama_memory_clear(mem, true);
                prev_tokens.clear();
                std::string raw = user_input + "\n";
                std::vector<llama_token> raw_toks;
                raw_toks.reserve(raw.size() + 1);
                if (add_bos) raw_toks.push_back(bos_id);
                const auto extra = tokenize_render(vocab, raw);
                raw_toks.insert(raw_toks.end(), extra.begin(), extra.end());
                if (!decode_tokens(raw_toks)) continue;
                prev_tokens = extra;
                printf("\033[33m");
                std::string resp;
                GenStats stats;
                generate(resp, stats);
                printf("\n\033[0m");
                finish_turn(resp);
                continue;
            }
            if (!start_turn(formatted)) continue;

            printf("\033[33m");
            std::string resp;
            GenStats stats;
            generate(resp, stats);
            printf("\n\033[0m");
            if (stats.tokens_generated > 0) {
                fprintf(stderr, "  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                        stats.tokens_generated, stats.tps(), stats.elapsed_sec);
            }
            total_tokens_generated += stats.tokens_generated;
            total_gen_time += stats.elapsed_sec;
            finish_turn(resp);
        }
    } else {
        // ─── Single-shot mode ──────────────────────────────────────────────
        history.push_back({"user", cli.prompt});
        std::string formatted;
        bool have_render = render_conversation(true, formatted);
        std::string prompt_text = have_render ? formatted : (cli.prompt + "\n");
        if (have_render && formatted.empty()) prompt_text = cli.prompt + "\n";

        std::vector<llama_token> all = tokenize_render(vocab, prompt_text);
        if (all.empty()) {
            fprintf(stderr, "\033[31mtokenization failed\033[0m\n");
            return 1;
        }
        std::vector<llama_token> full;
        full.reserve(all.size() + 1);
        if (add_bos && (all.empty() || all[0] != bos_id)) full.push_back(bos_id);
        full.insert(full.end(), all.begin(), all.end());
        if (!decode_tokens(full)) return 1;
        prev_tokens = std::move(all);

        printf("\033[33m");
        std::string resp;
        GenStats stats;
        generate(resp, stats);
        printf("\n\033[0m");
        if (stats.tokens_generated > 0) {
            fprintf(stderr, "\n  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                    stats.tokens_generated, stats.tps(), stats.elapsed_sec);
        }
        finish_turn(resp);
    }
    printf("\nExiting.\n");
    return 0;
}
// ──── src/main.cpp ────


int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    signal(SIGINT, anvil_signal_handler);
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fprintf(stderr, "%s", text);
    }, nullptr);

    CliArgs cli = parse_args(argc, argv);
    if (cli.invalid) { print_usage(); return 1; }
    if (cli.help)    { print_usage(); return 0; }
    if (cli.version) { printf("anvil %s\n", ANVIL_VERSION); return 0; }

    // Subcommand dispatch (registry / profile / pull / rm).
    if (cli.sub == "models")  return cmd_models(cli.sub_args);
    if (cli.sub == "profile") return cmd_profile(cli.sub_args);
    if (cli.sub == "rm")      return cmd_rm(cli.sub_args);
    if (cli.sub == "pull")    return cmd_pull(cli.sub_args);

    if (cli.model.empty()) {
        fprintf(stderr, "error: no model specified\n\n");
        print_usage();
        return 1;
    }

    // Resolve a friendly registry name or a local file path.
    std::string resolved_path;
    std::string friendly;
    if (!resolve_model_arg(cli.model, resolved_path, friendly)) {
        fprintf(stderr, "\033[31merror: '%s' is not a file and not a registered model\033[0m\n",
                cli.model.c_str());
        fprintf(stderr, "  Try: anvil pull ollama:%s | anvil pull hf:<repo> | anvil models import <file.gguf> --name %s\n",
                cli.model.c_str(), cli.model.c_str());
        return 1;
    }
    cli.model = resolved_path;
    if (friendly.empty()) {
        // Local file: derive a friendly name for the registry.
        friendly = slugify(std::filesystem::path(cli.model).stem().string());
    }
    cli.friendly = friendly;

    if (!validate_gguf(cli.model)) {
        fprintf(stderr, "\033[31merror: '%s' is not a valid GGUF file\033[0m\n", cli.model.c_str());
        return 1;
    }

    const HWInfo hw = probe_hw();

    // Read GGUF header metadata (vocab only) for context defaults + registry.
    const ModelMeta meta = read_model_meta(cli.model);
    const int max_ctx = static_cast<int>(meta.trained_ctx);

    LlamaBackend backend;

    AnvilConfig cfg;
    if (cli.setup || !config_exists()) {
        cfg = run_setup_tui(hw, max_ctx);
        cfg.model = cli.model;
        write_config(cfg);
        fprintf(stderr, "\nConfig saved to %s\n", config_path().c_str());
    } else {
        cfg = load_config();
    }

    fprintf(stderr, "Hardware: %s | %s | %" PRIu64 " GB RAM | %d threads\n",
            hw.cpu.c_str(), hw.arch.c_str(),
            static_cast<uint64_t>(hw.ram_bytes / (1024ULL * 1024 * 1024)),
            hw.cpu_threads);
    for (const auto & gpu : hw.gpus) {
        fprintf(stderr, "GPU: %s (%s) %" PRIu64 " MB VRAM%s\n",
                gpu.name.c_str(), gpu.vendor.c_str(),
                gpu.vram_mb,
                gpu.is_discrete ? " [discrete]" : "");
    }

    // Registry: auto-register local files and apply the model's persistent
    // profile (explicit profile keys win over the global config).
    std::vector<ModelEntry> models = load_models();
    ModelEntry * entry = find_model(models, friendly);
    if (!entry) {
        entry = auto_register(models, cli.model, friendly, meta);
        if (!save_models(models)) return 1;
        fprintf(stderr, "Registered local model as '%s' (see: anvil models)\n", friendly.c_str());
    }
    if (entry) {
        // The entry may pre-exist under a different name (canonical-path match).
        if (cli.friendly != entry->name) cli.friendly = entry->name;
        friendly = entry->name;
        if (!std::filesystem::exists(entry->path)) {
            fprintf(stderr, "\033[33mwarning: registered model file missing: %s\033[0m\n", entry->path.c_str());
        }
        // Refresh metadata + file size from disk (pulls may update the file).
        entry->desc = meta.desc;
        entry->trained_ctx = meta.trained_ctx;
        if (!meta.gguf_meta.empty()) entry->gguf_meta = meta.gguf_meta;
        std::error_code ec;
        const auto sz = std::filesystem::file_size(cli.model, ec);
        if (!ec) entry->size_bytes = sz;
        entry->profile.apply_to(cfg);
        fprintf(stderr, "Profile '%s': %d setting(s) applied\n", friendly.c_str(),
                static_cast<int>(entry->profile.settings.size()));
    }

    // CLI overrides win over profile and global config.
    if (cli.n_ctx > 0)          cfg.n_ctx = cli.n_ctx;
    if (cfg.n_ctx <= 0 && max_ctx > 0) cfg.n_ctx = max_ctx;
    if (cli.ngl >= 0)           cfg.ngl = cli.ngl;
    else if (cfg.ngl < 0)       cfg.ngl = derive_ngl(hw);
    if (cli.temp >= 0)          cfg.temp = cli.temp;
    if (cli.top_k >= 0)         cfg.top_k = cli.top_k;
    if (cli.top_p >= 0)         cfg.top_p = cli.top_p;
    if (cli.repeat_penalty >= 0) cfg.repeat_penalty = cli.repeat_penalty;
    if (cli.n_threads > 0)      cfg.n_threads = cli.n_threads;
    if (cli.flash_attn)         cfg.flash_attn = true;
    if (cli.no_flash_attn)      cfg.flash_attn = false;
    if (cli.mtp)                cfg.mtp = true;
    if (!cli.type_k.empty())    cfg.type_k = kv_type_from_name(cli.type_k);
    if (!cli.type_v.empty())    cfg.type_v = kv_type_from_name(cli.type_v);

    // --save: persist explicit CLI overrides into the model's profile.
    if (cli.save_profile && entry) {
        ModelProfile & p = entry->profile;
        if (cli.n_ctx > 0)               p.set("n_ctx", cli.n_ctx);
        if (cli.ngl >= 0)                p.set("ngl", cli.ngl);
        if (cli.n_threads > 0)           p.set("n_threads", cli.n_threads);
        if (cli.temp >= 0)               p.set("temp", cli.temp);
        if (cli.top_k >= 0)              p.set("top_k", cli.top_k);
        if (cli.top_p >= 0)              p.set("top_p", cli.top_p);
        if (cli.repeat_penalty >= 0)     p.set("repeat_penalty", cli.repeat_penalty);
        if (cli.flash_attn)              p.set("flash_attn", true);
        if (cli.no_flash_attn)           p.set("flash_attn", false);
        if (cli.mtp)                     p.set("mtp", true);
        // Normalize KV names so a bad --type-k can't poison the profile.
        if (!cli.type_k.empty() && valid_kv_name(cli.type_k)) p.set("type_k", cli.type_k);
        if (!cli.type_v.empty() && valid_kv_name(cli.type_v)) p.set("type_v", cli.type_v);
        if (!cli.system_prompt.empty())  p.set("system_prompt", cli.system_prompt);
        save_models(models);
        fprintf(stderr, "Saved %zu setting(s) to profile '%s'\n", p.settings.size(), friendly.c_str());
    } else if (entry) {
        save_models(models);   // persist refreshed metadata
    }

    cfg.model = cli.model;
    if (cfg.system_prompt.empty() && !cli.system_prompt.empty()) {
        cfg.system_prompt = cli.system_prompt;
    }
    write_config(cfg);

    const int rc = run_chat(cli, cfg, hw);
    return rc;
}
