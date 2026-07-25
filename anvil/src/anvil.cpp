#include "llama.h"
#include "ggml.h"
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <IOKit/IOKitLib.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <glob.h>
#include <fstream>
#include <regex>
#include <unistd.h>
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi.h>
#include <process.h>
#pragma comment(lib, "dxgi.lib")
#endif
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
static std::atomic<bool> g_interrupted{false};

static void signal_handler(int) {
    if (g_interrupted.load()) {
        _exit(130);
    }
    g_interrupted.store(true);
}

// ─────────────────────────────────────────────
// KV Cache Type Helpers
// ─────────────────────────────────────────────

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
    { "Recommended   (K=q8_0,   V=turbo3)  ~4.6x  <1.5% PPL", 1, 3 },
    { "Balanced      (K=turbo4, V=turbo3)  ~4.2x  <2% PPL",   2, 3 },
    { "Max Compress  (K=turbo4, V=turbo2)  ~6.1x  ~3% PPL",   2, 4 },
    { "Quality       (K=f16,    V=f16)     1x     baseline",   0, 0 },
    { "Custom...",                                              -1, -1 },
};
static const int KV_PRESET_COUNT = 5;

static ggml_type kv_type_from_name(const std::string & name) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (name == KV_OPTIONS[i].short_name) return KV_OPTIONS[i].type;
    }
    return GGML_TYPE_F16;
}

static const char * kv_type_short(ggml_type t) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (KV_OPTIONS[i].type == t) return KV_OPTIONS[i].short_name;
    }
    return "f16";
}

static int kv_type_index(ggml_type t) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (KV_OPTIONS[i].type == t) return i;
    }
    return 0;
}

// ─────────────────────────────────────────────
// GPU / Hardware Info
// ─────────────────────────────────────────────

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

#ifdef __APPLE__
static void detect_gpus_macos(HWInfo & hw) {
    CFMutableDictionaryRef matching = IOServiceMatching("IOGPU");
    io_iterator_t iter;
    kern_return_t kr = IOServiceGetMatchingServices(kIOMasterPortDefault, matching, &iter);
    if (kr != KERN_SUCCESS) return;
    io_object_t device;
    while ((device = IOIteratorNext(iter)) != 0) {
        GPUInfo gpu;
        gpu.vendor = "Apple";
        gpu.is_discrete = false;
        gpu.vram_mb = 0;
        gpu.name = "Apple GPU (unified " + std::to_string(hw.ram_bytes / (1024ULL * 1024 * 1024)) + " GB)";
        hw.gpus.push_back(gpu);
        IOObjectRelease(device);
    }
    IOObjectRelease(iter);
}
#endif

#ifdef __linux__
static void run_cmd(const char * cmd, std::string & out) {
    FILE * pipe = popen(cmd, "r");
    if (!pipe) return;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
}

static void detect_gpus_linux(HWInfo & hw) {
    // NVIDIA
    {
        std::string out;
        run_cmd("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null", out);
        if (!out.empty()) {
            std::istringstream ss(out);
            std::string line;
            while (std::getline(ss, line)) {
                while (!line.empty() && line.back() <= ' ') line.pop_back();
                if (line.empty()) continue;
                GPUInfo gpu;
                gpu.vendor = "NVIDIA";
                gpu.is_discrete = true;
                auto comma = line.rfind(',');
                if (comma != std::string::npos) {
                    gpu.name = line.substr(0, comma);
                    while (!gpu.name.empty() && gpu.name.back() == ' ') gpu.name.pop_back();
                    gpu.vram_mb = std::stoull(line.substr(comma + 1));
                } else {
                    gpu.name = line;
                }
                hw.gpus.push_back(gpu);
            }
        }
    }
    // AMD / Intel via sysfs
    {
        glob_t globbuf;
        if (glob("/sys/class/drm/card*/device/vendor", 0, nullptr, &globbuf) == 0) {
            for (size_t i = 0; i < globbuf.gl_pathc; i++) {
                std::ifstream vf(globbuf.gl_pathv[i]);
                std::string vendor_id;
                std::getline(vf, vendor_id);
                while (!vendor_id.empty() && vendor_id.back() <= ' ') vendor_id.pop_back();

                std::string base(globbuf.gl_pathv[i]);
                base = base.substr(0, base.rfind("/device/vendor"));

                GPUInfo gpu;
                if (vendor_id == "0x1002") {
                    gpu.vendor = "AMD";
                    gpu.is_discrete = true;
                    std::ifstream df(base + "/device/device");
                    std::string did; std::getline(df, did);
                    gpu.name = "AMD GPU (" + did + ")";
                } else if (vendor_id == "0x8086") {
                    gpu.vendor = "Intel";
                    gpu.is_discrete = false;
                    std::ifstream df(base + "/device/device");
                    std::string did; std::getline(df, did);
                    gpu.name = "Intel GPU (" + did + ")";
                } else {
                    continue;
                }
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
}
#endif

#ifdef _WIN32
static void detect_gpus_windows(HWInfo & hw) {
    IDXGIFactory * factory = nullptr;
    if (CreateDXGIFactory(__uuidof(IDXGIFactory), (void **)&factory) != S_OK) return;
    IDXGIAdapter * adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC desc;
        if (adapter->GetDesc(&desc) == S_OK) {
            GPUInfo gpu;
            char name_buf[256];
            wcstombs(name_buf, desc.Description, sizeof(name_buf));
            gpu.name = name_buf;
            gpu.vram_mb = desc.DedicatedVideoMemory / (1024 * 1024);
            gpu.is_discrete = (desc.VendorId == 0x10DE);
            if (desc.VendorId == 0x10DE)      gpu.vendor = "NVIDIA";
            else if (desc.VendorId == 0x1002)  gpu.vendor = "AMD";
            else if (desc.VendorId == 0x8086)  gpu.vendor = "Intel";
            else                               gpu.vendor = "Other";
            hw.gpus.push_back(gpu);
        }
        adapter->Release();
    }
    factory->Release();
}
#endif

static HWInfo probe_hw() {
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

    hw.cpu_threads = (int)std::thread::hardware_concurrency();
    if (hw.cpu_threads <= 0) hw.cpu_threads = 4;

#ifdef __APPLE__
    {
        char buf[256];
        size_t len = sizeof(buf);
        if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0) {
            hw.cpu = buf;
            hw.apple_silicon = (hw.cpu.find("Apple") != std::string::npos);
        }
    }
    {
        uint64_t ram = 0;
        size_t len = sizeof(ram);
        sysctlbyname("hw.memsize", &ram, &len, nullptr, 0);
        hw.ram_bytes = ram;
    }
    detect_gpus_macos(hw);

#elif defined(__linux__)
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("model name") != std::string::npos) {
                auto pos = line.find(':');
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
                uint64_t kb = 0;
                sscanf(line.c_str(), "MemTotal: %" SCNu64 " kB", &kb);
                hw.ram_bytes = kb * 1024;
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
        hw.cpu_threads = (int)si.dwNumberOfProcessors;
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

static int derive_ngl(const HWInfo & hw) {
    if (hw.apple_silicon) return 99;
    for (const auto & gpu : hw.gpus) {
        if (gpu.is_discrete && gpu.vram_mb >= 4096) return 99;
    }
    for (const auto & gpu : hw.gpus) {
        if (gpu.vram_mb >= 4096) return 99;
    }
    return 0;
}

// ─────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────

struct AnvilConfig {
    int         version    = CONFIG_VERSION;
    int         ngl        = 99;
    int         n_ctx      = 8192;
    int         n_threads  = 0;   // 0 = auto
    float       temp       = 0.8f;
    bool        flash_attn = true;
    bool        mtp        = false;
    ggml_type   type_k     = GGML_TYPE_Q8_0;
    ggml_type   type_v     = GGML_TYPE_TURBO3_0;
    bool        triattn    = false;
    std::string model;
};

static std::string config_dir() {
    const char * home = getenv("HOME");
    if (!home) home = ".";
    return std::string(home) + "/.anvil";
}

static std::string config_path() {
    return config_dir() + "/config.json";
}

static std::string sessions_dir() {
    return config_dir() + "/sessions";
}

static void write_config(const AnvilConfig & cfg) {
    namespace fs = std::filesystem;
    fs::create_directories(config_dir());
    std::ofstream f(config_path());
    if (!f) {
        fprintf(stderr, "\033[33mwarning: could not write config to %s\033[0m\n", config_path().c_str());
        return;
    }
    f << "{\n";
    f << "  \"version\": "    << cfg.version    << ",\n";
    f << "  \"ngl\": "        << cfg.ngl        << ",\n";
    f << "  \"n_ctx\": "      << cfg.n_ctx      << ",\n";
    f << "  \"n_threads\": "  << cfg.n_threads  << ",\n";
    f << "  \"temp\": "       << cfg.temp       << ",\n";
    f << "  \"flash_attn\": " << (cfg.flash_attn ? "true" : "false") << ",\n";
    f << "  \"mtp\": "        << (cfg.mtp ? "true" : "false") << ",\n";
    f << "  \"triattn\": "    << (cfg.triattn ? "true" : "false") << ",\n";
    f << "  \"type_k\": \""   << kv_type_short(cfg.type_k) << "\",\n";
    f << "  \"type_v\": \""   << kv_type_short(cfg.type_v) << "\",\n";
    f << "  \"model\": \""    << cfg.model      << "\"\n";
    f << "}\n";
}

static std::string json_get(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
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
                result += json[pos];
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

static AnvilConfig load_config() {
    AnvilConfig cfg;
    std::ifstream f(config_path());
    if (!f) return cfg;
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    auto s = json_get(json, "version");
    if (!s.empty()) cfg.version = std::stoi(s);

    s = json_get(json, "ngl");
    if (!s.empty()) cfg.ngl = std::stoi(s);

    s = json_get(json, "n_ctx");
    if (!s.empty()) cfg.n_ctx = std::stoi(s);

    s = json_get(json, "n_threads");
    if (!s.empty()) cfg.n_threads = std::stoi(s);

    s = json_get(json, "temp");
    if (!s.empty()) cfg.temp = std::stof(s);

    s = json_get(json, "flash_attn");
    if (!s.empty()) cfg.flash_attn = (s == "true");

    s = json_get(json, "mtp");
    if (!s.empty()) cfg.mtp = (s == "true");

    s = json_get(json, "triattn");
    if (!s.empty()) cfg.triattn = (s == "true");

    s = json_get(json, "type_k");
    if (!s.empty()) cfg.type_k = kv_type_from_name(s);

    s = json_get(json, "type_v");
    if (!s.empty()) cfg.type_v = kv_type_from_name(s);

    s = json_get(json, "model");
    if (!s.empty()) cfg.model = s;

    // Migrate v1 configs (had no_turbo bool instead of type_k/type_v)
    if (cfg.version < 2) {
        s = json_get(json, "no_turbo");
        if (!s.empty() && s == "true") {
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

static bool config_exists() {
    std::ifstream f(config_path());
    return f.good();
}

// ─────────────────────────────────────────────
// CLI Args
// ─────────────────────────────────────────────

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
    bool  version       = false;
    std::string type_k;
    std::string type_v;
    std::string system_prompt;
    std::string prompt;
    std::string grammar;
    int   max_tokens    = -1;   // -1 = unlimited
};

static void print_usage() {
    printf("anvil %s — Forge anything.\n\n", ANVIL_VERSION);
    printf("Usage:\n");
    printf("  anvil run <model> [options]     Run a model with chat REPL\n");
    printf("  anvil run <model> -p \"prompt\"   Single-shot generation\n");
    printf("  anvil --help                    Show this help\n");
    printf("  anvil --version                 Show version\n\n");
    printf("Options:\n");
    printf("  -c, --ctx <n>            Context size (default: 8192)\n");
    printf("  -ngl, --n-gpu-layers <n> GPU layers to offload (default: auto)\n");
    printf("  -t, --temp <f>           Sampling temperature (default: 0.8)\n");
    printf("  --threads <n>            CPU threads (default: auto)\n");
    printf("  --flash-attn             Enable flash attention (default: on)\n");
    printf("  --no-flash-attn          Disable flash attention\n");
    printf("  --type-k <type>          K cache type: f16|q8_0|turbo4|turbo3|turbo2\n");
    printf("  --type-v <type>          V cache type: f16|q8_0|turbo4|turbo3|turbo2\n");
    printf("  --mtp                    Enable MTP speculative decoding\n");
    printf("  --triattn                Enable TriAttention KV eviction\n");
    printf("  --grammar <file>         GBNF grammar file for constrained output\n");
    printf("  -s, --system <text>      System prompt\n");
    printf("  -p, --prompt <text>      User prompt (non-interactive mode)\n");
    printf("  -n, --max-tokens <n>     Max tokens to generate (default: unlimited)\n\n");
    printf("TurboQuant KV presets (from setup TUI):\n");
    printf("  Recommended:  K=q8_0   V=turbo3  (~4.6x, <1.5%% PPL loss)\n");
    printf("  Balanced:     K=turbo4 V=turbo3  (~4.2x)\n");
    printf("  Max Compress: K=turbo4 V=turbo2  (~6.1x)\n\n");
    printf("Examples:\n");
    printf("  anvil run model.gguf\n");
    printf("  anvil run model.gguf --ctx 131072 --ngl 99 --type-k q8_0 --type-v turbo3\n");
    printf("  anvil run model.gguf -p \"Explain quantum computing\" -n 200\n");
    printf("  anvil run model.gguf --grammar json.gbnf -p \"List 3 colors\"\n");
}

static CliArgs parse_args(int argc, char ** argv) {
    CliArgs a;
    if (argc < 2) { a.help = true; return a; }
    int i = 1;
    if (i < argc && std::string(argv[i]) == "run") i++;
    for (; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--help" || arg == "-h")                          { a.help = true; }
        else if (arg == "--version")                                      { a.version = true; }
        else if ((arg == "-c" || arg == "--ctx") && i+1 < argc)           { a.n_ctx = std::stoi(argv[++i]); }
        else if ((arg == "-ngl" || arg == "--ngl" || arg == "--n-gpu-layers") && i+1 < argc) { a.ngl = std::stoi(argv[++i]); }
        else if ((arg == "-t" || arg == "--temp") && i+1 < argc)          { a.temp = std::stof(argv[++i]); }
        else if (arg == "--threads" && i+1 < argc)                        { a.n_threads = std::stoi(argv[++i]); }
        else if (arg == "--flash-attn")                                   { a.flash_attn = true; }
        else if (arg == "--no-flash-attn")                                { a.no_flash_attn = true; a.flash_attn = false; }
        else if (arg == "--type-k" && i+1 < argc)                         { a.type_k = argv[++i]; }
        else if (arg == "--type-v" && i+1 < argc)                         { a.type_v = argv[++i]; }
        else if (arg == "--mtp")                                          { a.mtp = true; }
        else if (arg == "--triattn")                                      { a.triattn = true; }
        else if (arg == "--grammar" && i+1 < argc)                        { a.grammar = argv[++i]; }
        else if ((arg == "-s" || arg == "--system") && i+1 < argc)        { a.system_prompt = argv[++i]; }
        else if ((arg == "-p" || arg == "--prompt") && i+1 < argc)        { a.prompt = argv[++i]; }
        else if ((arg == "-n" || arg == "--max-tokens") && i+1 < argc)    { a.max_tokens = std::stoi(argv[++i]); }
        else if (arg[0] != '-')                                           { a.model = arg; }
        else { fprintf(stderr, "Unknown option: %s\n", arg.c_str()); a.help = true; }
    }
    return a;
}

// ─────────────────────────────────────────────
// UTF-8 Streaming Buffer
// ─────────────────────────────────────────────

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

// ─────────────────────────────────────────────
// Session Persistence
// ─────────────────────────────────────────────

struct ChatMessage {
    std::string role;
    std::string content;
};

static std::string session_path() {
    namespace fs = std::filesystem;
    fs::create_directories(sessions_dir());
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
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

// ─────────────────────────────────────────────
// TUI Setup Wizard
// ─────────────────────────────────────────────

static AnvilConfig run_setup_tui(const HWInfo & hw, int max_ctx) {
    AnvilConfig cfg;
    cfg.ngl = derive_ngl(hw);
    cfg.n_ctx = 8192;

    // Context options
    std::vector<int> ctx_options;
    for (int c = 2048; c <= max_ctx; c *= 2) ctx_options.push_back(c);
    ctx_options.push_back(0); // Custom
    int custom_idx = (int)ctx_options.size() - 1;

    int ctx_sel = 0;
    for (size_t i = 0; i < ctx_options.size(); i++)
        if (ctx_options[i] == cfg.n_ctx) { ctx_sel = (int)i; break; }

    std::vector<std::string> ctx_labels;
    for (int c : ctx_options) {
        if (c == 0) ctx_labels.push_back("Custom...");
        else if (c >= 1024) ctx_labels.push_back(std::to_string(c / 1024) + "K tokens");
        else ctx_labels.push_back(std::to_string(c) + " tokens");
    }

    // KV preset labels
    std::vector<std::string> kv_preset_labels;
    for (int i = 0; i < KV_PRESET_COUNT; i++)
        kv_preset_labels.push_back(KV_PRESETS[i].label);

    // Individual KV type labels
    std::vector<std::string> kv_type_labels;
    for (int i = 0; i < KV_OPTIONS_COUNT; i++)
        kv_type_labels.push_back(KV_OPTIONS[i].label);

    std::vector<std::string> fa_labels   = {"on (recommended)", "off"};
    std::vector<std::string> temp_labels = {"0.7 (focused)", "0.8 (balanced)", "0.9 (creative)", "1.0 (wild)"};

    int kv_preset_sel = 0;
    int kv_k_sel      = 1;  // q8_0
    int kv_v_sel      = 3;  // turbo3
    int fa_sel        = 0;
    int temp_sel      = 1;
    bool custom_kv    = false;

    // Menu: 0=ctx, 1=kv_preset, 2=fa, 3=temp, (4=k_type, 5=v_type when custom)
    int menu_idx = 0;
    int menu_count = 4;

    // HW info lines
    std::vector<std::string> hw_lines;
    hw_lines.push_back("CPU  : " + hw.cpu);
    hw_lines.push_back("RAM  : " + std::to_string(hw.ram_bytes / (1024ULL * 1024 * 1024)) + " GB");
    hw_lines.push_back("Threads: " + std::to_string(hw.cpu_threads));
    for (const auto & gpu : hw.gpus)
        hw_lines.push_back("GPU  : " + gpu.name + " (" + std::to_string(gpu.vram_mb) + " MB)");

    auto screen = ftxui::ScreenInteractive::FitComponent();

    auto component = ftxui::Renderer([&]() {
        ftxui::Elements rows;
        rows.push_back(ftxui::text(ANVIL_LOGO) | ftxui::bold | ftxui::color(ftxui::Color::Yellow));
        rows.push_back(ftxui::text("  First-time setup  v" + std::string(ANVIL_VERSION)) | ftxui::bold | ftxui::color(ftxui::Color::Cyan));
        rows.push_back(ftxui::text(" "));
        rows.push_back(ftxui::text("  Detected hardware:") | ftxui::bold);
        for (const auto & line : hw_lines)
            rows.push_back(ftxui::text("    " + line));
        rows.push_back(ftxui::text(" "));

        auto render_setting = [&](const std::string & label, const std::vector<std::string> & opts,
                                  int sel, int idx, int indent = 2) {
            std::string pad(indent, ' ');
            bool active = (idx == menu_idx);
            auto lbl = ftxui::text(pad + label);
            if (active) lbl = lbl | ftxui::bold | ftxui::color(ftxui::Color::Cyan);
            else        lbl = lbl | ftxui::bold;
            rows.push_back(lbl);
            for (int i = 0; i < (int)opts.size(); i++) {
                bool selected = (i == sel);
                auto prefix = selected
                    ? ftxui::text(pad + "  ▸ ") | ftxui::bold | ftxui::color(ftxui::Color::Green)
                    : ftxui::text(pad + "  · ");
                auto label_el = ftxui::text(opts[i]);
                if (selected && active)
                    label_el = label_el | ftxui::bold | ftxui::color(ftxui::Color::Green);
                else if (selected)
                    label_el = label_el | ftxui::color(ftxui::Color::Green);
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

        // Live KV summary
        std::string kv_summary;
        if (!custom_kv) {
            int ki = KV_PRESETS[kv_preset_sel].k_idx;
            int vi = KV_PRESETS[kv_preset_sel].v_idx;
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

        menu_count = custom_kv ? 6 : 4;

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

        auto cycle = [&](int & sel, int count, int dir) {
            sel = (sel + dir + count) % count;
        };

        if (e == ftxui::Event::ArrowUp) {
            switch (menu_idx) {
                case 0: cycle(ctx_sel, (int)ctx_labels.size(), -1); break;
                case 1:
                    cycle(kv_preset_sel, KV_PRESET_COUNT, -1);
                    custom_kv = (kv_preset_sel == KV_PRESET_COUNT - 1);
                    break;
                case 2: cycle(fa_sel, (int)fa_labels.size(), -1); break;
                case 3: cycle(temp_sel, (int)temp_labels.size(), -1); break;
                case 4: cycle(kv_k_sel, KV_OPTIONS_COUNT, -1); break;
                case 5: cycle(kv_v_sel, KV_OPTIONS_COUNT, -1); break;
            }
            return true;
        }
        if (e == ftxui::Event::ArrowDown) {
            switch (menu_idx) {
                case 0: cycle(ctx_sel, (int)ctx_labels.size(), 1); break;
                case 1:
                    cycle(kv_preset_sel, KV_PRESET_COUNT, 1);
                    custom_kv = (kv_preset_sel == KV_PRESET_COUNT - 1);
                    break;
                case 2: cycle(fa_sel, (int)fa_labels.size(), 1); break;
                case 3: cycle(temp_sel, (int)temp_labels.size(), 1); break;
                case 4: cycle(kv_k_sel, KV_OPTIONS_COUNT, 1); break;
                case 5: cycle(kv_v_sel, KV_OPTIONS_COUNT, 1); break;
            }
            return true;
        }
        return false;
    });

    screen.Loop(wrapped);

    // Apply selections
    if (ctx_sel == custom_idx) {
        printf("\033[?25h");
        printf("Enter context size (tokens): ");
        fflush(stdout);
        std::string input;
        std::getline(std::cin, input);
        cfg.n_ctx = std::stoi(input);
    } else {
        cfg.n_ctx = ctx_options[ctx_sel];
    }

    // KV types
    if (custom_kv) {
        cfg.type_k = KV_OPTIONS[kv_k_sel].type;
        cfg.type_v = KV_OPTIONS[kv_v_sel].type;
    } else {
        int ki = KV_PRESETS[kv_preset_sel].k_idx;
        int vi = KV_PRESETS[kv_preset_sel].v_idx;
        cfg.type_k = KV_OPTIONS[ki].type;
        cfg.type_v = KV_OPTIONS[vi].type;
    }

    cfg.flash_attn = (fa_sel == 0);
    float temps[] = {0.7f, 0.8f, 0.9f, 1.0f};
    cfg.temp = temps[temp_sel];

    return cfg;
}

// ─────────────────────────────────────────────
// GGUF Validation
// ─────────────────────────────────────────────

static bool validate_gguf(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[4];
    bool ok = (fread(magic, 1, 4, f) == 4 && memcmp(magic, "GGUF", 4) == 0);
    fclose(f);
    return ok;
}

// ─────────────────────────────────────────────
// Token Helpers
// ─────────────────────────────────────────────

static std::string token_to_str(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, true);
    if (n < 0) return "";
    buf[n] = '\0';
    return std::string(buf);
}

// ─────────────────────────────────────────────
// Context Usage Bar
// ─────────────────────────────────────────────

static void print_ctx_bar(int used, int total) {
    if (total <= 0) return;
    float pct = (float)used / (float)total;
    int bar_width = 30;
    int filled = (int)(pct * bar_width);
    if (filled > bar_width) filled = bar_width;

    const char * color;
    if (pct < 0.5f)      color = "\033[32m";
    else if (pct < 0.8f) color = "\033[33m";
    else                 color = "\033[31m";

    fprintf(stderr, "  %sctx [", color);
    for (int i = 0; i < bar_width; i++)
        fprintf(stderr, i < filled ? "█" : "░");
    fprintf(stderr, "] %d%% (%d/%d)\033[0m\n", (int)(pct * 100), used, total);
}

// ─────────────────────────────────────────────
// Generation Stats
// ─────────────────────────────────────────────

struct GenStats {
    int    tokens_generated = 0;
    double elapsed_sec      = 0.0;

    double tps() const {
        return elapsed_sec > 0.0 ? tokens_generated / elapsed_sec : 0.0;
    }
};

// ─────────────────────────────────────────────
// Chat Engine
// ─────────────────────────────────────────────

static int run_chat(const CliArgs & cli, AnvilConfig cfg, const HWInfo & hw) {
    // ── Model params ──
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.ngl;
    mparams.use_mmap = true;

    fprintf(stderr, "Loading model: %s ...\n", cli.model.c_str());
    auto load_start = std::chrono::steady_clock::now();

    llama_model * model = llama_model_load_from_file(cli.model.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "\033[31merror: failed to load model '%s'\033[0m\n", cli.model.c_str());
        return 1;
    }

    auto load_end = std::chrono::steady_clock::now();
    double load_sec = std::chrono::duration<double>(load_end - load_start).count();

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_ctx_train = llama_model_n_ctx_train(model);
    bool has_encoder = llama_model_has_encoder(model);
    bool has_decoder = llama_model_has_decoder(model);

    // ── Model info ──
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

    if (cfg.flash_attn)
        fprintf(stderr, "  flash attn  : \033[32menabled\033[0m\n");
    else
        fprintf(stderr, "  flash attn  : \033[33mdisabled\033[0m (perf will suffer)\n");

    fprintf(stderr, "  KV cache    : K=\033[32m%s\033[0m V=\033[32m%s\033[0m\n",
            kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));

    if (cfg.mtp)
        fprintf(stderr, "  MTP         : \033[33menabled\033[0m\n");

    if (cfg.triattn)
        fprintf(stderr, "  TriAttention: \033[32menabled\033[0m (intelligent KV eviction)\n");

    fprintf(stderr, "\n");

    // ── Context params ──
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = cfg.n_ctx;
    cparams.n_batch   = std::min(cfg.n_ctx, 2048);
    cparams.n_threads = cfg.n_threads > 0 ? cfg.n_threads : hw.cpu_threads;
    cparams.flash_attn_type = cfg.flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k = cfg.type_k;
    cparams.type_v = cfg.type_v;

    if (cfg.mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "\033[31merror: failed to create context\033[0m\n");
        llama_model_free(model);
        return 1;
    }

    // ── Sampler ──
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ── Grammar (optional) ──
    llama_grammar * grammar = nullptr;
    if (!cli.grammar.empty()) {
        std::ifstream gf(cli.grammar);
        if (!gf) {
            fprintf(stderr, "\033[31merror: cannot open grammar file '%s'\033[0m\n", cli.grammar.c_str());
        } else {
            std::string grammar_str((std::istreambuf_iterator<char>(gf)), std::istreambuf_iterator<char>());
            grammar = llama_grammar_init(nullptr, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0);
            if (grammar) {
                llama_sampler_chain_add(smpl, llama_sampler_init_grammar(grammar, "root", false));
                fprintf(stderr, "  grammar     : \033[32m%s\033[0m\n", cli.grammar.c_str());
            } else {
                fprintf(stderr, "\033[31merror: failed to parse grammar\033[0m\n");
            }
        }
    }

    // ── Banner ──
    printf("\033[1;33m%s\033[0m", ANVIL_LOGO);
    printf("  model   : %s\n", cli.model.c_str());
    printf("  backend : GPU layers=%d | flash=%s | threads=%d\n",
           cfg.ngl, cfg.flash_attn ? "on" : "off", cparams.n_threads);
    printf("  ctx     : %d tokens\n", cfg.n_ctx);
    printf("  KV      : K=%s V=%s\n", kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));
    printf("  temp    : %.2f\n", cfg.temp);
    if (cfg.mtp)     printf("  spec    : MTP\n");
    if (cfg.triattn) printf("  triattn : on\n");
    if (grammar)     printf("  grammar : %s\n", cli.grammar.c_str());
    printf("  commands: /exit /clear /stats /undo /export /model /temp <f> /ctx\n\n");

    // ── Chat state ──
    std::vector<ChatMessage> history;
    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(cparams.n_ctx * 4);
    int prev_len = 0;
    int total_tokens_generated = 0;
    double total_gen_time = 0.0;

    if (!cli.system_prompt.empty()) {
        history.push_back({"system", cli.system_prompt});
        messages.push_back({"system", strdup(cli.system_prompt.c_str())});
    }

    const char * tmpl = llama_model_chat_template(model, nullptr);
    Utf8Buffer utf8_buf;

    // ── Generation lambda ──
    auto generate = [&](const std::string & prompt_text) -> std::pair<std::string, GenStats> {
        std::string response;
        GenStats stats;
        utf8_buf = Utf8Buffer();

        const bool is_first = (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1);
        int n_tokens = -llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(),
                                       nullptr, 0, is_first, true);
        if (n_tokens < 0) {
            fprintf(stderr, "\033[31mtokenization error\033[0m\n");
            return {"", stats};
        }

        std::vector<llama_token> tokens(n_tokens);
        llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(),
                       tokens.data(), tokens.size(), is_first, true);

        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        auto gen_start = std::chrono::steady_clock::now();

        while (true) {
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (n_ctx_used + batch.n_tokens > (int)cparams.n_ctx) {
                fprintf(stderr, "\n\033[33m⚠ context window full (%d/%d)\033[0m\n",
                        n_ctx_used, cparams.n_ctx);
                fprintf(stderr, "  Use /clear to reset or /undo to remove last turn.\n");
                break;
            }

            // Enforce max_tokens only when explicitly set (> 0)
            if (cli.max_tokens > 0 && stats.tokens_generated >= cli.max_tokens) {
                fprintf(stderr, "\n\033[33m⚠ max tokens reached (%d)\033[0m\n", cli.max_tokens);
                break;
            }

            if (g_interrupted.load()) break;
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "\033[31mdecode error\033[0m\n");
                break;
            }

            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;

            std::string piece = token_to_str(vocab, id);
            std::string printable = utf8_buf.feed(piece);
            if (!printable.empty()) {
                printf("%s", printable.c_str());
                fflush(stdout);
            }
            response += piece;
            stats.tokens_generated++;
            llama_sampler_accept(smpl, id);
            batch = llama_batch_get_one(&id, 1);
        }

        // Flush remaining UTF-8
        std::string tail = utf8_buf.flush();
        if (!tail.empty()) {
            printf("%s", tail.c_str());
            fflush(stdout);
        }

        auto gen_end = std::chrono::steady_clock::now();
        stats.elapsed_sec = std::chrono::duration<double>(gen_end - gen_start).count();

        return {response, stats};
    };

    // ── Helper: rebuild messages from history ──
    auto rebuild_messages = [&]() {
        for (auto & msg : messages) free(const_cast<char *>(msg.content));
        messages.clear();
        prev_len = 0;
        for (const auto & h : history) {
            messages.push_back({h.role.c_str(), strdup(h.content.c_str())});
        }
        if (!messages.empty()) {
            prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                  false, nullptr, 0);
            if (prev_len < 0) prev_len = 0;
        }
    };

    // ── Interactive REPL ──
    if (cli.prompt.empty()) {
        while (true) {
            g_interrupted.store(false);

            // Context usage
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (n_ctx_used > 0) print_ctx_bar(n_ctx_used, cfg.n_ctx);

            printf("\033[32m> \033[0m");
            fflush(stdout);

            std::string user_input;
            if (!std::getline(std::cin, user_input)) break;
            while (!user_input.empty() && (user_input.back() == '\n' || user_input.back() == '\r'))
                user_input.pop_back();
            if (user_input.empty()) continue;

            // ── Commands ──
            if (user_input == "/exit" || user_input == "/quit") break;

            if (user_input == "/clear") {
                history.clear();
                rebuild_messages();
                llama_memory_clear(llama_get_memory(ctx), true);
                if (!cli.system_prompt.empty()) {
                    history.push_back({"system", cli.system_prompt});
                    messages.push_back({"system", strdup(cli.system_prompt.c_str())});
                    prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                          false, nullptr, 0);
                    if (prev_len < 0) prev_len = 0;
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
                int n_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
                printf("  ctx used        : %d / %d (%.1f%%)\n", n_used, cfg.n_ctx,
                       100.0 * n_used / cfg.n_ctx);
                printf("  KV cache        : K=%s V=%s\n", kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));
                printf("  temp            : %.2f\n", cfg.temp);
                printf("\n");
                continue;
            }

            if (user_input == "/undo") {
                if (history.size() >= 2 &&
                    history.back().role == "assistant" &&
                    history[history.size()-2].role == "user") {
                    history.pop_back();
                    history.pop_back();
                    rebuild_messages();
                    llama_memory_clear(llama_get_memory(ctx), true);
                    if (!messages.empty()) {
                        int full_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                                  true, formatted.data(), formatted.size());
                        if (full_len > (int)formatted.size()) {
                            formatted.resize(full_len + 256);
                            full_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                                  true, formatted.data(), formatted.size());
                        }
                        if (full_len > 0) {
                            std::string full_prompt(formatted.begin(), formatted.begin() + full_len);
                            const bool is_first = true;
                            int nt = -llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(),
                                                     nullptr, 0, is_first, true);
                            if (nt > 0) {
                                std::vector<llama_token> toks(nt);
                                llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(),
                                               toks.data(), toks.size(), is_first, true);
                                llama_batch b = llama_batch_get_one(toks.data(), toks.size());
                                llama_decode(ctx, b);
                            }
                        }
                        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                              false, nullptr, 0);
                        if (prev_len < 0) prev_len = 0;
                    }
                    printf("Undid last turn.\n\n");
                } else {
                    printf("Nothing to undo.\n\n");
                }
                continue;
            }

            if (user_input == "/export") {
                std::string path = session_path();
                export_session(history, path);
                continue;
            }

            if (user_input == "/model") {
                printf("\n\033[1;36m── Model Info ──\033[0m\n");
                printf("  file       : %s\n", cli.model.c_str());
                printf("  arch       : %s\n", llama_model_arch(model));
                printf("  trained ctx: %d\n", n_ctx_train);
                printf("  encoder    : %s\n", has_encoder ? "yes" : "no");
                printf("  decoder    : %s\n", has_decoder ? "yes" : "no");
                printf("  ngl        : %d\n", cfg.ngl);
                printf("\n");
                continue;
            }

            if (user_input.substr(0, 6) == "/temp ") {
                float new_temp = std::stof(user_input.substr(6));
                cfg.temp = new_temp;
                llama_sampler_free(smpl);
                smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(new_temp));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
                if (grammar) {
                    llama_sampler_chain_add(smpl, llama_sampler_init_grammar(grammar, "root", false));
                }
                printf("Temperature set to %.2f\n\n", new_temp);
                continue;
            }

            if (user_input == "/ctx") {
                int n_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
                print_ctx_bar(n_used, cfg.n_ctx);
                printf("\n");
                continue;
            }

            if (user_input[0] == '/') {
                printf("Unknown command: %s\n", user_input.c_str());
                printf("Available: /exit /clear /stats /undo /export /model /temp <f> /ctx\n\n");
                continue;
            }

            // ── Normal message ──
            history.push_back({"user", user_input});
            messages.push_back({"user", strdup(user_input.c_str())});

            int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                     true, formatted.data(), formatted.size());
            if (new_len > (int)formatted.size()) {
                formatted.resize(new_len + 256);
                new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                     true, formatted.data(), formatted.size());
            }

            if (new_len < 0) {
                fprintf(stderr, "\033[33mchat template failed, using raw prompt\033[0m\n");
                std::string raw = user_input + "\n";
                history.pop_back();
                free(const_cast<char *>(messages.back().content));
                messages.pop_back();
                printf("\033[33m");
                auto [resp, stats] = generate(raw);
                printf("\n\033[0m");
                continue;
            }

            std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);
            printf("\033[33m");
            auto [resp, stats] = generate(prompt);
            printf("\n\033[0m");

            if (stats.tokens_generated > 0) {
                fprintf(stderr, "  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                        stats.tokens_generated, stats.tps(), stats.elapsed_sec);
            }

            total_tokens_generated += stats.tokens_generated;
            total_gen_time += stats.elapsed_sec;

            history.push_back({"assistant", resp});
            messages.push_back({"assistant", strdup(resp.c_str())});
            prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                  false, nullptr, 0);
            if (prev_len < 0) prev_len = 0;
        }
    }
    // ── Single-shot mode ──
    else {
        history.push_back({"user", cli.prompt});
        messages.push_back({"user", strdup(cli.prompt.c_str())});

        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                 true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len + 256);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(),
                                                 true, formatted.data(), formatted.size());
        }

        std::string prompt_text;
        if (new_len > 0)
            prompt_text.assign(formatted.begin(), formatted.begin() + new_len);
        else
            prompt_text = cli.prompt;

        printf("\033[33m");
        auto [resp, stats] = generate(prompt_text);
        printf("\n\033[0m");

        if (stats.tokens_generated > 0) {
            fprintf(stderr, "\n  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                    stats.tokens_generated, stats.tps(), stats.elapsed_sec);
        }
    }

    // ── Cleanup ──
    for (auto & msg : messages) free(const_cast<char *>(msg.content));
    if (grammar) llama_grammar_free(grammar);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    printf("\nExiting.\n");
    return 0;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    signal(SIGINT, signal_handler);

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fprintf(stderr, "%s", text);
    }, nullptr);

    CliArgs cli = parse_args(argc, argv);
    if (cli.help)    { print_usage(); return 0; }
    if (cli.version) { printf("anvil %s\n", ANVIL_VERSION); return 0; }
    if (cli.model.empty()) {
        fprintf(stderr, "error: no model specified\n\n");
        print_usage();
        return 1;
    }

    if (!validate_gguf(cli.model)) {
        fprintf(stderr, "\033[31merror: '%s' is not a valid GGUF file\033[0m\n", cli.model.c_str());
        return 1;
    }

    HWInfo hw = probe_hw();
    AnvilConfig cfg;

    llama_backend_init();

    // Read model metadata without loading weights
    int max_ctx = 262144;
    {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        mparams.vocab_only = true;
        llama_model * meta_model = llama_model_load_from_file(cli.model.c_str(), mparams);
        if (meta_model) {
            max_ctx = llama_model_n_ctx_train(meta_model);
            if (max_ctx <= 0) max_ctx = 262144;
            llama_model_free(meta_model);
        }
    }

    if (!config_exists()) {
        cfg = run_setup_tui(hw, max_ctx);
        cfg.model = cli.model;
        write_config(cfg);
        fprintf(stderr, "\nConfig saved to %s\n", config_path().c_str());
    } else {
        cfg = load_config();
    }

    fprintf(stderr, "Hardware: %s | %s | %" PRIu64 " GB RAM | %d threads\n",
            hw.cpu.c_str(), hw.arch.c_str(),
            hw.ram_bytes / (1024ULL * 1024 * 1024),
            hw.cpu_threads);
    for (const auto & gpu : hw.gpus) {
        fprintf(stderr, "GPU: %s (%s) %" PRIu64 " MB VRAM%s\n",
                gpu.name.c_str(), gpu.vendor.c_str(),
                gpu.vram_mb,
                gpu.is_discrete ? " [discrete]" : "");
    }

    if (cli.n_ctx > 0)         cfg.n_ctx = cli.n_ctx;
    else if (cfg.n_ctx == 0)   cfg.n_ctx = 8192;
    if (cli.ngl >= 0)          cfg.ngl = cli.ngl;
    else if (cfg.ngl == 0)     cfg.ngl = derive_ngl(hw);
    if (cli.temp >= 0)         cfg.temp = cli.temp;
    if (cli.n_threads > 0)     cfg.n_threads = cli.n_threads;
    if (cli.flash_attn)        cfg.flash_attn = true;
    if (cli.no_flash_attn)     cfg.flash_attn = false;
    if (cli.mtp)               cfg.mtp = true;
    if (cli.triattn)           cfg.triattn = true;
    if (!cli.type_k.empty())   cfg.type_k = kv_type_from_name(cli.type_k);
    if (!cli.type_v.empty())   cfg.type_v = kv_type_from_name(cli.type_v);
    cfg.model = cli.model;

    bool has_overrides = cli.n_ctx > 0 || cli.ngl >= 0 || cli.temp >= 0 ||
                         cli.n_threads > 0 || cli.flash_attn || cli.no_flash_attn ||
                         cli.mtp || cli.triattn ||
                         !cli.type_k.empty() || !cli.type_v.empty();
    if (has_overrides) {
        write_config(cfg);
    }

    int rc = run_chat(cli, cfg, hw);
    llama_backend_free();
    return rc;
}
