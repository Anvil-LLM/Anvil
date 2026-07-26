#include "config.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

std::string config_dir() {
    const char * home = getenv("HOME");
    if (!home) home = ".";
    return std::string(home) + "/.anvil";
}

std::string config_path() {
    return config_dir() + "/config.json";
}

std::string sessions_dir() {
    return config_dir() + "/sessions";
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

bool config_exists() {
    std::ifstream f(config_path());
    return f.good();
}

AnvilConfig load_config() {
    AnvilConfig cfg;
    std::ifstream f(config_path());
    if (!f) return cfg;
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto s = json_get(json, "version");
    if (!s.empty()) parse_int(s, cfg.version);
    s = json_get(json, "ngl");
    if (!s.empty()) parse_int(s, cfg.ngl);
    s = json_get(json, "n_ctx");
    if (!s.empty()) parse_int(s, cfg.n_ctx);
    s = json_get(json, "n_threads");
    if (!s.empty()) parse_int(s, cfg.n_threads);
    s = json_get(json, "temp");
    if (!s.empty()) parse_float(s, cfg.temp);
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

void write_config(const AnvilConfig & cfg) {
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
