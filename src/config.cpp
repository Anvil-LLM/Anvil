#include "config.hpp"

#include <cstdio>

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
