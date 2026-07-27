#include "common.h"
#include "hardware.h"
#include "config.h"
#include "cli.h"
#include "tui.h"
#include "session.h"

#include "llama.h"
#include "ggml.h"

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
#include <utility>
#include <vector>

std::atomic<bool> g_interrupted{false};

// ── RAII wrappers for llama objects ──

struct LlamaModel {
    llama_model* p = nullptr;
    explicit LlamaModel(llama_model* p_ = nullptr) : p(p_) {}
    ~LlamaModel() { if (p) llama_model_free(p); }
    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;
    LlamaModel(LlamaModel&& o) noexcept : p(o.p) { o.p = nullptr; }
    LlamaModel& operator=(LlamaModel&& o) noexcept {
        if (this != &o) { if (p) llama_model_free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    llama_model* get() const { return p; }
    operator llama_model*() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

struct LlamaContext {
    llama_context* p = nullptr;
    explicit LlamaContext(llama_context* p_ = nullptr) : p(p_) {}
    ~LlamaContext() { if (p) llama_free(p); }
    LlamaContext(const LlamaContext&) = delete;
    LlamaContext& operator=(const LlamaContext&) = delete;
    LlamaContext(LlamaContext&& o) noexcept : p(o.p) { o.p = nullptr; }
    LlamaContext& operator=(LlamaContext&& o) noexcept {
        if (this != &o) { if (p) llama_free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    llama_context* get() const { return p; }
    operator llama_context*() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

struct LlamaSampler {
    llama_sampler* p = nullptr;
    explicit LlamaSampler(llama_sampler* p_ = nullptr) : p(p_) {}
    ~LlamaSampler() { if (p) llama_sampler_free(p); }
    LlamaSampler(const LlamaSampler&) = delete;
    LlamaSampler& operator=(const LlamaSampler&) = delete;
    LlamaSampler(LlamaSampler&& o) noexcept : p(o.p) { o.p = nullptr; }
    LlamaSampler& operator=(LlamaSampler&& o) noexcept {
        if (this != &o) { if (p) llama_sampler_free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    void reset(llama_sampler* p_) { if (p) llama_sampler_free(p); p = p_; }
    llama_sampler* get() const { return p; }
    operator llama_sampler*() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

// RAII owner for llama_chat_message C-strings
struct ChatMessages {
    std::vector<llama_chat_message> m;
    ~ChatMessages() { clear(); }
    ChatMessages() = default;
    ChatMessages(const ChatMessages&) = delete;
    ChatMessages& operator=(const ChatMessages&) = delete;
    ChatMessages(ChatMessages&&) = default;
    ChatMessages& operator=(ChatMessages&&) = default;
    void clear() {
        for (auto& x : m) free(const_cast<char*>(x.content));
        m.clear();
    }
    void push_back(llama_chat_message msg) { m.push_back(msg); }
    void pop_back() {
        if (!m.empty()) { free(const_cast<char*>(m.back().content)); m.pop_back(); }
    }
    llama_chat_message* data() { return m.data(); }
    const llama_chat_message* data() const { return m.data(); }
    size_t size() const { return m.size(); }
    bool empty() const { return m.empty(); }
    llama_chat_message& back() { return m.back(); }
    const llama_chat_message& back() const { return m.back(); }
};

// ── Helpers ──

static bool validate_gguf(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 16) { fclose(f); return false; }
    char magic[4];
    bool ok = (fread(magic, 1, 4, f) == 4 && memcmp(magic, "GGUF", 4) == 0);
    fclose(f);
    return ok;
}

static std::string token_to_str(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, true);
    if (n < 0) return "";
    buf[n] = '\0';
    return std::string(buf);
}

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

// ── Chat Engine ──

static int run_chat(const CliArgs & cli, AnvilConfig cfg, const HWInfo & hw) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.ngl;
    mparams.use_mmap = true;
    fprintf(stderr, "Loading model: %s ...\n", cli.model.c_str());
    auto load_start = std::chrono::steady_clock::now();
    LlamaModel model(llama_model_load_from_file(cli.model.c_str(), mparams));
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
    LlamaContext ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        fprintf(stderr, "\033[31merror: failed to create context\033[0m\n");
        return 1;
    }
    LlamaSampler smpl(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    bool        grammar_active = false;
    std::string grammar_src;
    if (!cli.grammar.empty()) {
        std::ifstream gf(cli.grammar);
        if (!gf) {
            fprintf(stderr, "\033[31merror: cannot open grammar file '%s'\033[0m\n", cli.grammar.c_str());
        } else {
            grammar_src.assign((std::istreambuf_iterator<char>(gf)), std::istreambuf_iterator<char>());
            llama_sampler * grammar_smpl = llama_sampler_init_grammar(vocab, grammar_src.c_str(), "root");
            if (grammar_smpl) {
                llama_sampler_chain_add(smpl, grammar_smpl);
                grammar_active = true;
                fprintf(stderr, "  grammar     : \033[32m%s\033[0m\n", cli.grammar.c_str());
            } else {
                fprintf(stderr, "\033[31merror: failed to parse grammar\033[0m\n");
            }
        }
    }
    printf("\033[1;33m%s\033[0m", ANVIL_LOGO);
    printf("  model   : %s\n", cli.model.c_str());
    printf("  backend : GPU layers=%d | flash=%s | threads=%d\n",
           cfg.ngl, cfg.flash_attn ? "on" : "off", cparams.n_threads);
    printf("  ctx     : %d tokens\n", cfg.n_ctx);
    printf("  KV      : K=%s V=%s\n", kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));
    printf("  temp    : %.2f\n", cfg.temp);
    if (cfg.mtp)     printf("  spec    : MTP\n");
    if (cfg.triattn) printf("  triattn : on\n");
    if (grammar_active) printf("  grammar : %s\n", cli.grammar.c_str());
    printf("  commands: /exit /clear /stats /undo /export /model /temp <f> /ctx\n\n");
    std::vector<ChatMessage> history;
    ChatMessages messages;
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
        std::string tail = utf8_buf.flush();
        if (!tail.empty()) {
            printf("%s", tail.c_str());
            fflush(stdout);
        }
        auto gen_end = std::chrono::steady_clock::now();
        stats.elapsed_sec = std::chrono::duration<double>(gen_end - gen_start).count();

        return {response, stats};
    };
    auto rebuild_messages = [&]() {
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
    if (cli.prompt.empty()) {
        while (true) {
            g_interrupted.store(false);
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (n_ctx_used > 0) print_ctx_bar(n_ctx_used, cfg.n_ctx);
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
                char arch_buf[256];
                llama_model_desc(model, arch_buf, sizeof(arch_buf));
                printf("  arch       : %s\n", arch_buf);
                printf("  trained ctx: %d\n", n_ctx_train);
                printf("  encoder    : %s\n", has_encoder ? "yes" : "no");
                printf("  decoder    : %s\n", has_decoder ? "yes" : "no");
                printf("  ngl        : %d\n", cfg.ngl);
                printf("\n");
                continue;
            }
            if (user_input.substr(0, 6) == "/temp ") {
                float new_temp = 0;
                if (!parse_float(user_input.substr(6), new_temp) || new_temp < 0) {
                    printf("Invalid temperature.\n\n");
                    continue;
                }
                cfg.temp = new_temp;
                smpl.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
                llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(new_temp));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
                if (grammar_active) {
                    llama_sampler_chain_add(smpl, llama_sampler_init_grammar(vocab, grammar_src.c_str(), "root"));
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
    printf("\nExiting.\n");
    return 0;
}

// ── Entry Point ──

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    signal(SIGINT, signal_handler);
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) fprintf(stderr, "%s", text);
    }, nullptr);
    CliArgs cli = parse_args(argc, argv);
    if (cli.invalid) { print_usage(); return 1; }
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
    int max_ctx = 262144;
    {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        mparams.vocab_only = true;
        LlamaModel meta_model(llama_model_load_from_file(cli.model.c_str(), mparams));
        if (meta_model) {
            max_ctx = llama_model_n_ctx_train(meta_model);
            if (max_ctx <= 0) max_ctx = 262144;
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
            (uint64_t)(hw.ram_bytes / (1024ULL * 1024 * 1024)),
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
