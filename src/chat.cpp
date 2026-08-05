#include "chat.hpp"
#include "config.hpp"
#include "hardware.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>

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
    printf("  model   : %s\n", cli.model.c_str());
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

    if (!cli.system_prompt.empty()) {
        history.push_back({"system", cli.system_prompt});
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
                if (!cli.system_prompt.empty()) {
                    history.push_back({"system", cli.system_prompt});
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
