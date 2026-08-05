#include "common.hpp"
#include "cli.hpp"
#include "config.hpp"
#include "hardware.hpp"
#include "setup.hpp"
#include "chat.hpp"

#include <cinttypes>
#include <cstdio>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    signal(SIGINT, anvil_signal_handler);
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fprintf(stderr, "%s", text);
    }, nullptr);

    const CliArgs cli = parse_args(argc, argv);
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

    const HWInfo hw = probe_hw();

    // Read model metadata (vocab only) to pick a sensible default context.
    int max_ctx = 0;
    {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        mparams.vocab_only = true;
        LlamaModel meta_model(llama_model_load_from_file(cli.model.c_str(), mparams));
        if (meta_model) {
            max_ctx = llama_model_n_ctx_train(meta_model);
        }
    }

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

    // CLI overrides win over saved config.
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

    cfg.model = cli.model;
    write_config(cfg);

    const int rc = run_chat(cli, cfg, hw);
    return rc;
}
