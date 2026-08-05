#include "cli.hpp"

#include <cstdio>

void print_usage() {
    printf("anvil %s — Forge anything.\n\n", ANVIL_VERSION);
    printf("Usage:\n");
    printf("  anvil run <model> [options]     Run a model with chat REPL\n");
    printf("  anvil run <model> -p \"prompt\"   Single-shot generation\n");
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
    printf("  -n, --max-tokens <n>     Max tokens to generate (default: unlimited)\n\n");
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
    if (argc < 2) { a.help = true; return a; }
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
