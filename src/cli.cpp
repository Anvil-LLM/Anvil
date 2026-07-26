#include "cli.h"

#include <cstdio>
#include <cstring>
#include <string>

void print_usage() {
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
    printf("  Recommended:  K=turbo4 V=turbo3  (4.2x, ~1%% quality loss)\n");
    printf("  Quality+:     K=q8_0   V=turbo3  (~3x,  <1%% quality loss)\n");
    printf("  High Compression: K=turbo4 V=turbo2  (6.1x, ~3%% quality loss)\n\n");
    printf("Examples:\n");
    printf("  anvil run model.gguf\n");
    printf("  anvil run model.gguf --ctx 131072 --ngl 99 --type-k q8_0 --type-v turbo3\n");
    printf("  anvil run model.gguf -p \"Explain quantum computing\" -n 200\n");
    printf("  anvil run model.gguf --grammar json.gbnf -p \"List 3 colors\"\n");
}

CliArgs parse_args(int argc, char ** argv) {
    CliArgs a;
    if (argc < 2) { a.help = true; return a; }
    int i = 1;
    if (i < argc && std::string(argv[i]) == "run") i++;
    for (; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--help" || arg == "-h")                          { a.help = true; }
        else if (arg == "--version")                                      { a.version = true; }
        else if ((arg == "-c" || arg == "--ctx") && i+1 < argc)           { if (!parse_int(argv[++i], a.n_ctx)) { fprintf(stderr, "error: invalid value for %s\n", arg.c_str()); a.invalid = true; a.help = true; } }
        else if ((arg == "-ngl" || arg == "--ngl" || arg == "--n-gpu-layers") && i+1 < argc) { if (!parse_int(argv[++i], a.ngl)) { fprintf(stderr, "error: invalid value for %s\n", arg.c_str()); a.invalid = true; a.help = true; } }
        else if ((arg == "-t" || arg == "--temp") && i+1 < argc)          { if (!parse_float(argv[++i], a.temp)) { fprintf(stderr, "error: invalid value for %s\n", arg.c_str()); a.invalid = true; a.help = true; } }
        else if (arg == "--threads" && i+1 < argc)                        { if (!parse_int(argv[++i], a.n_threads)) { fprintf(stderr, "error: invalid value for %s\n", arg.c_str()); a.invalid = true; a.help = true; } }
        else if (arg == "--flash-attn")                                   { a.flash_attn = true; }
        else if (arg == "--no-flash-attn")                                { a.no_flash_attn = true; a.flash_attn = false; }
        else if (arg == "--type-k" && i+1 < argc)                         { a.type_k = argv[++i]; }
        else if (arg == "--type-v" && i+1 < argc)                         { a.type_v = argv[++i]; }
        else if (arg == "--mtp")                                          { a.mtp = true; }
        else if (arg == "--triattn")                                      { a.triattn = true; }
        else if (arg == "--grammar" && i+1 < argc)                        { a.grammar = argv[++i]; }
        else if ((arg == "-s" || arg == "--system") && i+1 < argc)        { a.system_prompt = argv[++i]; }
        else if ((arg == "-p" || arg == "--prompt") && i+1 < argc)        { a.prompt = argv[++i]; }
        else if ((arg == "-n" || arg == "--max-tokens") && i+1 < argc)    { if (!parse_int(argv[++i], a.max_tokens)) { fprintf(stderr, "error: invalid value for %s\n", arg.c_str()); a.invalid = true; a.help = true; } }
        else if (arg[0] != '-')                                           { a.model = arg; }
        else { fprintf(stderr, "Unknown option: %s\n", arg.c_str()); a.invalid = true; a.help = true; }
    }
    return a;
}
