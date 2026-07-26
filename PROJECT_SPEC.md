# Anvil — Forge Anything.

**A transparent, modular, terminal-first local AI runtime.**

Anvil is a single-binary, zero-dependency local LLM tool that links `llama.cpp` directly at the native level. It is designed to be the fastest, most transparent, and most jank-free local AI tool in existence. It is the definitive Ollama alternative for developers who refuse slow defaults, opaque protocols, or resource hogs.

---

## 1. Core Philosophy

- **Janklessness above all.** A dropped TUI framerate, a 3-second startup delay, a JSON roundtrip per token — all of it is unacceptable. Every millisecond must be justified.
- **Transparent by default.** No hidden blob storage, no opaque protocols. Models are plain GGUF files. Configs are plain JSON. Every default is documented and overridable.
- **Everything is opt-in.** The user chooses what they want. Nothing is forced. Not the TUI, not the API server, not even TurboQuant. We suggest the best defaults, but the user decides.
- **Maximum speed, zero bloat.** Inference runs in-process via native C++ FFI to `llama.cpp`. No subprocess spawn overhead. No HTTP serialization per token.
- **Terminal-first, human-friendly.** Drop a beginner into a TUI chat with one command. Give a pro total control via flags, config files.
- **One binary.** The user downloads `anvil`. It works. No extra files, no runtime dependencies.

---

## 2. The Problem with the Status Quo

### Ollama
- Opaque blob storage, custom non-standard API, always-on daemon, slow defaults, closed telemetry, unoptimized inference.

### llama.cpp
- Brilliant but a nightmare to configure. The user must manually set quants, context size, GPU layers, backend, and flags. Beginners give up.

### Anvil bridges both worlds.
| Status Quo Problem | Anvil's Fix |
|---|---|
| Opaque storage | Models are plain GGUF files in `~/.anvil/models/` |
| Custom API | Direct C++ integration, no HTTP layer needed |
| Slow/unoptimized defaults | Hardware probe + auto-optimization on first run |
| Always-on daemon | Zero idle overhead. The binary runs when you tell it to. |
| Hard to configure | `anvil run model.gguf` — it just works at max speed |
| Resource hog | TurboQuant + speculative enabled by default where applicable |

---

## 3. Architecture

Anvil is a **single C++ binary** that compiles the `llama.cpp` C++ backend (`backends/llama-turbo`) into itself as a static library. Inference is a native function call, not a subprocess or HTTP request.

### 3.1 The Core Engine (In-Process)

The `llama.cpp` codebase (`backends/llama-turbo` fork) is compiled as a static library via CMake. The resulting `anvil` binary contains:

- The full `llama.cpp` inference engine (CPU, Metal, CUDA, Vulkan backends)
- TurboQuant KV cache compression (WHT-rotated low-bit quantization)
- MTP (Multi-Token Prediction) speculative decoding for Gemma 4
- NextN speculative decoding for Qwen 3.x
- All model architectures supported by upstream llama.cpp

**Interface:** Direct C++ calls via `llama.h` (pure C API). Zero serialization overhead.

### 3.2 The Application Layer (`src/anvil.cpp`)

Single translation unit containing:

| Component | Description |
|---|---|
| **CLI Parsing** | `parse_args()` — manual flag parsing with friendly aliases |
| **Hardware Probing** | `probe_hw()` — cross-platform CPU/GPU/RAM detection |
| **Config Management** | `load_config()` / `write_config()` — JSON config at `~/.anvil/config.json` |
| **Setup TUI** | `run_setup_tui()` — FTXUI-based first-run interactive wizard |
| **Chat Engine** | `run_chat()` — interactive REPL with 9 commands, grammar support, streaming |
| **Session Export** | `export_session()` — saves conversations to `~/.anvil/sessions/` |
| **RAII Wrappers** | `LlamaModel`, `LlamaContext`, `LlamaSampler`, `ChatMessages` — automatic resource management |

### 3.3 Directory Structure

```
~/.anvil/
  config.json              # Global user defaults (readable, editable JSON)
  sessions/                # Chat session exports (Markdown)
  models/                  # Plain GGUF files (user-managed)
  cache/                   # HuggingFace download cache (future)
  logs/                    # Plain text logs (future)
```

The Anvil source tree:
```
Anvil/
├── src/
│   └── anvil.cpp              # Single translation unit — the entire application
├── backends/
│   └── llama-turbo/            # Git submodule: llama.cpp fork with TurboQuant
│       ├── include/llama.h     # C API header (FFI target)
│       ├── src/                # llama.cpp core engine
│       ├── ggml/               # GGML tensor library backends
│       ├── common/             # Shared utilities (sampling, chat, console)
│       └── conversion/         # HF→GGUF model conversion scripts (Python)
├── docs/
│   ├── index.html              # Project website / landing page
│   └── install.sh              # POSIX-compliant installer script
├── tests/
│   └── smoke.sh                # CLI smoke tests
├── .github/workflows/
│   └── anvil.yml               # CI pipeline (Linux, macOS, Windows)
├── CMakeLists.txt              # Build system
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
└── README.md                   # Project overview & usage
```

---

## 4. Build Pipeline

```bash
# Clone with submodules
git clone --recursive https://github.com/anvil-llm/anvil
cd anvil

# Build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Result: build/anvil (single binary, fully self-contained)
```

### 4.1 Build Details

1. **CMake** configures the project:
   - Builds `backends/llama-turbo` as a static library (`libllama.a`) with:
     - Metal backend on macOS
     - Vulkan backend on Linux (dlopen'd at runtime, no hard link)
     - CPU backend always available
   - Fetches FTXUI v7.0.1 for the setup TUI
   - Links everything into the final `anvil` binary

2. **Platform-specific linking:**
   - macOS: Metal, Foundation, Accelerate, QuartzCore, CoreFoundation, IOKit frameworks
   - Linux: pthread, dl, m
   - Windows: DXGI (planned)

### 4.2 GPU Backend Selection

| Detected Hardware | Backend | --ngl | Notes |
|---|---|---|---|
| Apple Silicon (M1-M4) | Metal (auto) | 99 | Full offload |
| Apple Intel | Metal (auto) | 0 | No GPU memory |
| NVIDIA GPU | CUDA (opt-in) | 99 | Requires `-DGGML_CUDA=ON` |
| AMD GPU (ROCm) | HIP (opt-in) | 99 | Requires `-DGGML_HIP=ON` |
| Intel Arc (SYCL) | SYCL (opt-in) | 99 | Requires `-DGGML_SYCL=ON` |
| No GPU / Vulkan | CPU / Vulkan | 0 | Vulkan if available |

---

## 5. The User Experience

### 5.1 Installation

```bash
curl -sSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/install.sh | sh
```

The installer is a single POSIX-compliant shell script that:
- Detects OS/arch
- Downloads the latest release binary
- Installs to `~/.anvil/bin/anvil`
- Optionally adds to PATH
- Runs hardware probe
- Done in < 10 seconds

Or build from source (for tinkerers):
```bash
git clone --recursive https://github.com/anvil-llm/anvil
cd anvil
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 5.2 First Run: `anvil run <model>`

```bash
anvil run model.gguf
```

1. **Hardware Probe (runs once):** Detects CPU features (AVX2/AVX-512), GPU (VRAM, type), RAM.
2. **Setup TUI:** Interactive configuration wizard for:
   - Context size (2K to max trained context)
   - KV cache compression presets (Recommended, Quality+, Max Compress, No Compress, Custom)
   - Flash attention toggle
   - Temperature presets (0.7–1.0)
3. **Launch:** Chat REPL opens with optimized settings.

### 5.3 Pro Mode: Full Control

```
bash
anvil run mymodel.gguf \
  --ctx 131072 \
  --ngl 99 \
  --temp 0.5 \
  --type-k q8_0 \
  --type-v turbo3 \
  --flash-attn \
  --mtp \
  --triattn \
  --grammar json.gbnf \
  -p "Explain quantum computing" \
  -n 200
```

### 5.4 Chat Commands

| Command | Function |
|---|---|
| `/exit` / `/quit` | Exit the application |
| `/clear` | Reset conversation and context |
| `/stats` | Show session statistics (tokens, speed, context usage) |
| `/undo` | Remove last user/assistant turn |
| `/export` | Save session to `~/.anvil/sessions/` as Markdown |
| `/model` | Display model architecture info |
| `/temp <f>` | Change temperature at runtime |
| `/ctx` | Show context usage bar |

---

## 6. Key Features & Implementation Details

### 6.1 TurboQuant KV Cache Compression

- **What:** WHT-rotated low-bit quantization for KV cache.
- **How:** Native in `llama.cpp` core (our fork). Enabled at runtime via `llama_context` params.
- **Options:**
  | Type | Compression | Quality Impact |
  |------|-------------|----------------|
  | f16 | 1x (baseline) | None |
  | q8_0 | ~2x | <0.1% loss |
  | turbo4 | ~4x | ~1% loss |
  | turbo3 | ~5.3x | ~2% loss |
  | turbo2 | ~8x | ~3% loss |
- **Presets:**
  1. **Recommended**: K=turbo4, V=turbo3 (4.2x, ~1% loss)
  2. **Quality+**: K=q8_0, V=turbo3 (~3x, <1% loss)
  3. **Max Compress**: K=turbo4, V=turbo2 (6.1x, ~3% loss)
  4. **No Compress**: K=f16, V=f16 (baseline)
  5. **Custom**: Manual K/V type selection
- **Default:** `turbo3` for both K and V caches.
- **Override:** `--type-k <type> --type-v <type>` or compile without the `turbo` feature.

### 6.2 MTP & NextN Speculative Decoding

- **MTP (Gemma 4):** Separate `gemma4_assistant` head loaded alongside the target model, cross-attending into the target's KV cache. Enabled via `--mtp` flag.
- **NextN (Qwen 3.x):** Shared-model draft context using the `_MTP.gguf` variant. Enabled via `--triattn` flag.
- **Both:** Configurable at runtime. No model changes needed.

### 6.3 Hardware-Adaptive Optimization

- **Probe:** Platform-specific APIs (IOKit on macOS, nvidia-smi/sysfs on Linux, DXGI on Windows).
- **Profile:** Maps hardware to best backend, GPU layer count, and context size.
- **Cached:** Result stored in `~/.anvil/config.json` after first run.
- **Override:** User can override any setting via CLI flags.

### 6.4 Flash Attention

- **Default:** Enabled (auto-selected for Metal backend).
- **Override:** `--flash-attn` / `--no-flash-attn`.

### 6.5 Grammar Support

- **Format:** GBNF grammar files for constrained generation.
- **Usage:** `--grammar json.gbnf -p "List 3 colors"`
- **Integration:** Added to the sampler chain at runtime.

### 6.6 Session Management

- **Auto-export:** Sessions saved to `~/.anvil/sessions/YYYYMMDD_HHMMSS.md`
- **Format:** Markdown with role-labeled messages (System, You, Assistant).
- **Trigger:** `/export` command in the chat REPL.

---

## 7. Code Quality & Safety

### 7.1 RAII Resource Management

All llama.cpp resources are wrapped in RAII types:

```cpp
struct LlamaModel {
    llama_model* p;
    ~LlamaModel() { if (p) llama_model_free(p); }
    // Move-only, no copies
};

struct LlamaContext {
    llama_context* p;
    ~LlamaContext() { if (p) llama_free(p); }
    // Move-only, no copies
};

struct LlamaSampler {
    llama_sampler* p;
    ~LlamaSampler() { if (p) llama_sampler_free(p); }
    // Move-only, no copies
    void reset(llama_sampler* p_);  // Safe replacement
};

struct ChatMessages {
    std::vector<llama_chat_message> m;
    ~ChatMessages() { clear(); }  // Frees all C-strings
    // Move-only, no copies
};
```

### 7.2 Safe Numeric Parsing

All user-provided numeric input is validated through safe helpers:

```cpp
static bool parse_int(const std::string& s, int& out);
static bool parse_float(const std::string& s, float& out);
static bool parse_uint64(const std::string& s, uint64_t& out);
```

These use try-catch blocks and verify the entire string was consumed, preventing crashes from malformed input.

### 7.3 UTF-8 Safety

The `Utf8Buffer` class ensures multi-byte UTF-8 characters are never split across token boundaries during streaming output.

---

## 8. Testing

### 8.1 Smoke Tests (`tests/smoke.sh`)

Tests 5 critical scenarios:
1. `--version` returns successfully
2. `--help` returns successfully
3. Invalid model file returns non-zero exit code
4. Invalid numeric argument returns non-zero exit code
5. Unknown option returns non-zero exit code

### 8.2 CI Pipeline (`.github/workflows/anvil.yml`)

GitHub Actions workflow that:
- Builds on ubuntu-latest, macos-latest, windows-latest
- Installs dependencies (CMake, compilers)
- Configures and builds with CMake
- Runs smoke tests
- Caches dependencies for faster subsequent builds

---

## 9. Feature Matrix

| Feature | Status |
|---|---|
| Core in-process inference | ✅ |
| TurboQuant KV cache compression | ✅ |
| Metal / CUDA / Vulkan / CPU backends | ✅ |
| Hardware auto-probe | ✅ |
| Setup TUI (first-run wizard) | ✅ |
| Chat REPL (`anvil run`) | ✅ |
| Session export | ✅ |
| Grammar support | ✅ |
| Flash attention | ✅ |
| MTP speculative decoding | ✅ |
| TriAttention KV eviction | ✅ |
| CLI argument validation | ✅ |
| RAII resource management | ✅ |
| Smoke tests + CI | ✅ |
| Speculative decoding (NextN) | 🛠️ |
| Monitoring dashboard | 🛠️ |
| OpenAI API server (`anvil serve`) | 🛠️ |
| Self-updater (`anvil self-update`) | 🛠️ |
| Cloud proxy | 🛠️ |
| Finetuning | 🛠️ |

---

## 10. MVP Roadmap

### Phase 1: The Core Runner ✅ (Complete)
- [x] C++ project scaffold with CMake + llama.cpp static dependency
- [x] Direct C++ calls to `llama.h` API (model load, decode, sampling)
- [x] `anvil run <model>` with in-process inference and interactive REPL
- [x] Hardware prober (OS, CPU, GPU, RAM)
- [x] First-run setup TUI (FTXUI)
- [x] Config persistence (`~/.anvil/config.json`)
- [x] RAII wrappers for all llama.cpp resources
- [x] Safe numeric parsing for all user input
- [x] UTF-8 safe streaming output

### Phase 2: The TUI (Enhancements)
- [ ] ratatui-based multi-pane chat interface (future)
- [ ] Real-time memory/performance visualization
- [ ] Model list / monitor mode

### Phase 3: The API Server
- [ ] axum OpenAI-compatible server (future Rust integration)
- [ ] Streaming and non-streaming completions
- [ ] In-process token generation (no serialization)

### Phase 4: Smarts
- [ ] Auto-optimizer based on hardware probe (enhanced)
- [ ] Self-updater (`anvil self-update`)
- [ ] Model downloader from HuggingFace

### Phase 5: Power Tools
- [ ] Cloud proxy (OpenAI, Anthropic)
- [ ] Finetuning pipeline
- [ ] Cross-compilation CI (macOS Intel/ARM, Linux x64, Windows x64)

---

## 11. Design Principles

1. **Janklessness is non-negotiable.** If it's slow, it doesn't ship.
2. **Opt-in, not opt-out.** Every feature beyond the core engine is optional.
3. **No hidden magic.** Auto-selected settings are visible and overridable.
4. **Files are plain.** Models are files. Configs are JSON. Logs are text.
5. **One binary.** The user downloads `anvil`. The end.
6. **Single translation unit.** Keep the core runtime in `src/anvil.cpp` for simplicity and fast compilation.

---

## 12. Brand

- **Name:** Anvil
- **CLI:** `anvil`
- **Tagline:** Forge anything.
- **Color Identity:** Deep charcoal, steel grey, forge-orange. Industrial, solid, no-bullshit.

---

*Anvil: the anvil doesn't ask questions. It just works.*
