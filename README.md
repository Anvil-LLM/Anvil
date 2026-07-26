# Anvil

> **Forge anything.**
> A zero-jank, single-binary local AI runtime. The definitive Ollama alternative for developers who refuse slow defaults, opaque protocols, and resource hogs.

---

## What is it?

Anvil is a terminal-first local LLM tool that runs natively. One files, One binary. Zero runtime dependencies. No background daemon. No hidden blob storage. No telemetry. Just pure, raw, in-process inference.

| Status Quo | Anvil |
|---|---|
| Opaque blob storage | Models are plain GGUF files in `~/.anvil/models/` |
| Proprietary API | Drop-in OpenAI-compatible API, in-process |
| Always-on daemon | Zero idle overhead; the binary runs when you tell it to |
| Slow, unoptimized defaults | Hardware probe + auto-optimization on first run |
| Hard to configure | `anvil run llama3` — it just works at max speed |
| Resource hog | TurboQuant + speculative enabled, where it matters |

---

## Install

```bash
curl -sSL https://raw.githubusercontent.com/gondaliyashreyan1/Anvil/main/install.sh | sh
```

Or build from source:

```bash
git clone https://github.com/gondaliyashreyan1/Anvil
cd Anvil
cargo build --release
```

---

## Usage

### Run a model (TUI chat)

```bash
anvil run llama3.1
```

### Run a model (raw REPL)

```bash
anvil run llama3.1 --no-tui
```

### Full control

```bash
anvil run mymodel.gguf \
  --backend metal \
  --quant Q4_K_M \
  --ctx 128000 \
  --spec-type mtp \
  --ngl 99
```

### Serve an API

```bash
anvil serve
# OpenAI-compatible API on localhost:11434
# In-process. No subprocess. No serialization.
```

---

## Why Anvil?

- **In-process by default.** Inference runs natively via C++ compiled binary. No HTTP roundtrip, no JSON per token, no subprocess overhead.
- **Zero jank.** 60 fps TUI. <2s startup. Background threads sleep when idle.
- **Everything is opt-in.** The core engine is the only thing that ships by default. TUI, API server, cloud proxy, finetuning — enabled and loaded only when you choose.
- **One binary.** Download `anvil` or run the curl installer. It just works. No extra files, no tracking, no telemetry.
- **Hardware-adaptive.** Auto-detects your CPU, GPU, RAM, and picks the fastest backend and settings. Zero config.
- **Transparent.** Plain GGUF models. Plain JSON config. Plain text logs. No hidden magic.

---

## Architecture

Anvil compiles `llama.cpp` as a static library and links it directly into a single Rust binary.

| Layer | Implementation |
|---|---|
| **Engine** | `llama.cpp` (TurboQuant, speculative decoding, Metal/CUDA/Vulkan/CPU) |
| **Interface** | Direct C++ `llama.h` (pure C API). Zero serialization overhead. |
| **Orchestrator** | Rust: model lifecycle, inference loop, hardware probing, configuration |
| **TUI** | `ratatui` — multi-pane chat, monitoring, real-time stats |
| **API** | `axum` — OpenAI-compatible HTTP, in-process routing |
| **Cloud** | Optional feature-gated proxy (OpenAI, Anthropic, Gemini) |
| **Finetune** | Optional feature-gated training pipeline |

---

## Feature Matrix

| Feature | Status |
|---|---|
| Core in-process inference | ✅ |
| TurboQuant | ✅ |
| Metal / CUDA / Vulkan / CPU backends | ✅ |
| Speculative decoding (MTP/NextN) | ✅ |
| TUI chat (`anvil run`) | ✅ |
| Monitoring dashboard | 🛠️ |
| OpenAI API server (`anvil serve`) | 🛠️ |
| Hardware auto-probe | 🛠️ |
| Self-updater (`anvil self-update`) | 🛠️ |
| Cloud proxy | 🛠️ |
| Finetuning | 🛠️ |

---

## Contributing

We welcome PRs, issues, and feature requests. See the full [Project Spec](./PROJECT_SPEC.md) for architecture and roadmap details.

---

## License

MIT License.

---

*The anvil doesn't ask questions. It just works.*
