# Anvil

> **Forge anything.**
> A zero-jank, single-binary local AI runtime. The definitive Ollama alternative for developers who refuse slow defaults, opaque protocols, and resource hogs.

---

## What is it?

Anvil is a terminal-first local LLM tool that runs natively. One files, One binary. Zero runtime dependencies. No background daemon. No hidden blob storage. No telemetry. Just pure, raw, in-process inference.

| Status Quo | Anvil |
|---|---|
| Opaque blob storage | Models are plain GGUF files in `~/.anvil/models/` |
| Proprietary API | Drop-in OpenAI-compatible API, in-process (roadmap) |
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
git clone --recursive https://github.com/anvil-llm/anvil
cd anvil
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
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

---

## Why Anvil?

- **In-process by default.** Inference runs natively via C++ compiled binary. No HTTP roundtrip, no JSON per token, no subprocess overhead.
- **Zero jank.** 60 fps TUI. <2s startup. Background threads sleep when idle.
- **Everything is opt-in.** The core engine is the only thing that ships by default. Extras — API server, cloud proxy, finetuning — will be enabled and loaded only when you choose.
- **One binary.** Download `anvil` or run the curl installer. It just works. No extra files, no tracking, no telemetry.
- **Hardware-adaptive.** Auto-detects your CPU, GPU, RAM, and picks the fastest backend and settings. Zero config.
- **Transparent.** Plain GGUF models. Plain JSON config. Plain text logs. No hidden magic.

---

## Architecture

Anvil compiles a custom `llama.cpp` fork as a static library and links it directly into a single C++ binary.

| Layer | Implementation |
|---|---|
| **Engine** | `llama.cpp` fork with TurboQuant KV-cache types |
| **Interface** | Direct C++ `llama.h` (pure C API). Zero serialization overhead. |
| **TUI** | `FTXUI` — first-time setup wizard and chat REPL |

## Status

Currently implemented:

- `anvil run <model.gguf>` — interactive REPL and single-shot generation.
- First-time hardware-probing setup TUI.
- KV-cache compression presets, flash attention, MTP.
- Session export and basic REPL commands.

Not yet implemented (roadmap):

- `anvil serve` / OpenAI-compatible API
- Monitoring dashboard
- Cloud proxy
- Finetuning
- Self-updater

---

## Feature Matrix

| Feature | Status |
|---|---|
| Core in-process inference | ✅ |
| TurboQuant | ✅ |
| Metal / CUDA / Vulkan / CPU backends | ✅ |
| Session export | ✅ |
| Speculative decoding (MTP/NextN) | 🛠️ |
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
