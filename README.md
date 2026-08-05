# Anvil

> **Forge anything.**
> A zero-jank, single-binary local AI runtime. The definitive Ollama alternative for developers who refuse slow defaults, opaque protocols, and resource hogs.

---

## What is it?

Anvil is a terminal-first local LLM tool that runs natively. One file. One binary. Zero runtime dependencies. No background daemon. No hidden blob storage. No telemetry. Just pure, raw, in-process inference.

| Status Quo | Anvil |
|---|---|
| Opaque blob storage | Models are plain GGUF files in `~/.anvil/models/` |
| Proprietary API | Drop-in OpenAI-compatible API, in-process (roadmap) |
| Always-on daemon | Zero idle overhead; the binary runs when you tell it to |
| Slow, unoptimized defaults | Hardware probe + auto-optimization on first run |
| Hard to configure | `anvil run model.gguf` — it just works at max speed |
| Resource hog | TurboQuant + speculative enabled, where it matters |

---

## Install

```bash
curl -fsSL https://anvil-llm.github.io/anvil/install.sh | sh
```

Or build from source:

```bash
git clone --recursive https://github.com/anvil-llm/anvil
cd anvil
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

---

## NVIDIA / CUDA (Linux)

On Linux x86_64 with an NVIDIA GPU, the installer automatically **prefers the
CUDA prebuilt** (`anvil-linux-x86_64-cuda`) and falls back to the CPU/Vulkan
build if it isn't available. The CUDA build needs **only the NVIDIA driver** —
the CUDA runtime is bundled, so there's no CUDA toolkit to install.

If the driver is missing, anvil points you at the driver helper
(`docs/anvil-nvidia-install.sh`), which installs NVIDIA's official driver on
**any distro, any NVIDIA card** — no distro-specific packages needed:

```bash
curl -fsSL -O https://raw.githubusercontent.com/anvil-llm/anvil/main/docs/anvil-nvidia-install.sh
chmod +x anvil-nvidia-install.sh
./anvil-nvidia-install.sh --check      # report only, changes nothing
sudo ./anvil-nvidia-install.sh         # universal install (NVIDIA .run + DKMS)
```

It reads your GPU generation and picks the right driver branch automatically:
Turing and newer (RTX, GTX 16xx) get the latest driver with open kernel
modules; Pascal/Maxwell (GTX 10xx/9xx) get the 580 series; Kepler (GTX 6xx/7xx)
gets the 470 series. DKMS keeps the module rebuilt across kernel updates, and
Secure Boot (MOK) is handled. Distro-package installs remain available via
`--distro`, and Frogging-Family's nvidia-all via `--nvidia-all` (Arch).

Force the CUDA build on NVIDIA hardware (or any x86_64 Linux):

```bash
curl -fsSL https://anvil-llm.github.io/anvil/install.sh | sh -s -- --nvidia
```

---

## Usage

### Run a model (chat REPL)

```bash
anvil run model.gguf
```

### Full control

```bash
anvil run model.gguf --ctx 128000 --ngl -1 --type-k turbo4 --type-v turbo3 --mtp
```

### Model registry, profiles & pulls

Models get **friendly names** with persistent **per-model profiles** (settings
remembered across runs). Running a local file auto-registers it:

```bash
anvil models                          # list registered models
anvil models import model.gguf --name llama3.2
anvil run llama3.2                    # friendly name, profile auto-applied
anvil run llama3.2 --temp 0.9 --save  # ...and persist this override
anvil profile llama3.2                # show the profile
anvil profile llama3.2 set n_ctx=131072 type_k=turbo4
anvil rm llama3.2 [--yes]             # unregister (--yes also deletes the file)
```

Pull models straight from the **Ollama registry** — Ollama models *are* GGUF, so
anvil downloads the weights blob and extracts the chat template, license and
sampling params (which become the profile defaults):

```bash
anvil pull ollama:llama3.2:3b        # registry pull (namespace defaults to library)
```

Already use `ollama`? Import without re-downloading — anvil hardlinks the
blobs into its own store (falling back to direct references across
filesystems), and respects the `OLLAMA_MODELS` env var:

```bash
anvil pull ollama-local:llama3.2:3b
```

Pull GGUF quants straight from **HuggingFace**. With no file given, anvil
lists the repo's `.gguf` files (smallest first) and asks which quant to take;
pass the file explicitly to skip the picker, or use `--list` to just browse:

```bash
anvil pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF
anvil pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf
anvil pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF --list
```

Every download is verified against its registry/LFS sha256 and resumable
(`.part` files); a corrupt or stale partial is detected and restarted
automatically. Safetensors-only repos need the HF converter:
`backends/llama-turbo/convert_hf_to_gguf.py`.

**Precedence:** CLI flags > model profile > global config. Profiles live in
`~/.anvil/models.json`; every write is atomic (tmp + rename).

---

## Why Anvil?

- **In-process by default.** Inference runs natively via C++ compiled binary. No HTTP roundtrip, no JSON per token, no subprocess overhead.
- **Zero jank.** <2s startup. No background threads when idle.
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
| Metal (macOS) / CUDA & Vulkan (opt-in builds) / CPU | ✅ |
| Session export | ✅ |
| Speculative decoding (MTP/NextN) | 🛠️ |
| Chat REPL (`anvil run`) | ✅ |
| Monitoring dashboard | 🛠️ |
| OpenAI API server (`anvil serve`) | 🛠️ |
| Hardware auto-probe | 🛠️ |
| Self-updater (`anvil self-update`) | 🛠️ |
| Cloud proxy | 🛠️ |
| Finetuning | 🛠️ |

---

## Contributing

We welcome PRs, issues, and feature requests for architecture details.

---

## License

MIT License.

---

*The anvil doesn't ask questions. It just works.*
