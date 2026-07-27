# Contributing to Anvil

Thanks for taking the time to contribute.

## Building

Anvil is a single C++ binary. It needs a C++17 compiler, CMake 3.22+, and the
`backends/llama-turbo` submodule.

```bash
git clone --recursive https://github.com/anvil-llm/anvil
cd anvil
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Testing

Run the smoke tests after a build:

```bash
bash tests/smoke.sh
```

## Architecture

```
src/
├── common.h        # Shared types, utilities, version info
├── anvil.cpp       # Core inference engine (monolithic, depends on llama.cpp)
├── hardware.cpp    # Cross-platform hardware probing (no llama deps)
├── config.cpp      # JSON config read/write (no llama deps)
├── cli.cpp         # CLI argument parsing (no llama deps)
├── tui.cpp         # Setup TUI wizard (FTXUI, no llama deps)
└── session.cpp     # Session export to Markdown (no llama deps)
```

External feature modules (`hardware`, `config`, `cli`, `tui`, `session`) have
**zero dependency on `llama.h`/`ggml.h`** and can be compiled and tested
independently.

## Coding style

- Keep the core inference engine in `src/anvil.cpp` (monolithic).
- External features go in their own `.cpp`/`.h` files under `src/`.
- Use RAII for resource management.
- Never use `std::stoi`/`std::stof` without validation; use the existing
  `parse_int`/`parse_float` helpers from `common.h`.
- Update this document if you add new build or test steps.
