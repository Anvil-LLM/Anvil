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

## Coding style

- Keep related code colocated. Split into modules when a function exceeds ~200 lines.
- Use RAII for resource management.
- Never use `std::stoi`/`std::stof` without validation; use the existing
  `parse_int`/`parse_float` helpers.
- Update this document if you add new build or test steps.
