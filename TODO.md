# Refactoring Plan: Extract External Features from anvil.cpp

## Progress

- [x] Step 1: Create `src/common.h` — Shared Types & Utilities
- [x] Step 2: Create `src/hardware.h` + `src/hardware.cpp` — Hardware Probing
- [x] Step 3: Create `src/config.h` + `src/config.cpp` — Configuration
- [x] Step 4: Create `src/cli.h` + `src/cli.cpp` — CLI Argument Parsing
- [x] Step 5: Create `src/tui.h` + `src/tui.cpp` — Setup TUI
- [x] Step 6: Create `src/session.h` + `src/session.cpp` — Session Export
- [x] Step 7: Modify `src/anvil.cpp` — Core Engine (remove extracted code, add includes)
- [x] Step 8: Update `CMakeLists.txt` — Add new source files
- [x] Step 9: Update `CONTRIBUTING.md` — Architecture docs
- [x] Step 10: Update `PROJECT_SPEC.md` — Architecture docs
- [x] Step 11: Build & Test
- [ ] Step 12: Push to GitHub
