#pragma once

#include "common.hpp"

HWInfo probe_hw();
int derive_ngl(const HWInfo & hw);
bool validate_gguf(const std::string & path);
