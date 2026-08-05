#pragma once

#include "common.hpp"

ggml_type kv_type_from_name(const std::string & name);
const char * kv_type_short(ggml_type type);

void write_config(const AnvilConfig & cfg);
AnvilConfig load_config();
bool config_exists();
