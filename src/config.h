#pragma once
#include "common.h"

std::string config_dir();
std::string config_path();
std::string sessions_dir();
bool config_exists();
AnvilConfig load_config();
void write_config(const AnvilConfig & cfg);
