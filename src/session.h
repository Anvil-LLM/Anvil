#pragma once
#include "common.h"

std::string session_path();
void export_session(const std::vector<ChatMessage> & msgs, const std::string & path);
