#include "session.h"
#include "config.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

std::string session_path() {
    namespace fs = std::filesystem;
    fs::create_directories(sessions_dir());
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
    return sessions_dir() + "/" + std::string(buf) + ".md";
}

void export_session(const std::vector<ChatMessage> & msgs, const std::string & path) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "\033[31mcould not write session to %s\033[0m\n", path.c_str());
        return;
    }
    f << "# Anvil Chat Session\n\n";
    for (const auto & m : msgs) {
        if (m.role == "system")    f << "**[System]** " << m.content << "\n\n";
        else if (m.role == "user") f << "**[You]** " << m.content << "\n\n";
        else                       f << "**[Assistant]** " << m.content << "\n\n";
    }
    fprintf(stderr, "\033[32mSession exported to %s\033[0m\n", path.c_str());
}
