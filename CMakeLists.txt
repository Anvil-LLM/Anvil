cmake_minimum_required(VERSION 3.22)
project(anvil C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── llama.cpp TurboQuant backend (git submodule) ──
set(LLAMA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/backends/llama-turbo)

set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)

# GPU backends default OFF for portable binaries; CI/users opt in.
set(GGML_METAL  OFF CACHE BOOL "")
set(GGML_VULKAN OFF CACHE BOOL "")
set(GGML_CUDA   OFF CACHE BOOL "")

# macOS always has Metal.
if(APPLE)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(${LLAMA_DIR} ${CMAKE_BINARY_DIR}/llama-build EXCLUDE_FROM_ALL)

# ── FTXUI ──
include(FetchContent)
FetchContent_Declare(ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
    GIT_TAG        v7.0.1
    GIT_SHALLOW    TRUE
)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ftxui)

# ── anvil ──
add_executable(anvil src/anvil.cpp)

target_include_directories(anvil PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

target_link_libraries(anvil PRIVATE llama ftxui::ftxui)

if(APPLE)
    target_link_libraries(anvil PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework Accelerate"
        "-framework QuartzCore"
        "-framework CoreFoundation"
        "-framework IOKit"
        "-lc++"
        "-lobjc"
    )
elseif(UNIX)
    target_link_libraries(anvil PRIVATE pthread dl m)
    # ggml-vulkan dlopens libvulkan.so at runtime and falls back to CPU if
    # absent. Do NOT link Vulkan::Vulkan here — that would hard-link libvulkan
    # and crash on machines without it. Headers only, for the build.
    if(GGML_VULKAN)
        find_package(Vulkan QUIET)
    endif()
endif()

install(TARGETS anvil RUNTIME DESTINATION bin)
