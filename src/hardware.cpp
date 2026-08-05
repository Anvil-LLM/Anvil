#include "hardware.hpp"

#include <algorithm>
#include <sstream>

#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif
#ifdef __linux__
#include <glob.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi.h>
#ifdef _MSC_VER
#pragma comment(lib, "dxgi.lib")
#endif
#endif

#ifdef __APPLE__
static void detect_gpus_macos(HWInfo & hw) {
    CFMutableDictionaryRef matching = IOServiceMatching("IOGPU");
    if (!matching) return;

    mach_port_t mp = MACH_PORT_NULL;
    const kern_return_t main_ret = IOMainPort(MACH_PORT_NULL, &mp);
    if (main_ret != KERN_SUCCESS) { CFRelease(matching); return; }

    io_iterator_t iter = 0;
    const kern_return_t kr = IOServiceGetMatchingServices(mp, matching, &iter);
    if (kr != KERN_SUCCESS) return;  // matching consumed by the call

    io_object_t device;
    while ((device = IOIteratorNext(iter)) != 0) {
        GPUInfo gpu;
        gpu.vendor = "Apple";
        gpu.is_discrete = false;
        gpu.vram_mb = 0;
        gpu.name = "Apple GPU (unified " +
                   std::to_string(hw.ram_bytes / (1024ULL * 1024 * 1024)) + " GB)";
        hw.gpus.push_back(gpu);
        IOObjectRelease(device);
    }
    IOObjectRelease(iter);
}
#endif

#ifdef __linux__
static std::string run_cmd(const std::string & cmd) {
    std::string out;
    FILE * pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    char buf[4096];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        out.append(buf, n);
    }
    pclose(pipe);
    return out;
}

static void detect_gpus_linux(HWInfo & hw) {
    // nvidia-smi (when the proprietary driver is present)
    const std::string out = run_cmd(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null");
    if (!out.empty()) {
        std::istringstream ss(out);
        std::string line;
        while (std::getline(ss, line)) {
            while (!line.empty() && line.back() <= ' ') line.pop_back();
            if (line.empty()) continue;
            GPUInfo gpu;
            gpu.vendor = "NVIDIA";
            gpu.is_discrete = true;

            const auto comma = line.rfind(',');
            if (comma != std::string::npos) {
                std::string vram_str = line.substr(comma + 1);
                while (!vram_str.empty() && vram_str.back() <= ' ') vram_str.pop_back();
                while (!vram_str.empty() && vram_str.front() == ' ') vram_str.erase(vram_str.begin());
                uint64_t vram = 0;
                if (parse_uint64(vram_str, vram)) gpu.vram_mb = vram;
                gpu.name = line.substr(0, comma);
                while (!gpu.name.empty() && gpu.name.back() == ' ') gpu.name.pop_back();
            } else {
                gpu.name = line;
            }
            hw.gpus.push_back(gpu);
        }
    }

    // sysfs PCI enumeration (works with any driver, incl. nouveau/amdgpu)
    glob_t globbuf;
    if (glob("/sys/class/drm/card*/device/vendor", 0, nullptr, &globbuf) == 0) {
        for (size_t i = 0; i < globbuf.gl_pathc; i++) {
            std::ifstream vf(globbuf.gl_pathv[i]);
            std::string vendor_id;
            std::getline(vf, vendor_id);
            while (!vendor_id.empty() && vendor_id.back() <= ' ') vendor_id.pop_back();

            std::string base(globbuf.gl_pathv[i]);
            const auto pos = base.rfind("/device/vendor");
            if (pos == std::string::npos) continue;
            base = base.substr(0, pos);

            GPUInfo gpu;
            if (vendor_id == "0x1002") {
                gpu.vendor = "AMD";
                gpu.is_discrete = true;
            } else if (vendor_id == "0x8086") {
                gpu.vendor = "Intel";
                gpu.is_discrete = false;
            } else {
                continue;  // vendor card* entries may duplicate; skip non-GPU vendors
            }
            std::ifstream df(base + "/device/device");
            std::string did;
            std::getline(df, did);
            gpu.name = gpu.vendor + " GPU (" + did + ")";

            std::ifstream vf2(base + "/device/mem_info_vram_total");
            if (vf2.good()) {
                uint64_t bytes = 0;
                vf2 >> bytes;
                gpu.vram_mb = bytes / (1024 * 1024);
            }
            hw.gpus.push_back(gpu);
        }
        globfree(&globbuf);
    }
}
#endif

#ifdef _WIN32
static void detect_gpus_windows(HWInfo & hw) {
    IDXGIFactory * factory = nullptr;
    if (CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void **>(&factory)) != S_OK) return;
    IDXGIAdapter * adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC desc;
        if (adapter->GetDesc(&desc) == S_OK) {
            GPUInfo gpu;
            const int needed = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, nullptr, 0, nullptr, nullptr);
            if (needed > 1) {
                std::string name_buf(static_cast<size_t>(needed), '\0');
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf.data(), needed, nullptr, nullptr);
                if (!name_buf.empty() && name_buf.back() == '\0') name_buf.pop_back();
                gpu.name = std::move(name_buf);
            } else {
                gpu.name = "Unknown";
            }
            gpu.vram_mb = std::max(desc.DedicatedVideoMemory, desc.SharedSystemMemory) / (1024 * 1024);
            gpu.is_discrete = (desc.VendorId == 0x10DE || desc.VendorId == 0x1002);
            if (desc.VendorId == 0x10DE)      gpu.vendor = "NVIDIA";
            else if (desc.VendorId == 0x1002) gpu.vendor = "AMD";
            else if (desc.VendorId == 0x8086) gpu.vendor = "Intel";
            else                              gpu.vendor = "Other";
            hw.gpus.push_back(gpu);
        }
        adapter->Release();
    }
    factory->Release();
}
#endif

HWInfo probe_hw() {
    HWInfo hw;

#if defined(__APPLE__)
    hw.os = "macos";
#elif defined(__linux__)
    hw.os = "linux";
#elif defined(_WIN32)
    hw.os = "windows";
#else
    hw.os = "unknown";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    hw.arch = "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    hw.arch = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    hw.arch = "i386";
#else
    hw.arch = "unknown";
#endif

    hw.cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hw.cpu_threads <= 0) hw.cpu_threads = 0;

#ifdef __APPLE__
    {
        std::string buf;
        size_t len = 0;
        if (sysctlbyname("machdep.cpu.brand_string", nullptr, &len, nullptr, 0) == 0 && len > 0) {
            buf.resize(len);
            if (sysctlbyname("machdep.cpu.brand_string", &buf[0], &len, nullptr, 0) == 0) {
                while (!buf.empty() && buf.back() == '\0') buf.pop_back();
                hw.cpu = buf;
                hw.apple_silicon = (hw.cpu.find("Apple") != std::string::npos);
            }
        }
    }
    {
        uint64_t ram = 0;
        size_t len = sizeof(ram);
        if (sysctlbyname("hw.memsize", &ram, &len, nullptr, 0) == 0) {
            hw.ram_bytes = ram;
        }
    }
    detect_gpus_macos(hw);
#elif defined(__linux__)
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("model name") != std::string::npos) {
                const auto pos = line.find(':');
                if (pos != std::string::npos) {
                    hw.cpu = line.substr(pos + 2);
                    while (!hw.cpu.empty() && hw.cpu.back() <= ' ') hw.cpu.pop_back();
                }
                break;
            }
        }
    }
    {
        std::ifstream f("/proc/meminfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("MemTotal") != std::string::npos) {
                std::istringstream iss(line);
                std::string key;
                uint64_t kb = 0;
                std::string unit;
                if ((iss >> key >> kb >> unit) && key == "MemTotal" && unit == "kB") {
                    hw.ram_bytes = kb * 1024;
                }
                break;
            }
        }
    }
    detect_gpus_linux(hw);
#elif defined(_WIN32)
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        hw.arch = (si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64) ? "aarch64" : "x86_64";
        hw.cpu_threads = static_cast<int>(si.dwNumberOfProcessors);
        hw.cpu = "Windows CPU";
    }
    {
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            hw.ram_bytes = ms.ullTotalPhys;
        }
    }
    detect_gpus_windows(hw);
#endif
    return hw;
}

int derive_ngl(const HWInfo & hw) {
    // Apple Silicon and any discrete GPU: offload everything (-1 = auto).
    if (hw.apple_silicon) return -1;
    if (!hw.gpus.empty()) return -1;
    return 0;
}

bool validate_gguf(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4] = {0, 0, 0, 0};
    f.read(magic, 4);
    return f.gcount() == 4 && memcmp(magic, "GGUF", 4) == 0;
}
