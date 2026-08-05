#include "setup.hpp"
#include "hardware.hpp"
#include "cli.hpp"

#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <iostream>

AnvilConfig run_setup_tui(const HWInfo & hw, int max_ctx) {
    AnvilConfig cfg;
    cfg.ngl = derive_ngl(hw);
    cfg.n_ctx = max_ctx > 0 ? max_ctx : 0;

    // Context options: powers of two up to the model's trained context.
    std::vector<int> ctx_options;
    for (int c = 1; c > 0 && c <= max_ctx && c <= INT32_MAX / 2; c *= 2) {
        ctx_options.push_back(c);
    }
    ctx_options.push_back(0);  // "Custom..."
    const int custom_idx = static_cast<int>(ctx_options.size()) - 1;
    int ctx_sel = 0;
    for (size_t i = 0; i < ctx_options.size(); i++) {
        if (ctx_options[i] == cfg.n_ctx) { ctx_sel = static_cast<int>(i); break; }
    }

    std::vector<std::string> ctx_labels;
    for (const int c : ctx_options) {
        if (c == 0) ctx_labels.push_back("Custom...");
        else if (c >= 1024) ctx_labels.push_back(std::to_string(c / 1024) + "K tokens");
        else ctx_labels.push_back(std::to_string(c) + " tokens");
    }

    std::vector<std::string> kv_preset_labels;
    for (int i = 0; i < KV_PRESETS_COUNT; i++) kv_preset_labels.push_back(KV_PRESETS[i].label);

    std::vector<std::string> kv_type_labels;
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) kv_type_labels.push_back(KV_OPTIONS[i].label);

    const std::vector<std::string> fa_labels   = {"on (recommended)", "off"};
    const std::vector<std::string> temp_labels = {"0.7 (focused)", "0.8 (balanced)", "0.9 (creative)", "1.0 (wild)"};

    int kv_preset_sel = 0;
    int kv_k_sel      = 1;
    int kv_v_sel      = 3;
    int fa_sel        = 0;
    int temp_sel      = 1;
    bool custom_kv    = false;
    int menu_idx      = 0;

    std::vector<std::string> hw_lines;
    hw_lines.push_back("CPU  : " + hw.cpu);
    hw_lines.push_back("RAM  : " + std::to_string(hw.ram_bytes / (1024ULL * 1024 * 1024)) + " GB");
    hw_lines.push_back("Threads: " + std::to_string(hw.cpu_threads));
    for (const auto & gpu : hw.gpus) {
        hw_lines.push_back("GPU  : " + gpu.name + " (" + std::to_string(gpu.vram_mb) + " MB)");
    }

    auto screen = ftxui::ScreenInteractive::FitComponent();
    auto component = ftxui::Renderer([&]() {
        ftxui::Elements rows;
        rows.push_back(ftxui::text(ANVIL_LOGO) | ftxui::bold | ftxui::color(ftxui::Color::Yellow));
        rows.push_back(ftxui::text("  First-time setup  v" + std::string(ANVIL_VERSION)) | ftxui::bold | ftxui::color(ftxui::Color::Cyan));
        rows.push_back(ftxui::text(" "));
        rows.push_back(ftxui::text("  Detected hardware:") | ftxui::bold);
        for (const auto & line : hw_lines) rows.push_back(ftxui::text("    " + line));
        rows.push_back(ftxui::text(" "));

        auto render_setting = [&](const std::string & label, const std::vector<std::string> & opts,
                                  int sel, int idx, int indent = 2) {
            const std::string pad(static_cast<size_t>(indent), ' ');
            const bool active = (idx == menu_idx);
            auto lbl = ftxui::text(pad + label);
            if (active) lbl = lbl | ftxui::bold | ftxui::color(ftxui::Color::Cyan);
            else        lbl = lbl | ftxui::bold;
            rows.push_back(lbl);
            for (int i = 0; i < static_cast<int>(opts.size()); i++) {
                const bool selected = (i == sel);
                auto prefix = selected
                    ? ftxui::text(pad + "  ▸ ") | ftxui::bold | ftxui::color(ftxui::Color::Green)
                    : ftxui::text(pad + "  · ");
                auto label_el = ftxui::text(opts[static_cast<size_t>(i)]);
                if (selected && active) label_el = label_el | ftxui::bold | ftxui::color(ftxui::Color::Green);
                else if (selected)      label_el = label_el | ftxui::color(ftxui::Color::Green);
                rows.push_back(ftxui::hbox({prefix, label_el}));
            }
            rows.push_back(ftxui::text(" "));
        };

        render_setting("Context size", ctx_labels, ctx_sel, 0);
        render_setting("KV cache compression", kv_preset_labels, kv_preset_sel, 1);
        if (custom_kv) {
            render_setting("  K cache type", kv_type_labels, kv_k_sel, 4, 4);
            render_setting("  V cache type", kv_type_labels, kv_v_sel, 5, 4);
        }
        render_setting("Flash attention", fa_labels, fa_sel, 2);
        render_setting("Temperature", temp_labels, temp_sel, 3);

        std::string kv_summary;
        if (!custom_kv) {
            const int ki = KV_PRESETS[kv_preset_sel].k_idx;
            const int vi = KV_PRESETS[kv_preset_sel].v_idx;
            if (ki >= 0 && vi >= 0)
                kv_summary = "  Active: K=" + std::string(KV_OPTIONS[ki].short_name) +
                             " V=" + std::string(KV_OPTIONS[vi].short_name);
        } else {
            kv_summary = "  Active: K=" + std::string(KV_OPTIONS[kv_k_sel].short_name) +
                         " V=" + std::string(KV_OPTIONS[kv_v_sel].short_name);
        }
        rows.push_back(ftxui::text(kv_summary) | ftxui::dim);
        rows.push_back(ftxui::text(" "));
        rows.push_back(ftxui::text("  Tab switch  ↑/↓ change  Enter confirm  q cancel") | ftxui::dim);

        return ftxui::vbox(std::move(rows));
    });

    auto wrapped = component | ftxui::CatchEvent([&](ftxui::Event e) {
        if (e == ftxui::Event::Character('q')) { screen.Exit(); return true; }
        if (e == ftxui::Event::Return) { screen.Exit(); return true; }

        const int menu_count = custom_kv ? 6 : 4;
        if (e == ftxui::Event::Tab) {
            menu_idx = (menu_idx + 1) % menu_count;
            if (!custom_kv && menu_idx >= 4) menu_idx = 0;
            return true;
        }
        if (e == ftxui::Event::TabReverse) {
            menu_idx = (menu_idx - 1 + menu_count) % menu_count;
            if (!custom_kv && menu_idx >= 4) menu_idx = 3;
            return true;
        }
        auto cycle = [](int & sel, int count, int dir) {
            sel = (sel + dir + count) % count;
        };
        if (e == ftxui::Event::ArrowUp) {
            switch (menu_idx) {
                case 0: cycle(ctx_sel, static_cast<int>(ctx_labels.size()), -1); break;
                case 1:
                    cycle(kv_preset_sel, KV_PRESETS_COUNT, -1);
                    custom_kv = (kv_preset_sel == KV_PRESETS_COUNT - 1);
                    break;
                case 2: cycle(fa_sel, static_cast<int>(fa_labels.size()), -1); break;
                case 3: cycle(temp_sel, static_cast<int>(temp_labels.size()), -1); break;
                case 4: cycle(kv_k_sel, KV_OPTIONS_COUNT, -1); break;
                case 5: cycle(kv_v_sel, KV_OPTIONS_COUNT, -1); break;
                default: break;
            }
            return true;
        }
        if (e == ftxui::Event::ArrowDown) {
            switch (menu_idx) {
                case 0: cycle(ctx_sel, static_cast<int>(ctx_labels.size()), 1); break;
                case 1:
                    cycle(kv_preset_sel, KV_PRESETS_COUNT, 1);
                    custom_kv = (kv_preset_sel == KV_PRESETS_COUNT - 1);
                    break;
                case 2: cycle(fa_sel, static_cast<int>(fa_labels.size()), 1); break;
                case 3: cycle(temp_sel, static_cast<int>(temp_labels.size()), 1); break;
                case 4: cycle(kv_k_sel, KV_OPTIONS_COUNT, 1); break;
                case 5: cycle(kv_v_sel, KV_OPTIONS_COUNT, 1); break;
                default: break;
            }
            return true;
        }
        return false;
    });

    screen.Loop(wrapped);

    if (ctx_sel == custom_idx) {
        printf("\033[?25h");
        printf("Enter context size (tokens): ");
        fflush(stdout);
        std::string input;
        std::getline(std::cin, input);
        if (!parse_int(input, cfg.n_ctx) || cfg.n_ctx <= 0 || cfg.n_ctx > MAX_CTX) {
            fprintf(stderr, "\033[33mwarning: invalid context size '%s', using auto\033[0m\n", input.c_str());
            cfg.n_ctx = max_ctx > 0 ? max_ctx : 0;
        }
    } else {
        cfg.n_ctx = ctx_options[static_cast<size_t>(ctx_sel)];
    }

    if (custom_kv) {
        cfg.type_k = KV_OPTIONS[kv_k_sel].type;
        cfg.type_v = KV_OPTIONS[kv_v_sel].type;
    } else {
        const int ki = KV_PRESETS[kv_preset_sel].k_idx;
        const int vi = KV_PRESETS[kv_preset_sel].v_idx;
        cfg.type_k = KV_OPTIONS[ki].type;
        cfg.type_v = KV_OPTIONS[vi].type;
    }
    cfg.flash_attn = (fa_sel == 0);
    const float temps[] = {0.7f, 0.8f, 0.9f, 1.0f};
    cfg.temp = temps[temp_sel];

    return cfg;
}
