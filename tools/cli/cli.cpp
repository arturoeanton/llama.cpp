#include "chat.h"
#include "common.h"
#include "arg.h"
#include "console.h"
#include "fit.h"
#include "log.h"
#include "slotted-inference.h"

#include <sys/resource.h>

#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

#include <array>
#include <atomic>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <thread>
#include <signal.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

const char * LLAMA_ASCII_LOGO = R"(
▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀
)";

static std::atomic<bool> g_is_interrupted = false;
static bool should_stop() {
    return g_is_interrupted.load();
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void signal_handler(int) {
    if (g_is_interrupted.load()) {
        // second Ctrl+C - exit immediately
        // make sure to clear colors before exiting (not using LOG or console.cpp here to avoid deadlock)
        fprintf(stdout, "\033[0m\n");
        fflush(stdout);
        std::exit(130);
    }
    g_is_interrupted.store(true);
}
#endif

struct cli_context {
    server_context ctx_server;
    json messages = json::array();
    std::vector<raw_buffer> input_files;
    task_params defaults;
    bool verbose_prompt;

    // thread for showing "loading" animation
    std::atomic<bool> loading_show;

    cli_context(const common_params & params) {
        defaults.sampling    = params.sampling;
        defaults.speculative = params.speculative;
        defaults.n_keep      = params.n_keep;
        defaults.n_predict   = params.n_predict;
        defaults.antiprompt  = params.antiprompt;

        defaults.stream = true; // make sure we always use streaming mode
        defaults.timings_per_token = true; // in order to get timings even when we cancel mid-way
        // defaults.return_progress = true; // TODO: show progress

        verbose_prompt = params.verbose_prompt;
    }

    std::string generate_completion(result_timings & out_timings) {
        server_response_reader rd = ctx_server.get_response_reader();
        auto chat_params = format_chat();
        {
            // TODO: reduce some copies here in the future
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id         = rd.get_new_id();
            task.index      = 0;
            task.params     = defaults;           // copy
            task.cli_prompt = chat_params.prompt; // copy
            task.cli_files  = input_files;        // copy
            task.cli        = true;

            // chat template settings
            task.params.chat_parser_params = common_chat_parser_params(chat_params);
            task.params.chat_parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
            if (!chat_params.parser.empty()) {
                task.params.chat_parser_params.parser.load(chat_params.parser);
            }

            // reasoning budget sampler
            if (!chat_params.thinking_end_tag.empty()) {
                const llama_vocab * vocab = llama_model_get_vocab(
                    llama_get_model(ctx_server.get_llama_context()));

                task.params.sampling.reasoning_budget_tokens = defaults.sampling.reasoning_budget_tokens;
                task.params.sampling.generation_prompt = chat_params.generation_prompt;

                if (!chat_params.thinking_start_tag.empty()) {
                    task.params.sampling.reasoning_budget_start =
                        common_tokenize(vocab, chat_params.thinking_start_tag, false, true);
                }
                task.params.sampling.reasoning_budget_end =
                    common_tokenize(vocab, chat_params.thinking_end_tag, false, true);
                task.params.sampling.reasoning_budget_forced =
                    common_tokenize(vocab, defaults.sampling.reasoning_budget_message + chat_params.thinking_end_tag, false, true);
            }

            rd.post_task({std::move(task)});
        }

        if (verbose_prompt) {
            console::set_display(DISPLAY_TYPE_PROMPT);
            console::log("%s\n\n", chat_params.prompt.c_str());
            console::set_display(DISPLAY_TYPE_RESET);
        }

        // wait for first result
        console::spinner::start();
        server_task_result_ptr result = rd.next(should_stop);

        console::spinner::stop();
        std::string curr_content;
        bool is_thinking = false;

        while (result) {
            if (should_stop()) {
                break;
            }
            if (result->is_error()) {
                json err_data = result->to_json();
                if (err_data.contains("message")) {
                    console::error("Error: %s\n", err_data["message"].get<std::string>().c_str());
                } else {
                    console::error("Error: %s\n", err_data.dump().c_str());
                }
                return curr_content;
            }
            auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                out_timings = std::move(res_partial->timings);
                for (const auto & diff : res_partial->oaicompat_msg_diffs) {
                    if (!diff.content_delta.empty()) {
                        if (is_thinking) {
                            console::log("\n[End thinking]\n\n");
                            console::set_display(DISPLAY_TYPE_RESET);
                            is_thinking = false;
                        }
                        curr_content += diff.content_delta;
                        console::log("%s", diff.content_delta.c_str());
                        console::flush();
                    }
                    if (!diff.reasoning_content_delta.empty()) {
                        console::set_display(DISPLAY_TYPE_REASONING);
                        if (!is_thinking) {
                            console::log("[Start thinking]\n");
                        }
                        is_thinking = true;
                        console::log("%s", diff.reasoning_content_delta.c_str());
                        console::flush();
                    }
                }
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                out_timings = std::move(res_final->timings);
                break;
            }
            result = rd.next(should_stop);
        }
        g_is_interrupted.store(false);
        // server_response_reader automatically cancels pending tasks upon destruction
        return curr_content;
    }

    // TODO: support remote files in the future (http, https, etc)
    std::string load_input_file(const std::string & fname, bool is_media) {
        std::ifstream file(fname, std::ios::binary);
        if (!file) {
            return "";
        }
        if (is_media) {
            raw_buffer buf;
            buf.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            input_files.push_back(std::move(buf));
            return get_media_marker();
        } else {
            std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            return content;
        }
    }

    common_chat_params format_chat() {
        auto meta = ctx_server.get_meta();
        auto & chat_params = meta.chat_params;

        auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());

        common_chat_templates_inputs inputs;
        inputs.messages              = common_chat_msgs_parse_oaicompat(messages);
        inputs.tools                 = {}; // TODO
        inputs.tool_choice           = COMMON_CHAT_TOOL_CHOICE_NONE;
        inputs.json_schema           = ""; // TODO
        inputs.grammar               = ""; // TODO
        inputs.use_jinja             = chat_params.use_jinja;
        inputs.parallel_tool_calls   = caps["supports_parallel_tool_calls"];
        inputs.add_generation_prompt = true;
        inputs.reasoning_format      = COMMON_REASONING_FORMAT_DEEPSEEK;
        inputs.force_pure_content    = chat_params.force_pure_content;
        inputs.enable_thinking       = chat_params.enable_thinking ? common_chat_templates_support_enable_thinking(chat_params.tmpls.get()) : false;

        // Apply chat template to the list of messages
        return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
    }
};

// TODO?: Make this reusable, enums, docs
static const std::array<std::string_view, 7> cmds = {
    "/audio ",
    "/clear",
    "/exit",
    "/glob ",
    "/image ",
    "/read ",
    "/regen",
};

static std::vector<std::pair<std::string, size_t>> auto_completion_callback(std::string_view line, size_t cursor_byte_pos) {
    std::vector<std::pair<std::string, size_t>> matches;
    std::string cmd;

    if (line.length() > 1 && line.front() == '/' && !std::any_of(cmds.begin(), cmds.end(), [line](std::string_view prefix) {
        return string_starts_with(line, prefix);
    })) {
        auto it = cmds.begin();

        while ((it = std::find_if(it, cmds.end(), [line](std::string_view cmd_line) {
            return string_starts_with(cmd_line, line);
        })) != cmds.end()) {
            matches.emplace_back(*it, it->length());
            ++it;
        }
    } else {
        auto it = std::find_if(cmds.begin(), cmds.end(), [line](std::string_view prefix) {
            return prefix.back() == ' ' && string_starts_with(line, prefix);
        });

        if (it != cmds.end()) {
            cmd = *it;
        }
    }

    if (!cmd.empty() && cmd != "/glob " && line.length() >= cmd.length() && cursor_byte_pos >= cmd.length()) {
        const std::string path_prefix  = std::string(line.substr(cmd.length(), cursor_byte_pos - cmd.length()));
        const std::string path_postfix = std::string(line.substr(cursor_byte_pos));
        auto cur_dir = std::filesystem::current_path();
        std::string cur_dir_str = cur_dir.string();
        std::string expanded_prefix = path_prefix;

#if !defined(_WIN32)
        if (string_starts_with(path_prefix, '~')) {
            const char * home = std::getenv("HOME");
            if (home && home[0]) {
                expanded_prefix = home + path_prefix.substr(1);
            }
        }
        if (string_starts_with(expanded_prefix, '/')) {
#else
        if (std::isalpha(expanded_prefix[0]) && expanded_prefix.find(':') == 1) {
#endif
            cur_dir = std::filesystem::path(expanded_prefix).parent_path();
            cur_dir_str.clear();
        } else if (!path_prefix.empty()) {
            cur_dir /= std::filesystem::path(path_prefix).parent_path();
        }

        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(cur_dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.exists(ec)) {
                ec.clear();
                continue;
            }

            const std::string path_full = entry.path().string();
            std::string path_entry = !cur_dir_str.empty() && string_starts_with(path_full, cur_dir_str) ? path_full.substr(cur_dir_str.length() + 1) : path_full;

            if (entry.is_directory(ec)) {
                path_entry.push_back(std::filesystem::path::preferred_separator);
            }

            if (expanded_prefix.empty() || string_starts_with(path_entry, expanded_prefix)) {
                const std::string updated_line = cmd + path_entry;
                matches.emplace_back(updated_line + path_postfix, updated_line.length());
            }

            if (ec) {
                ec.clear();
            }
        }

        if (matches.empty()) {
            const std::string updated_line = cmd + path_prefix;
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        // Add the longest common prefix
        if (!expanded_prefix.empty() && matches.size() > 1) {
            const std::string_view match0(matches[0].first);
            const std::string_view match1(matches[1].first);
            auto it = std::mismatch(match0.begin(), match0.end(), match1.begin(), match1.end());
            size_t len = it.first - match0.begin();

            for (size_t i = 2; i < matches.size(); ++i) {
                const std::string_view matchi(matches[i].first);
                auto cmp = std::mismatch(match0.begin(), match0.end(), matchi.begin(), matchi.end());
                len = std::min(len, static_cast<size_t>(cmp.first - match0.begin()));
            }

            const std::string updated_line = std::string(match0.substr(0, len));
            matches.emplace_back(updated_line + path_postfix, updated_line.length());
        }

        std::sort(matches.begin(), matches.end(), [](const auto & a, const auto & b) {
            return a.first.compare(0, a.second, b.first, 0, b.second) < 0;
        });
    }

    return matches;
}

static constexpr size_t FILE_GLOB_MAX_RESULTS = 100;

int main(int argc, char ** argv) {
    common_params params;

    params.verbosity = LOG_LEVEL_ERROR; // by default, less verbose logs

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)) {
        return 1;
    }

    // TODO: maybe support it later?
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
        console::error("--no-conversation is not supported by llama-cli\n");
        console::error("please use llama-completion instead\n");
    }

    // experimental: FASE 3 — wire the slotted-inference eval callback BEFORE the
    // model is loaded, so the llama_context picks it up via common_params.
    // The runtime is initialized below, once the model and slot plan exist.
    common_slot_runtime slotted_rt;
    if (params.slotted_test) {
        params.cb_eval           = common_slotted_cb_eval;
        params.cb_eval_user_data = &slotted_rt;
    }

    // experimental: FASE 4A — slotted-real path.
    //
    // FASE 4A-1: load filtered model, report peak RSS, exit. Bypass cli_context.
    // FASE 4A-2b: when combined with --slotted-decode-test, install the layer
    //   filter on params and fall through to the normal cli_context path. The
    //   context constructor's sched_reserve becomes a no-op via cparams.
    //   slotted_real_skip_sched_reserve (set in common_context_params_to_llama).
    //
    // These need to outlive load_model, so we keep them in main's scope.
    static common_slot_plan         g_slotted_real_plan;
    static common_slot_filter_state g_slotted_real_filter_state;
    if (params.slotted_real) {
        if (params.slot_layers <= 0 && params.slot_size_mb <= 0) {
            console::error("--slotted-real requires --slot-layers or --slot-size-mb\n");
            return 1;
        }
        if (params.slots_resident <= 0) {
            console::error("--slots-resident must be > 0\n");
            return 1;
        }
        g_slotted_real_plan = common_slotted_build_plan_from_gguf(params.model.path, params);
        if (!g_slotted_real_plan.enabled) {
            console::error("--slotted-real: failed to build plan from GGUF\n");
            return 1;
        }
        g_slotted_real_filter_state.plan           = &g_slotted_real_plan;
        g_slotted_real_filter_state.slots_resident = params.slots_resident;

        params.layer_filter           = common_slotted_layer_filter_cb;
        params.layer_filter_user_data = &g_slotted_real_filter_state;
        params.fit_params             = false;

        if (params.slotted_decode_test || params.slotted_chat_poc) {
            // FASE 4A-2b extras: disable repack so raw GGUF bytes can be hot-swapped
            // into the pool's buffers, and rely on slotted_real_skip_sched_reserve
            // (derived in common_context_params_to_llama from these two flags).
            params.no_extra_bufts = true;
            LOG("[slotted-real-4A-2b] forcing --no-repack (no_extra_bufts=true) and skip_sched_reserve\n");
        }
    }

    // FASE 4A-1 path: load only, measure RSS, and exit before context creation.
    if (params.slotted_real && !params.slotted_decode_test && !params.slotted_decode_baseline && !params.slotted_chat_poc) {
        common_slot_plan & plan = g_slotted_real_plan;
        llama_model_params mparams = common_model_params_to_llama(params);
        mparams.layer_filter           = common_slotted_layer_filter_cb;
        mparams.layer_filter_user_data = &g_slotted_real_filter_state;

        llama_backend_init();
        llama_numa_init(params.numa);

        console::log(("\nLoading model (slotted-real, " +
                      std::to_string(params.slots_resident) + "/" +
                      std::to_string(plan.slots.size()) + " slots)... ").c_str());
        console::spinner::start();
        llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
        console::spinner::stop();
        console::log("\n");
        if (model == nullptr) {
            console::error("\nFailed to load the model (slotted-real)\n");
            llama_backend_free();
            return 1;
        }

        struct rusage ru = {};
        getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
        const double peak_rss_mb = (double) ru.ru_maxrss / (1024.0 * 1024.0); // bytes on macOS
#else
        const double peak_rss_mb = (double) ru.ru_maxrss / 1024.0;            // KB on Linux
#endif
        const int nshow = (int) std::min<size_t>(params.slots_resident, plan.slots.size());
        double resident_mb = 0;
        for (int i = 0; i < nshow; ++i) {
            resident_mb += (double) plan.slots[i].weight_bytes / (1024.0 * 1024.0);
        }
        common_slotted_print_plan(plan);
        LOG("[slotted-real]\n");
        LOG("  slots_resident          = %d / %zu\n", params.slots_resident, plan.slots.size());
        LOG("  loaded slots            =");
        for (int s = 0; s < nshow; ++s) {
            const auto & sd = plan.slots[s];
            LOG(" %d(%d..%d)", sd.slot_id, sd.layer_start, sd.layer_end);
        }
        LOG("\n");
        LOG("  resident_weight_mb      = %.2f (sum of resident slots)\n", resident_mb);
        LOG("  non_layer_weight_mb     = %.2f (embeddings + final norm + lm_head)\n",
            (double) plan.non_layer_bytes / (1024.0 * 1024.0));
        LOG("  expected_layer_ram_mb   = %.2f (resident + non-layer)\n",
            resident_mb + (double) plan.non_layer_bytes / (1024.0 * 1024.0));
        LOG("  peak_rss_mb             = %.2f\n", peak_rss_mb);
        LOG("  page_reclaims           = %ld\n", (long) ru.ru_minflt);
        LOG("  hard_page_faults        = %ld\n", (long) ru.ru_majflt);
        LOG("  voluntary_ctx_switches  = %ld\n", (long) ru.ru_nvcsw);
        LOG("  inference_supported     = no (FASE 4A-2 will add per-slot graph execution)\n");
        if (!params.slotted_json_path.empty()) {
            if (common_slotted_write_plan_json(plan, params.slotted_json_path)) {
                LOG("slotted: wrote plan JSON to '%s'\n", params.slotted_json_path.c_str());
            }
        }

        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    // struct that contains llama context and inference
    cli_context ctx_cli(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    // TODO: avoid using atexit() here by making `console` a singleton
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    console::set_display(DISPLAY_TYPE_RESET);
    console::set_completion_callback(auto_completion_callback);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    console::log("\nLoading model... "); // followed by loading animation
    console::spinner::start();
    if (!ctx_cli.ctx_server.load_model(params)) {
        console::spinner::stop();
        console::error("\nFailed to load the model\n");
        return 1;
    }

    console::spinner::stop();
    console::log("\n");

    // experimental: FASE 4A-2b — interactive chat PoC over the slotted-real runtime.
    // Each turn: read a line from stdin, tokenize, reset KV, run slotted decode,
    // greedy-sample n_predict tokens, print. The hot-swap runtime is initialised
    // ONCE so we only pay model-load cost once.
    if (params.slotted_chat_poc) {
        if (!params.slotted_real) {
            console::error("--slotted-chat-poc requires --slotted-real\n");
            return 1;
        }
        llama_context * lctx = ctx_cli.ctx_server.get_llama_context();
        const llama_model * model = llama_get_model(lctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_predict = std::max(1, (int) params.n_predict);

        llama_slotted_hot_swap * hs = llama_slotted_hot_swap_init(
                const_cast<llama_model *>(model),
                params.model.path.c_str(),
                params.slot_layers,
                params.slot_size_mb,
                params.slots_resident,
                params.slotted_round_robin ? 0 : 1,
                params.slotted_async_prefetch ? 1 : 0);
        if (hs == nullptr) {
            console::error("slotted-chat-poc: hot-swap init failed\n");
            return 1;
        }

        LOG("\n");
        LOG("=======================================================================\n");
        LOG("  Slotted-real chat loop. Each turn streams non-resident slots from\n");
        LOG("  disk; expect wall-clock-dominated latency. KV is reset every turn.\n");
        LOG("  Each turn: %d tokens.\n", n_predict);
        LOG("  Type a prompt and press Enter. Empty line / Ctrl-D exits.\n");
        LOG("=======================================================================\n");

        std::string line;
        int turn = 0;
        while (true) {
            fprintf(stdout, "\n> ");
            fflush(stdout);
            if (!std::getline(std::cin, line)) break;  // EOF / Ctrl-D
            if (line.empty()) break;

            ++turn;
            LOG("\n[turn %d] prompt=\"%s\"\n", turn, line.c_str());

            // Wipe KV cache so each turn starts from scratch.
            llama_memory_clear(llama_get_memory(lctx), /*data=*/true);
            llama_memory_seq_rm(llama_get_memory(lctx), 0, 0, -1);

            std::vector<llama_token> prompt_tokens =
                common_tokenize(vocab, line, /*add_special=*/true, /*parse_special=*/false);
            if (prompt_tokens.empty()) {
                LOG("[turn %d] empty tokenization, skipping\n", turn);
                continue;
            }

            // Feed prompt one token at a time.
            bool failed = false;
            for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                llama_token t = prompt_tokens[i];
                llama_batch b = llama_batch_get_one(&t, 1);
                if (llama_decode_slotted_real_test(lctx, b, hs) != 0) {
                    console::error("slotted-chat-poc: prompt decode failed\n");
                    failed = true;
                    break;
                }
            }
            if (failed) continue;

            // Greedy-sample n_predict tokens.
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            std::string reply;
            for (int step = 0; step < n_predict; ++step) {
                const float * logits = llama_get_logits_ith(lctx, -1);
                if (logits == nullptr) break;
                int   argmax = 0;
                float maxv   = logits[0];
                for (int v = 1; v < n_vocab; ++v) {
                    if (logits[v] > maxv) { maxv = logits[v]; argmax = v; }
                }
                std::string piece = common_token_to_piece(vocab, argmax);
                reply += piece;
                fprintf(stdout, "%s", piece.c_str());
                fflush(stdout);
                if (step + 1 < n_predict) {
                    llama_token t = argmax;
                    llama_batch b = llama_batch_get_one(&t, 1);
                    if (llama_decode_slotted_real_test(lctx, b, hs) != 0) {
                        console::error("\nslotted-chat-poc: gen decode failed\n");
                        break;
                    }
                }
            }
            fprintf(stdout, "\n");
            LOG("[turn %d] reply (%d tok): %s\n", turn, n_predict, reply.c_str());
        }

        llama_slotted_hot_swap_free(hs);
        return 0;
    }

    // experimental: FASE 4A-2a — slotted decode test harness.
    // Tokenises a fixed prompt and runs either the baseline path or the slotted
    // path (or both), prints the top-1 logit + decoded text, and exits.
    if (params.slotted_decode_test || params.slotted_decode_baseline) {
        llama_context * lctx = ctx_cli.ctx_server.get_llama_context();
        const llama_model * model = llama_get_model(lctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        // Tokenise the prompt. Use `-p` if provided, else a small default.
        // BOS is added so the setup matches conventional decode flow.
        const std::string prompt = params.prompt.empty() ? std::string("Hola") : params.prompt;
        std::vector<llama_token> tokens = common_tokenize(vocab, prompt, /*add_special=*/true, /*parse_special=*/false);
        if (tokens.empty()) {
            console::error("slotted-decode-test: tokenization produced no tokens\n");
            return 1;
        }
        // FASE 4A-2a accepts batch_size == 1; we feed tokens one-by-one and
        // only sample after the LAST token.
        LOG("\n[slotted-decode-test] prompt=\"%s\" tokens=%zu\n", prompt.c_str(), tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            LOG("  token[%zu] = %d (%s)\n", i, (int) tokens[i],
                common_token_to_piece(vocab, tokens[i]).c_str());
        }

        // n_predict comes from `-n` (default in common_params). We generate at
        // least 1 token. Greedy sampling throughout.
        const int n_predict = std::max(1, (int) params.n_predict);

        // FASE 4A-2b: when --slotted-real is also active, initialise the hot-swap
        // runtime. The slot plan inside `hs` matches the plan we used for the
        // load-time filter (same flags -> same grouping).
        llama_slotted_hot_swap * hs = nullptr;
        if (params.slotted_real && params.slotted_decode_test) {
            hs = llama_slotted_hot_swap_init(const_cast<llama_model *>(model),
                                             params.model.path.c_str(),
                                             params.slot_layers,
                                             params.slot_size_mb,
                                             params.slots_resident,
                                             params.slotted_round_robin ? 0 : 1,
                                             params.slotted_async_prefetch ? 1 : 0);
            if (hs == nullptr) {
                console::error("slotted-decode-test: hot-swap init failed\n");
                return 1;
            }
            LOG("[slotted-real-4A-2b] hot-swap state initialised (slots_resident=%d)\n", params.slots_resident);
        }

        auto run_one_pass = [&](const char * tag, bool slotted) -> std::vector<llama_token> {
            // Wipe KV cache between baseline and slotted runs so they start fresh.
            llama_memory_clear(llama_get_memory(lctx), /*data=*/true);
            llama_memory_seq_rm(llama_get_memory(lctx), 0, 0, -1);

            auto decode_one = [&](llama_token tk) -> int {
                llama_batch batch = llama_batch_get_one(&tk, 1);
                if (slotted) {
                    if (hs != nullptr) {
                        return llama_decode_slotted_real_test(lctx, batch, hs);
                    }
                    return llama_decode_slotted_test(lctx, batch, params.slot_layers, params.slot_size_mb);
                }
                return llama_decode(lctx, batch);
            };

            // Feed the prompt tokens one-by-one.
            for (size_t i = 0; i < tokens.size(); ++i) {
                const int rc = decode_one(tokens[i]);
                if (rc != 0) {
                    console::error(("[" + std::string(tag) + "] prompt decode failed at token " +
                                    std::to_string(i) + " rc=" + std::to_string(rc) + "\n").c_str());
                    return {};
                }
            }

            const int32_t n_vocab = llama_vocab_n_tokens(vocab);

            // Greedy sample n_predict tokens. After each sample, feed it back.
            std::vector<llama_token> out;
            out.reserve(n_predict);
            llama_token tk = 0;
            float       tk_logit = 0.0f;
            for (int step = 0; step < n_predict; ++step) {
                const float * logits = llama_get_logits_ith(lctx, -1);
                if (logits == nullptr) {
                    console::error("[%s] no logits available after step\n");
                    return out;
                }
                int   argmax = 0;
                float maxv   = logits[0];
                for (int v = 1; v < n_vocab; ++v) {
                    if (logits[v] > maxv) { maxv = logits[v]; argmax = v; }
                }
                tk       = argmax;
                tk_logit = maxv;
                out.push_back(tk);
                const std::string piece = common_token_to_piece(vocab, tk);
                LOG("[%s] step=%2d tok=%6d logit=%10.6f text=%s\n",
                    tag, step, tk, tk_logit, piece.c_str());

                // Feed the sampled token back as the next decode input. Skip on
                // the LAST iteration to avoid an unused decode call.
                if (step + 1 < n_predict) {
                    const int rc = decode_one(tk);
                    if (rc != 0) {
                        console::error(("[" + std::string(tag) + "] gen decode failed at step " +
                                        std::to_string(step) + " rc=" + std::to_string(rc) + "\n").c_str());
                        return out;
                    }
                }
            }
            return out;
        };

        std::vector<llama_token> base_seq, slot_seq;
        if (params.slotted_decode_baseline) {
            base_seq = run_one_pass("baseline", /*slotted=*/false);
        }
        if (params.slotted_decode_test) {
            slot_seq = run_one_pass("slotted ", /*slotted=*/true);
        }

        // If both ran, do a sequence-level comparison.
        if (params.slotted_decode_baseline && params.slotted_decode_test) {
            if ((int) base_seq.size() != n_predict || (int) slot_seq.size() != n_predict) {
                LOG("[compare] truncated run: baseline=%zu slotted=%zu (expected %d)\n",
                    base_seq.size(), slot_seq.size(), n_predict);
            }
            int n_match = 0;
            const int n_cmp = (int) std::min(base_seq.size(), slot_seq.size());
            for (int i = 0; i < n_cmp; ++i) {
                if (base_seq[i] == slot_seq[i]) {
                    ++n_match;
                }
            }
            LOG("[compare] tokens matched = %d / %d (%.1f%%)\n",
                n_match, n_cmp, n_cmp > 0 ? 100.0 * n_match / n_cmp : 0.0);
            std::string text_base, text_slot;
            for (auto t : base_seq) text_base += common_token_to_piece(vocab, t);
            for (auto t : slot_seq) text_slot += common_token_to_piece(vocab, t);
            LOG("[compare] baseline text: %s\n", text_base.c_str());
            LOG("[compare] slotted  text: %s\n", text_slot.c_str());
            if (n_match == n_cmp && n_cmp == n_predict) {
                LOG("[compare] PASS: sequences are identical\n");
            } else {
                LOG("[compare] FAIL: sequences differ\n");
                for (int i = 0; i < n_cmp; ++i) {
                    if (base_seq[i] != slot_seq[i]) {
                        LOG("           first divergence at step %d: baseline=%d slotted=%d\n",
                            i, base_seq[i], slot_seq[i]);
                        break;
                    }
                }
            }
        }

        if (hs != nullptr) {
            llama_slotted_hot_swap_free(hs);
        }
        return 0;
    }

    // experimental: slotted-inference (FASE 1+3) — plan, print, initialize runtime.
    if (params.slotted_test) {
        const llama_model * model = llama_get_model(ctx_cli.ctx_server.get_llama_context());
        if (params.slot_layers <= 0 && params.slot_size_mb <= 0) {
            LOG("slotted: --slotted-test set without --slot-layers or --slot-size-mb; "
                "falling back to a single slot covering all layers\n");
        }
        common_slot_plan plan = common_slotted_build_plan(model, params);
        common_slotted_print_plan(plan);
        // FASE 3: initialize runtime. cb_eval is already wired to &slotted_rt above.
        // After this returns successfully, the callback starts collecting timings.
        if (!common_slotted_runtime_init(slotted_rt, std::move(plan), params, params.model.path)) {
            LOG("slotted: runtime init failed; per-slot timing will be disabled\n");
        }
    }

    std::thread inference_thread([&ctx_cli]() {
        ctx_cli.ctx_server.start_loop();
    });

    auto inf = ctx_cli.ctx_server.get_meta();
    std::string modalities = "text";
    if (inf.has_inp_image) {
        modalities += ", vision";
    }
    if (inf.has_inp_audio) {
        modalities += ", audio";
    }

    auto add_system_prompt = [&]() {
        if (!params.system_prompt.empty()) {
            ctx_cli.messages.push_back({
                {"role",    "system"},
                {"content", params.system_prompt}
            });
        }
    };
    add_system_prompt();

    console::log("\n");
    console::log("%s\n", LLAMA_ASCII_LOGO);
    console::log("build      : %s\n", inf.build_info.c_str());
    console::log("model      : %s\n", inf.model_name.c_str());
    console::log("modalities : %s\n", modalities.c_str());
    if (!params.system_prompt.empty()) {
        console::log("using custom system prompt\n");
    }
    console::log("\n");
    console::log("available commands:\n");
    console::log("  /exit or Ctrl+C     stop or exit\n");
    console::log("  /regen              regenerate the last response\n");
    console::log("  /clear              clear the chat history\n");
    console::log("  /read <file>        add a text file\n");
    console::log("  /glob <pattern>     add text files using globbing pattern\n");
    if (inf.has_inp_image) {
        console::log("  /image <file>       add an image file\n");
    }
    if (inf.has_inp_audio) {
        console::log("  /audio <file>       add an audio file\n");
    }
    console::log("\n");

    // interactive loop
    std::string cur_msg;

    auto add_text_file = [&](const std::string & fname) -> bool {
        std::string marker = ctx_cli.load_input_file(fname, false);
        if (marker.empty()) {
            console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
            return false;
        }
        if (inf.fim_sep_token != LLAMA_TOKEN_NULL) {
            cur_msg += common_token_to_piece(ctx_cli.ctx_server.get_llama_context(), inf.fim_sep_token, true);
            cur_msg += fname;
            cur_msg.push_back('\n');
        } else {
            cur_msg += "--- File: ";
            cur_msg += fname;
            cur_msg += " ---\n";
        }
        cur_msg += marker;
        console::log("Loaded text from '%s'\n", fname.c_str());
        return true;
    };

    while (true) {
        std::string buffer;
        console::set_display(DISPLAY_TYPE_USER_INPUT);
        if (params.prompt.empty()) {
            console::log("\n> ");
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        } else {
            // process input prompt from args
            for (auto & fname : params.image) {
                std::string marker = ctx_cli.load_input_file(fname, true);
                if (marker.empty()) {
                    console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                    break;
                }
                console::log("Loaded media from '%s'\n", fname.c_str());
                cur_msg += marker;
            }
            buffer = params.prompt;
            if (buffer.size() > 500) {
                console::log("\n> %s ... (truncated)\n", buffer.substr(0, 500).c_str());
            } else {
                console::log("\n> %s\n", buffer.c_str());
            }
            params.prompt.clear(); // only use it once
        }
        console::set_display(DISPLAY_TYPE_RESET);
        console::log("\n");

        if (should_stop()) {
            g_is_interrupted.store(false);
            break;
        }

        // remove trailing newline
        if (!buffer.empty() &&buffer.back() == '\n') {
            buffer.pop_back();
        }

        // skip empty messages
        if (buffer.empty()) {
            continue;
        }

        bool add_user_msg = true;

        // process commands
        if (string_starts_with(buffer, "/exit")) {
            break;
        } else if (string_starts_with(buffer, "/regen")) {
            if (ctx_cli.messages.size() >= 2) {
                size_t last_idx = ctx_cli.messages.size() - 1;
                ctx_cli.messages.erase(last_idx);
                add_user_msg = false;
            } else {
                console::error("No message to regenerate.\n");
                continue;
            }
        } else if (string_starts_with(buffer, "/clear")) {
            ctx_cli.messages.clear();
            add_system_prompt();

            ctx_cli.input_files.clear();
            console::log("Chat history cleared.\n");
            continue;
        } else if (
                (string_starts_with(buffer, "/image ") && inf.has_inp_image) ||
                (string_starts_with(buffer, "/audio ") && inf.has_inp_audio)) {
            // just in case (bad copy-paste for example), we strip all trailing/leading spaces
            std::string fname = string_strip(buffer.substr(7));
            std::string marker = ctx_cli.load_input_file(fname, true);
            if (marker.empty()) {
                console::error("file does not exist or cannot be opened: '%s'\n", fname.c_str());
                continue;
            }
            cur_msg += marker;
            console::log("Loaded media from '%s'\n", fname.c_str());
            continue;
        } else if (string_starts_with(buffer, "/read ")) {
            std::string fname = string_strip(buffer.substr(6));
            add_text_file(fname);
            continue;
        } else if (string_starts_with(buffer, "/glob ")) {
            std::error_code ec;
            size_t count = 0;
            auto curdir = std::filesystem::current_path();
            std::string pattern = string_strip(buffer.substr(6));
            std::filesystem::path rel_path;

            auto startglob = pattern.find_first_of("![*?");
            if (startglob != std::string::npos && startglob != 0) {
                auto endpath = pattern.substr(0, startglob).find_last_of('/');
                if (endpath != std::string::npos) {
                    std::string rel_pattern = pattern.substr(0, endpath);
#if !defined(_WIN32)
                    if (string_starts_with(rel_pattern, '~')) {
                        const char * home = std::getenv("HOME");
                        if (home && home[0]) {
                            rel_pattern = home + rel_pattern.substr(1);
                        }
                    }
#endif
                    rel_path = rel_pattern;
                    pattern.erase(0, endpath + 1);
                    curdir /= rel_path;
                }
            }

            for (const auto & entry : std::filesystem::recursive_directory_iterator(curdir,
                    std::filesystem::directory_options::skip_permission_denied, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                std::string rel = std::filesystem::relative(entry.path(), curdir, ec).string();
                if (ec) {
                    ec.clear();
                    continue;
                }
                std::replace(rel.begin(), rel.end(), '\\', '/');

                if (!glob_match(pattern, rel)) {
                    continue;
                }

                if (!add_text_file((rel_path / rel).string())) {
                    continue;
                }

                if (++count >= FILE_GLOB_MAX_RESULTS) {
                    console::error("Maximum number of globbed files allowed (%zu) reached.\n", FILE_GLOB_MAX_RESULTS);
                    break;
                }
            }
            continue;
        } else {
            // not a command
            cur_msg += buffer;
        }

        // generate response
        if (add_user_msg) {
            ctx_cli.messages.push_back({
                {"role",    "user"},
                {"content", cur_msg}
            });
            cur_msg.clear();
        }
        result_timings timings;
        std::string assistant_content = ctx_cli.generate_completion(timings);
        ctx_cli.messages.push_back({
            {"role",    "assistant"},
            {"content", assistant_content}
        });
        console::log("\n");

        if (params.show_timings) {
            console::set_display(DISPLAY_TYPE_INFO);
            console::log("\n");
            console::log("[ Prompt: %.1f t/s | Generation: %.1f t/s ]\n", timings.prompt_per_second, timings.predicted_per_second);
            console::set_display(DISPLAY_TYPE_RESET);
        }

        if (params.single_turn) {
            break;
        }
    }

    console::set_display(DISPLAY_TYPE_RESET);

    console::log("\nExiting...\n");
    ctx_cli.ctx_server.terminate();
    inference_thread.join();

    // bump the log level to display timings
    common_log_set_verbosity_thold(LOG_LEVEL_INFO);
    common_memory_breakdown_print(ctx_cli.ctx_server.get_llama_context());

    // experimental: slotted-inference (FASE 3) — flush the last forward pass,
    // print aggregated stats, and optionally write the summary JSON.
    if (params.slotted_test) {
        common_slotted_runtime_flush(slotted_rt);
        common_slotted_print_summary(slotted_rt);
        if (!params.slotted_json_path.empty()) {
            if (common_slotted_write_summary_json(slotted_rt, params.slotted_json_path)) {
                LOG("slotted: wrote summary JSON to '%s'\n", params.slotted_json_path.c_str());
            }
        }
        common_slotted_runtime_shutdown(slotted_rt);
    }

    return 0;
}
