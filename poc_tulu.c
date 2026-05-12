/*
 * poc_tulu.c -- slotted-real demo launcher for Llama 3.1 Tulu-3 405B Q3_K_M.
 *
 * Sibling of poc.c (which targets the 70B Q3_K_XL). Same idea: hardcode every
 * flag llama-cli needs so a single `./poc_tulu` boots the slotted-real chat
 * demo on a multi-shard 405B model with a peak memory footprint of ~12.6 GB
 * on a 24 GB MacBook Air M4.
 *
 * Compiles with:
 *     clang poc_tulu.c -o poc_tulu
 *
 * Run from the repo root (where ./build/bin/llama-cli exists):
 *     ./poc_tulu          # 4 tokens per turn (default, ~9 min on 24 GB M4)
 *     ./poc_tulu 8        # 8 tokens per turn (~20 min)
 *     ./poc_tulu 1        # 1 token (smoke test)
 *
 * The model is split into 5 GGUF shards. We point llama-cli at shard 00001;
 * the runtime auto-discovers the rest via the `split.count` metadata key
 * and llama_split_path (FASE 4A-3 in src/llama-slotted-runtime.cpp).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>

#define LLAMA_CLI  "./build/bin/llama-cli"
#define MODEL_REL  "/models/llama31-405b-tulu-q3km/Llama-3.1-Tulu-3-405B-Q3_K_M/Llama-3.1-Tulu-3-405B-Q3_K_M-00001-of-00005.gguf"

static int file_exists(const char * path) {
    struct stat st;
    return stat(path, &st) == 0;
}

int main(int argc, char ** argv) {
    /* Resolve model path: $HOME/... */
    const char * home = getenv("HOME");
    if (home == NULL || home[0] == '\0') {
        fprintf(stderr, "poc_tulu: HOME is not set\n");
        return 1;
    }
    char model_path[4096];
    int n = snprintf(model_path, sizeof(model_path), "%s%s", home, MODEL_REL);
    if (n < 0 || (size_t) n >= sizeof(model_path)) {
        fprintf(stderr, "poc_tulu: model path too long\n");
        return 1;
    }

    /* Sanity checks before we fork llama-cli. */
    if (!file_exists(LLAMA_CLI)) {
        fprintf(stderr, "poc_tulu: %s not found.\n", LLAMA_CLI);
        fprintf(stderr, "          Build it first with:\n");
        fprintf(stderr, "            cmake -B build -DGGML_METAL=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF\n");
        fprintf(stderr, "            cmake --build build --config Release -j --target llama-cli\n");
        return 1;
    }
    if (!file_exists(model_path)) {
        fprintf(stderr, "poc_tulu: shard 1 not found at %s\n", model_path);
        fprintf(stderr, "          All 5 shards must live alongside it in the same directory.\n");
        return 1;
    }

    /* Pick number of tokens per turn. Default 4 (the 405B path is ~138 s/token
     * on a 24 GB M4, so the default keeps a demo under ~10 minutes). */
    const char * n_predict = "4";
    if (argc >= 2) {
        n_predict = argv[1];
    }

    fputs(
        "\n"
        "=========================================================================\n"
        "  llama.cpp slotted-real demo  --  405B edition\n"
        "=========================================================================\n"
        "  Model:     Llama-3.1-Tulu-3-405B-Q3_K_M (~200 GB on disk, 5 shards)\n"
        "  Mode:      CPU only, --no-mmap, --no-repack\n"
        "  Slot plan: 42 slots x 3 layers,  --slots-resident 2\n"
        "\n"
        "  This is much slower than the 70B PoC (~138 s/token on a 24 GB M4).\n"
        "  Goal: peak memory footprint ~12.6 GB even though the model file is\n"
        "  ~200 GB on disk.  Multi-shard + mixed-quant fixes (FASE 4A-3) make\n"
        "  this work.  Each turn generates the requested number of tokens then\n"
        "  waits for the next prompt.  KV cache is reset between turns.\n"
        "=========================================================================\n"
        "\n",
        stderr);

    /* Build the argv for execv. Notes:
     *  - --slot-layers 3 maps evenly into Llama 3.1 405B's 126 layers
     *    (126 / 3 = 42 uniform slots).
     *  - --no-repack is forced off by the slotted-real path anyway, but we
     *    keep it explicit to match poc.c's surface.
     *  - --slots-resident 2 yields ~12.6 GB peak. Raising it reduces hot-swap
     *    frequency but blows past the 24 GB envelope. */
    char * cli_argv[] = {
        (char *) LLAMA_CLI,
        (char *) "-m",                       (char *) model_path,
        (char *) "--n-gpu-layers",           (char *) "0",
        (char *) "--no-mmap",
        (char *) "--no-warmup",
        (char *) "--no-repack",
        (char *) "--ctx-size",               (char *) "128",
        (char *) "--batch-size",             (char *) "8",
        (char *) "--ubatch-size",            (char *) "8",
        (char *) "--slot-layers",            (char *) "3",
        (char *) "--slotted-real",
        (char *) "--slots-resident",         (char *) "2",
        (char *) "--slotted-chat-poc",
        (char *) "-n",                       (char *) n_predict,
        NULL
    };

    execv(cli_argv[0], cli_argv);
    /* If execv returns, it failed. */
    fprintf(stderr, "poc_tulu: execv failed: %s\n", strerror(errno));
    return 1;
}
