/*
 * poc.c -- slotted-real demo launcher.
 *
 * Compiles with:
 *     clang poc.c -o poc
 *
 * Run from the repo root (where ./build/bin/llama-cli exists):
 *     ./poc
 *
 * Hardcodes every flag needed to make Meta-Llama-3.1-70B-Instruct (Q3_K_XL,
 * 35 GB on disk) run on a 24-32 GB Mac/PC with only 2 slots resident at any
 * given moment. Peak memory footprint stays around 11 GB even though the
 * model file is bigger than RAM.
 *
 * This is intentionally a launcher: it just execs the modified llama-cli with
 * the right flags. The slotted-real / hot-swap implementation lives inside
 * llama.cpp; this file is here only so a single ./poc is enough to demo.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>

#define LLAMA_CLI  "./build/bin/llama-cli"
#define MODEL_REL  "/models/llama31-70b/Meta-Llama-3.1-70B-Instruct-Q3_K_XL.gguf"

static int file_exists(const char * path) {
    struct stat st;
    return stat(path, &st) == 0;
}

int main(int argc, char ** argv) {
    /* Resolve model path: $HOME/models/... */
    const char * home = getenv("HOME");
    if (home == NULL || home[0] == '\0') {
        fprintf(stderr, "poc: HOME is not set\n");
        return 1;
    }
    char model_path[4096];
    int n = snprintf(model_path, sizeof(model_path), "%s%s", home, MODEL_REL);
    if (n < 0 || (size_t) n >= sizeof(model_path)) {
        fprintf(stderr, "poc: model path too long\n");
        return 1;
    }

    /* Sanity checks before we fork llama-cli. */
    if (!file_exists(LLAMA_CLI)) {
        fprintf(stderr, "poc: %s not found.\n", LLAMA_CLI);
        fprintf(stderr, "     Build it first with:\n");
        fprintf(stderr, "       cmake -B build -DGGML_METAL=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF\n");
        fprintf(stderr, "       cmake --build build --config Release -j --target llama-cli\n");
        return 1;
    }
    if (!file_exists(model_path)) {
        fprintf(stderr, "poc: model not found at %s\n", model_path);
        return 1;
    }

    /* Pick number of tokens per turn. Default 16; override with `./poc 32` etc. */
    const char * n_predict = "16";
    if (argc >= 2) {
        n_predict = argv[1];
    }

    fputs(
        "\n"
        "=========================================================================\n"
        "  llama.cpp slotted-real demo\n"
        "=========================================================================\n"
        "  Model:     Meta-Llama-3.1-70B-Instruct Q3_K_XL (~35 GB on disk)\n"
        "  Mode:      CPU only, --no-mmap, --no-repack\n"
        "  Slot plan: 8 slots x 10 layers,  --slots-resident 2\n"
        "\n"
        "  This is intentionally slow. It proves a 35GB 70B model can run\n"
        "  with a reduced memory footprint (~11 GB peak instead of ~35 GB).\n"
        "  Each turn generates the requested number of tokens then waits for\n"
        "  the next prompt. KV cache is reset between turns.\n"
        "=========================================================================\n"
        "\n",
        stderr);

    /* Build the argv for execv. NOTE: --slot-layers 10 maps evenly into Llama
     * 3.1 70B's 80 layers and avoids the heterogeneous-slot mismatch we hit
     * with size-based grouping. */
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
        (char *) "--slot-layers",            (char *) "10",
        (char *) "--slotted-real",
        (char *) "--slots-resident",         (char *) "2",
        (char *) "--slotted-chat-poc",
        (char *) "-n",                       (char *) n_predict,
        NULL
    };

    execv(cli_argv[0], cli_argv);
    /* If execv returns, it failed. */
    fprintf(stderr, "poc: execv failed: %s\n", strerror(errno));
    return 1;
}
