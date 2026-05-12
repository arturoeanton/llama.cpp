// FASE 4B harness: time one or more Q4_K_M-style matmul kernels on a single
// FFN row from a Llama-class model. Single-threaded, no ggml linkage.
//
// Workload (default):
//   M = 256 rows
//   K = 8192 = 32 super-blocks of 256 weights each
//   x is a length-K fp32 vector
//   each row produces one fp32 dot product
//
// Per kernel: warm-up + N timed iterations, report ns/dot and GFLOPS.

#include "q4k_layout.h"
#include "bench/timing.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

extern "C" {
// kernels that consume FP32 input
float kernel_dot_q4k_fp32_scalar     (const block_q4_K * row, const float * x, int n_blocks);
float kernel_dot_q4k_fp32_scalar_opt (const block_q4_K * row, const float * x, int n_blocks);
// kernel that consumes Q8_K input (ggml)
float kernel_dot_q4k_q8k_ggml        (const block_q4_K * row, const void  * x_q8k, int n_blocks);
// Q8_K quantiser (also in ggml). 292 bytes per Q8_K block.
void  q4k_lab_quantize_input_q8_k    (const float * x, void * y, int K);
}

struct workload {
    int M;          // rows
    int K;          // columns (must be multiple of QK_K)
    int n_blocks;   // K / QK_K
    int iters;      // timed iterations per kernel
    int warmup;     // warm-up iterations before timing
};

static void fill_random_fp32(std::vector<float> & v, unsigned seed) {
    srand(seed);
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = ((float)(rand() & 0xFFFF) / 65536.0f) - 0.5f;
    }
}

enum input_kind { IN_FP32, IN_Q8K };

struct kernel_entry {
    const char *      name;
    enum input_kind   in_kind;
    float (*fn_fp32)(const block_q4_K * row, const float * x,    int n_blocks);
    float (*fn_q8k) (const block_q4_K * row, const void *  x_q8, int n_blocks);
};

struct measurement {
    double ns_min     = 0.0;
    double ns_median  = 0.0;
    double ns_max     = 0.0;
    double gflops_min = 0.0;
    double gflops_max = 0.0;
};

static measurement time_kernel(const kernel_entry & k,
                               const std::vector<block_q4_K> & W,
                               const std::vector<float> & x,
                               const std::vector<uint8_t> & x_q8k,
                               const workload & wl,
                               std::vector<float> & out) {
    auto call_one = [&](int m) -> float {
        if (k.in_kind == IN_FP32) {
            return k.fn_fp32(&W[m * wl.n_blocks], x.data(), wl.n_blocks);
        } else {
            return k.fn_q8k(&W[m * wl.n_blocks], x_q8k.data(), wl.n_blocks);
        }
    };
    // Warm up: process every row a few times so caches stabilise.
    for (int it = 0; it < wl.warmup; ++it) {
        for (int m = 0; m < wl.M; ++m) {
            out[m] = call_one(m);
        }
    }
    // Timed pass: time the whole M-row matmul, then divide by M to get
    // per-dot-product nanoseconds.
    std::vector<double> samples;
    samples.reserve(wl.iters);
    for (int it = 0; it < wl.iters; ++it) {
        const double t0 = timing_now_ns();
        for (int m = 0; m < wl.M; ++m) {
            out[m] = call_one(m);
        }
        const double t1 = timing_now_ns();
        samples.push_back((t1 - t0) / (double) wl.M);
    }
    std::sort(samples.begin(), samples.end());
    measurement r;
    r.ns_min    = samples.front();
    r.ns_median = samples[samples.size() / 2];
    r.ns_max    = samples.back();
    // FLOPS per dot product = 2 * K (one multiply + one add per element).
    const double flops_per_dot = 2.0 * (double) wl.K;
    r.gflops_min = flops_per_dot / r.ns_max;       // slowest -> lowest gflops
    r.gflops_max = flops_per_dot / r.ns_min;       // fastest -> highest gflops
    return r;
}

int main(int argc, char ** argv) {
    workload wl;
    wl.M        = 256;
    wl.K        = 8192;
    wl.iters    = 50;
    wl.warmup   = 5;
    if (argc >= 2) wl.M     = std::atoi(argv[1]);
    if (argc >= 3) wl.K     = std::atoi(argv[2]);
    if (argc >= 4) wl.iters = std::atoi(argv[3]);

    if (wl.K % QK_K != 0) {
        fprintf(stderr, "K (%d) must be a multiple of %d\n", wl.K, QK_K);
        return 1;
    }
    wl.n_blocks = wl.K / QK_K;

    const size_t W_blocks = (size_t) wl.M * (size_t) wl.n_blocks;
    const size_t W_bytes  = W_blocks * sizeof(block_q4_K);

    printf("\n");
    printf("=========================================================\n");
    printf("  m4-kernel-lab  (FASE 4B)\n");
    printf("=========================================================\n");
    printf("  M (rows)     = %d\n", wl.M);
    printf("  K (cols)     = %d  (= %d super-blocks)\n", wl.K, wl.n_blocks);
    printf("  iters/kernel = %d  (+ %d warmup)\n", wl.iters, wl.warmup);
    printf("  W size       = %zu super-blocks = %.2f MiB\n",
           W_blocks, (double) W_bytes / (1024.0 * 1024.0));
    printf("\n");

    // Allocate. Use posix_memalign so SIMD-friendly variants do not pay for
    // unaligned loads.
    block_q4_K * W = nullptr;
    if (posix_memalign((void **)&W, 64, W_bytes) != 0 || !W) {
        fprintf(stderr, "alloc W failed\n");
        return 1;
    }
    q4k_fill_random(W, W_blocks, /*seed=*/ 1);

    std::vector<float> x(wl.K);
    std::vector<float> out_ref(wl.M, 0.0f);
    std::vector<float> out_cur(wl.M, 0.0f);
    fill_random_fp32(x, /*seed=*/ 7);

    // Pre-quantise x to Q8_K so the ggml kernel sees the format it expects.
    // 292 bytes per Q8_K block.
    const size_t q8k_block_size = 4 + QK_K + 2 * (QK_K / 16); // sizeof(block_q8_K)
    std::vector<uint8_t> x_q8k(q8k_block_size * (size_t) wl.n_blocks, 0);
    q4k_lab_quantize_input_q8_k(x.data(), x_q8k.data(), wl.K);

    std::vector<kernel_entry> kernels = {
        { "scalar (C reference)",      IN_FP32, kernel_dot_q4k_fp32_scalar,     nullptr                       },
        { "scalar_opt (branch-free)",  IN_FP32, kernel_dot_q4k_fp32_scalar_opt, nullptr                       },
        { "ggml q4_K x q8_K",          IN_Q8K,  nullptr,                        kernel_dot_q4k_q8k_ggml       },
    };

    // The first kernel is the reference; later kernels are compared against it.
    bool first = true;
    measurement baseline_meas;
    for (const auto & k : kernels) {
        printf("--- %s ---\n", k.name);
        // For correctness comparison we run one untimed pass into either ref or cur.
        std::vector<float> & dst = first ? out_ref : out_cur;
        for (int m = 0; m < wl.M; ++m) {
            if (k.in_kind == IN_FP32) {
                dst[m] = k.fn_fp32(&W[m * wl.n_blocks], x.data(),    wl.n_blocks);
            } else {
                dst[m] = k.fn_q8k (&W[m * wl.n_blocks], x_q8k.data(), wl.n_blocks);
            }
        }
        if (!first) {
            double max_abs = 0.0, sum_abs = 0.0;
            for (int m = 0; m < wl.M; ++m) {
                const double e = std::fabs(out_ref[m] - out_cur[m]);
                if (e > max_abs) max_abs = e;
                sum_abs += e;
            }
            const double rel = max_abs / (std::fabs(out_ref[0]) + 1e-9);
            printf("  vs reference: max_abs=%.4g  mean_abs=%.4g  max_rel=%.4g\n",
                   max_abs, sum_abs / wl.M, rel);
        }

        const measurement meas = time_kernel(k, std::vector<block_q4_K>(W, W + W_blocks),
                                             x, x_q8k, wl, dst);
        printf("  ns/dot:  min=%9.1f  median=%9.1f  max=%9.1f\n",
               meas.ns_min, meas.ns_median, meas.ns_max);
        printf("  GFLOPS:  min=%6.2f  max=%6.2f\n", meas.gflops_min, meas.gflops_max);
        if (!first) {
            const double speedup = baseline_meas.ns_median / meas.ns_median;
            printf("  speedup vs %-22s = %.2fx\n", kernels[0].name, speedup);
        } else {
            baseline_meas = meas;
        }
        printf("\n");
        first = false;
    }

    free(W);
    return 0;
}
