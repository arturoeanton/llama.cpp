// FASE 4B kernel #3: thin wrapper around ggml's production Q4_K x Q8_K dot.
//
// The harness calls us with FP32 input. ggml's kernel wants the input
// quantised to Q8_K, so we quantise once per call (outside the inner time
// loop -- see main.cpp). The cost we time is JUST the matmul, the same
// way ggml times it during decode (Q8_K quantisation of inputs is amortised
// across the whole matmul row).
//
// Linkage: we declare the ggml-cpu entry points manually so this file does
// not need any ggml header. The actual symbols come from libggml-cpu.dylib
// (built by the main llama.cpp cmake project).

#include "../q4k_layout.h"

#include <stdint.h>
#include <stddef.h>

// Q8_K block layout from ggml-common.h. Replicated here so we do not need
// to pull in ggml headers. Binary-compatible with ggml's struct.
#define Q8K_BLOCK_QK 256
typedef struct {
    float   d;
    int8_t  qs[Q8K_BLOCK_QK];
    int16_t bsums[Q8K_BLOCK_QK / 16];
} block_q8_K;

// ggml-cpu public(ish) entry points. These are exported from libggml-cpu.dylib
// (we verified with `nm`). Signatures from ggml/src/ggml-cpu/quants.h.
extern void quantize_row_q8_K     (const float * x, void * y, int64_t k);
extern void ggml_vec_dot_q4_K_q8_K(int n, float * s, size_t bs,
                                   const void * vx, size_t bx,
                                   const void * vy, size_t by, int nrc);

// Quantise a length-K fp32 vector into Q8_K blocks. Writes (K/QK_K) blocks
// into y. The lab calls this once per kernel invocation so the per-call cost
// is amortised the same way ggml amortises it across an N-row matmul.
void q4k_lab_quantize_input_q8_k(const float * x, void * y, int K);

void q4k_lab_quantize_input_q8_k(const float * x, void * y, int K) {
    quantize_row_q8_K(x, y, K);
}

float kernel_dot_q4k_q8k_ggml(const block_q4_K * row, const void * x_q8k, int n_blocks);

float kernel_dot_q4k_q8k_ggml(const block_q4_K * row, const void * x_q8k, int n_blocks) {
    const int n = n_blocks * QK_K;
    float out = 0.0f;
    ggml_vec_dot_q4_K_q8_K(n, &out, /*bs=*/ 0,
                           /*vx=*/ row,    /*bx=*/ 0,
                           /*vy=*/ x_q8k,  /*by=*/ 0,
                           /*nrc=*/ 1);
    return out;
}
