// FASE 4B kernel #1: scalar reference implementation of one matmul row.
//
// Per row: out = sum_k(W_q4k[k] * x[k]) where
//   - W_q4k is a sequence of `n_blocks` Q4_K_M super-blocks (256 weights each)
//   - x     is a contiguous fp32 vector of length K = n_blocks * 256
//   - out   is a single fp32 scalar
//
// This is intentionally written as a clear, line-by-line dequant + multiply +
// accumulate. It is the correctness baseline that NEON and other variants are
// validated against.

#include "../q4k_layout.h"

float kernel_dot_q4k_fp32_scalar(const block_q4_K * row, const float * x, int n_blocks);

float kernel_dot_q4k_fp32_scalar(const block_q4_K * row, const float * x, int n_blocks) {
    float acc = 0.0f;
    for (int bi = 0; bi < n_blocks; ++bi) {
        const block_q4_K * blk = &row[bi];
        const float d    = q4k_fp16_to_fp32(blk->d);
        const float dmin = q4k_fp16_to_fp32(blk->dmin);
        uint8_t scales[8], mins[8];
        q4k_unpack_scales(blk->scales, scales, mins);

        const float * xb = x + bi * QK_K;
        for (int sb = 0; sb < QK_K_SCALES; ++sb) {
            const float s_real = d    * (float) scales[sb];
            const float m_real = dmin * (float) mins[sb];
            const uint8_t * q  = blk->qs + (sb / 2) * 32;
            const int       hi = sb & 1;
            const float   * xs = xb + sb * 32;
            for (int i = 0; i < 32; ++i) {
                const int q4 = hi ? (q[i] >> 4) : (q[i] & 0x0F);
                const float w = s_real * (float) q4 - m_real;
                acc += w * xs[i];
            }
        }
    }
    return acc;
}
