// FASE 4B kernel #2: branch-free scalar with per-sub-block accumulators.
//
// Same math as kernel_scalar.c but written so the inner loop has no branches,
// no per-element division between hi/lo nibbles, and only two scalar multiplies
// per sub-block instead of 32. The compiler can autovectorise the inner loops
// of nibble extraction and the FMA accumulators without us writing intrinsics.
//
// Algorithm:
//   for each super-block:
//     1. Pre-extract all 256 nibbles into nibbles[256] using a tight 32-iter loop.
//        Order: nibbles[sb*32 .. sb*32+31] holds sub-block `sb`'s quants.
//     2. Decode 8 (scale_real, min_real) pairs once.
//     3. For each sub-block of 32:
//          sum_qx = sum(nibble[i] * x[i])   // 32 int8*fp32 FMAs
//          sum_x  = sum(x[i])               // 32 fp32 adds
//          partial = scale_real * sum_qx - min_real * sum_x
//          acc    += partial
//
// Reduces the inner loop from "1 mul + 1 sub + 1 mul + 1 add per element"
// (4 ops) down to "1 fma + 1 add per element" (2 ops), plus 2 muls per
// sub-block factored out.

#include "../q4k_layout.h"

float kernel_dot_q4k_fp32_scalar_opt(const block_q4_K * row, const float * x, int n_blocks);

float kernel_dot_q4k_fp32_scalar_opt(const block_q4_K * row, const float * x, int n_blocks) {
    float acc = 0.0f;

    // Stack-local scratch. 256 int8 nibbles fits in a single cache line set.
    int8_t nibbles[QK_K];

    for (int bi = 0; bi < n_blocks; ++bi) {
        const block_q4_K * blk = &row[bi];

        // Step 1: extract nibbles in sub-block order. Each pass of `qs`
        // contributes two sub-blocks (low+high nibbles) for 32 weights each.
        for (int pass = 0; pass < 4; ++pass) {
            const uint8_t * q = blk->qs + pass * 32;
            int8_t * lo = nibbles + (pass * 2)     * 32;
            int8_t * hi = nibbles + (pass * 2 + 1) * 32;
            for (int i = 0; i < 32; ++i) {
                lo[i] = (int8_t)(q[i] & 0x0F);
                hi[i] = (int8_t)(q[i] >> 4);
            }
        }

        // Step 2: precompute the 8 (scale_real, min_real) pairs.
        const float d    = q4k_fp16_to_fp32(blk->d);
        const float dmin = q4k_fp16_to_fp32(blk->dmin);
        uint8_t scales[8], mins[8];
        q4k_unpack_scales(blk->scales, scales, mins);
        float s_real[8], m_real[8];
        for (int sb = 0; sb < 8; ++sb) {
            s_real[sb] = d    * (float) scales[sb];
            m_real[sb] = dmin * (float) mins[sb];
        }

        // Step 3: per-sub-block dot + accumulator-of-x.
        const float * xb = x + bi * QK_K;
        for (int sb = 0; sb < 8; ++sb) {
            const int8_t * q  = nibbles + sb * 32;
            const float  * xs = xb      + sb * 32;
            // Two scalar accumulators. The compiler is free to vectorise
            // these as long as they remain associative-equivalent.
            float sum_qx = 0.0f;
            float sum_x  = 0.0f;
            for (int i = 0; i < 32; ++i) {
                sum_qx += (float) q[i] * xs[i];
                sum_x  += xs[i];
            }
            acc += s_real[sb] * sum_qx - m_real[sb] * sum_x;
        }
    }
    return acc;
}
