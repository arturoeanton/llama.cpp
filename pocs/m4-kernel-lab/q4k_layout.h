#pragma once

// FASE 4B: standalone Q4_K_M layout for the kernel microbenchmark.
//
// This header replicates the ggml Q4_K_M block layout (ggml/src/ggml-common.h)
// without including any ggml header. The struct is binary-compatible with
// ggml's `block_q4_K` so the same bytes can be fed to either our kernels or
// (later) ggml's own kernel.
//
// Per super-block (256 weights):
//   - d:        fp16 scale for sub-block scales
//   - dmin:     fp16 scale for sub-block mins
//   - scales:   12 bytes packing eight pairs of (6-bit scale, 6-bit min)
//   - qs:       128 bytes of 4-bit quants (256 nibbles)
// Total: 144 bytes / super-block = 4.5 bits per weight.

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define QK_K        256
#define QK_K_SCALES   8      // 8 sub-blocks of 32 weights each
#define K_SCALE_SIZE 12

typedef uint16_t fp16_t;

typedef struct {
    fp16_t  d;
    fp16_t  dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

#define Q4_K_BLOCK_SIZE sizeof(block_q4_K)
_Static_assert(Q4_K_BLOCK_SIZE == 144, "block_q4_K must be 144 bytes");

// fp16<->fp32 helpers. We use a tiny portable converter so the lab does not
// depend on any compiler/platform fp16 intrinsics.
static inline float q4k_fp16_to_fp32(fp16_t h) {
    const uint32_t s = (h & 0x8000) << 16;
    const uint32_t e =  (h & 0x7C00) >> 10;
    const uint32_t m =  (h & 0x03FF);

    uint32_t out;
    if (e == 0) {
        if (m == 0) {
            out = s;
        } else {
            // subnormal -> normalise
            uint32_t mm = m;
            int      ee = -14 + 127;
            while ((mm & 0x0400) == 0) { mm <<= 1; --ee; }
            mm &= 0x03FF;
            out = s | ((uint32_t)ee << 23) | (mm << 13);
        }
    } else if (e == 0x1F) {
        // inf / NaN
        out = s | 0x7F800000u | (m << 13);
    } else {
        // normal
        out = s | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

static inline fp16_t q4k_fp32_to_fp16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t s  = (u & 0x80000000u) >> 16;
    int            e  = (int)((u & 0x7F800000u) >> 23) - 127 + 15;
    uint32_t       m  =  u & 0x007FFFFFu;
    if (e <= 0) {
        // subnormal or zero -> flush to zero for simplicity
        return (fp16_t)s;
    }
    if (e >= 0x1F) {
        return (fp16_t)(s | 0x7C00); // inf
    }
    return (fp16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

// Unpack the 12-byte `scales[]` field into 8 sub-block scales (in [0..63])
// and 8 sub-block mins (in [0..63]). Mirrors the bit-shuffle used by ggml's
// reference scalar kernel.
static inline void q4k_unpack_scales(const uint8_t scales[12],
                                     uint8_t out_scales[8],
                                     uint8_t out_mins[8]) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;
    uint32_t utmp[4];
    memcpy(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    memcpy(out_scales, &utmp[0], 8);
    memcpy(out_mins,   &utmp[2], 8);
}

// Pack 8 scales + 8 mins (each 6-bit, value in [0..63]) into the 12-byte
// layout above. Used only by the random generator; not on the hot path.
static inline void q4k_pack_scales(const uint8_t in_scales[8],
                                   const uint8_t in_mins[8],
                                   uint8_t out_scales[12]) {
    // The 12-byte layout produced by q4k_unpack_scales's inverse. Easiest
    // way: start from the desired utmp[0..3] = (scales lo, scales hi, mins lo, mins hi)
    // and reverse the bit-shuffle.
    //
    // utmp[0] holds scales[0..3] (each <=63 in low 6 bits), utmp[1] holds
    // scales[4..7], utmp[2] holds mins[0..3], utmp[3] holds mins[4..7].
    // The packing puts:
    //   bytes[0..3]  = utmp[0] | ((utmp[1] high bits) << 6) -- but only top 2 bits
    //   bytes[4..7]  = utmp[1] (mins?) ... see code below
    // We just invert the shuffle from q4k_unpack_scales.
    uint32_t s_lo, s_hi, m_lo, m_hi;
    memcpy(&s_lo, &in_scales[0], 4);
    memcpy(&s_hi, &in_scales[4], 4);
    memcpy(&m_lo, &in_mins[0],   4);
    memcpy(&m_hi, &in_mins[4],   4);

    // From the unpacker we have:
    //   out_scales (==utmp[0..1]) low6
    //   out_mins   (==utmp[2..3]) low6
    // and the on-disk bytes b[0..11] correspond to a permuted/packed form.
    // Re-derive on-disk bytes such that q4k_unpack_scales(bytes) reproduces
    // (in_scales, in_mins).
    uint32_t b0, b1, b2;
    b0 = s_lo & 0x3f3f3f3f;
    // bits 6..7 of b[0..3] hold mins[0..3] high2; bits 6..7 of b[4..7] hold mins[4..7] high2
    b0 |= ((m_lo & 0x30303030u) << 2);
    b1 = s_hi & 0x3f3f3f3f;
    b1 |= ((m_hi & 0x30303030u) << 2);
    b2 = (m_lo & 0x0f0f0f0fu) | ((m_hi & 0x0f0f0f0fu) << 4);
    memcpy(&out_scales[0], &b0, 4);
    memcpy(&out_scales[4], &b1, 4);
    memcpy(&out_scales[8], &b2, 4);
}

// Reference scalar dequant of one super-block into 256 fp32 values.
// out[i] = (scale_real * q_4bit) - (min_real)
// where scale_real = d * scale_6bit, min_real = dmin * min_6bit.
static inline void q4k_dequant_block(const block_q4_K * b, float * out) {
    const float d    = q4k_fp16_to_fp32(b->d);
    const float dmin = q4k_fp16_to_fp32(b->dmin);
    uint8_t scales[8], mins[8];
    q4k_unpack_scales(b->scales, scales, mins);
    for (int sb = 0; sb < QK_K_SCALES; ++sb) {
        const float s_real = d    * (float) scales[sb];
        const float m_real = dmin * (float) mins[sb];
        // Each sub-block is 32 weights. The 4-bit quants are packed two per
        // byte; sub-block 0 uses low nibbles of qs[0..31], sub-block 1 uses
        // high nibbles of the same bytes, sub-block 2 uses low nibbles of
        // qs[32..63], and so on -- same ordering as ggml's reference.
        const uint8_t * q = b->qs + (sb / 2) * 32;
        const int       hi = sb & 1;
        for (int i = 0; i < 32; ++i) {
            const int q4 = hi ? (q[i] >> 4) : (q[i] & 0x0F);
            out[sb * 32 + i] = s_real * (float) q4 - m_real;
        }
    }
}

// Fill `n_blocks` consecutive Q4_K_M super-blocks with plausibly-distributed
// random values. The numbers do not correspond to a real quantisation of a
// known float matrix -- they are just there to exercise the same compute
// pattern.
static inline void q4k_fill_random(block_q4_K * blocks, size_t n_blocks, unsigned seed) {
    srand(seed);
    for (size_t b = 0; b < n_blocks; ++b) {
        block_q4_K * blk = &blocks[b];
        // d, dmin: small positive fp16 values.
        blk->d    = q4k_fp32_to_fp16(0.02f + 0.001f * (float)(rand() & 0xFF));
        blk->dmin = q4k_fp32_to_fp16(0.005f + 0.0002f * (float)(rand() & 0xFF));
        // 6-bit scales and mins in [0..63].
        uint8_t s[8], m[8];
        for (int i = 0; i < 8; ++i) {
            s[i] = (uint8_t)(rand() & 0x3F);
            m[i] = (uint8_t)(rand() & 0x3F);
        }
        q4k_pack_scales(s, m, blk->scales);
        // 256 nibbles -> 128 bytes.
        for (int i = 0; i < QK_K/2; ++i) {
            blk->qs[i] = (uint8_t)(rand() & 0xFF);
        }
    }
}
