# m4-kernel-lab — Q-quant matmul microbenchmark

Standalone microbenchmark to measure Q-quantised matmul kernels on Apple M4
(or any aarch64 host). Self-contained: it never reaches into the slotted-real
runtime, never re-builds llama-cli. The only coupling is an optional link
against the already-built `libggml-cpu.dylib` so we can compare our kernels
against the production ggml implementation.

## Scope

| Iter | Kernel | Input | Status |
|---|---|---|---|
| A | `scalar` (C reference)         | FP32 | done |
| A | `scalar_opt` (branch-free)     | FP32 | done |
| C | `ggml_q4_K_q8_K` wrapper       | Q8_K | done |
| B | `neon` (NEON intrinsics)       | Q8_K | TODO |

The point of running A and C before B is to know what we are targeting. The
ggml production kernel is the number to beat (or to get close to).

## Workload

Default: one Llama-class FFN-up row.
- M = 256 rows (gives statistical variance without saturating caches)
- K = 8192 = 32 super-blocks of 256 weights each
- 50 timed iterations + 5 warmup

Override at runtime:
```
./build/m4_kernel_lab <M> <K> <iters>
```

K must be a multiple of `QK_K` (256).

## Build

Requires the main llama.cpp build at `../../build/` (specifically
`bin/libggml-cpu.dylib` and `bin/libggml-base.dylib`). If you built llama.cpp
elsewhere, pass `-DLLAMA_BUILD_DIR=/path/to/build`.

```bash
cmake -B pocs/m4-kernel-lab/build pocs/m4-kernel-lab
cmake --build pocs/m4-kernel-lab/build
./pocs/m4-kernel-lab/build/m4_kernel_lab
```

If `libggml-cpu` is missing the build still works but `kernel_ggml.c` will
fail to link (you can comment it out of CMakeLists.txt to test just the
scalar variants).

## Measured numbers (Apple M4 / 24 GB, single-thread)

```
M=256  K=8192  iters=50

scalar (C reference)         ns/dot median = 6973   GFLOPS max =   2.56   1.00x
scalar_opt (branch-free)     ns/dot median = 2503   GFLOPS max =   6.64   2.79x
ggml q4_K x q8_K (NEON)      ns/dot median =  124   GFLOPS max = 145.89  56.23x
```

The ~56x gap between our scalar reference and ggml has two sources:

1. **Algorithm:** ggml uses `Q8_K` input (int8-quantised activations) and
   does int8 multiply-accumulate into int32, applying the fp32 scale only
   once per sub-block of 32. Our scalar paths consume FP32 input and pay a
   per-element FP32 multiply.
2. **Vectorisation:** ggml's per-arch implementation uses NEON dot-product
   instructions on Apple Silicon (vdotq_s32 family).

Closing the gap requires both: (a) consume Q8_K input, (b) use NEON. That is
the FASE 4B Iter B target.

## Why Q4_K is the lab and Q3_K_XL is the real target

The model this whole experiment validates is
`Meta-Llama-3.1-70B-Instruct-Q3_K_XL.gguf` (35 GB on disk), running through
the slotted-real path with `--no-repack`. Most of its weights are
**Q3_K**, not Q4_K. The two quantisations share the same super-block layout
philosophy (256 elements per block, 8 sub-blocks of 32, two fp16 super
scales, packed 6-bit per-sub-block scales/mins, packed quantised values) but
differ in:
- bits per weight (3 vs 4)
- the on-disk packing of the quantised values (Q3_K uses 2-bit qs plus 1-bit
  hmask = 256 bits per sub-block; Q4_K uses 4-bit qs)
- the matching activation format (`q3_K` pairs with `q8_K` just like `q4_K`)

We are doing the lab on Q4_K first because:
- It is the simpler of the two layouts (no separate high-bit mask).
- The ggml kernel for Q4_K x Q8_K is heavily exercised and is a stable
  comparison target.
- The NEON techniques (vld1q_u8, vshr, vand, vmlaq, vdot) transfer directly
  to Q3_K once the bit-shuffling is understood here.

Iter B (NEON) on Q4_K will produce a kernel that is **NOT directly useful**
for the 70B target. A follow-up phase (Iter D, optional) would port that
NEON kernel to Q3_K_XL. The reason we stop at Q4_K NEON for now is to keep
each iteration small and decide whether to invest in Q3_K based on real
numbers, not speculation.

## File layout

```
.
├── CMakeLists.txt
├── q4k_layout.h               # block_q4_K, fp16 helpers, scales unpack/pack, fake-data gen
├── bench/
│   └── timing.h               # mach_absolute_time / clock_gettime wrapper
├── kernels/
│   ├── kernel_scalar.c        # FASE 4B Iter A baseline (FP32 input)
│   ├── kernel_scalar_opt.c    # FASE 4B Iter A branch-free (FP32 input)
│   └── kernel_ggml.c          # FASE 4B Iter C, calls into libggml-cpu (Q8_K input)
└── main.cpp                   # harness: random data, warmup, timing, correctness report
```

## Correctness notes

- `scalar_opt` vs `scalar`: `max_abs ~0.015`, `max_rel ~1.5e-4`. The
  difference is summation re-association in fp32 -- not a bug. Both compute
  the same mathematical expression with the same FP32 inputs.
- `ggml` vs `scalar`: `max_abs ~11`, `max_rel ~0.11`. This is the Q8_K
  quantisation of the activations -- a real lossy conversion. In production,
  ggml does this everywhere and the loss is part of the cost we already pay.

There is no way to make the ggml number bit-comparable to the FP32-input
scalar without changing the algorithm (either both scalars run on Q8_K, or
both kernels run on FP32). The lab keeps them apart and reports max_abs so
the reader knows where the divergence comes from.

## Out of scope

- Multi-threading. Lab is single-thread.
- Metal / CUDA / Vulkan. Lab is CPU only.
- Handwritten assembly. Lab uses C and NEON intrinsics only.
- Integration with the runtime. The lab does not affect `llama-cli`.

## Next steps

1. **Iter B**: implement `kernel_neon.c` (Q8_K input). Target: <= 200 ns/dot
   on K=8192, >= 80 GFLOPS. Stretch: match ggml.
2. **Iter D (optional)**: port the working Iter B NEON kernel to Q3_K_XL,
   so the technique applies to the actual 70B target. Only worth doing if
   Iter B gets within 2x of ggml on Q4_K.
3. **Iter E (much later)**: if and only if Iter B/D are clearly throughput-
   bound and not memory-bound, evaluate whether handwritten assembly buys
   anything beyond intrinsics. Today the evidence says memory bandwidth is
   the bottleneck on M4 for this workload, so assembly probably does not
   help.
