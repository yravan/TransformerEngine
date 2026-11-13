/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file cast_bidim.cuh
 *  \brief Register-resident MXFP8 cast kernel and its launcher.
 */

#ifndef TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_BIDIM_CUH_
#define TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_BIDIM_CUH_

#include <cuda_runtime.h>

#include "../../../common.h"
#include "../../../util/math.h"
#include "../../../util/ptx.cuh"
#include "../../../utils.cuh"
#include "../../../util/activation_x2.cuh"
#include "../swizzle.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp8 {
namespace quantize_kernel {
namespace specialized {

// Implementation details.  Both register-resident kernels define constants and
// helpers of the same name; now that they are headers included into the same
// translation unit, each needs its own scope.
namespace bidim_detail {

namespace ptx = transformer_engine::ptx;

// Bidimensional MXFP8 produces two independent quantizations of one BF16
// tensor: a rowwise one, where 32 consecutive elements of a row share a scale,
// and a colwise one, where 32 consecutive elements of a column share a scale.
// Both outputs keep the input's [M, K] row-major layout -- the colwise result
// is not transposed, only scaled differently.
//
// A CTA owns a 32-row band crossed with a strip of columns.  Thirty-two rows is
// exactly the colwise block height, so the colwise reduction closes inside the
// CTA: the tile is read from global memory once, into registers, and drives
// both passes.  That is what lets this kernel drop the TMA pipeline the
// specialized kernel uses -- there is no second pass to stage for.
//
// Within a tile:
//   - each warp owns kRowsPerWarp consecutive rows of the band,
//   - each lane owns COLS_PER_LANE consecutive columns of those rows,
//   - the rowwise scale needs a reduction across the lanes that share a
//     32-column block, which shuffles handle,
//   - the colwise scale needs a reduction down all 32 rows, hence across warps,
//     which is the one thing shuffles cannot reach and the only reason this
//     kernel touches shared memory.

// Elements in one MX block, and equivalently the colwise block height.
constexpr int32_t kBlockElems = 32;
// Rows a CTA covers.  Must equal kBlockElems to keep the colwise reduction
// CTA-local.
constexpr int32_t kRowsPerTile = kBlockElems;

constexpr int32_t kWarpsPerCta = 8;
constexpr int32_t kThreadsPerCta = kWarpsPerCta * THREADS_PER_WARP;
constexpr int32_t kRowsPerWarp = kRowsPerTile / kWarpsPerCta;
static_assert(kRowsPerTile % kWarpsPerCta == 0, "Warps must divide the row band evenly.");

// BF16 packs two elements per 32-bit word, and all the packed math below works
// on those words, so a shared-memory "slot" holds one column pair.
constexpr int32_t kElemsPerWord = sizeof(uint32_t) / sizeof(bf16);

//
// MX scale arithmetic.
//
// The E8M0 scale of a block is the smallest power of two with
// amax / scale <= 448, i.e. biased_exponent = ceil(log2(amax / 448)) + 127.
// For a BF16 amax that has an exact closed form in the bit pattern.
//
// Write amax = m * 2^(e - 127), with the biased exponent e in bits 7..14 and
// the stored mantissa in bits 0..6.  Then
//
//     ceil(log2(amax / 448)) + 127  =  e - 8   for m <= 1.75
//                                   =  e - 7   for m >  1.75
//
// and "m > 1.75" is exactly "mantissa field >= 97", because 1.75 is mantissa 96
// and 97 is the next representable step.  Adding 31 to the mantissa field
// carries into the exponent for mantissa >= 97 and for nothing below it, so a
// single add and mask round e by precisely that rule.
//
// This is not an approximation of the usual amax * (1/448) route: it is exact,
// and better behaved, because 1/448 is not representable in FP32 and that
// product can round across the boundary.
//
// The helpers return the reciprocal scale, ready to multiply by, since that is
// what the packed conversion consumes.  mx_scale_byte recovers the stored E8M0
// byte from it.

constexpr uint32_t kMantissaRoundUp = 31u;
constexpr uint32_t kBf16ExponentMask = 0x7F80u;

// Exponent of the output type's largest normal: 8 for E4M3 (448 = 1.75 * 2^8)
// and 15 for E5M2 (57344 = 1.75 * 2^15).  Both share the 1.75 mantissa, which
// is exactly why the "mantissa field >= 97" rule above holds for either and
// only this offset has to change.
template <typename OType>
constexpr uint32_t kMaxNormExponent = Quantized_Limits<OType>::max_unbiased_exponent;

// Smallest exponent field allowed, so the reciprocal stays a normal BF16.
template <typename OType>
constexpr uint32_t kMinExponentField = kMaxNormExponent<OType> << 7;

// Encoding of 2^(127 + offset); subtracting the rounded exponent field yields
// the reciprocal scale directly.
template <typename OType>
constexpr uint32_t kReciprocalBias = (254u + kMaxNormExponent<OType>) << 7;
// BF16 magnitude at or above which a value is Inf or NaN.
constexpr uint32_t kBf16InfBits = 0x7F80u;
// The two reciprocals the exponent subtraction cannot produce, matching
// ptx::exp2f_rcp<bf16> for E8M0 codes 254 and 255: 2^-127, which is subnormal
// and so lives in the leading mantissa bit rather than the exponent field, and
// BF16 NaN.  Both also give the right scale byte through mx_scale_byte below.
constexpr uint32_t kInfReciprocal = 0x0040u;
constexpr uint32_t kNaNReciprocal = 0x7FFFu;
constexpr uint32_t kBf16MagnitudeMask = 0x7FFFu;
constexpr uint32_t kMagnitudeMaskPair = 0x7FFF7FFFu;
// Added to a masked magnitude, this lifts bit 15 exactly for Inf and NaN.
constexpr uint32_t kInfProbePair = 0x00800080u;
// Bias such that kScaleByteBias - (reciprocal >> 7) is the E8M0 scale byte.
constexpr uint32_t kScaleByteBias = 254u;

/*! \brief Reciprocal MX scale for one BF16 amax, as BF16 bits.
 *  \param amax_bits  Block amax magnitude bits, sign already cleared. */
template <typename OType>
__device__ __forceinline__ uint32_t mx_scale_reciprocal(uint32_t amax_bits) {
  if (amax_bits >= kBf16InfBits) {
    return amax_bits == kBf16InfBits ? kInfReciprocal : kNaNReciprocal;
  }
  const uint32_t rounded_exponent =
      max((amax_bits + kMantissaRoundUp) & kBf16ExponentMask, kMinExponentField<OType>);
  return kReciprocalBias<OType> - rounded_exponent;
}

/*! \brief E8M0 scale byte matching a reciprocal from mx_scale_reciprocal.
 *
 * kNaNReciprocal is the one input whose subtraction goes negative, and it wraps
 * to exactly 255, the E8M0 NaN code.  Every other reciprocal lands in [0, 254].
 */
__device__ __forceinline__ e8m0_t mx_scale_byte(uint32_t reciprocal_bits) {
  return static_cast<e8m0_t>(kScaleByteBias - (reciprocal_bits >> 7));
}

/*! \brief mx_scale_reciprocal applied to two columns at once.
 *
 * Valid only when neither half is Inf or NaN.  Callers test that first, which
 * costs one comparison for the pair.
 */
template <typename OType>
__device__ __forceinline__ uint32_t mx_scale_reciprocal_x2(uint32_t amax_pair) {
  constexpr uint32_t kMantissaRoundUpPair = 0x001F001Fu;
  constexpr uint32_t kExponentMaskPair = 0xFF80FF80u;
  constexpr uint32_t kMinExponentPair = (kMinExponentField<OType> << 16) | kMinExponentField<OType>;
  constexpr uint32_t kReciprocalBiasPair = (kReciprocalBias<OType> << 16) | kReciprocalBias<OType>;

  // Each half is at most 0x7FFF, so adding 31 cannot carry out of the low half
  // into the high one; the two halves round independently.
  const uint32_t magnitudes = amax_pair & kMagnitudeMaskPair;
  ptx::bf16x2 rounded, floor_pair, clamped;
  reinterpret_cast<uint32_t &>(rounded) = (magnitudes + kMantissaRoundUpPair) & kExponentMaskPair;
  reinterpret_cast<uint32_t &>(floor_pair) = kMinExponentPair;
  // Both operands are positive BF16 patterns, so the magnitude-max doubles as a
  // per-half clamp against the minimum exponent.
  ptx::abs_max_2x(clamped, rounded, floor_pair);
  return kReciprocalBiasPair - reinterpret_cast<const uint32_t &>(clamped);
}

/*! \brief The two E8M0 scale bytes of a packed reciprocal pair, in the low
 *         16 bits of the result. */
__device__ __forceinline__ uint32_t mx_scale_byte_x2(uint32_t reciprocal_pair) {
  // The scalar form relies on 254 - 255 wrapping to the NaN code.  Packed, that
  // borrow would run out of the low half and corrupt the other column's byte,
  // so each half gets its own carry headroom and is masked back afterwards.
  constexpr uint32_t kScaleByteBiasPair = 0x01FE01FEu;
  const uint32_t bytes =
      (kScaleByteBiasPair - ((reciprocal_pair >> 7) & 0x00FF00FFu)) & 0x00FF00FFu;
  // Gather the two scale bytes, at byte 0 and byte 2, into the low half.
  return __byte_perm(bytes, 0u, 0x4420);
}

/*! \brief Fold a BF16 pair's halves together and return the magnitude.
 *
 * `max.xorsign.abs` sets the result sign to the XOR of its inputs, so an
 * accumulator built from it can come out negative even when its magnitude is
 * right.  Only the magnitude matters for a scale.
 */
__device__ __forceinline__ uint32_t fold_pair_magnitude(uint32_t pair) {
  ptx::bf16x2 lo, hi, folded;
  reinterpret_cast<uint32_t &>(lo) = pair;
  reinterpret_cast<uint32_t &>(hi) = __byte_perm(pair, pair, 0x1032);
  ptx::abs_max_2x(folded, lo, hi);
  return reinterpret_cast<const uint32_t &>(folded) & kBf16MagnitudeMask;
}

/*! \brief Broadcast a BF16 reciprocal scale into both halves of a pair. */
__device__ __forceinline__ ptx::bf16x2 broadcast_pair(uint32_t scale_bits) {
  ptx::bf16x2 result;
  reinterpret_cast<uint32_t &>(result) = __byte_perm(scale_bits, 0u, 0x1010);
  return result;
}

/*! \brief Scale one lane's BF16 words and pack them into FP8E4M3 words.
 *
 * \param scales  One reciprocal per input word.  The rowwise pass passes the
 *                same broadcast value throughout; the colwise pass gives each
 *                column pair its own.
 */
template <typename OType, int32_t WORDS_IN>
__device__ __forceinline__ void scale_and_convert(const uint32_t (&in)[WORDS_IN],
                                                  const ptx::bf16x2 (&scales)[WORDS_IN],
                                                  uint32_t (&out)[WORDS_IN / 2]) {
#pragma unroll
  for (int32_t i = 0; i < WORDS_IN / 2; ++i) {
    ptx::mul_cvt_4x(reinterpret_cast<ptx::FPx4<OType> &>(out[i]),
                    reinterpret_cast<const ptx::bf16x2 &>(in[2 * i]), scales[2 * i],
                    reinterpret_cast<const ptx::bf16x2 &>(in[2 * i + 1]), scales[2 * i + 1]);
  }
}

/*! \brief Store a lane's output words with the shared L2 policy. */
template <int32_t WORDS>
__device__ __forceinline__ void store_words(void *dst, const uint32_t (&words)[WORDS],
                                            uint64_t policy) {
  static_assert(WORDS == 2 || WORDS == 4, "Output stores are 64- or 128-bit.");
  if constexpr (WORDS == 4) {
    ptx::st_global_b32x4(dst, words, policy);
  } else {
    ptx::st_global_b32x2(dst, words, policy);
  }
}

//
// Shared-memory access for the colwise fold, as one vector operation per call.
//
// WORDS is how many consecutive 32-bit slots a lane owns in each half of the
// slot array: 4 for the wide tile (LDS/STS.128) and 2 for the narrow one
// (LDS/STS.64).  Spelling the vector type out, rather than looping over
// scalars and hoping the compiler merges them, is what guarantees the single
// wide access the half-split layout was designed around.
//

template <int32_t WORDS>
__device__ __forceinline__ void store_shared_words(uint32_t *dst, const uint32_t *src) {
  static_assert(WORDS == 2 || WORDS == 4, "Shared-memory access is 64- or 128-bit.");
  if constexpr (WORDS == 4) {
    *reinterpret_cast<uint4 *>(dst) = make_uint4(src[0], src[1], src[2], src[3]);
  } else {
    *reinterpret_cast<uint2 *>(dst) = make_uint2(src[0], src[1]);
  }
}

template <int32_t WORDS>
__device__ __forceinline__ void load_shared_words(uint32_t *dst, const uint32_t *src) {
  static_assert(WORDS == 2 || WORDS == 4, "Shared-memory access is 64- or 128-bit.");
  if constexpr (WORDS == 4) {
    const uint4 v = *reinterpret_cast<const uint4 *>(src);
    dst[0] = v.x;
    dst[1] = v.y;
    dst[2] = v.z;
    dst[3] = v.w;
  } else {
    const uint2 v = *reinterpret_cast<const uint2 *>(src);
    dst[0] = v.x;
    dst[1] = v.y;
  }
}

/*! \brief The activation (and its gradient) of one BF16 word pair, rounded back to BF16 pairs.
 *
 * \param value  FP32 result before the BF16 rounding, kept for the bias gradient.
 */
template <bool IS_ACT, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
__device__ __forceinline__ uint32_t activate_word(const uint32_t in_word, const uint32_t act_word,
                                                  float2 &value) {
  const float2 in = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 &>(in_word));
  float2 act_in = make_float2(0.f, 0.f);
  if constexpr (IS_DACT) {
    act_in = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 &>(act_word));
  }
  value = activate_x2<IS_ACT, IS_DACT, ParamOP, OP>(in, act_in);
  const __nv_bfloat162 rounded = __floats2bfloat162_rn(value.x, value.y);
  return reinterpret_cast<const uint32_t &>(rounded);
}

/*! \brief Quantize a 32-row band crossed with a strip of columns.
 *
 * \tparam COLS_PER_LANE     Columns one lane owns.  16 gives a 32x512 tile read
 *                           with 256-bit loads; 8 gives 32x256 with 128-bit
 *                           loads, trading load width for more CTAs and so
 *                           better latency hiding on deep shapes.
 * \tparam K_COMPILE_TIME    When non-zero, the column count is a compile-time
 *                           constant, turning the row stride, the scale strides
 *                           and the grid width into immediates.  Zero takes
 *                           them at run time.
 * \tparam MIN_BLOCKS_PER_SM Occupancy target for __launch_bounds__.
 */
template <typename OType, int32_t COLS_PER_LANE, int32_t K_COMPILE_TIME, int32_t MIN_BLOCKS_PER_SM,
          bool SWIZZLED_SCALES, bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
__global__ __launch_bounds__(kThreadsPerCta, MIN_BLOCKS_PER_SM) void quantize_bidim_kernel(
    const uint8_t *__restrict__ input, const uint8_t *__restrict__ act_input,
    uint8_t *__restrict__ output_rowwise, uint8_t *__restrict__ scales_rowwise,
    uint8_t *__restrict__ output_colwise, uint8_t *__restrict__ scales_colwise,
    float *__restrict__ dbias_workspace, int32_t rows, int32_t cols_rt,
    int32_t scale_stride_rowwise_rt, int32_t scale_stride_colwise_rt) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int32_t cols = K_COMPILE_TIME ? K_COMPILE_TIME : cols_rt;
  const int32_t scale_stride_rowwise =
      K_COMPILE_TIME ? DIVUP_TO_MULTIPLE(K_COMPILE_TIME / kBlockElems, 4) : scale_stride_rowwise_rt;
  const int32_t scale_stride_colwise =
      K_COMPILE_TIME ? DIVUP_TO_MULTIPLE(K_COMPILE_TIME, 128) : scale_stride_colwise_rt;

  constexpr int32_t kWordsPerLane = COLS_PER_LANE / kElemsPerWord;
  constexpr int32_t kOutWordsPerLane = kWordsPerLane / 2;
  constexpr int32_t kColsPerTile = THREADS_PER_WARP * COLS_PER_LANE;
  // Lanes that must cooperate to cover one 32-column rowwise block.
  constexpr int32_t kLanesPerRowBlock = kBlockElems / COLS_PER_LANE;
  // Column pairs in the tile, i.e. shared-memory slots for the colwise fold.
  constexpr int32_t kColumnSlots = kColsPerTile / kElemsPerWord;
  // Half-split layout: a lane deposits its partials as two vector stores, one
  // into each half of the slot array, which keeps the store and the later
  // read-back free of bank conflicts.  Four slots per half for the wide tile
  // (COLS_PER_LANE 16, so 128-bit accesses) and two for the narrow one
  // (COLS_PER_LANE 8, so 64-bit) -- both instantiations are live, the narrow
  // tile being what deep grids and K-not-a-multiple-of-512 shapes use.
  constexpr int32_t kSlotsPerLaneHalf = kWordsPerLane / 2;
  constexpr int32_t kHalfSlots = kColumnSlots / 2;

  // Both arrays are indexed by column *pair*, one 32-bit slot each, because
  // that is the granularity the packed BF16 math works at.
  //
  // s_column_partial holds per-warp amax candidates: two BF16 magnitudes.
  // s_column_scale holds the finished *reciprocal* scales: again two BF16, one
  // per column, not the one-byte E8M0 encodings.  BF16 is what mul_cvt_4x
  // multiplies by, so keeping the reciprocal in that form avoids a byte-to-BF16
  // expansion inside the conversion loop; the E8M0 bytes are produced once, by
  // mx_scale_byte_x2, only when warp 0 writes the scale array out.
  __shared__ uint32_t s_column_partial[kWarpsPerCta][kColumnSlots];
  __shared__ uint32_t s_column_scale[kColumnSlots];

  const int32_t tid = threadIdx.x;
  const int32_t warp = tid / THREADS_PER_WARP;
  const int32_t lane = tid % THREADS_PER_WARP;

  const int32_t row0 = blockIdx.y * kRowsPerTile + warp * kRowsPerWarp;
  const int32_t col0 = blockIdx.x * kColsPerTile;

  // One policy for every stream.  Each byte is touched once, but marking them
  // evict_last measured better than streaming: holding the tile's lines through
  // the CTA's lifetime is what keeps the three write bursts coalesced.
  const uint64_t policy = ptx::create_l2_policy_evict_last();

  const size_t row_stride_bytes = static_cast<size_t>(cols) * sizeof(bf16);
  const size_t lane_offset = static_cast<size_t>(row0) * cols + col0 + COLS_PER_LANE * lane;
  const uint8_t *tile_in = input + lane_offset * sizeof(bf16);
  uint8_t *out_row = output_rowwise + lane_offset;
  uint8_t *out_col = output_colwise + lane_offset;
  // One lane of each cooperating group writes the shared rowwise scale byte.
  const bool owns_row_scale = (lane % kLanesPerRowBlock) == 0;
  // Where this lane's scale byte for row0 goes, and how far apart the rows are.  In the swizzled
  // layout the band's 32 rows are the 32 consecutive 16-byte groups of one tile, and a warp's
  // rows never cross a band, so the index is computed once and the rest is a stride; the 64-bit
  // swizzle arithmetic otherwise cost more instructions per row than the quantization did.
  const int32_t row_scale_col = col0 / kBlockElems + lane / kLanesPerRowBlock;
  uint8_t *const scale_row = SWIZZLED_SCALES
      ? scales_rowwise + swizzle::gemm_swizzled_scale_idx(row0, row_scale_col, DIVUP(cols, 128))
      : scales_rowwise + static_cast<size_t>(row0) * scale_stride_rowwise + row_scale_col;
  const int32_t scale_row_stride = SWIZZLED_SCALES ? swizzle::kSwizzledRowStride : scale_stride_rowwise;

  // Read the whole tile into registers; both passes run off these.
  uint32_t tile[kRowsPerWarp][kWordsPerLane];
#pragma unroll
  for (int32_t i = 0; i < kRowsPerWarp; ++i) {
    const void *src = tile_in + i * row_stride_bytes;
    if constexpr (kWordsPerLane == 8) {
      ptx::ld_global_nc_b32x8(tile[i], src, policy);
    } else {
      ptx::ld_global_nc_b32x4(tile[i], src, policy);
    }
  }

  constexpr bool kActivates = IS_ACT || IS_DACT;
  if constexpr (kActivates) {
    // The activation is evaluated in FP32 and rounded to BF16 once; the quantization below only
    // ever sees the rounded values, exactly as if the activation had run as a separate kernel.
    uint32_t act_tile[kRowsPerWarp][IS_DACT ? kWordsPerLane : 1];
    if constexpr (IS_DACT) {
      const uint8_t *act_in = act_input + lane_offset * sizeof(bf16);
#pragma unroll
      for (int32_t i = 0; i < kRowsPerWarp; ++i) {
        const void *src = act_in + i * row_stride_bytes;
        if constexpr (kWordsPerLane == 8) {
          ptx::ld_global_nc_b32x8(act_tile[i], src, policy);
        } else {
          ptx::ld_global_nc_b32x4(act_tile[i], src, policy);
        }
      }
    }
    float2 column_sum[IS_DBIAS ? kWordsPerLane : 1];
    if constexpr (IS_DBIAS) {
#pragma unroll
      for (int32_t k = 0; k < kWordsPerLane; ++k) {
        column_sum[k] = make_float2(0.f, 0.f);
      }
    }
#pragma unroll
    for (int32_t i = 0; i < kRowsPerWarp; ++i) {
#pragma unroll
      for (int32_t k = 0; k < kWordsPerLane; ++k) {
        float2 value;
        tile[i][k] = activate_word<IS_ACT, IS_DACT, ParamOP, OP>(
            tile[i][k], IS_DACT ? act_tile[i][k] : 0u, value);
        if constexpr (IS_DBIAS) {
          column_sum[k] = __fadd2_rn(column_sum[k], value);
        }
      }
    }
    if constexpr (IS_DBIAS) {
      __shared__ float s_dbias[kWarpsPerCta][kColsPerTile];
#pragma unroll
      for (int32_t k = 0; k < kWordsPerLane; ++k) {
        s_dbias[warp][COLS_PER_LANE * lane + 2 * k] = column_sum[k].x;
        s_dbias[warp][COLS_PER_LANE * lane + 2 * k + 1] = column_sum[k].y;
      }
      __syncthreads();
      for (int32_t c = tid; c < kColsPerTile; c += kThreadsPerCta) {
        float sum = s_dbias[0][c];
#pragma unroll
        for (int32_t w = 1; w < kWarpsPerCta; ++w) {
          sum += s_dbias[w][c];
        }
        dbias_workspace[static_cast<size_t>(blockIdx.y) * cols + col0 + c] = sum;
      }
    }
  }

  // ---- rowwise pass -------------------------------------------------------
#pragma unroll
  for (int32_t i = 0; i < kRowsPerWarp; ++i) {
    ptx::bf16x2 amax;
    reinterpret_cast<uint32_t &>(amax) = tile[i][0];
#pragma unroll
    for (int32_t k = 1; k < kWordsPerLane; ++k) {
      ptx::abs_max_2x(amax, amax, reinterpret_cast<const ptx::bf16x2 &>(tile[i][k]));
    }
    // Butterfly across the lanes that share this 32-column block.
#pragma unroll
    for (int32_t d = 1; d < kLanesPerRowBlock; d <<= 1) {
      ptx::bf16x2 partner;
      reinterpret_cast<uint32_t &>(partner) =
          __shfl_xor_sync(0xFFFFFFFFu, reinterpret_cast<const uint32_t &>(amax), d);
      ptx::abs_max_2x(amax, amax, partner);
    }

    const uint32_t reciprocal =
        mx_scale_reciprocal<OType>(fold_pair_magnitude(reinterpret_cast<const uint32_t &>(amax)));
    const ptx::bf16x2 broadcast = broadcast_pair(reciprocal);
    ptx::bf16x2 scales[kWordsPerLane];
#pragma unroll
    for (int32_t k = 0; k < kWordsPerLane; ++k) {
      scales[k] = broadcast;
    }
    uint32_t out_words[kOutWordsPerLane];
    scale_and_convert<OType>(tile[i], scales, out_words);
    store_words(out_row + static_cast<size_t>(i) * cols, out_words, policy);

    if (owns_row_scale) {
      ptx::st_global_b8(scale_row + i * scale_row_stride, mx_scale_byte(reciprocal), policy);
    }
  }

  // ---- colwise pass -------------------------------------------------------
  // Each warp reduces its own rows first; the warps then meet in shared memory,
  // the only cross-warp step in the kernel.
  uint32_t column_partial[kWordsPerLane];
#pragma unroll
  for (int32_t k = 0; k < kWordsPerLane; ++k) {
    ptx::bf16x2 acc;
    reinterpret_cast<uint32_t &>(acc) = tile[0][k];
#pragma unroll
    for (int32_t i = 1; i < kRowsPerWarp; ++i) {
      ptx::abs_max_2x(acc, acc, reinterpret_cast<const ptx::bf16x2 &>(tile[i][k]));
    }
    column_partial[k] = reinterpret_cast<const uint32_t &>(acc);
  }

  store_shared_words<kSlotsPerLaneHalf>(&s_column_partial[warp][kSlotsPerLaneHalf * lane],
                                        &column_partial[0]);
  store_shared_words<kSlotsPerLaneHalf>(
      &s_column_partial[warp][kHalfSlots + kSlotsPerLaneHalf * lane],
      &column_partial[kSlotsPerLaneHalf]);
  __syncthreads();

  // Fold the per-warp partials into one scale per column pair.  Threads take
  // stride-1 slots so the strided reads across warps stay conflict-free.
  for (int32_t slot = tid; slot < kColumnSlots; slot += kThreadsPerCta) {
    const uint32_t *column = &s_column_partial[0][slot];
    ptx::bf16x2 acc;
    reinterpret_cast<uint32_t &>(acc) = column[0];
#pragma unroll
    for (int32_t w = 1; w < kWarpsPerCta; ++w) {
      ptx::abs_max_2x(acc, acc, reinterpret_cast<const ptx::bf16x2 &>(column[w * kColumnSlots]));
    }
    const uint32_t amax_pair = reinterpret_cast<const uint32_t &>(acc);

    uint32_t reciprocal_pair;
    // Test both halves directly rather than folding them first: the fold uses
    // max.xorsign.abs, which returns the *other* operand when one is NaN, so a
    // NaN column paired with a finite one would slip into the packed path.
    // magnitude + 0x0080 sets bit 15 exactly when magnitude >= 0x7F80, and the
    // sum cannot carry between halves.
    const uint32_t exceptional = ((amax_pair & kMagnitudeMaskPair) + kInfProbePair) & 0x80008000u;
    if (__builtin_expect(exceptional == 0u, 1)) {
      reciprocal_pair = mx_scale_reciprocal_x2<OType>(amax_pair);
    } else {
      // At least one of the two columns is Inf or NaN; take them separately.
      const uint32_t lo = mx_scale_reciprocal<OType>(amax_pair & kBf16MagnitudeMask);
      const uint32_t hi = mx_scale_reciprocal<OType>((amax_pair >> 16) & kBf16MagnitudeMask);
      reciprocal_pair = lo | (hi << 16);
    }
    s_column_scale[slot] = reciprocal_pair;
  }
  __syncthreads();

  // Read back the reciprocals covering this lane's own columns.
  uint32_t column_reciprocal[kWordsPerLane];
  load_shared_words<kSlotsPerLaneHalf>(&column_reciprocal[0],
                                       &s_column_scale[kSlotsPerLaneHalf * lane]);
  load_shared_words<kSlotsPerLaneHalf>(&column_reciprocal[kSlotsPerLaneHalf],
                                       &s_column_scale[kHalfSlots + kSlotsPerLaneHalf * lane]);

  // Warp 0 emits the colwise scale row: one byte per column, one coalesced
  // vector store per lane.
  if (warp == 0 && SWIZZLED_SCALES) {
    // The lane's columns are consecutive within one 32-column group, so their swizzled slots
    // are one stride apart from the first column's.
    uint8_t *const scale_col = scales_colwise + swizzle::gemm_swizzled_scale_idx(
                                                    col0 + COLS_PER_LANE * lane, blockIdx.y,
                                                    DIVUP(rows, 128));
#pragma unroll
    for (int32_t c = 0; c < COLS_PER_LANE; c += 2) {
      const uint32_t pair_bytes = mx_scale_byte_x2(column_reciprocal[c / 2]);
#pragma unroll
      for (int32_t h = 0; h < 2; ++h) {
        ptx::st_global_b8(scale_col + (c + h) * swizzle::kSwizzledRowStride,
                          (pair_bytes >> (8 * h)) & 0xFFu, policy);
      }
    }
  } else if (warp == 0) {
    uint32_t packed[kOutWordsPerLane];
#pragma unroll
    for (int32_t j = 0; j < kOutWordsPerLane; ++j) {
      packed[j] = mx_scale_byte_x2(column_reciprocal[2 * j]) |
                  (mx_scale_byte_x2(column_reciprocal[2 * j + 1]) << 16);
    }
    store_words(scales_colwise + static_cast<size_t>(blockIdx.y) * scale_stride_colwise + col0 +
                    COLS_PER_LANE * lane,
                packed, policy);
  }

  // ---- colwise quantized output -------------------------------------------
  ptx::bf16x2 colwise_scales[kWordsPerLane];
#pragma unroll
  for (int32_t k = 0; k < kWordsPerLane; ++k) {
    reinterpret_cast<uint32_t &>(colwise_scales[k]) = column_reciprocal[k];
  }
#pragma unroll
  for (int32_t i = 0; i < kRowsPerWarp; ++i) {
    uint32_t out_words[kOutWordsPerLane];
    scale_and_convert<OType>(tile[i], colwise_scales, out_words);
    store_words(out_col + static_cast<size_t>(i) * cols, out_words, policy);
  }
#endif  // (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

namespace {

// Tile shapes.  The wide tile reads 256 bits per lane and amortises the per-row scale work
// over 512 columns; the narrow tile halves the register footprint, which is what the
// bias-gradient variant needs: its second input and column sums do not fit the wide tile's
// 64-register budget without spilling (0.51 vs 0.55 ms at 86016x4096).
constexpr int32_t kWideColsPerLane = 16;
constexpr int32_t kNarrowColsPerLane = 8;
// Four CTAs of 256 threads, i.e. 64 registers; ptxas accepts higher targets, but the extra
// occupancy they buy was slower than the instructions they cost.
constexpr int32_t kWideMinBlocksPerSm = 4;
constexpr int32_t kNarrowMinBlocksPerSm = 4;

// Thread-block cluster widths.  Clustering does not reduce L2 traffic; what it
// buys is delivery rate when co-scheduled CTAs read adjacent columns of the same
// rows, and only a cast-only kernel at the memory roofline collects it.  The
// fused activations are issue-bound and lose 5-8% to any cluster (gelu_erf
// 0.326 vs 0.342 ms at 86016x4096), so they launch unclustered.
constexpr int32_t kActivationCluster = 1;
constexpr int32_t kCastCluster = 4;

/*! \brief Launch quantize_bidim_kernel, specializing on K where we can.
 *
 * The handful of column counts below cover the shapes that matter in practice;
 * anything else takes the run-time path, which costs a few address
 * computations rather than a recompile.
 */
template <typename OType, int32_t COLS_PER_LANE, int32_t MIN_BLOCKS_PER_SM,
          bool SWIZZLED_SCALES, bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void launch_tiled(const void *input, const void *act_input, void *output_rowwise,
                  void *scales_rowwise, void *output_colwise, void *scales_colwise,
                  float *dbias_workspace, int32_t rows, int32_t cols,
                  int32_t scale_stride_rowwise, int32_t scale_stride_colwise, int32_t cluster_width,
                  cudaStream_t stream) {
  constexpr int32_t kColsPerTile = THREADS_PER_WARP * COLS_PER_LANE;
  const dim3 grid(cols / kColsPerTile, rows / kRowsPerTile);

  // A cluster launch is rejected outright unless its width divides the grid, so
  // narrow the requested width to the largest power of two that does.  Column
  // counts that are not a multiple of the tile width times the cluster width --
  // 7168 columns with the narrow tile, for instance, giving a grid of 28 -- are
  // otherwise a hard launch failure rather than a slow path.
  int32_t cluster_x = cluster_width;
  while (cluster_x > 1 && static_cast<int32_t>(grid.x) % cluster_x != 0) {
    --cluster_x;
  }

  cudaLaunchAttribute cluster_attr;
  cluster_attr.id = cudaLaunchAttributeClusterDimension;
  cluster_attr.val.clusterDim.x = cluster_x;
  cluster_attr.val.clusterDim.y = 1;
  cluster_attr.val.clusterDim.z = 1;

  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = dim3(kThreadsPerCta);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &cluster_attr;
  config.numAttrs = 1;

  const uint8_t *in = reinterpret_cast<const uint8_t *>(input);
  const uint8_t *act_in = reinterpret_cast<const uint8_t *>(act_input);
  uint8_t *qrow = reinterpret_cast<uint8_t *>(output_rowwise);
  uint8_t *srow = reinterpret_cast<uint8_t *>(scales_rowwise);
  uint8_t *qcol = reinterpret_cast<uint8_t *>(output_colwise);
  uint8_t *scol = reinterpret_cast<uint8_t *>(scales_colwise);

#define NVTE_LAUNCH_BIDIM(K_CONST)                                                             \
  do {                                                                                         \
    auto kernel = quantize_bidim_kernel<OType, COLS_PER_LANE, K_CONST, MIN_BLOCKS_PER_SM,      \
                                        SWIZZLED_SCALES, IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>; \
    if (cluster_x > 1) {                                                                       \
      NVTE_CHECK_CUDA(cudaLaunchKernelEx(&config, kernel, in, act_in, qrow, srow, qcol, scol,  \
                                         dbias_workspace, rows, cols, scale_stride_rowwise,    \
                                         scale_stride_colwise));                               \
    } else {                                                                                   \
      kernel<<<grid, kThreadsPerCta, 0, stream>>>(in, act_in, qrow, srow, qcol, scol,          \
                                                  dbias_workspace, rows, cols,                 \
                                                  scale_stride_rowwise, scale_stride_colwise); \
      NVTE_CHECK_CUDA(cudaGetLastError());                                                     \
    }                                                                                          \
  } while (0)

  // Specialising on K folds `cols` and both scale strides into constants, which
  // cuts a noticeable share of the executed instructions.  This kernel is
  // memory-bound, so that only becomes time where the memory system has slack --
  // which means small K, not the large shapes the specialisation looks aimed at.
  // Keep the small cases; everything else takes the runtime path.  All column
  // counts work either way: `default` is the runtime instantiation, not a
  // rejection.
  switch (cols) {
    case 2048:
      NVTE_LAUNCH_BIDIM(2048);
      break;
    case 4096:
      NVTE_LAUNCH_BIDIM(4096);
      break;
    default:
      NVTE_LAUNCH_BIDIM(0);
      break;
  }
#undef NVTE_LAUNCH_BIDIM
}

}  // namespace

/*! \brief Cast BF16 to MXFP8 with both rowwise and colwise scales.
 *
 * Produces two quantizations of the same tensor: a rowwise one, where 32
 * consecutive elements of a row share a scale, and a colwise one, where 32
 * consecutive elements of a column share a scale.  Both outputs keep the
 * input's row-major [rows, cols] layout; the colwise result is not transposed.
 *
 * Unlike the TMA kernel in mxfp8/specialized this one is register-resident: a
 * CTA reads its 32-row tile once and drives both passes from registers.  See
 * cast_bidim.cu for the layout and tuning rationale.
 *
 * Requires SM 10.0+ (Blackwell), matching MXFP8 support in the rest of TE.
 *
 *  \tparam     OType                 FP8 output type.
 *  \param[in]  input                 BF16 input, [rows, cols], row-major.
 *  \param[out] output_rowwise        FP8E4M3 rowwise-scaled output, [rows, cols].
 *  \param[out] scales_rowwise        E8M0 rowwise scales, one per 32 columns.
 *  \param[out] output_colwise        FP8E4M3 colwise-scaled output, [rows, cols].
 *  \param[out] scales_colwise        E8M0 colwise scales, one per 32 rows.
 *  \param[in]  rows                  Row count; must be a multiple of 32.
 *  \param[in]  cols                  Column count; must be a multiple of 256.
 *  \param[in]  scale_stride_rowwise  Rowwise scale elements per row.
 *  \param[in]  scale_stride_colwise  Colwise scale elements per 32-row band.
 *  \param[in]  stream                CUDA stream.
 */
template <typename OType, bool SWIZZLED_SCALES, bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ParamOP,
          float (*OP)(float, const ParamOP &)>
void launch_cast_bidim_packed(const void *input, const void *act_input, void *output_rowwise,
                              void *scales_rowwise, void *output_colwise, void *scales_colwise,
                              float *dbias_workspace, int rows, int cols, int scale_stride_rowwise,
                              int scale_stride_colwise, cudaStream_t stream) {
  constexpr int32_t kWideColsPerTile = THREADS_PER_WARP * kWideColsPerLane;
  constexpr int32_t kNarrowColsPerTile = THREADS_PER_WARP * kNarrowColsPerLane;

  NVTE_CHECK(rows % kRowsPerTile == 0, "Bidimensional MXFP8 requires the row count (", rows,
             ") to be a multiple of the MX block size (", kRowsPerTile, ").");
  NVTE_CHECK(cols % kNarrowColsPerTile == 0, "Bidimensional MXFP8 requires the column count (",
             cols, ") to be a multiple of ", kNarrowColsPerTile, ".");

  constexpr int32_t kCluster = (IS_ACT || IS_DACT) ? kActivationCluster : kCastCluster;
  if (cols % kWideColsPerTile == 0 && !IS_DBIAS) {
    launch_tiled<OType, kWideColsPerLane, kWideMinBlocksPerSm, SWIZZLED_SCALES, IS_DBIAS, IS_DACT,
                 IS_ACT, ParamOP, OP>(input, act_input, output_rowwise, scales_rowwise,
                                      output_colwise, scales_colwise, dbias_workspace, rows, cols,
                                      scale_stride_rowwise, scale_stride_colwise, kCluster, stream);
    return;
  }

  // The bias gradient, or a column count that only the narrow tile divides.
  launch_tiled<OType, kNarrowColsPerLane, kNarrowMinBlocksPerSm, SWIZZLED_SCALES, IS_DBIAS,
               IS_DACT, IS_ACT, ParamOP, OP>(
      input, act_input, output_rowwise, scales_rowwise, output_colwise, scales_colwise,
      dbias_workspace, rows, cols, scale_stride_rowwise, scale_stride_colwise, kCluster, stream);
}

/*! \brief Entry point: cast, optionally fused with an activation or its gradient and the bias
 *         gradient. */
template <typename OType, bool SWIZZLED_SCALES, bool IS_DBIAS = false, bool IS_DACT = false,
          bool IS_ACT = false, typename ParamOP = Empty,
          float (*OP)(float, const ParamOP &) = nullptr>
inline void launch_cast_bidim(const void *input, const void *act_input, void *output_rowwise,
                              void *scales_rowwise, void *output_colwise, void *scales_colwise,
                              float *dbias_workspace, int rows, int cols, int scale_stride_rowwise,
                              int scale_stride_colwise, cudaStream_t stream) {
  launch_cast_bidim_packed<OType, SWIZZLED_SCALES, IS_DBIAS, IS_DACT, IS_ACT, ParamOP, OP>(
      input, act_input, output_rowwise, scales_rowwise, output_colwise, scales_colwise,
      dbias_workspace, rows, cols, scale_stride_rowwise, scale_stride_colwise, stream);
}

}  // namespace bidim_detail

constexpr size_t kBidimRows = bidim_detail::kRowsPerTile;
constexpr size_t kBidimCols = 256;

using bidim_detail::launch_cast_bidim;
using bidim_detail::launch_cast_bidim_packed;

}  // namespace specialized
}  // namespace quantize_kernel
}  // namespace mxfp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_MXFP8_SPECIALIZED_CAST_BIDIM_CUH_
