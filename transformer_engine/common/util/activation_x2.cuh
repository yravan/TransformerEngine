/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file activation_x2.cuh
 *  \brief GELU, dGELU and their erf variants for two FP32 lanes at a time.
 *
 *  Each function reproduces, bit for bit, what util/math.h computes for one lane (including the
 *  libdevice expansions of tanhf, erff and expf), but issues the arithmetic as packed f32x2
 *  instructions on Blackwell.
 */

#ifndef TRANSFORMER_ENGINE_UTIL_ACTIVATION_X2_CUH_
#define TRANSFORMER_ENGINE_UTIL_ACTIVATION_X2_CUH_

#include <type_traits>

#include "../common.h"
#include "math.h"

namespace transformer_engine {

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ float2 splat2(const float v) { return make_float2(v, v); }

// tanhf as libdevice computes it, two lanes at a time on the paired f32x2 FMA path.
__device__ __forceinline__ float2 tanhf_x2(const float2 v) {
  const float2 a = make_float2(fabsf(v.x), fabsf(v.y));
  float2 e = __fmul2_rn(a, splat2(__int_as_float(0x4038AA3B)));
  asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(e.x));
  asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(e.y));
  e = __fadd2_rn(e, splat2(1.f));
  asm("rcp.approx.ftz.f32 %0, %0;" : "+f"(e.x));
  asm("rcp.approx.ftz.f32 %0, %0;" : "+f"(e.y));
  e = __ffma2_rn(e, splat2(-2.f), splat2(1.f));
  const float big = __int_as_float(0x41102CB4);
  e = make_float2(copysignf(a.x >= big ? 1.f : e.x, v.x), copysignf(a.y >= big ? 1.f : e.y, v.y));
  const float2 s = __fmul2_rn(v, v);
  float2 p = __ffma2_rn(s, splat2(__int_as_float(0x3C80F082)), splat2(__int_as_float(0xBD563CAE)));
  p = __ffma2_rn(p, s, splat2(__int_as_float(0x3E085941)));
  p = __ffma2_rn(p, s, splat2(__int_as_float(0xBEAAA9ED)));
  p = __ffma2_rn(p, s, splat2(0.f));
  p = __ffma2_rn(p, v, v);
  const float small = __int_as_float(0x3F19999A);
  return make_float2(a.x >= small ? e.x : p.x, a.y >= small ? e.y : p.y);
}

__device__ __forceinline__ float2 gelu_x2(const float2 c) {
  const float2 u = __ffma2_rn(c, __fmul2_rn(c, splat2(0.03567741f)), splat2(0.79788456f));
  const float2 t = tanhf_x2(__fmul2_rn(c, u));
  return __fmul2_rn(c, __ffma2_rn(t, splat2(0.5f), splat2(0.5f)));
}

__device__ __forceinline__ float2 dgelu_x2(const float2 c) {
  const float2 u = __ffma2_rn(c, __fmul2_rn(c, splat2(0.044715f)), splat2(1.f));
  const float2 t = tanhf_x2(__fmul2_rn(__fmul2_rn(c, splat2(0.79788456f)), u));
  const float2 one_m_t2 = __ffma2_rn(make_float2(-t.x, -t.y), t, splat2(1.f));
  const float2 w = __ffma2_rn(c, __fmul2_rn(c, splat2(0.1070322243f)), splat2(0.79788456f));
  const float2 l = __fmul2_rn(__fmul2_rn(c, splat2(0.5f)), __fmul2_rn(w, one_m_t2));
  return __ffma2_rn(__fadd2_rn(t, splat2(1.f)), splat2(0.5f), l);
}

// erff as libdevice computes it, two lanes at a time.
__device__ __forceinline__ float2 erff_x2(const float2 v) {
  const float2 a = make_float2(fabsf(v.x), fabsf(v.y));
  const float edge = __int_as_float(0x3F8060FE);
  const bool big[2] = {a.x >= edge, a.y >= edge};
  const float2 s = __fmul2_rn(v, v);
  const float2 t = make_float2(big[0] ? a.x : s.x, big[1] ? a.y : s.y);
  const auto c = [&](const int big_c, const int small_c) {
    return make_float2(__int_as_float(big[0] ? big_c : small_c),
                       __int_as_float(big[1] ? big_c : small_c));
  };
  float2 p = __ffma2_rn(c(0x38EB4C3A, 0x38B1E96A), t, c(0xBAAE005B, 0xBA574D20));
  p = __ffma2_rn(p, t, c(0x3C09919F, 0x3BAAD5EA));
  p = __ffma2_rn(p, t, c(0xBD24D99A, 0xBCDC1BE7));
  p = __ffma2_rn(p, t, c(0x3E235519, 0x3DE718AF));
  p = __ffma2_rn(p, t, c(0x3F69B4F9, 0xBEC093AC));
  p = __ffma2_rn(p, t, c(0x3F210A14, 0x3E0375D3));
  const float2 u = make_float2(big[0] ? -t.x : v.x, big[1] ? -t.y : v.y);
  float2 r = __ffma2_rn(p, u, u);
  if (big[0]) {
    asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(r.x));
    r.x = copysignf(1.f - r.x, v.x);
  }
  if (big[1]) {
    asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(r.y));
    r.y = copysignf(1.f - r.y, v.y);
  }
  return r;
}

__device__ __forceinline__ float2 gelu_erf_x2(const float2 c) {
  const float2 e = erff_x2(__fmul2_rn(c, splat2(0.70710678118654752440f)));
  return __fmul2_rn(__fmul2_rn(c, splat2(0.5f)), __fadd2_rn(e, splat2(1.f)));
}

// dgelu_erf with expf(-x * x / 2) as libdevice computes it, two lanes at a time.
__device__ __forceinline__ float2 dgelu_erf_x2(const float2 c) {
  const float2 cdf2 =
      __fadd2_rn(erff_x2(__fmul2_rn(c, splat2(0.70710678118654752440f))), splat2(1.f));
  const float2 h = __fmul2_rn(c, __fmul2_rn(c, splat2(-0.5f)));
  const float2 t = __ffma2_rn(h, splat2(__int_as_float(0x3BBB989D)), splat2(0.5f));
  const float2 j = __ffma2_rd(make_float2(__saturatef(t.x), __saturatef(t.y)),
                              splat2(__int_as_float(0x437C0000)),
                              splat2(__int_as_float(0x4B400001)));
  float2 r = __ffma2_rn(j, splat2(-1.f), splat2(__int_as_float(0x4B40007F)));
  r = __ffma2_rn(h, splat2(__int_as_float(0x3FB8AA3B)), r);
  r = __ffma2_rn(h, splat2(__int_as_float(0x32A57060)), r);
  asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(r.x));
  asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(r.y));
  const float2 two_j = make_float2(__int_as_float(__float_as_int(j.x) << 23),
                                   __int_as_float(__float_as_int(j.y) << 23));
  const float2 pdf = __fmul2_rn(__fmul2_rn(r, two_j), splat2(0.39894228040143267794f));
  return __ffma2_rn(cdf2, splat2(0.5f), __fmul2_rn(c, pdf));
}
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

template <typename ParamOP, float (*OP)(float, const ParamOP &)>
__device__ __forceinline__ float2 op_x2(const float2 v) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if constexpr (std::is_same_v<ParamOP, Empty>) {
    if constexpr (OP == &gelu<fp32, fp32>) {
      return gelu_x2(v);
    } else if constexpr (OP == &dgelu<fp32, fp32>) {
      return dgelu_x2(v);
    } else if constexpr (OP == &gelu_erf<fp32, fp32>) {
      return gelu_erf_x2(v);
    } else if constexpr (OP == &dgelu_erf<fp32, fp32>) {
      return dgelu_erf_x2(v);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  return make_float2(OP(v.x, {}), OP(v.y, {}));
}

template <bool IS_ACT, bool IS_DACT, typename ParamOP, float (*OP)(float, const ParamOP &)>
__device__ __forceinline__ float2 activate_x2(float2 elt, const float2 act_in) {
  if constexpr (IS_ACT) {
    elt = op_x2<ParamOP, OP>(elt);
  }
  if constexpr (IS_DACT) {
    const float2 d = op_x2<ParamOP, OP>(act_in);
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    elt = __fmul2_rn(elt, d);
#else
    elt = make_float2(elt.x * d.x, elt.y * d.y);
#endif
  }
  return elt;
}

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_UTIL_ACTIVATION_X2_CUH_
