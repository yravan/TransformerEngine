/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file gated_fp8.cuh
 *  \brief CUDA kernels to cast to FP8 with gated activations.
 */

#ifndef TRANSFORMER_ENGINE_GATED_FP8_CUH_
#define TRANSFORMER_ENGINE_GATED_FP8_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/transformer_engine.h>

#include "../../common.h"
#include "../../util/cuda_runtime.h"
#include "../../util/math.h"
#include "../../util/ptx.cuh"
#include "../../util/vectorized_pointwise.h"
#include "../../utils.cuh"

namespace transformer_engine {
namespace dispatch {
namespace fp8 {
namespace kernel {

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 512;
constexpr size_t THREADS_PER_CHUNK_X = CHUNK_DIM_X;
constexpr size_t THREADS_PER_CHUNK_Y = THREADS_PER_CHUNK / THREADS_PER_CHUNK_X;  // 4 = 512 / 128
constexpr size_t BUFFERS_NUM = 2;
constexpr size_t BUFFER_DIM_Y = 32;
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 32
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t BUFFER_STAGES_NUM = BUFFER_DIM_Y / THREADS_PER_CHUNK_Y;  //  8 =  32 / 4
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                 //  4 = 128 / 32
static_assert(ITERATIONS >= 1);
static_assert(BUFFER_STAGES_NUM % 2 == 0);

template <typename ParamOP, float (*OP)(float, const ParamOP &), float (*REF)(float, const Empty &)>
__device__ constexpr bool is_op() {
  if constexpr (std::is_same<ParamOP, Empty>::value) {
    return OP == REF;
  } else {
    return false;
  }
}

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ float2 splat_x2(const float v) { return make_float2(v, v); }

// __frcp_rn for d >= 1, two lanes at a time: one Newton step from rcp.approx, with the
// correctly rounded scalar path where the approximation flushes (d >= 2^126) or d is NaN.
__device__ __forceinline__ float2 rcp_rn_x2(const float2 d) {
  float2 r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r.x) : "f"(d.x));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r.y) : "f"(d.y));
  r = __ffma2_rn(r, __ffma2_rn(make_float2(-d.x, -d.y), r, splat_x2(1.f)), r);
  constexpr float limit = 0x1p126f;
  if (!(d.x < limit && d.y < limit)) {
    r = make_float2(__frcp_rn(d.x), __frcp_rn(d.y));
  }
  return r;
}

// silu<fp32, fp32> (x / (1 + expf(-x))) as libdevice computes it, two lanes at a time.
__device__ __forceinline__ float2 silu_x2(const float2 x) {
  const float2 t = __ffma2_rn(x, splat_x2(__int_as_float(0xBBBB989D)), splat_x2(0.5f));
  const float2 j = __ffma2_rd(make_float2(__saturatef(t.x), __saturatef(t.y)),
                              splat_x2(__int_as_float(0x437C0000)),
                              splat_x2(__int_as_float(0x4B400001)));
  float2 r = __ffma2_rn(j, splat_x2(-1.f), splat_x2(__int_as_float(0x4B40007F)));
  r = __ffma2_rn(x, splat_x2(__int_as_float(0xBFB8AA3B)), r);
  r = __ffma2_rn(x, splat_x2(__int_as_float(0xB2A57060)), r);
  asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(r.x));
  asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(r.y));
  const float2 two_j = make_float2(__int_as_float(__float_as_int(j.x) << 23),
                                   __int_as_float(__float_as_int(j.y) << 23));
  return __fmul2_rn(x, rcp_rn_x2(__ffma2_rn(r, two_j, splat_x2(1.f))));
}

// sigmoidf(x), then x * s and dsilu = x * s * (1 - s) + s, two lanes at a time.
__device__ __forceinline__ void silu_dsilu_x2(const float2 x, float2 &act, float2 &dact) {
  float2 e = __fmul2_rn(x, splat_x2(__int_as_float(0xBFB8AA3B)));
  asm("ex2.approx.f32 %0, %0;" : "+f"(e.x));
  asm("ex2.approx.f32 %0, %0;" : "+f"(e.y));
  const float2 s = rcp_rn_x2(__fadd2_rn(e, splat_x2(1.f)));
  act = __fmul2_rn(x, s);
  dact = __ffma2_rn(act, __ffma2_rn(s, splat_x2(-1.f), splat_x2(1.f)), s);
}
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &), typename IType, typename OType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    cast_fp8_gated_kernel(const __grid_constant__ CUtensorMap tensor_map_grad,
                          const __grid_constant__ CUtensorMap tensor_map_input_act,
                          const __grid_constant__ CUtensorMap tensor_map_input_gate,
                          const __grid_constant__ CUtensorMap tensor_map_output_act,
                          const __grid_constant__ CUtensorMap tensor_map_output_gate,
                          float *const amax_ptr, float *const scale_inv_ptr,
                          const float *const scale_ptr, const size_t rows, const size_t cols,
                          const ParamOP p) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  const size_t chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const size_t chunk_offset_X = blockIdx.x * CHUNK_DIM_X;

  const size_t tid_Y = threadIdx.x / THREADS_PER_CHUNK_X;
  const size_t tid_X = threadIdx.x % THREADS_PER_CHUNK_X;

  const size_t thread_offset_Y = tid_Y;
  const size_t thread_offset_X = tid_X;

  float amax = 0;
  const float scale = (scale_ptr != nullptr) ? *scale_ptr : 1;

  extern __shared__ char dynamic_shmem[];
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  char *dshmem = align_up(dynamic_shmem, TMA_SHMEM_ALIGNMENT);

  constexpr size_t buff_elems = SHMEM_DIM_Y * SHMEM_DIM_X;
  constexpr size_t buff_elems_total = BUFFERS_NUM * buff_elems;
  constexpr size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(IType), TMA_SHMEM_ALIGNMENT);
  constexpr size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(buff_elems_total * sizeof(OType), TMA_SHMEM_ALIGNMENT);

  constexpr size_t grad_mem = IS_BWD ? buff_size_aligned_in : 0;

  constexpr size_t in_act_mem = buff_size_aligned_in;
  constexpr size_t in_gate_mem = buff_size_aligned_in;
  constexpr size_t in_mem = in_act_mem + in_gate_mem;

  constexpr size_t out_act_mem = buff_size_aligned_out;
  constexpr size_t in_transaction_size = buff_elems * sizeof(IType);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  IType *in_grad_sh = reinterpret_cast<IType *>(dshmem);
  IType *in_act_sh = reinterpret_cast<IType *>(dshmem + grad_mem);
  IType *in_gate_sh = reinterpret_cast<IType *>(dshmem + grad_mem + in_act_mem);
  OType *out_act_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem);
  OType *out_gate_sh = reinterpret_cast<OType *>(dshmem + grad_mem + in_mem + out_act_mem);

  const uint64_t *TMAP_grad_in = reinterpret_cast<const uint64_t *>(&tensor_map_grad);
  const uint64_t *TMAP_in_act = reinterpret_cast<const uint64_t *>(&tensor_map_input_act);
  const uint64_t *TMAP_in_gate = reinterpret_cast<const uint64_t *>(&tensor_map_input_gate);
  const uint64_t *TMAP_output_act = reinterpret_cast<const uint64_t *>(&tensor_map_output_act);
  const uint64_t *TMAP_output_gate = reinterpret_cast<const uint64_t *>(&tensor_map_output_gate);

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  initialize_barriers<ITERATIONS, THREADS_PER_CHUNK>(mbar, is_master_thread);

  int parity = 0;

  // Prefetch data of the first stage

  if constexpr (IS_BWD) {
    copy_2d_to_sharedx3(in_grad_sh, TMAP_grad_in, chunk_offset_X, chunk_offset_Y, in_act_sh,
                        TMAP_in_act, chunk_offset_X, chunk_offset_Y, in_gate_sh, TMAP_in_gate,
                        chunk_offset_X, chunk_offset_Y, in_transaction_size, &mbar[0],
                        is_master_thread);
  } else {
    copy_2d_to_sharedx2(in_act_sh, TMAP_in_act, chunk_offset_X, chunk_offset_Y, in_gate_sh,
                        TMAP_in_gate, chunk_offset_X, chunk_offset_Y, in_transaction_size, &mbar[0],
                        is_master_thread);
  }

#pragma unroll
  for (int it = 0; it < ITERATIONS; ++it) {
    const size_t buff = it % BUFFERS_NUM;
    const size_t next_it = it + 1;
    if (next_it < ITERATIONS) {
      const size_t next_buff = next_it % BUFFERS_NUM;
      const size_t chunk_it_offset_y = chunk_offset_Y + next_it * BUFFER_DIM_Y;
      const size_t chunk_it_offset_x = chunk_offset_X;
      if constexpr (IS_BWD) {
        copy_2d_to_sharedx3(
            &in_grad_sh[next_buff * buff_elems], TMAP_grad_in, chunk_it_offset_x, chunk_it_offset_y,
            &in_act_sh[next_buff * buff_elems], TMAP_in_act, chunk_it_offset_x, chunk_it_offset_y,
            &in_gate_sh[next_buff * buff_elems], TMAP_in_gate, chunk_it_offset_x, chunk_it_offset_y,
            in_transaction_size, &mbar[next_it], is_master_thread);
      } else {
        copy_2d_to_sharedx2(&in_act_sh[next_buff * buff_elems], TMAP_in_act, chunk_it_offset_x,
                            chunk_it_offset_y, &in_gate_sh[next_buff * buff_elems], TMAP_in_gate,
                            chunk_it_offset_x, chunk_it_offset_y, in_transaction_size,
                            &mbar[next_it], is_master_thread);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[it], parity);

    IType *in_grad_sh_curr = in_grad_sh + buff * buff_elems;
    IType *in_act_sh_curr = in_act_sh + buff * buff_elems;
    IType *in_gate_sh_curr = in_gate_sh + buff * buff_elems;
    OType *out_act_sh_curr = out_act_sh + buff * buff_elems;
    OType *out_gate_sh_curr = out_gate_sh + buff * buff_elems;
#pragma unroll
    for (int stage = 0; stage < BUFFER_STAGES_NUM; stage += 2) {
      size_t shmem_idx[2];
      float act_elt[2];
      float gate_elt[2];
      float dgate_elt[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const size_t stage_offset_Y = (stage + i) * THREADS_PER_CHUNK_Y;
        const size_t shmem_offset_y = thread_offset_Y + stage_offset_Y;
        const size_t shmem_offset_x = thread_offset_X;
        shmem_idx[i] = shmem_offset_y * SHMEM_DIM_X + shmem_offset_x;

        act_elt[i] = static_cast<float>(in_act_sh_curr[shmem_idx[i]]);
        gate_elt[i] = static_cast<float>(in_gate_sh_curr[shmem_idx[i]]);
        dgate_elt[i] = 1.0f;  // gating is ideally an identity function
        if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
          if (gate_elt[i] > p.limit || gate_elt[i] < -p.limit) {
            dgate_elt[i] = 0.f;
          }
          gate_elt[i] = min(max(-p.limit, gate_elt[i]), p.limit) + p.glu_linear_offset;
        } else if constexpr (std::is_same<ParamOP, SiTUGLUParam>::value) {
          dgate_elt[i] = dsitu_up<float, float>(gate_elt[i], p);
          gate_elt[i] = situ_up<float, float>(gate_elt[i], p);
        }
      }

      if constexpr (IS_BWD) {
        float2 act_x2;
        float2 dact_x2;
        if constexpr (is_op<ParamOP, ActOP, silu<fp32, fp32>>() &&
                      is_op<ParamOP, DActOP, dsilu<fp32, fp32>>()) {
          silu_dsilu_x2(make_float2(act_elt[0], act_elt[1]), act_x2, dact_x2);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          float grad_elt = static_cast<float>(in_grad_sh_curr[shmem_idx[i]]);

          const float x = act_elt[i];
          float act_x;
          float dact_x;
          if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
            const float x = min(act_elt[i], p.limit);
            const float s = sigmoidf(p.alpha * x);
            act_x = x * s;
            if (act_elt[i] <= p.limit) {
              dact_x = s + s * (1 - s) * p.alpha * x;
            } else {
              dact_x = 0.0f;
            }
          } else if constexpr (std::is_same<ParamOP, SiTUGLUParam>::value) {
            act_x = ActOP(x, p);
            dact_x = DActOP(x, p);
          } else {
            if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
              act_x = i == 0 ? act_x2.x : act_x2.y;
              dact_x = i == 0 ? dact_x2.x : dact_x2.y;
            } else {
              act_x = ActOP(x, p);
              dact_x = DActOP(x, p);
            }
          }
          float after_dact = dact_x * grad_elt * gate_elt[i];
          float after_dgate = dgate_elt[i] * act_x * grad_elt;

          out_act_sh_curr[shmem_idx[i]] = static_cast<OType>(scale * after_dact);
          out_gate_sh_curr[shmem_idx[i]] = static_cast<OType>(scale * after_dgate);

          amax = fmaxf(amax, fabsf(after_dact));
          amax = fmaxf(amax, fabsf(after_dgate));
        }
      } else {
        float act_x[2];
        if constexpr (is_op<ParamOP, ActOP, silu<fp32, fp32>>()) {
          const float2 silu_x = silu_x2(make_float2(act_elt[0], act_elt[1]));
          act_x[0] = silu_x.x;
          act_x[1] = silu_x.y;
        } else {
          act_x[0] = ActOP(act_elt[0], p);
          act_x[1] = ActOP(act_elt[1], p);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const float after_act = act_x[i] * gate_elt[i];
          out_act_sh_curr[shmem_idx[i]] = static_cast<OType>(scale * after_act);
          amax = fmaxf(amax, fabsf(after_act));
        }
      }
    }

    // Wait for shared memory writes to be visible to TMA engine (cross-proxy fence)
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const size_t chunk_it_offset_y = chunk_offset_Y + it * BUFFER_DIM_Y;
      const size_t chunk_it_offset_x = chunk_offset_X;

      // dGeLU
      ptx::cp_async_bulk_tensor_2d_shared_to_global(TMAP_output_act, chunk_it_offset_x,
                                                    chunk_it_offset_y,
                                                    reinterpret_cast<uint64_t *>(out_act_sh_curr));

      if constexpr (IS_BWD) {
        // dGate
        ptx::cp_async_bulk_tensor_2d_shared_to_global(
            TMAP_output_gate, chunk_it_offset_x, chunk_it_offset_y,
            reinterpret_cast<uint64_t *>(out_gate_sh_curr));
      }

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<BUFFERS_NUM - 1>();
    }
    // The bulk async-group is owned by the issuing thread. Hand its completion
    // off to all cooperative writers before the output ring can be reused.
    if (next_it < ITERATIONS && next_it >= BUFFERS_NUM) {
      __syncthreads();
    }
  }
  if (is_master_thread) {
    ptx::cp_async_bulk_wait_group();
  }
  __syncthreads();

  if (amax_ptr != nullptr) {
    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    // Reduce the amax over the block
    amax = reduce_max<THREADS_PER_CHUNK / THREADS_PER_WARP>(amax, warp_id);
    // Update the global amax
    if (is_master_thread) {
      atomicMaxFloat(amax_ptr, amax);
    }
  }

  // Update scale-inverse
  if (is_master_thread && blockIdx.x == 0 && (scale_inv_ptr != nullptr)) {
    reciprocal<float>(scale_inv_ptr, scale);
  }

  // Destroy the barriers. This invalidates the memory region of the barrier.
  // If further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int it = 0; it < ITERATIONS; ++it) {
      ptx::mbarrier_invalid(&mbar[it]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}
}  // namespace kernel

// Dynamic shared memory requested by cast_fp8_gated_kernel. Must match the
// buffer layout inside the kernel.
inline size_t cast_gated_tma_dynamic_shmem_size(const bool is_bwd, const DType itype,
                                                const DType otype) {
  using namespace kernel;
  const size_t buff_elems_total = BUFFERS_NUM * SHMEM_DIM_Y * SHMEM_DIM_X;
  const size_t buff_size_aligned_in =
      DIVUP_TO_MULTIPLE(buff_elems_total * typeToNumBits(itype) / 8, TMA_SHMEM_ALIGNMENT);
  const size_t buff_size_aligned_out =
      DIVUP_TO_MULTIPLE(buff_elems_total * typeToNumBits(otype) / 8, TMA_SHMEM_ALIGNMENT);
  const size_t grad_mem = (is_bwd ? buff_size_aligned_in : 0);
  const size_t in_act_mem = buff_size_aligned_in;
  const size_t in_gate_mem = buff_size_aligned_in;
  const size_t out_act_mem = buff_size_aligned_out;
  const size_t out_gate_mem = buff_size_aligned_out;
  return grad_mem + (in_act_mem + in_gate_mem) + (out_act_mem + out_gate_mem) + TMA_SHMEM_ALIGNMENT;
}

// Whether cast_fp8_gated_kernel fits in the shared memory of the current device.
// Devices with compute capability 10.0+ differ in shared memory per block
// (e.g. SM 12.0 has much less than SM 10.0), so FP32 configurations may not fit.
// The static shared memory is read from the compiled kernel so that __shared__
// arrays in the device functions it calls (e.g. reduce_max) are included.
template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
bool cast_gated_tma_fits_device(const DType itype, const DType otype) {
  using namespace kernel;
  size_t static_shmem_size = 0;
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      itype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          otype, OType,
          static_shmem_size = cuda::static_shared_memory_size(reinterpret_cast<const void *>(
              &cast_fp8_gated_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType>));););
  const size_t required =
      cast_gated_tma_dynamic_shmem_size(IS_BWD, itype, otype) + static_shmem_size;
  return required <= cuda::max_shared_memory_per_block_optin();
}

template <bool IS_BWD, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_gated_tma(const Tensor &gated_input, const Tensor &grad, Tensor *output, ParamOP &p,
                    cudaStream_t stream) {
  using namespace kernel;
  checkCuDriverContext(stream);

  NVTE_CHECK(!output->has_columnwise_data(), "Only rowwise cast supported in this function.");
  const size_t rows = gated_input.flat_first_dim();
  const size_t cols = gated_input.flat_last_dim() / 2;
  const size_t output_cols = (IS_BWD ? 2 : 1) * cols;

  const size_t blocks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t blocks_X = DIVUP(cols, CHUNK_DIM_X);

  float *const amax_ptr = reinterpret_cast<float *>(output->amax.dptr);
  float *const scale_inv_ptr = reinterpret_cast<float *>(output->scale_inv.dptr);
  float *const scale_ptr = reinterpret_cast<float *>(output->scale.dptr);

  const dim3 block_dim(THREADS_PER_CHUNK);
  const dim3 grid_dim(blocks_X, blocks_Y);

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      gated_input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OType,

          alignas(64) CUtensorMap tensor_map_grad{};
          alignas(64) CUtensorMap tensor_map_input_act{};
          alignas(64) CUtensorMap tensor_map_input_gate{};
          alignas(64) CUtensorMap tensor_map_output_act{};
          alignas(64) CUtensorMap tensor_map_output_gate{};

          if constexpr (IS_BWD) {
            create_2D_tensor_map(tensor_map_grad, grad.data, rows, cols, SHMEM_DIM_Y, SHMEM_DIM_X,
                                 cols, 0, typeToNumBits(gated_input.dtype()));
          }

          const uint32_t tensor_stride_elems = output_cols;

          create_2D_tensor_map(tensor_map_input_act, gated_input.data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, cols * 2, 0, typeToNumBits(gated_input.dtype()));
          create_2D_tensor_map(tensor_map_input_gate, gated_input.data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, cols * 2, cols, typeToNumBits(gated_input.dtype()));
          create_2D_tensor_map(tensor_map_output_act, output->data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, tensor_stride_elems, 0, typeToNumBits(output->dtype()));
          create_2D_tensor_map(tensor_map_output_gate, output->data, rows, cols, SHMEM_DIM_Y,
                               SHMEM_DIM_X, tensor_stride_elems, cols,
                               typeToNumBits(output->dtype()));

          const size_t shmem_size =
              cast_gated_tma_dynamic_shmem_size(IS_BWD, gated_input.dtype(), output->dtype());

          auto kernel = cast_fp8_gated_kernel<IS_BWD, ParamOP, ActOP, DActOP, IType, OType>;
          NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               shmem_size));

          kernel<<<grid_dim, block_dim, shmem_size, stream>>>(
              tensor_map_grad, tensor_map_input_act, tensor_map_input_gate, tensor_map_output_act,
              tensor_map_output_gate, amax_ptr, scale_inv_ptr, scale_ptr, rows, cols, p);
          NVTE_CHECK_CUDA(cudaGetLastError()););  // NOLINT(*)
  );                                              // NOLINT(*)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void cast_gated_fwd(const Tensor &input, Tensor *output, ParamOP &p, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OType,

          constexpr int nvec = 32 / sizeof(IType);
          GatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP>(
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<OType *>(output->data.dptr),
              reinterpret_cast<const fp32 *>(output->scale.dptr),
              reinterpret_cast<fp32 *>(output->amax.dptr),
              reinterpret_cast<fp32 *>(output->scale_inv.dptr), input.flat_first_dim(),
              output->flat_last_dim(), p, stream););  // NOLINT(*)
  );                                                  // NOLINT(*)
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void cast_gated_bwd(const Tensor &input, const Tensor &grad, Tensor *output, ParamOP &p,
                    cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OType,

          constexpr int nvec = 32 / sizeof(IType);
          DGatedActivationKernelLauncher<nvec, fp32, ParamOP, ActOP, DActOP>(
              reinterpret_cast<const IType *>(grad.data.dptr),
              reinterpret_cast<const IType *>(input.data.dptr),
              reinterpret_cast<OType *>(output->data.dptr),
              reinterpret_cast<const fp32 *>(output->scale.dptr),
              reinterpret_cast<fp32 *>(output->amax.dptr),
              reinterpret_cast<fp32 *>(output->scale_inv.dptr), grad.flat_first_dim(),
              grad.flat_last_dim(), p, stream););  // NOLINT(*)
  );                                               // NOLINT(*)
}
}  // namespace fp8
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_GATED_FP8_CUH_
