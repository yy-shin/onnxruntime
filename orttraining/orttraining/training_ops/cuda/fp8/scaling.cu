// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/fp8/scaling.h"

#include "core/providers/cuda/cu_inc/elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

template <bool is_fwd>
struct OpDefaultScalingFactor {
  OpDefaultScalingFactor(const fp32* amax, const fp32* scale) : amax_(amax), scale_(scale) {}

  __device__ __inline__ fp32 operator()(size_t idx) const {
    // fp8_max_fwd = 448, fp8_max_bwd = 57344, margin = 0
    // sf = (fp8_max / amax) / (2 ** margin)
    // sf = torch.where(amax > 0.0, sf, scale)
    // sf = torch.where(torch.isfinite(amax), sf, scale)
    // return sf
    float sf = (is_fwd ? 448.0f : 57344.0f) / amax_[idx];
    if (amax_[idx] <= 0.0f) sf = scale_[idx];
    return isfinite(amax_[idx]) ? sf : scale_[idx];
  }

  const fp32* amax_;
  const fp32* scale_;
};

struct OpScalingFactorInverse {
  OpScalingFactorInverse(const fp32* scale) : scale_(scale) {}

  __device__ __inline__ fp32 operator()(size_t idx) const { return 1.0f / scale_[idx]; }

  const fp32* scale_;
};

void ComputeDefaultScalingFactor(cudaStream_t stream, const fp32* amax, const fp32* scale_in, fp32* scale_out,
                                 fp32* scale_inv, size_t count, bool is_fwd) {
  if (is_fwd) {
    OpDefaultScalingFactor<true> op_scale(amax, scale_in);
    LaunchElementwiseKernel<fp32, OpDefaultScalingFactor<true>, size_t>(stream, scale_out, op_scale, count);
  } else {
    OpDefaultScalingFactor<false> op_scale(amax, scale_in);
    LaunchElementwiseKernel<fp32, OpDefaultScalingFactor<false>, size_t>(stream, scale_out, op_scale, count);
  }
  OpScalingFactorInverse op_scale_inv(scale_out);
  LaunchElementwiseKernel<fp32, OpScalingFactorInverse, size_t>(stream, scale_inv, op_scale_inv, count);
}

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
