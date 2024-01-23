// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/training_ops/cuda/fp8/scaling.h"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

constexpr size_t kFwdCount = 3;
constexpr size_t kBwdCount = 2;
constexpr size_t kLength = 1024;

class FP8Linear : public CudaKernel {
 public:
  FP8Linear(const OpKernelInfo& info) : CudaKernel(info), scaling(kFwdCount, kLength) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  // amax[1024][3](0), scale[3](1), scale_inv[3](1)
  mutable Scaling scaling;
  mutable bool update_scaling = false;
};

class FP8LinearGrad : public CudaKernel {
 public:
  FP8LinearGrad(const OpKernelInfo& info) : CudaKernel(info), scaling(kBwdCount, kLength, false) {}
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  // amax[1024][2](0), scale[2](1), scale_inv[2](1)
  mutable Scaling scaling;
};

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
