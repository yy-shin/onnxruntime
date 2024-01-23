// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "orttraining/training_ops/cuda/fp8/common.h"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

struct WrappedTensor {
  void* data_ptr;
  std::vector<size_t> shape;
  DType dtype;
  float* amax_ptr;
  float* scale_ptr;
  float* scale_inv_ptr;

  WrappedTensor(void* data_ptr, const std::vector<size_t>& shape, DType dtype, float* amax_ptr = nullptr,
                float* scale_ptr = nullptr, float* scale_inv_ptr = nullptr)
      : data_ptr(data_ptr),
        shape(shape),
        dtype(dtype),
        amax_ptr(amax_ptr),
        scale_ptr(scale_ptr),
        scale_inv_ptr(scale_inv_ptr) {}
  WrappedTensor() : WrappedTensor(nullptr, {}, DType::kFloat32) {}
};

class FP8GemmWorkspace {
 public:
  static FP8GemmWorkspace& Instance() {
    static FP8GemmWorkspace instance;
    return instance;
  }

  size_t SizeInBytes() const { return workspace_size_bytes; }

  void* GetWorkspace() const { return workspace; }

 private:
  FP8GemmWorkspace();
  ~FP8GemmWorkspace();

  static constexpr size_t workspace_size_bytes = 33554432;
  void* workspace = nullptr;
};

Status FP8Gemm(cudaStream_t stream, const WrappedTensor input_a, const WrappedTensor input_b,
               const WrappedTensor input_bias, WrappedTensor output_d, WrappedTensor pre_gelu_out, bool trans_a,
               bool trans_b, bool grad, bool accumulate, bool use_split_accumulator, int math_sm_count);

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
