// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/fp8/scaling.h"

#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/reduction/reduction_ops.h"

namespace onnxruntime {
namespace cuda {
namespace fp8 {

Scaling::Scaling(size_t count, size_t length, bool is_fwd) : count(count), length(length), is_fwd(is_fwd) {
  // TODO: need to reset data to 0 or 1.
  CUDA_CALL_THROW(cudaMalloc(&scale, sizeof(fp32) * count * 2));
  std::vector<fp32> cpu_scale(count * 2, 1.0f);
  CUDA_CALL_THROW(cudaMemcpy(scale, cpu_scale.data(), sizeof(fp32) * count * 2, cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMalloc(&scale_inv, sizeof(fp32) * count));
  std::vector<fp32> cpu_scale_inv(count, 1.0f);
  CUDA_CALL_THROW(cudaMemcpy(scale_inv, cpu_scale_inv.data(), sizeof(fp32) * count, cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMalloc(&amax, sizeof(fp32) * count));
  CUDA_CALL_THROW(cudaMalloc(&amax_history, sizeof(fp32) * length * count * 2));
  CUDA_CALL_THROW(cudaMemset(amax_history, 0, sizeof(fp32) * length * count * 2));
}

Scaling::~Scaling() {
  CUDA_CALL_THROW(cudaFree(scale));
  CUDA_CALL_THROW(cudaFree(scale_inv));
  CUDA_CALL_THROW(cudaFree(amax));
  CUDA_CALL_THROW(cudaFree(amax_history));
}

Status Scaling::Update(const CudaKernel* kernel, OpKernelContext* context) {
  // Reduce to amax.
  cudaStream_t stream = kernel->Stream(context);
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(amax, 0, sizeof(fp32) * count, stream));
  size_t workspace_bytes = 0;
  size_t indices_bytes = 0;
  CudnnTensor input_tensor;
  CudnnTensor output_tensor;
  CudnnReduceDescriptor reduce_desc;
  cudnnDataType_t cudnn_type_X = CUDNN_DATA_FLOAT;
  ORT_RETURN_IF_ERROR(reduce_desc.Set(CUDNN_REDUCE_TENSOR_MAX, cudnn_type_X, CUDNN_REDUCE_TENSOR_NO_INDICES));
  ORT_RETURN_IF_ERROR(input_tensor.Set(
      std::vector<int64_t>{1, static_cast<int64_t>(length), static_cast<int64_t>(count)}, cudnn_type_X));
  ORT_RETURN_IF_ERROR(output_tensor.Set(std::vector<int64_t>{1, 1, static_cast<int64_t>(count)}, cudnn_type_X));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionWorkspaceSize(kernel->GetCudnnHandle(context), reduce_desc, input_tensor,
                                                       output_tensor, &workspace_bytes));
  CUDNN_RETURN_IF_ERROR(cudnnGetReductionIndicesSize(kernel->GetCudnnHandle(context), reduce_desc, input_tensor,
                                                     output_tensor, &indices_bytes));
  IAllocatorUniquePtr<void> workspace_cuda =
      workspace_bytes == 0 ? nullptr : kernel->GetScratchBuffer<void>(workspace_bytes, context->GetComputeStream());
  IAllocatorUniquePtr<void> indices_cuda =
      indices_bytes == 0 ? nullptr : kernel->GetScratchBuffer<void>(indices_bytes, context->GetComputeStream());
  const auto one = Consts<float>::One;
  const auto zero = Consts<float>::Zero;
  CUDNN_RETURN_IF_ERROR(cudnnReduceTensor(kernel->GetCudnnHandle(context), reduce_desc, indices_cuda.get(),
                                          indices_bytes, workspace_cuda.get(), workspace_bytes, &one, input_tensor,
                                          AmaxHistory(), &zero, output_tensor, amax));
  // Roll amax_history.
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(NextAmaxHistory(), AmaxHistory() + count, sizeof(fp32) * (length - 1) * count,
                                       cudaMemcpyDeviceToDevice, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(NextAmaxHistory() + (length - 1) * count, AmaxHistory(), sizeof(fp32) * count,
                                       cudaMemcpyDeviceToDevice, stream));
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(NextAmaxHistory(), 0, sizeof(fp32) * count, stream));
  // Compute scale and scale_inv.
  ComputeDefaultScalingFactor(stream, amax, Scale(), NextScale(), ScaleInv(), count, is_fwd);
  // Update amax_history_idx.
  amax_history_idx = 1 - amax_history_idx;
  return Status::OK();
}

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
