// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/fp8/linear.h"

#include "orttraining/training_ops/cuda/fp8/gemm/gemm.h"
#include "orttraining/training_ops/cuda/fp8/transpose/transpose.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(FP8Linear, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<MLFloat16>()),
                        fp8::FP8Linear);

ONNX_OPERATOR_KERNEL_EX(FP8LinearGrad, kMSDomain, 1, kCudaExecutionProvider,
                        (*KernelDefBuilder::Create()).TypeConstraint("T", BuildKernelDefConstraints<MLFloat16>()),
                        fp8::FP8LinearGrad);

namespace fp8 {

namespace {

template <typename T>
struct DispatchFP8LinearImpl {
  Status operator()(const FP8Linear* fp8_linear, OpKernelContext* context, const Tensor* input, const Tensor* weight,
                    const Tensor* bias, Tensor* output, Tensor* trans_input, Tensor* trans_weight,
                    Tensor* scale_inv_fwd, float* scale, float* scale_inv, float* amax_history) {
    // inputmat, inputmat_t = fp8_cast_transpose_fused(input, scaling_fw)
    // weight_fp8, weight_t_fp8 = fp8_cast_transpose_fused(weight, scaling_fw)
    // out = fp8_gemm(weight_fp8, scale_inv_fw, inputmat, scale_inv_fw)
    // Op outputs: out(fp16), inputmat_t(fp8),  weight_t_fp8(fp8), scale_inv_fw.clone()
    cudaStream_t stream = fp8_linear->Stream(context);
    const T* input_data = input->Data<T>();
    const T* weight_data = weight->Data<T>();
    const T* bias_data = bias ? bias->Data<T>() : nullptr;
    T* output_data = output->MutableData<T>();
    Float8E4M3FN* trans_input_data = trans_input->MutableData<Float8E4M3FN>();
    Float8E4M3FN* trans_weight_data = trans_weight->MutableData<Float8E4M3FN>();
    float* scale_inv_fwd_data = scale_inv_fwd->MutableData<float>();
    const TensorShape& input_shape = input->Shape();
    size_t input_rank = input_shape.NumDimensions();
    size_t m = static_cast<size_t>(input_shape.SizeToDimension(input_rank - 1));
    size_t k = static_cast<size_t>(input_shape[input_rank - 1]);
    const TensorShape& weight_shape = weight->Shape();
    size_t n = static_cast<size_t>(weight_shape[0]);
    ORT_ENFORCE(static_cast<size_t>(weight_shape[1]) == k);
    if (bias_data) {
      ORT_ENFORCE(static_cast<size_t>(bias->Shape().Size()) == n);
    }
    IAllocatorUniquePtr<Float8E4M3FN> fp8_input_data =
        fp8_linear->GetScratchBuffer<Float8E4M3FN>(m * k, context->GetComputeStream());
    IAllocatorUniquePtr<Float8E4M3FN> fp8_weight_data =
        fp8_linear->GetScratchBuffer<Float8E4M3FN>(k * n, context->GetComputeStream());
    IAllocatorUniquePtr<float> single_fp32 = fp8_linear->GetScratchBuffer<float>(1, context->GetComputeStream());
    CastTranspose(stream, input_data, fp8_input_data.get(), trans_input_data, scale, amax_history, k, m);
    CastTranspose(stream, weight_data, fp8_weight_data.get(), trans_weight_data, scale + 1, amax_history + 1, k, n);
    WrappedTensor weight_tensor(fp8_weight_data.get(), {n, k}, MappedType<Float8E4M3FN>::dtype, nullptr, nullptr,
                                scale_inv + 1);
    WrappedTensor input_tensor(fp8_input_data.get(), {m, k}, MappedType<Float8E4M3FN>::dtype, nullptr, nullptr,
                               scale_inv);
    WrappedTensor bias_tensor;
    if (bias_data) {
      bias_tensor.data_ptr = const_cast<void*>(reinterpret_cast<const void*>(bias_data));
      bias_tensor.shape = {n};
      bias_tensor.dtype = MappedType<T>::dtype;
    }
    WrappedTensor output_tensor(output_data, {m, n}, MappedType<T>::dtype, single_fp32.get(), single_fp32.get());
    WrappedTensor pre_gelu_tensor;
    ORT_RETURN_IF_ERROR(FP8Gemm(stream, weight_tensor, input_tensor, bias_tensor, output_tensor, pre_gelu_tensor, true,
                                false, false, false, false, 0));
    CUDA_RETURN_IF_ERROR(
        cudaMemcpyAsync(scale_inv_fwd_data, scale_inv, sizeof(float) * kFwdCount, cudaMemcpyDeviceToDevice, stream));
    return Status::OK();
  }
};

template <typename T>
struct DispatchFP8LinearGradImpl {
  Status operator()(const FP8LinearGrad* fp8_linear_grad, OpKernelContext* context, const Tensor* grad_output,
                    const Tensor* trans_input, const Tensor* trans_weight, const Tensor* scale_inv_fwd,
                    Tensor* grad_input, Tensor* grad_weight, Tensor* grad_bias, float* scale, float* scale_inv,
                    float* amax_history) {
    // grad_bias, grad_output_c, grad_output_t = fp8_cast_transpose_bgrad_fused(grad_output_mat, scaling_bw)
    //   or
    // grad_output_c, grad_output_t = fp8_cast_transpose_fused(grad_output_mat, scaling_bw)
    // dgrad = fp8_gemm(weight_t_fp8, scale_inv_fw, grad_output_c, scale_inv_bw)
    // wgrad = fp8_gemm(inputmat_t, scale_inv_fw, grad_output_t, scale_inv_bw)
    cudaStream_t stream = fp8_linear_grad->Stream(context);
    const T* grad_output_data = grad_output->Data<T>();
    const Float8E4M3FN* trans_input_data = trans_input->Data<Float8E4M3FN>();
    const Float8E4M3FN* trans_weight_data = trans_weight->Data<Float8E4M3FN>();
    float* scale_inv_fwd_data = const_cast<float*>(scale_inv_fwd->Data<float>());
    T* grad_input_data = grad_input ? grad_input->MutableData<T>() : nullptr;
    T* grad_weight_data = grad_weight ? grad_weight->MutableData<T>() : nullptr;
    T* grad_bias_data = grad_bias ? grad_bias->MutableData<T>() : nullptr;
    const TensorShape& grad_output_shape = grad_output->Shape();
    size_t grad_output_rank = grad_output_shape.NumDimensions();
    size_t m = static_cast<size_t>(grad_output_shape.SizeToDimension(grad_output_rank - 1));
    size_t n = static_cast<size_t>(grad_output_shape[grad_output_rank - 1]);
    const TensorShape& trans_weight_shape = trans_weight->Shape();
    size_t k = static_cast<size_t>(trans_weight_shape[0]);
    ORT_ENFORCE(static_cast<size_t>(trans_weight_shape[1]) == n);
    IAllocatorUniquePtr<Float8E5M2> fp8_grad_output_data =
        fp8_linear_grad->GetScratchBuffer<Float8E5M2>(m * n, context->GetComputeStream());
    IAllocatorUniquePtr<Float8E5M2> trans_grad_output_data =
        fp8_linear_grad->GetScratchBuffer<Float8E5M2>(m * n, context->GetComputeStream());
    IAllocatorUniquePtr<float> single_fp32 = fp8_linear_grad->GetScratchBuffer<float>(1, context->GetComputeStream());
    if (grad_bias_data) {
      ORT_ENFORCE(static_cast<size_t>(grad_bias->Shape().Size()) == n);
      size_t workspace_size = GetCastTransposeBiasWorkspaceSize<Float8E5M2>(n, m);
      IAllocatorUniquePtr<float> workspace =
          fp8_linear_grad->GetScratchBuffer<float>(workspace_size * sizeof(float), context->GetComputeStream());
      CastTransposeBias(stream, grad_output_data, fp8_grad_output_data.get(), trans_grad_output_data.get(),
                        grad_bias_data, workspace.get(), scale, amax_history, n, m);
    } else {
      CastTranspose(stream, grad_output_data, fp8_grad_output_data.get(), trans_grad_output_data.get(), scale,
                    amax_history, n, m);
    }
    if (grad_input_data) {
      WrappedTensor trans_weight_tensor(const_cast<void*>(reinterpret_cast<const void*>(trans_weight_data)), {k, n},
                                        MappedType<Float8E4M3FN>::dtype, nullptr, nullptr, scale_inv_fwd_data + 1);
      WrappedTensor grad_output_tensor(fp8_grad_output_data.get(), {m, n}, MappedType<Float8E5M2>::dtype, nullptr,
                                       nullptr, scale_inv);
      WrappedTensor bias_tensor;
      WrappedTensor grad_input_tensor(grad_input_data, {m, k}, MappedType<T>::dtype, single_fp32.get(),
                                      single_fp32.get());
      WrappedTensor pre_gelu_tensor;
      ORT_RETURN_IF_ERROR(FP8Gemm(stream, trans_weight_tensor, grad_output_tensor, bias_tensor, grad_input_tensor,
                                  pre_gelu_tensor, true, false, false, false, true, 0));
    }
    if (grad_weight_data) {
      WrappedTensor trans_input_tensor(const_cast<void*>(reinterpret_cast<const void*>(trans_input_data)), {k, m},
                                       MappedType<Float8E4M3FN>::dtype, nullptr, nullptr, scale_inv_fwd_data);
      WrappedTensor trans_grad_output_tensor(trans_grad_output_data.get(), {n, m}, MappedType<Float8E5M2>::dtype,
                                             nullptr, nullptr, scale_inv);
      WrappedTensor bias_tensor;
      WrappedTensor grad_weight_tensor(grad_weight_data, {n, k}, MappedType<T>::dtype, single_fp32.get(),
                                       single_fp32.get());
      WrappedTensor pre_gelu_tensor;
      ORT_RETURN_IF_ERROR(FP8Gemm(stream, trans_input_tensor, trans_grad_output_tensor, bias_tensor, grad_weight_tensor,
                                  pre_gelu_tensor, true, false, false, false, true, 0));
    }
    return Status::OK();
  }
};

}  // namespace

Status FP8Linear::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weight = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const TensorShape& input_shape = input->Shape();
  const TensorShape& weight_shape = weight->Shape();
  size_t input_rank = input_shape.NumDimensions();

  TensorShapeVector output_shape_vec = input_shape.AsShapeVector();
  output_shape_vec[input_rank - 1] = weight_shape[0];
  TensorShapeVector input_t_shape_vec = TensorShapeVector(2);
  input_t_shape_vec[0] = input_shape[input_rank - 1];
  input_t_shape_vec[1] = input_shape.SizeToDimension(input_rank - 1);
  TensorShapeVector weight_t_shape_vec = TensorShapeVector(2);
  weight_t_shape_vec[0] = weight_shape[1];
  weight_t_shape_vec[1] = weight_shape[0];
  TensorShapeVector scale_inv_fwd_shape_vec = TensorShapeVector(1);
  scale_inv_fwd_shape_vec[0] = static_cast<int64_t>(kFwdCount);
  Tensor* output = context->Output(0, TensorShape(output_shape_vec));
  Tensor* trans_input = context->Output(1, TensorShape(input_t_shape_vec));
  Tensor* trans_weight = context->Output(2, TensorShape(weight_t_shape_vec));
  Tensor* scale_inv_fwd = context->Output(3, TensorShape(scale_inv_fwd_shape_vec));

  if (update_scaling) {
    ORT_RETURN_IF_ERROR(scaling.Update(this, context));
  }
  update_scaling = true;

  utils::MLTypeCallDispatcher<MLFloat16> t_disp(input->GetElementType());
  return t_disp.InvokeRet<Status, DispatchFP8LinearImpl>(this, context, input, weight, bias, output, trans_input,
                                                         trans_weight, scale_inv_fwd, scaling.Scale(),
                                                         scaling.ScaleInv(), scaling.AmaxHistory());
}

Status FP8LinearGrad::ComputeInternal(OpKernelContext* context) const {
  const Tensor* grad_output = context->Input<Tensor>(0);
  const Tensor* trans_input = context->Input<Tensor>(1);
  const Tensor* trans_weight = context->Input<Tensor>(2);
  const Tensor* scale_inv_fwd = context->Input<Tensor>(3);
  const TensorShape& grad_output_shape = grad_output->Shape();
  const TensorShape& trans_weight_shape = trans_weight->Shape();
  size_t grad_output_rank = grad_output_shape.NumDimensions();

  TensorShapeVector grad_input_shape_vec = grad_output_shape.AsShapeVector();
  grad_input_shape_vec[grad_output_rank - 1] = trans_weight_shape[0];
  TensorShapeVector grad_weight_shape_vec = TensorShapeVector(2);
  grad_weight_shape_vec[0] = trans_weight_shape[1];
  grad_weight_shape_vec[1] = trans_weight_shape[0];
  TensorShapeVector grad_bias_shape_vec = TensorShapeVector(1);
  grad_bias_shape_vec[0] = trans_weight_shape[1];
  Tensor* grad_input = context->Output(0, TensorShape(grad_input_shape_vec));
  Tensor* grad_weight = context->Output(1, TensorShape(grad_weight_shape_vec));
  Tensor* grad_bias = context->Output(2, TensorShape(grad_bias_shape_vec));

  ORT_RETURN_IF_ERROR(scaling.Update(this, context));

  utils::MLTypeCallDispatcher<MLFloat16> t_disp(grad_output->GetElementType());
  return t_disp.InvokeRet<Status, DispatchFP8LinearGradImpl>(
      this, context, grad_output, trans_input, trans_weight, scale_inv_fwd, grad_input, grad_weight, grad_bias,
      scaling.Scale(), scaling.ScaleInv(), scaling.AmaxHistory());
}

}  // namespace fp8
}  // namespace cuda
}  // namespace onnxruntime
