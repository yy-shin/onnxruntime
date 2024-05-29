// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include <vector>
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

#include "QnnOpDef.h"
#include "QnnTypes.h"

namespace onnxruntime {
namespace qnn {

class MatMulOpBuilder : public BaseOpBuilder {
 public:
  MatMulOpBuilder() : BaseOpBuilder("MatMulOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation = false) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

// Move to qnn_utils if it's re-usable
static Status InsertConvertOp(QnnModelWrapper& qnn_model_wrapper,
                              const std::string& convert_input_name,
                              const std::string& convert_output_name,
                              Qnn_DataType_t input_qnn_data_type,
                              Qnn_DataType_t output_qnn_data_type,
                              int32_t input_offset,
                              float input_scale,
                              const std::vector<uint32_t>& output_shape,
                              bool do_op_validation) {
  // Assume input is already handled.
  float qmin = 0.0f;
  float qmax = 255.0f;
  ORT_RETURN_IF_ERROR(qnn::utils::GetQminQmax(input_qnn_data_type, qmin, qmax));
  double value_min = qnn::utils::Dequantize(input_offset, input_scale, qmin);
  double value_max = qnn::utils::Dequantize(input_offset, input_scale, qmax);
  float scale = 0.0f;
  int32_t offset = 0;
  ORT_RETURN_IF_ERROR(qnn::utils::GetQuantParams(static_cast<float>(value_min),
                                                 static_cast<float>(value_max),
                                                 output_qnn_data_type,
                                                 scale,
                                                 offset));

  std::vector<uint32_t> output_shape_copy = output_shape;
  QnnTensorWrapper convert_output_tensorwrapper(convert_output_name,
                                                QNN_TENSOR_TYPE_NATIVE,
                                                output_qnn_data_type,
                                                QnnQuantParamsWrapper(scale, offset),
                                                std::move(output_shape_copy));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(convert_output_tensorwrapper)), "Failed to add tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(convert_output_name,
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    "Convert",
                                                    {convert_input_name},
                                                    {convert_output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to add node.");
  return Status::OK();
}

Status MatMulOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  TensorInfo input0_info = {};
  TensorInfo input1_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input0_info));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input1_info));

  const bool use_fully_connected = !input0_info.is_initializer &&
                                   input0_info.shape.size() >= 2 &&
                                   input1_info.is_initializer &&
                                   input1_info.shape.size() >= 2;

  if (use_fully_connected) {
    // Process inputs for QNN's FullyConnected op, which may be faster than QNN's MatMul op.

    // Input 0 - dynamic input, may need to reshape to 2D.
    const std::string& input0_name = inputs[0].node_arg.Name();
    const bool input0_needs_reshape = input0_info.shape.size() != 2;
    const std::string fc_input0_name = !input0_needs_reshape ? input0_name
                                                             : input0_name + "_ort_qnn_ep_reshape";
    input_names.push_back(fc_input0_name);

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(fc_input0_name)) {
      ORT_RETURN_IF(input0_info.quant_param.IsPerChannel(),
                    "Dynamic MatMul inputs only support per-tensor quantization");

      std::vector<uint32_t> shape(2);
      if (input0_needs_reshape) {
        const size_t input0_original_rank = input0_info.shape.size();

        if (input0_original_rank < 2) {
          shape[0] = 1;
          shape[1] = input0_info.shape[0];
        } else {
          shape[0] = input0_info.shape[0];
          shape[1] = input0_info.shape[1];
        }
        if (input0_original_rank > 2) {
          shape[0] = 1;
          for (size_t i = 0; i < input0_original_rank - 1; i++) {
            shape[0] *= input0_info.shape[i];
          }
          shape[1] = input0_info.shape[input0_original_rank - 1];
        }

        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input0_name,
                                                             fc_input0_name,
                                                             input0_info.shape,
                                                             shape,
                                                             input0_info.qnn_data_type,
                                                             input0_info.quant_param,
                                                             do_op_validation,
                                                             qnn_model_wrapper.IsGraphInput(input0_name)));
      }

      Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(fc_input0_name);
      QnnTensorWrapper input_tensorwrapper(fc_input0_name, tensor_type, input0_info.qnn_data_type,
                                           std::move(input0_info.quant_param), std::move(shape));

      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    }

    // Input 1: weight, will need to transpose, may need to reshape to 2D.
    const std::string& input1_name = inputs[1].node_arg.Name();
    const size_t input1_original_rank = input1_info.shape.size();
    const bool input1_needs_reshape = input1_info.shape.size() != 2;
    input_names.push_back(input1_name);

    ORT_RETURN_IF_NOT(input1_original_rank >= 2);  // TODO: Support shape.size() == 1
    if (input1_info.quant_param.IsPerChannel()) {
      int32_t axis = 0;
      ORT_RETURN_IF_ERROR(input1_info.quant_param.GetAxis(axis));
      ORT_RETURN_IF_NOT(axis == input1_original_rank - 1);  // TODO: fix for 1D input
    }

    // Create perm to transpose the last two dimensions.
    std::vector<size_t> perm(input1_original_rank);
    for (size_t i = 0; i < input1_original_rank; i++) {
      perm[i] = i;
    }
    perm[input1_original_rank - 1] = input1_original_rank - 2;
    perm[input1_original_rank - 2] = input1_original_rank - 1;

    std::vector<uint8_t> initializer_data;
    ORT_RETURN_IF_ERROR(TransposeInitializer(qnn_model_wrapper, *input1_info.initializer_tensor, perm,
                                             initializer_data));
    std::vector<uint32_t> input1_shape(2);
    if (input1_needs_reshape) {
      input1_shape[1] = 1;
      for (size_t i = 0; i < input1_original_rank - 1; i++) {
        input1_shape[1] *= input1_info.shape[i];
      }
      input1_shape[0] = input1_info.shape[input1_original_rank - 1];
    } else {
      input1_shape[1] = input1_info.shape[0];
      input1_shape[0] = input1_info.shape[1];
    }

    // Transpose quantization parameter's axis if this is using per-channel quantization.
    if (input1_info.quant_param.IsPerChannel()) {
      std::vector<size_t> perm_inv(perm.size());
      ORT_RETURN_IF_ERROR(utils::InvertPerm<size_t>(perm, perm_inv));
      ORT_RETURN_IF_ERROR(input1_info.quant_param.HandleTranspose<size_t>(perm_inv));
    }

    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input1_name);
    QnnTensorWrapper input_tensorwrapper(input1_name, tensor_type, input1_info.qnn_data_type,
                                         std::move(input1_info.quant_param), std::move(input1_shape),
                                         std::move(initializer_data));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  } else {
    // Process inputs for QNN's MatMul op.
    ORT_RETURN_IF_ERROR(BaseOpBuilder::ProcessInputs(qnn_model_wrapper, node_unit, logger, input_names, do_op_validation));

    // Need to insert Convert op if both inputs are dynamic inputs and are ufixed_16
    if (!input0_info.is_initializer && !input1_info.is_initializer &&
        input0_info.qnn_data_type == input1_info.qnn_data_type &&
        input0_info.qnn_data_type == QNN_DATATYPE_UFIXED_POINT_16) {
      ORT_RETURN_IF_NOT(input1_info.quant_param.IsPerTensor(),
                        "MatMul's activation inputs only support per-tensor quantization");
      const Qnn_QuantizeParams_t& quant_param = input1_info.quant_param.Get();
      // insert Convert op after input1
      std::string convert_input_name = input_names.back();
      input_names.pop_back();
      const std::string& matmul_output_name = node_unit.Outputs()[0].node_arg.Name();
      std::string convert_output_name = convert_input_name + "_convert_" + matmul_output_name;
      ORT_RETURN_IF_ERROR(InsertConvertOp(qnn_model_wrapper,
                                          convert_input_name,
                                          convert_output_name,
                                          input1_info.qnn_data_type,
                                          QNN_DATATYPE_UFIXED_POINT_8,
                                          quant_param.scaleOffsetEncoding.offset,
                                          quant_param.scaleOffsetEncoding.scale,
                                          input1_info.shape,
                                          do_op_validation));
      input_names.push_back(convert_output_name);
    }
  }

  return Status::OK();
}

Status MatMulOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  TensorInfo input0_info = {};
  TensorInfo input1_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input0_info));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[1], input1_info));

  const bool use_fully_connected = !input0_info.is_initializer &&
                                   input0_info.shape.size() >= 2 &&
                                   input1_info.is_initializer &&
                                   input1_info.shape.size() >= 2;

  if (use_fully_connected) {
    const auto& output = node_unit.Outputs()[0];
    const auto& output_name = output.node_arg.Name();
    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(output, output_info));

    const size_t output_original_rank = output_info.shape.size();

    // TODO: Handle rank < 2
    ORT_RETURN_IF_NOT(output_original_rank >= 2, "Dont support rank < 2 yet");
    if (output_original_rank > 2) {
      const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
      std::vector<uint32_t> output_shape_2d = {
          output_info.shape[output_original_rank - 2],
          output_info.shape[output_original_rank - 1],
      };
      const std::string fc_output_name = output_name + "_ort_qnn_ep_fc";
      QnnTensorWrapper output_tensorwrapper(fc_output_name,
                                            QNN_TENSOR_TYPE_NATIVE,
                                            output_info.qnn_data_type,
                                            output_info.quant_param.Copy(),
                                            std::vector<uint32_t>(output_shape_2d));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit),
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_FULLY_CONNECTED,
                                                        std::move(input_names),
                                                        {fc_output_name},
                                                        {},
                                                        do_op_validation),
                        "Failed to add FullyConnected node.");

      // Add Reshape to convert QNN FullyConnected's output back to N-D.
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(fc_output_name,
                                                           output_name,
                                                           output_shape_2d,
                                                           output_info.shape,
                                                           output_info.qnn_data_type,
                                                           output_info.quant_param,
                                                           do_op_validation,
                                                           false,
                                                           is_graph_output));
    } else {
      const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
      Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
      QnnTensorWrapper output_tensorwrapper(output_name,
                                            tensor_type,
                                            output_info.qnn_data_type,
                                            std::move(output_info.quant_param),
                                            std::move(output_info.shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetNodeName(node_unit),
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_FULLY_CONNECTED,
                                                        std::move(input_names),
                                                        {output_name},
                                                        {},
                                                        do_op_validation),
                        "Failed to add FullyConnected node.");
    }

  } else {
    const std::string& op_type = node_unit.OpType();
    std::vector<std::string> param_tensor_names;

    Qnn_Scalar_t scalar_param = QNN_SCALAR_INIT;
    scalar_param.dataType = QNN_DATATYPE_BOOL_8;
    scalar_param.bool8Value = 0;
    QnnParamWrapper transpose_in0_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0, scalar_param);
    param_tensor_names.push_back(transpose_in0_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in0_param));

    QnnParamWrapper transpose_in1_param(node_unit.Index(), node_unit.Name(), QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1, scalar_param);
    param_tensor_names.push_back(transpose_in1_param.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(transpose_in1_param));

    ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit,
                                       std::move(input_names),
                                       std::move(param_tensor_names),
                                       logger, do_op_validation, GetQnnOpType(op_type)));
  }

  return Status::OK();
}

void CreateMatMulOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<MatMulOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
