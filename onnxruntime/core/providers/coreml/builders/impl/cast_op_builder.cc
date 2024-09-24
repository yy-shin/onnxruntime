// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class CastOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;
};

Status CastOpBuilder::AddToModelBuilderImpl([[maybe_unused]] ModelBuilder& model_builder,
                                            [[maybe_unused]] const Node& node,
                                            [[maybe_unused]] const logging::Logger& logger) const {
  // This is a special handling case for ArgMax Op, where argmax is followed by a cast to int32 type.
  // The ArgMax is fused with the Cast node and produces an int32 output.
  // Cast node is not provided in CoreML model, so we're skipping adding the Cast node here.
  #if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#module-coremltools.converters.mil.mil.ops.defs.iOS15.reduction

    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "cast");
    AddOperationInput(*op, "x", node.InputDefs()[0]->Name());
    NodeAttrHelper helper(node);
    const auto cast_to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (cast_to_type == ONNX_NAMESPACE::TensorProto::INT32) {
      AddOperationInput(*op, "dtype", "int32");
    } else if (cast_to_type == ONNX_NAMESPACE::TensorProto::FLOAT) {
      AddOperationInput(*op, "dtype", "fp32");
    } else if (cast_to_type == ONNX_NAMESPACE::TensorProto::FLOAT16) {
      AddOperationInput(*op, "dtype", "fp16");
    } else if (cast_to_type == ONNX_NAMESPACE::TensorProto::BOOL) {
      AddOperationInput(*op, "dtype", "bool");
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported cast type: ", cast_to_type);
    }

    AddOperationOutput(*op, *node.OutputDefs()[0]);
    model_builder.AddOperation(std::move(op));
  }
  #endif

  return Status::OK();
}

bool CastOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  if (node.GetInputEdgesCount() == 0) {
    LOGS(logger, VERBOSE) << "Cast has no preceding nodes.";
    return false;
  }

  const auto& prec_node = node.InputEdgesBegin()->GetNode();

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    NodeAttrHelper helper(node);
    const auto cast_to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto::UNDEFINED);
    if (prec_node.OpType() == "ArgMax" && cast_to_type == ONNX_NAMESPACE::TensorProto::INT32 &&
        prec_node.GetOutputEdgesCount() == 1) {
      fused_into_argmax_ = true;
    }
    return true;
  }
#endif

  /*Cast node is only aimed for supporting argmax and we are only handling the case where an argmax
    followed by a cast node. We need to check if the preceding node is an argmax and also if it's a
    supported argmax op type.*/
  if (prec_node.OpType() != "ArgMax") {
    LOGS(logger, VERBOSE) << "Cast's producing node is not ArgMax is not supported."
                          << "Current producing node: [" << prec_node.OpType()
                          << "]";
    return false;
  }
  if (!IsNodeSupported(prec_node, input_params, logger)) {
    LOGS(logger, VERBOSE) << "Cast's producing node ["
                          << prec_node.OpType()
                          << "] is not a supported op.";
    return false;
  }

  // Check if the output type of cast node is int32
  NodeAttrHelper helper(node);
  const auto cast_to_type = helper.Get("to", ONNX_NAMESPACE::TensorProto::UNDEFINED);
  if (cast_to_type != ONNX_NAMESPACE::TensorProto::INT32) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Output type: [" << cast_to_type
                          << "] is not supported.";
    return false;
  }

  return true;
}

bool CastOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                           const logging::Logger& logger) const {
  // We only check the type of input 0
  const auto& input = *node.InputDefs()[0];

  int32_t input_type;
  if (!GetType(input, input_type, logger))
    return false;

#if defined(COREML_ENABLE_MLPROGRAM)
  if (input_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
      input_type == ONNX_NAMESPACE::TensorProto_DataType_INT32 ||
      input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      input_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
      input_type == ONNX_NAMESPACE::TensorProto_DataType_bool) {
    return true;
  } else {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported.";
    return false;
  }
#endif

  // only support int64 coming from ArgMax (check for ArgMax is done in IsOpSupportedImpl())
  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported.";
    return false;
  }

  return true;
}

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<CastOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
