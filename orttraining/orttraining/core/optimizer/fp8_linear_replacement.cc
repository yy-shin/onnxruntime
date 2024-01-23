// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_FP8_TRAINING

#include "orttraining/core/optimizer/fp8_linear_replacement.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

Status Fp8LinearReplacement::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                       const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) continue;  // Node was removed.

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // Matching for Gemm or FusedMatMul node.
    if ((!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gemm", {13}) &&
         !graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedMatMul", {1}, kMSDomain)) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // Requires bias for now, will support case without bias in the future.
    Node* bias_node = nullptr;
    if (node.OpType() == "FusedMatMul") {
      const NodeArg* matmul_output = node.OutputDefs()[0];
      if (graph.IsOutput(matmul_output)) continue;
      auto consumer_nodes = graph.GetConsumerNodes(matmul_output->Name());
      if (consumer_nodes.size() != 1 ||
          !graph_utils::IsSupportedOptypeVersionAndDomain(*consumer_nodes[0], "Add", {13, 14}) ||
          consumer_nodes[0]->GetExecutionProviderType() != node.GetExecutionProviderType()) {
        continue;
      }
      bias_node = graph.GetNode(consumer_nodes[0]->Index());
    }

    NodeArg* input_arg = node.MutableInputDefs()[0];
    NodeArg* weight_arg = node.MutableInputDefs()[1];
    auto weight_comsumers = graph.GetConsumerNodes(weight_arg->Name());
    if (weight_comsumers.size() != 2) {
      continue;
    }

    Node& grad_input_node =
        *graph.GetNode(weight_comsumers[weight_comsumers[0]->Index() == node.Index() ? 1 : 0]->Index());
    if (grad_input_node.OpType() != node.OpType()) {
      continue;
    }

    NodeArg* grad_output_arg = grad_input_node.MutableInputDefs()[0];
    auto grad_output_consumers = graph.GetConsumerNodes(grad_output_arg->Name());
    if (grad_output_consumers.size() != 3) {
      continue;
    }

    Node* reduce_sum_node = nullptr;
    Node* grad_weight_node = nullptr;
    for (int i = 0; i < 3; ++i) {
      if (grad_output_consumers[i]->OpType() == "ReduceSum") {
        reduce_sum_node = graph.GetNode(grad_output_consumers[i]->Index());
      } else if (grad_output_consumers[i]->Index() != grad_input_node.Index()) {
        grad_weight_node = graph.GetNode(grad_output_consumers[i]->Index());
      }
    }

    if (!reduce_sum_node || !grad_weight_node ||
        (!graph_utils::IsSupportedOptypeVersionAndDomain(*grad_weight_node, "Gemm", {13}) &&
         !graph_utils::IsSupportedOptypeVersionAndDomain(*grad_weight_node, "Reshape", {13, 14, 19}))) {
      continue;
    }

    Node* lhs_reshape_node = nullptr;
    Node* rhs_reshape_node = nullptr;
    if (grad_weight_node->OpType() == "Reshape") {
      lhs_reshape_node = grad_weight_node;
      const NodeArg* reshape_output = lhs_reshape_node->OutputDefs()[0];
      auto consumer_nodes = graph.GetConsumerNodes(reshape_output->Name());
      if (consumer_nodes.size() != 1 ||
          !graph_utils::IsSupportedOptypeVersionAndDomain(*consumer_nodes[0], "Gemm", {13})) {
        continue;
      }
      grad_weight_node = graph.GetNode(consumer_nodes[0]->Index());
      auto producer_node = graph.GetProducerNode(grad_weight_node->MutableInputDefs()[1]->Name());
      if (!producer_node || producer_node->OpType() != "Reshape") {
        continue;
      }
      rhs_reshape_node = graph.GetNode(producer_node->Index());
    }

    InlinedVector<NodeArg*> fp8_linear_inputs;
    fp8_linear_inputs.emplace_back(input_arg);
    fp8_linear_inputs.emplace_back(weight_arg);
    fp8_linear_inputs.emplace_back(bias_node ? bias_node->MutableInputDefs()[1] : node.MutableInputDefs()[2]);
    InlinedVector<NodeArg*> fp8_gemm_outputs;
    fp8_gemm_outputs.emplace_back(bias_node ? bias_node->MutableOutputDefs()[0] : node.MutableOutputDefs()[0]);
    ONNX_NAMESPACE::TypeProto tensor_fp8;
    tensor_fp8.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E4M3FN);
    ONNX_NAMESPACE::TypeProto tensor_float;
    tensor_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    NodeArg& input_t_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("input_t"), &tensor_fp8);
    fp8_gemm_outputs.emplace_back(&input_t_def);
    NodeArg& weight_t_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("weight_t"), &tensor_fp8);
    fp8_gemm_outputs.emplace_back(&weight_t_def);
    NodeArg& scale_inv_def = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("scale_inv"), &tensor_float);
    fp8_gemm_outputs.emplace_back(&scale_inv_def);
    Node& fp8_linear_node = graph.AddNode(graph.GenerateNodeName("FP8Linear"), "FP8Linear", "FP8Linear",
                                          fp8_linear_inputs, fp8_gemm_outputs, nullptr, kMSDomain);
    fp8_linear_node.SetExecutionProviderType(node.GetExecutionProviderType());

    InlinedVector<NodeArg*> fp8_linear_grad_inputs{grad_output_arg, &input_t_def, &weight_t_def, &scale_inv_def};
    InlinedVector<NodeArg*> fp8_linear_grad_outputs{grad_input_node.MutableOutputDefs()[0],
                                                    grad_weight_node->MutableOutputDefs()[0],
                                                    reduce_sum_node->MutableOutputDefs()[0]};
    const std::string grad_op_type = "FP8LinearGrad";
    Node& fp8_linear_grad_node =
        graph.AddNode(graph.GenerateNodeName("FP8LinearGrad"), "FP8LinearGrad", "FP8LinearGrad", fp8_linear_grad_inputs,
                      fp8_linear_grad_outputs, nullptr, kMSDomain);
    fp8_linear_grad_node.SetExecutionProviderType(node.GetExecutionProviderType());

    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.Index());
    if (bias_node) {
      graph_utils::RemoveNodeOutputEdges(graph, *bias_node);
      graph.RemoveNode(bias_node->Index());
    }
    graph_utils::RemoveNodeOutputEdges(graph, grad_input_node);
    graph.RemoveNode(grad_input_node.Index());
    if (lhs_reshape_node) {
      graph_utils::RemoveNodeOutputEdges(graph, *lhs_reshape_node);
      graph.RemoveNode(lhs_reshape_node->Index());
      graph_utils::RemoveNodeOutputEdges(graph, *rhs_reshape_node);
      graph.RemoveNode(rhs_reshape_node->Index());
    }
    graph_utils::RemoveNodeOutputEdges(graph, *grad_weight_node);
    graph.RemoveNode(grad_weight_node->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *reduce_sum_node);
    graph.RemoveNode(reduce_sum_node->Index());
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime

#endif  // ENABLE_FP8_TRAINING
