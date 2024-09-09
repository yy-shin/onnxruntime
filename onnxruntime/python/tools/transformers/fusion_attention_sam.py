# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from typing import Tuple, Union

import os
import numpy as np
from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


# class FusionAttentionSam(Fusion):
#     """
#     Fuse Attention subgraph of Segment Anything Model (SAM) 2 into one Attention node.
#     """

#     def __init__(
#         self,
#         model: OnnxModel,
#         hidden_size: int,
#         num_heads: int,
#         is_cross_attention: bool,
#         enable_packed_qkv: bool,
#         enable_packed_kv: bool,
#     ):
#         super().__init__(
#             model,
#             "Attention" if is_cross_attention and enable_packed_qkv else "MultiHeadAttention",
#             ["LayerNormalization"],
#         )
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.is_cross_attention = is_cross_attention

#         # Note: pack Q/K/V or K/V weights into one tensor make it harder for updating initializers for LoRA.
#         # To support LoRA, it is better to use separated Q, K and V inputs in offline optimization,
#         # and CUDA operator pre-packs those tensors to preferred format based on available kernels.
#         # In this way, we can support LoRA and get optimal performance at same time.
#         self.enable_packed_qkv = enable_packed_qkv
#         self.enable_packed_kv = enable_packed_kv

#         # Flags to show warning only once
#         self.num_heads_warning = True
#         self.hidden_size_warning = True

#     def get_num_heads(self, reshape_q: NodeProto) -> int:
#         """Detect num_heads from a reshape node.

#         Args:
#             reshape_q (NodeProto): reshape node for Q
#         Returns:
#             int: num_heads, or 0 if not found
#         """
#         num_heads = 0

#         # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
#         shape_value = self.model.get_constant_value(reshape_q.input[1])
#         if shape_value is not None:
#             if isinstance(shape_value, np.ndarray) and list(shape_value.shape) == [4]:
#                 num_heads = int(shape_value[2])

#         if isinstance(num_heads, int) and num_heads > 0:
#             return num_heads

#         return 0

#     def get_hidden_size(self, layernorm_node):
#         """Detect hidden_size from LayerNormalization node.
#         Args:
#             layernorm_node (NodeProto): LayerNormalization node before Q, K and V
#         Returns:
#             int: hidden_size, or 0 if not found
#         """
#         layernorm_bias = self.model.get_initializer(layernorm_node.input[2])
#         if layernorm_bias:
#             return NumpyHelper.to_array(layernorm_bias).shape[0]

#         return 0

#     def get_num_heads_and_hidden_size(self, reshape_q: NodeProto, layernorm_node: NodeProto) -> Tuple[int, int]:
#         """Detect num_heads and hidden_size.

#         Args:
#             reshape_q (NodeProto): reshape node for Q
#             layernorm_node (NodeProto): LayerNormalization node before Q, K, V
#         Returns:
#             Tuple[int, int]: num_heads and hidden_size
#         """
#         num_heads = self.get_num_heads(reshape_q)
#         if num_heads <= 0:
#             num_heads = self.num_heads  # Fall back to user specified value

#         if self.num_heads > 0 and num_heads != self.num_heads:
#             if self.num_heads_warning:
#                 logger.warning(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")
#                 self.num_heads_warning = False  # Do not show the warning more than once

#         hidden_size = self.get_hidden_size(layernorm_node)
#         if hidden_size <= 0:
#             hidden_size = self.hidden_size  # Fall back to user specified value

#         if self.hidden_size > 0 and hidden_size != self.hidden_size:
#             if self.hidden_size_warning:
#                 logger.warning(
#                     f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
#                 )
#                 self.hidden_size_warning = False  # Do not show the warning more than once

#         return num_heads, hidden_size

#     def merge_qkv_bias(self, q_add: NodeProto, k_add: NodeProto, v_add: NodeProto, qkv_bias_name: str):
#         q_bias_tensor = self.model.get_initializer(q_add.input[1]) or self.model.get_initializer(q_add.input[0])
#         k_bias_tensor = self.model.get_initializer(k_add.input[1]) or self.model.get_initializer(k_add.input[0])
#         v_bias_tensor = self.model.get_initializer(v_add.input[1]) or self.model.get_initializer(v_add.input[0])

#         q_bias = numpy_helper.to_array(q_bias_tensor)
#         k_bias = numpy_helper.to_array(k_bias_tensor)
#         v_bias = numpy_helper.to_array(v_bias_tensor)

#         q_bias_shape = np.prod(q_bias.shape)
#         k_bias_shape = np.prod(k_bias.shape)
#         v_bias_shape = np.prod(v_bias.shape)
#         assert q_bias_shape == k_bias_shape == v_bias_shape
#         qkv_bias = np.stack((q_bias, k_bias, v_bias), axis=0)
#         qkv_bias_dim = 3 * q_bias_shape

#         # No bias, use zeros
#         # qkv_bias = np.zeros([3, hidden_size], dtype=np.float32)
#         # qkv_bias_dim = 3 * hidden_size

#         self.add_initializer(
#             name=qkv_bias_name,
#             data_type=TensorProto.FLOAT,
#             dims=[qkv_bias_dim],
#             vals=qkv_bias,
#         )

#     def merge_kv_bias(self, k_add: NodeProto, v_add: NodeProto, kv_bias_name: str):
#         k_bias_tensor = self.model.get_initializer(k_add.input[1]) or self.model.get_initializer(k_add.input[0])
#         v_bias_tensor = self.model.get_initializer(v_add.input[1]) or self.model.get_initializer(v_add.input[0])

#         k_bias = numpy_helper.to_array(k_bias_tensor)
#         v_bias = numpy_helper.to_array(v_bias_tensor)

#         k_bias_shape = np.prod(k_bias.shape)
#         v_bias_shape = np.prod(v_bias.shape)
#         assert k_bias_shape == v_bias_shape
#         kv_bias = np.stack((k_bias, v_bias), axis=0)
#         kv_bias_dim = k_bias_shape + v_bias_shape
#         self.add_initializer(
#             name=kv_bias_name,
#             data_type=TensorProto.FLOAT,
#             dims=[kv_bias_dim],
#             vals=kv_bias,
#         )

#     def create_attention_node(
#         self,
#         q_matmul: NodeProto,
#         q_add: NodeProto,
#         k_matmul: NodeProto,
#         k_add: NodeProto,
#         v_matmul: NodeProto,
#         v_add: NodeProto,
#         num_heads: int,
#         hidden_size: int,
#         input: str,
#         output: str,
#     ) -> Union[NodeProto, None]:
#         """Create an Attention node.

#         Args:
#             q_matmul (NodeProto): MatMul node in fully connection for Q
#             q_add (NodeProto): Add bias node in fully connection for Q
#             k_matmul (NodeProto): MatMul node in fully connection for K
#             k_add (NodeProto): Add bias node in fully connection for K
#             v_matmul (NodeProto): MatMul node in fully connection for V
#             v_add (NodeProto): Add bias node in fully connection for V
#             num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
#             hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
#             input (str): input name
#             output (str): output name

#         Returns:
#             Union[NodeProto, None]: the node created or None if failed.
#         """
#         is_self_attention = not self.is_cross_attention

#         if is_self_attention:
#             if q_matmul.input[0] != input or k_matmul.input[0] != input or v_matmul.input[0] != input:
#                 logger.debug(
#                     "For self attention, input hidden state for q and k/v shall be same. Got %s, %s, %s",
#                     q_matmul.input[0],
#                     k_matmul.input[0],
#                     v_matmul.input[0],
#                 )
#                 return None
#         # else:
#         #     if q_matmul.input[0] != input or (k_matmul.input[0] != v_matmul.input[0]) or (k_matmul.input[0] == input):
#         #         logger.debug(
#         #             "For cross attention, input hidden state for q and k/v are different. Got %s, %s, %s",
#         #             q_matmul.input[0],
#         #             k_matmul.input[0],
#         #             v_matmul.input[0],
#         #         )

#         if hidden_size > 0 and (hidden_size % num_heads) != 0:
#             logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
#             return None

#         q_weight = self.model.get_initializer(q_matmul.input[1])
#         k_weight = self.model.get_initializer(k_matmul.input[1])
#         v_weight = self.model.get_initializer(v_matmul.input[1])
#         if not (q_weight and k_weight and v_weight):
#             return None

#         # Sometimes weights are stored in fp16
#         float_type = q_weight.data_type

#         qw = NumpyHelper.to_array(q_weight)
#         kw = NumpyHelper.to_array(k_weight)
#         vw = NumpyHelper.to_array(v_weight)
#         logger.debug(f"qw={qw.shape} kw={kw.shape} vw={vw.shape} hidden_size={hidden_size}")

#         # assert q and k have same shape as expected
#         if is_self_attention:
#             if qw.shape != kw.shape or qw.shape != vw.shape:
#                 return None

#             qw_in_size = qw.shape[0]

#             if hidden_size > 0 and hidden_size != qw_in_size:
#                 raise ValueError(
#                     f"Input hidden size ({hidden_size}) is not same as weight dimension of q,k,v ({qw_in_size}). "
#                     "Please provide a correct input hidden size or pass in 0"
#                 )

#             # All the matrices can have the same shape or q, k matrics can have the same shape with v being different
#             # For 2d weights, the shapes would be [in_size, out_size].
#             # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
#             qw_out_size = int(np.prod(qw.shape[1:]))

#             if self.enable_packed_qkv:
#                 attention_node_name = self.model.create_node_name("MultiHeadAttention")

#                 c = qw_in_size
#                 n = num_heads
#                 h = qw_out_size // num_heads

#                 # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 3, H] shape
#                 qkv_weight = np.dstack([qw.reshape(c, n, h), kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(
#                     c, n * 3 * h
#                 )

#                 matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_QKV")
#                 self.add_initializer(
#                     name=matmul_node_name + "_weight",
#                     data_type=float_type,
#                     dims=[qkv_weight.shape[0], qkv_weight.shape[1]],
#                     vals=qkv_weight,
#                 )

#                 matmul_node = helper.make_node(
#                     "MatMul",
#                     inputs=[k_matmul.input[0], matmul_node_name + "_weight"],
#                     outputs=[matmul_node_name + "_out"],
#                     name=matmul_node_name,
#                 )
#                 self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

#                 qkv_bias_name = attention_node_name + "_qkv_bias"
#                 self.merge_qkv_bias(q_add, k_add, v_add, qkv_bias_name)
#                 add_node_name = self.model.create_node_name("Add", name_prefix="AddBias_QKV")
#                 add_bias_node = helper.make_node(
#                     "Add",
#                     inputs=[matmul_node_name + "_out", qkv_bias_name],
#                     outputs=[add_node_name + "_out"],
#                     name=add_node_name,
#                 )
#                 self.node_name_to_graph_name[add_node_name.name] = self.this_graph_name

#                 self.add_initializer(
#                     name=matmul_node_name + "_reshape_shape",
#                     data_type=TensorProto.INT64,
#                     dims=[5],
#                     vals=[0, 0, n, 3, h],
#                     raw=False,
#                 )

#                 reshape_node = helper.make_node(
#                     "Reshape",
#                     inputs=[
#                         add_node_name + "_out",
#                         add_node_name + "_reshape_shape",
#                     ],
#                     outputs=[attention_node_name + "_qkv_input"],
#                     name=matmul_node_name + "_reshape",
#                 )
#                 self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name

#                 self.nodes_to_add.extend([matmul_node, add_bias_node, reshape_node])
#                 self.nodes_to_remove.extend([q_matmul, k_matmul, v_matmul, q_add, k_add, v_add])
#             else:
#                 qkv_weight = np.stack((qw, kw, vw), axis=1)
#                 qkv_weight_dim = 3 * qw_out_size

#                 attention_node_name = self.model.create_node_name("Attention")

#                 self.add_initializer(
#                     name=attention_node_name + "_qkv_weight",
#                     data_type=float_type,
#                     dims=[qw_in_size, qkv_weight_dim],
#                     vals=qkv_weight,
#                 )

#                 qkv_bias_name = attention_node_name + "_qkv_bias"
#                 self.merge_qkv_bias(q_add, k_add, v_add, qkv_bias_name)
#         else:  # cross attention
#             attention_node_name = self.model.create_node_name("MultiHeadAttention")
#             if self.enable_packed_kv:
#                 if kw.shape != vw.shape:
#                     return None

#                 kw_in_size = kw.shape[0]
#                 vw_in_size = vw.shape[0]
#                 assert kw_in_size == vw_in_size

#                 qw_out_size = qw.shape[1]
#                 kw_out_size = kw.shape[1]
#                 vw_out_size = vw.shape[1]
#                 assert qw_out_size == vw_out_size and kw_out_size == vw_out_size

#                 c = kw_in_size
#                 n = num_heads
#                 h = kw_out_size // num_heads

#                 # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 2, H] shape
#                 kv_weight = np.dstack([kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(c, n * 2 * h)

#                 matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_KV")
#                 self.add_initializer(
#                     name=matmul_node_name + "_weight",
#                     data_type=float_type,
#                     dims=[kv_weight.shape[0], kv_weight.shape[1]],
#                     vals=kv_weight,
#                 )

#                 matmul_node = helper.make_node(
#                     "MatMul",
#                     inputs=[k_matmul.input[0], matmul_node_name + "_weight"],
#                     outputs=[matmul_node_name + "_out"],
#                     name=matmul_node_name,
#                 )
#                 self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

#                 add_node_name = self.model.create_node_name("Add", name_prefix="AddBias_KV")
#                 add_bias_node = helper.make_node(
#                     "Add",
#                     inputs=[matmul_node_name + "_out", add_node_name + "_kv_bias"],
#                     outputs=[add_node_name + "_out"],
#                     name=add_node_name,
#                 )
#                 self.node_name_to_graph_name[add_bias_node.name] = self.this_graph_name

#                 self.add_initializer(
#                     name=matmul_node_name + "_reshape_shape",
#                     data_type=TensorProto.INT64,
#                     dims=[5],
#                     vals=[0, 0, n, 2, h],
#                     raw=False,
#                 )

#                 reshape_node = helper.make_node(
#                     "Reshape",
#                     inputs=[
#                         add_node_name + "_out",
#                         add_node_name + "_reshape_shape",
#                     ],
#                     outputs=[attention_node_name + "_kv_input"],
#                     name=add_node_name + "_reshape",
#                 )
#                 self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name

#                 self.nodes_to_add.extend([matmul_node, add_bias_node, reshape_node])
#                 self.nodes_to_remove.extend([k_matmul, v_matmul, k_add, v_add])

#         if is_self_attention:
#             if not self.enable_packed_qkv:
#                 attention_inputs = [
#                     input,
#                     attention_node_name + "_qkv_weight",
#                     attention_node_name + "_qkv_bias",
#                 ]
#             else:
#                 attention_inputs = [attention_node_name + "_qkv_input"]
#         else:
#             if not self.enable_packed_kv:
#                 attention_inputs = [
#                     q_add.output[0],
#                     k_add.output[0],
#                     v_add.output[0],
#                 ]
#             else:
#                 attention_inputs = [
#                     q_add.output[0],
#                     attention_node_name + "_kv_input",
#                 ]

#         attention_node = helper.make_node(
#             "Attention" if (is_self_attention and not self.enable_packed_qkv) else "MultiHeadAttention",
#             inputs=attention_inputs,
#             outputs=[output],
#             name=attention_node_name,
#         )
#         attention_node.domain = "com.microsoft"
#         attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

#         counter_name = (
#             "Attention (self attention)"
#             if is_self_attention and not self.enable_packed_qkv
#             else "MultiHeadAttention ({})".format(
#                 "self attention with packed qkv"
#                 if self.enable_packed_qkv
#                 else "cross attention with packed kv" if self.enable_packed_kv else "cross attention"
#             )
#         )
#         self.increase_counter(counter_name)
#         return attention_node

#     def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
#         #self.model.save_model_to_file("e:\\before_att_fusion.onnx")       
#         root_input = normalize_node.output[0]

#         children_nodes = input_name_to_nodes[root_input]
#         skip_add = None
#         for node in children_nodes:
#             if node.op_type != "Add":
#                 continue

#             skip_add = node
#             match_qkv = self.match_attention_subgraph(skip_add, 1 if skip_add.input[0] == root_input else 0)
#             if match_qkv is None:
#                 continue

#             reshape_qkv, transpose_qkv, reshape_q, matmul_q, add_q, matmul_k, add_k, matmul_v, add_v = match_qkv

#             attention_last_node = reshape_qkv

#             q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q, normalize_node)
#             if q_num_heads <= 0:
#                 logger.debug("fuse_attention: failed to detect num_heads")
#                 return

#             # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
#             new_node = self.create_attention_node(
#                 matmul_q,
#                 add_q,
#                 matmul_k,
#                 add_k,
#                 matmul_v,
#                 add_v,
#                 q_num_heads,
#                 q_hidden_size,
#                 input=matmul_q.input[0],
#                 output=attention_last_node.output[0],
#             )
#             if new_node is None:
#                 return

#             self.nodes_to_add.append(new_node)
#             self.node_name_to_graph_name[new_node.name] = self.this_graph_name

#             self.nodes_to_remove.extend([attention_last_node, transpose_qkv])

#             # Use prune graph to remove nodes since they are shared by all attention nodes.
#             self.prune_graph = True

#     def match_attention_subgraph(self, node_after_output_projection, input_index):
#         """Match Q, K and V paths exported by SDPA of PyTorch 2.*"""
#         qkv_nodes = self.model.match_parent_path(
#             node_after_output_projection,
#             ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
#             [input_index, None, None, 0, 0],
#         )

#         if qkv_nodes is None:
#             return None

#         (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

#         v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
#         if v_nodes is None:
#             logger.debug("fuse_attention: failed to match v path")
#             return None
#         (_, _, add_v, matmul_v) = v_nodes

#         qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "MatMul"], [0, 0])
#         if qk_nodes is not None:
#             (_softmax_qk, matmul_qk) = qk_nodes
#         else:
#             logger.debug("fuse_attention: failed to match qk path")
#             return None

#         q_nodes = self.model.match_parent_path(
#             matmul_qk, ["Mul", "Transpose", "Reshape", "Add", "MatMul"], [0, None, 0, 0, None]
#         )
#         if q_nodes is None:
#             logger.debug("fuse_attention: failed to match q path")
#             return None
#         (mul_q, _transpose_q, reshape_q, add_q, matmul_q) = q_nodes

#         k_nodes = self.model.match_parent_path(
#             matmul_qk, ["Mul", "Transpose", "Reshape", "Add", "MatMul"], [1, None, 0, 0, None]
#         )
#         if k_nodes is None:
#             logger.debug("fuse_attention: failed to match k path")
#             return None

#         (_mul_k, _, _, add_k, matmul_k) = k_nodes

#         # The scalar for Q and K is sqrt(1.0/sqrt(head_size)).
#         mul_q_nodes = self.model.match_parent_path(
#             mul_q,
#             ["Sqrt", "Div", "Sqrt", "Cast", "Slice", "Shape", "Transpose", "Reshape"],
#             [None, 0, 1, 0, 0, 0, 0, 0],
#         )
#         if mul_q_nodes is None or mul_q_nodes[-1] != reshape_q:
#             logger.debug("fuse_attention: failed to match mul_q path")
#             return None

#         return reshape_qkv, transpose_qkv, reshape_q, matmul_q, add_q, matmul_k, add_k, matmul_v, add_v


class FusionMultiHeadAttentionSam(Fusion):
    """
    Fuse Attention subgraph of Segment Anything Model (SAM) 2 into one MultiHeadAttention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
    ):
        super().__init__(
            model,
            "MultiHeadAttention",
            ["LayerNormalization"],
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads(self, reshape_q: NodeProto) -> int:
        """Detect num_heads from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q
        Returns:
            int: num_heads, or 0 if not found
        """
        num_heads = 0

        # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
        shape_value = self.model.get_constant_value(reshape_q.input[1])
        if shape_value is not None:
            if isinstance(shape_value, np.ndarray) and list(shape_value.shape) == [4]:
                num_heads = int(shape_value[2])

        if isinstance(num_heads, int) and num_heads > 0:
            return num_heads

        return 0
    

    def get_encoder_num_heads(self, reshape_in: NodeProto) -> int:
        """Detect num_heads from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q
        Returns:
            int: num_heads, or 0 if not found
        """
        num_heads = 0

        shape_value = self.model.get_constant_value(reshape_in.input[1])
        if shape_value is not None:
            if isinstance(shape_value, np.ndarray) and list(shape_value.shape) == [5]:
                num_heads = int(shape_value[3])
        else:
            concat_shape = self.model.match_parent(reshape_in, "Concat", 1)
            if concat_shape is not None and len(concat_shape.input) == 5:
                # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
                shape_value = self.model.get_constant_value(concat_shape.input[3])
                if shape_value is not None:
                    if isinstance(shape_value, np.ndarray) and list(shape_value.shape) == [1]:
                        num_heads = int(shape_value[0])

        if isinstance(num_heads, int) and num_heads > 0:
            return num_heads

        return 0

    def get_hidden_size(self, layernorm_node):
        """Detect hidden_size from LayerNormalization node.
        Args:
            layernorm_node (NodeProto): LayerNormalization node before Q, K and V
        Returns:
            int: hidden_size, or 0 if not found
        """
        layernorm_bias = self.model.get_initializer(layernorm_node.input[2])
        if layernorm_bias:
            return NumpyHelper.to_array(layernorm_bias).shape[0]

        return 0

    def get_num_heads_and_hidden_size(self, reshape_q: NodeProto, layernorm_node: NodeProto, is_encoder:bool = False) -> Tuple[int, int]:
        """Detect num_heads and hidden_size.

        Args:
            reshape_q (NodeProto): reshape node for Q
            layernorm_node (NodeProto): LayerNormalization node before Q, K, V
        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """
        if is_encoder:
            num_heads = self.get_encoder_num_heads(reshape_q)
        else:
            num_heads = self.get_num_heads(reshape_q)
        if num_heads <= 0:
            num_heads = self.num_heads  # Fall back to user specified value

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")
                self.num_heads_warning = False  # Do not show the warning more than once

        hidden_size = self.get_hidden_size(layernorm_node)
        if hidden_size <= 0:
            hidden_size = self.hidden_size  # Fall back to user specified value

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = False  # Do not show the warning more than once

        return num_heads, hidden_size

    def create_attention_node(
        self,
        q_matmul: NodeProto,
        q_add: NodeProto,
        k_matmul: NodeProto,
        k_add: NodeProto,
        v_matmul: NodeProto,
        v_add: NodeProto,
        num_heads: int,
        hidden_size: int,
        output: str,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            q_matmul (NodeProto): MatMul node in fully connection for Q
            q_add (NodeProto): Add bias node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            k_add (NodeProto): Add bias node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            v_add (NodeProto): Add bias node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        # else:
        #     if q_matmul.input[0] != input or (k_matmul.input[0] != v_matmul.input[0]) or (k_matmul.input[0] == input):
        #         logger.debug(
        #             "For cross attention, input hidden state for q and k/v are different. Got %s, %s, %s",
        #             q_matmul.input[0],
        #             k_matmul.input[0],
        #             v_matmul.input[0],
        #         )
        print(f"enter create_attention_node with {hidden_size=} {num_heads=} {hidden_size > 0} {(hidden_size % num_heads) != 0}")

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        if not (q_weight and k_weight and v_weight):
            return None

        # Sometimes weights are stored in fp16
        float_type = q_weight.data_type

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)
        logger.debug(f"qw={qw.shape} kw={kw.shape} vw={vw.shape} hidden_size={hidden_size}")

        attention_node_name = self.model.create_node_name("MultiHeadAttention")

        attention_inputs = [
            q_add.output[0],
            k_add.output[0],
            v_add.output[0],
        ]

        attention_node = helper.make_node(
            "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        counter_name = (
            "MultiHeadAttention ({})".format(
                "cross attention"
            )
        )
        self.increase_counter(counter_name)
        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        if not os.path.exists("before_mha_fusion_v2.onnx"):
            self.model.save_model_to_file("before_mha_fusion_v2.onnx")

        if self.fuse_sam_encoder_pattern(normalize_node, input_name_to_nodes, output_name_to_node):
            return

        match_qkv = self.match_attention_subgraph(normalize_node)
        if match_qkv is None:
            if normalize_node.input[0] not in output_name_to_node:
                return

            skip_add = output_name_to_node[normalize_node.input[0]]
            if skip_add.op_type != "Add":
                return

            match_qkv = self.match_attention_subgraph(skip_add)

            if match_qkv is None:
                return

        reshape_qkv, transpose_qkv, reshape_q, matmul_q, add_q, matmul_k, add_k, matmul_v, add_v = match_qkv

        attention_last_node = reshape_qkv

        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q, normalize_node, False)
        if q_num_heads <= 0:
            logger.debug("fuse_attention: failed to detect num_heads")
            return

        # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
        new_node = self.create_attention_node(
            matmul_q,
            add_q,
            matmul_k,
            add_k,
            matmul_v,
            add_v,
            q_num_heads,
            q_hidden_size,
            output=attention_last_node.output[0],
        )
        if new_node is None:
            return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([attention_last_node, transpose_qkv])

        # Use prune graph to remove nodes since they are shared by all attention nodes.
        self.prune_graph = True

    def match_attention_subgraph(self, node_after_output_projection):
        """Match Q, K and V paths exported by PyTorch 2.*"""
        qkv_nodes = self.model.match_parent_path(
            node_after_output_projection,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [None, None, None, 0, 0],
        )

        if qkv_nodes is None:
            return None

        (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add", "MatMul"], [1, 0, 0, None])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return None
        (_, _, add_v, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "MatMul"], [0, 0])
        if qk_nodes is not None:
            (_softmax_qk, matmul_qk) = qk_nodes
        else:
            logger.debug("fuse_attention: failed to match qk path")
            return None

        q_nodes = self.model.match_parent_path(
            matmul_qk, ["Mul", "Transpose", "Reshape", "Add", "MatMul"], [0, None, 0, 0, None]
        )
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return None
        (mul_q, _transpose_q, reshape_q, add_q, matmul_q) = q_nodes

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Mul", "Transpose", "Reshape", "Add", "MatMul"], [1, None, 0, 0, None]
        )
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return None

        (_mul_k, _, _, add_k, matmul_k) = k_nodes

        # The scalar for Q and K is sqrt(1.0/sqrt(head_size)).
        mul_q_nodes = self.model.match_parent_path(
            mul_q,
            ["Sqrt", "Div", "Sqrt", "Cast", "Slice", "Shape", "Transpose", "Reshape"],
            [None, 0, 1, 0, 0, 0, 0, 0],
        )
        if mul_q_nodes is None or mul_q_nodes[-1] != reshape_q:
            logger.debug("fuse_attention: failed to match mul_q path")
            return None

        return reshape_qkv, transpose_qkv, reshape_q, matmul_q, add_q, matmul_k, add_k, matmul_v, add_v
    
    #--------------------------------------------------------
    # The following are for SAM encoder attention fusion
    #--------------------------------------------------------
    def fuse_sam_encoder_pattern(self, normalize_node, input_name_to_nodes, output_name_to_node):
        # SAM encoder attention layer pattern:        
        #           Add -----------+
        #            |             |
        #        LayerNorm         |
        #            |             |
        #        Reshape           |
        #            |             |
        #        Transpose         |
        #            |             |
        #        MatMul            |
        #            |             |
        #           Add            |
        #            |             |
        #         Reshape          |
        #            |             |
        #          Split           |
        #            |             |
        #  Self Attention subgraph |
        #            |             |
        #        Reshape           |
        #            |             |
        #        Transpose         |
        #            |             |
        #        Reshape           |
        #            |             |
        #            Add ----------+
        #            |
        #         LayerNorm (starts from here)

        nodes = self.model.match_parent_path(
            normalize_node,
            ["Add", "Reshape", "Transpose", "Reshape"],
            [0, None, 0, 0],
        )
        if nodes is None:
            nodes = self.model.match_parent_path(
                normalize_node,
                ["Add", "Slice", "Slice", "Reshape", "Transpose", "Reshape"],
                [0, None, 0, 0, 0, 0],
            )
        if nodes is None:
            nodes = self.model.match_parent_path(
                normalize_node,
                ["Add"],
                [0],
            )
        if nodes is None:        
            return False
                 
        node_after_output_projection = nodes[-1]
        matched_sdpa = self.match_sam_encoder_attention_subgraph(node_after_output_projection, input_index=1 if len(nodes) == 1 else None)
        if matched_sdpa is None:
            print("##8")
            return False

        reshape_out, transpose_out, split_qkv, transpose_q, transpose_k, transpose_v = matched_sdpa
 
        # B, S, N, H => B, N, S, H
        permutation_q = OnnxModel.get_node_attribute(transpose_q, "perm")
        if (not isinstance(permutation_q, list)) or permutation_q != [0, 2, 1, 3]:
            print("##Permutation Q")
            return False

        # B, S, N, H => B, N, H, S
        permutation_k = OnnxModel.get_node_attribute(transpose_k, "perm")
        if (not isinstance(permutation_k, list)) or permutation_k != [0, 2, 3, 1]:
            print("##Permutation K")
            return False

        # B, S, N, H => B, N, S, H
        permutation_v = OnnxModel.get_node_attribute(transpose_v, "perm")
        if (not isinstance(permutation_v, list)) or permutation_v != [0, 2, 1, 3]:
            print("##Permutation V")
            return False

        input_projection_nodes = self.model.match_parent_path(
            split_qkv,
            ["Reshape", "Add", "MatMul"],
            [0, 0, None],
        )
        if input_projection_nodes is None:
            print("##9")
            return False
        reshape_in, add_in, matmul_in = input_projection_nodes
        q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_in, normalize_node, True)
        if q_num_heads <= 0:
            logger.debug("fuse_attention: failed to detect num_heads")
            print("##10")
            return False

        # Add a shape to convert 4D BxSxNxH to 3D BxSxD, which is required by MHA operator.
        new_dims_name = "bsnh_to_bsd_reshape_dims"
        new_dims = self.model.get_initializer(new_dims_name)
        if new_dims is None:
            new_dims = numpy_helper.from_array(
                np.array([0, 0, -1], dtype="int64"), name=new_dims_name
            )
            self.model.add_initializer(new_dims, self.this_graph_name)
        reshape_q_name = self.model.create_node_name("Reshape")
        reshape_q = helper.make_node(
            "Reshape",
            inputs=[transpose_q.input[0], new_dims_name],
            outputs=[transpose_q.input[0] + "_BSD"],
            name=reshape_q_name,
        )
        self.nodes_to_add.append(reshape_q)
        self.node_name_to_graph_name[reshape_q.name] = self.this_graph_name

        # MHA do not accept BNHS format, so we transpose K to BNSH format instead.
        transpose_k_bnsh = transpose_q               
        transpose_k_bnsh.input[0] = transpose_k.input[0]
        transpose_k_bnsh.output[0] = transpose_k.output[0] + "_BNSH"

        print(f"Found MHA: {q_num_heads=} {q_hidden_size=}")

        # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
        # TODO: use new output instead since its shape is changed.
        new_node = self.create_mha_node(reshape_q, transpose_k_bnsh, transpose_v, q_num_heads, output=transpose_out.output[0])
        if new_node is None:
            print("##11")
            return False

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name
        self.nodes_to_remove.extend([transpose_out])

        # Use prune graph to remove nodes since they are shared by all attention nodes.
        self.prune_graph = True
        return True
    
    def match_sam_encoder_attention_subgraph(self, node_after_output_projection, input_index=None):
        """Match SDPA pattern in SAM2 enconder.*"""

        # nodes of output projection and the second MatMul in SDPA.
        out_nodes = self.model.match_parent_path(
            node_after_output_projection,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [input_index, None, None, 0, 0],
        )

        if out_nodes is None:
            print("##1")
            return None

        (_, _, reshape_out, transpose_out, matmul_qk_v) = out_nodes

        # Split and Reshape is for packed QKV
        v_nodes = self.model.match_parent_path(matmul_qk_v, ["Transpose", "Squeeze", "Split", "Reshape"], [1, 0, 0, 0])
        if v_nodes is None:
            logger.debug("failed to match v path")
            print("##2")
            return None
        (transpose_v, _, split_qkv, reshape_qkv) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qk_v, ["Softmax", "MatMul"], [0, 0])
        if qk_nodes is not None:
            (_softmax_qk, matmul_qk) = qk_nodes
        else:
            logger.debug("failed to match qk path")
            print("##3")
            return None

        q_nodes = self.model.match_parent_path(
            matmul_qk, ["Mul", "Transpose", "Squeeze", "Split"], [0, None, 0, 0]
        )
        if q_nodes is None:
            q_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Mul", "Transpose", "Reshape", "Transpose", "MaxPool", "Transpose", "Reshape", "Squeeze", "Split"],
                [0, None, 0, 0, 0, 0, 0, 0, 0]
            )
            if q_nodes is None:
                logger.debug("failed to match q path")
                print("##4")
                return None

        if q_nodes[-1] != split_qkv:
            print("##5")
            return None
        transpose_q = q_nodes[1]

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Mul", "Transpose", "Squeeze", "Split"], [1, None, 0, 0]
        )
        if k_nodes is None:
            logger.debug("failed to match k path")
            print("##6")
            return None
        if k_nodes[-1] != split_qkv:
            print("##7")
            return None
        (mul_k, transpose_k, _squeeze_k, _) = k_nodes

        # The scalar for Q and K is sqrt(1.0/sqrt(head_size)). Might be constant
        # mul_q_nodes = self.model.match_parent_path(
        #     mul_q,
        #     ["Sqrt", "Cast", "Div", "Sqrt", "Cast", "Slice", "Shape", "Transpose", "Squeeze", "Split"],
        #     [None, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        # )
        # if mul_q_nodes is None or mul_q_nodes[-1] != split_qkv:
        #     logger.debug("failed to match mul_q path")
        #     return None

        return reshape_out, transpose_out, split_qkv, transpose_q, transpose_k, transpose_v
    
    def create_mha_node(
        self,
        reshape_q: NodeProto,
        transpose_k: NodeProto,
        transpose_v: NodeProto,
        num_heads: int,
        output: str,
    ) -> Union[NodeProto, None]:
        """Create an Attention node.

        Args:
            reshape_q (NodeProto): Reshape node for Q, output is 3D BxSxNH format
            transpose_k (NodeProto): Transpose node for K, output is BNSH format
            transpose_v (NodeProto): Transpose node for V, output is BNSH format
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """

        attention_node_name = self.model.create_node_name("MultiHeadAttention")

        inputs = [
            reshape_q.output[0],
            transpose_k.output[0],
            transpose_v.output[0],
        ]

        attention_node = helper.make_node(
            "MultiHeadAttention",
            inputs=inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        counter_name = (
            "MultiHeadAttention ({})".format(
                "self attention"
            )
        )
        self.increase_counter(counter_name)
        return attention_node