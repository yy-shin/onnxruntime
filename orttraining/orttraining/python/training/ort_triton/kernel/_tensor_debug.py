# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import itertools

import torch
from onnx import helper

from onnxruntime.training.ortmodule import register_graph_optimizer


_DEBUG_NODES = {
    # "SoftmaxCrossEntropyLossInternalGrad": [(None, [0])],
}

_DEBUG_ARGS = [
    # "/_original_module/Sub_output_0",
    # "/_original_module/Sub_output_0_grad",
]


def tensor_debug(input, **kwargs):
    torch.cuda.synchronize()
    arg_name = kwargs.get("arg_name", "")
    print(f"#### Arg name: {arg_name} ####")
    print(f"Shape: {input.size()}")
    print(input)
    print(f"Is finite: {torch.all(torch.isfinite(input))}")
    print(f"#### End of {arg_name} ####")
    return input


def _get_type(graph, arg: str):
    value_infos = [
        value_info
        for value_info in itertools.chain(graph.input, graph.output, graph.value_info)
        if value_info.name == arg
    ]
    if len(value_infos) > 0:
        return value_infos[0].type.tensor_type.elem_type
    initializers = [initializer for initializer in graph.initializer if initializer.name == arg]
    if len(initializers) > 0:
        return initializers[0].data_type
    return None


@register_graph_optimizer(devices="cuda")
def transform_tensor_debug(graph):
    args = set(_DEBUG_ARGS)
    for node in graph.node:
        for config in _DEBUG_NODES.get(node.op_type, []):
            node_name, outputs = config
            if node_name is not None and node.name != node_name:
                continue
            args.extend([node.output[idx] for idx in outputs])
    triton_nodes = []
    value_infos = []
    for arg in args:
        dummy_arg = helper.make_tensor_value_info("dummy_" + arg, _get_type(graph, arg), None)
        value_infos.append(dummy_arg)
        triton_node = helper.make_node(
            "TritonOp",
            [arg],
            [dummy_arg.name],
            "tensor_debug_" + arg,
            None,
            "com.microsoft",
            func_name="tensor_debug",
            arg_name=arg,
        )
        triton_nodes.append(triton_node)
    all_nodes = [node for node in graph.node]
    for node in triton_nodes:
        all_nodes.append(node)  # noqa: PERF402
    graph.ClearField("node")
    graph.node.extend(all_nodes)
    graph.value_info.extend(value_infos)
