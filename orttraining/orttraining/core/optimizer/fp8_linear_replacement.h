// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_FP8_TRAINING

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class Fp8LinearReplacement : public GraphTransformer {
 public:
  Fp8LinearReplacement(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("Fp8LinearReplacement", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime

#endif  // ENABLE_FP8_TRAINING
