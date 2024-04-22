// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <absl/synchronization/mutex.h>
namespace onnxruntime {
using absl::Mutex = absl::Mutex;
using OrtCondVar = absl::CondVar;
};  // namespace onnxruntime
