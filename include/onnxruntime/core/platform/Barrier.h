// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Dmitry Vyukov <dvyukov@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#pragma once
#include <assert.h>

#include "core/common/spin_pause.h"
#include "core/platform/ort_mutex.h"
#include <absl/synchronization/barrier.h>
#include <absl/synchronization/notification.h>

namespace onnxruntime {
using Notification = absl::Notification;

}  // namespace onnxruntime
