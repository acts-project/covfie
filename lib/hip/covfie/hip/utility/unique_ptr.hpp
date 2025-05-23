/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>

#include <hip/hip_runtime.h>

#include <covfie/hip/error_check.hpp>

namespace covfie::utility::hip {
template <typename T>
struct device_deleter {
    static_assert(
        std::is_trivially_destructible_v<std::remove_extent_t<T>>,
        "Allocation pointer type must be trivially destructible."
    );

public:
    void operator()(void * p) const
    {
        hipErrorCheck(hipFree(p));
    }
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, device_deleter<T>>;
}
