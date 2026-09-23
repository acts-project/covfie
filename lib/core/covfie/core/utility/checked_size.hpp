/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2026 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace covfie::utility {
template <std::integral Index, std::integral Size>
Index checked_size(Size size)
{
    if (!std::in_range<Index>(size) || !std::in_range<std::size_t>(size)) {
        throw std::overflow_error("Array size is not representable.");
    }
    return static_cast<Index>(size);
}
}
