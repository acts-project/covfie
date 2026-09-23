/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>

#include <covfie/core/array.hpp>

namespace covfie::utility {
template <std::size_t N, typename index_t>
using nd_size = array::array<index_t, N>;
}
