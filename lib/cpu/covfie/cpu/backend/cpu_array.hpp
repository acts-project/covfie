/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <covfie/core/backend/layout/strided.hpp>
#include <covfie/core/backend/vector/input.hpp>
#include <covfie/cpu/backend/storage/c_array.hpp>

namespace covfie::backend {
template <std::size_t input_dimensions, typename _datatype_t>
using cpu_array = layout::strided<
    vector::input_vector<std::size_t, input_dimensions>,
    storage::c_array<_datatype_t>>;
}
