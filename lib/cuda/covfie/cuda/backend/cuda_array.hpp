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
#include <covfie/core/backend/modular.hpp>
#include <covfie/cuda/backend/storage/cuda_device_array.hpp>

namespace covfie::backend {
template <std::size_t input_dimensions, typename _datatype_t>
using cuda_array = _modular<
    layout::strided<input_dimensions>,
    storage::cuda_device_array,
    _datatype_t>;
}
