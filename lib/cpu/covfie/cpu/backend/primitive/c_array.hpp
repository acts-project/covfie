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

#include <memory>

#include <covfie/core/backend/primitive/array.hpp>

namespace covfie::backend {
template <
    concepts::vector_descriptor _output_vector_t,
    typename _index_t = std::size_t>
using c_array = array<_output_vector_t, _index_t>;
}
