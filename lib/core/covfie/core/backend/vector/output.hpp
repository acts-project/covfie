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

#include <covfie/core/concepts.hpp>

namespace covfie::backend::vector {
template <
    CONSTRAINT(concepts::output_scalar) _scalar_type,
    std::size_t _dimensions>
struct output_vector {
    static constexpr std::size_t dimensions = _dimensions;
    using output_scalar_t = _scalar_type;
    using vector_t = std::array<output_scalar_t, dimensions>;
};

namespace output {
using float1 = output_vector<float, 1>;
using float2 = output_vector<float, 2>;
using float3 = output_vector<float, 3>;
}
}