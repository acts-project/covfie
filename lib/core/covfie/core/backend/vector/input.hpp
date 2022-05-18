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

#include <array>

#include <covfie/core/concepts.hpp>

namespace covfie::backend::vector {
template <
    CONSTRAINT(concepts::output_scalar) _scalar_t,
    std::size_t _dimensions,
    template <typename, std::size_t> typename _vector_tc = std::array>
struct input_vector {
    static constexpr std::size_t dimensions = _dimensions;

    template <typename T>
    using reapply = input_vector<T, dimensions>;

    using scalar_t = _scalar_t;
    using vector_t = _vector_tc<scalar_t, dimensions>;
};

namespace input {
using float1 = input_vector<float, 1>;
using float2 = input_vector<float, 2>;
using float3 = input_vector<float, 3>;

using unsigned1 = input_vector<unsigned, 1>;
using unsigned2 = input_vector<unsigned, 2>;
using unsigned3 = input_vector<unsigned, 3>;

using ulong1 = input_vector<unsigned long, 1>;
using ulong2 = input_vector<unsigned long, 2>;
using ulong3 = input_vector<unsigned long, 3>;
}
}
