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

namespace covfie::backend::datatype {
template <typename _scalar_type, std::size_t _dimensions>
struct datatype {
    static constexpr std::size_t dimensions = _dimensions;
    using output_scalar_t = _scalar_type;
};

using float1 = datatype<float, 1>;
using float2 = datatype<float, 2>;
using float3 = datatype<float, 3>;
}
