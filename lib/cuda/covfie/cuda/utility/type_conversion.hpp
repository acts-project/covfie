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

#include <cuda_runtime.h>

#include <covfie/core/vector.hpp>

namespace covfie::utility {
template <typename T>
struct to_cuda_channel_t {
};

template <>
struct to_cuda_channel_t<covfie::vector::float1> {
    using type = float;
};

template <>
struct to_cuda_channel_t<covfie::vector::float2> {
    using type = ::float2;
};

template <>
struct to_cuda_channel_t<covfie::vector::float3> {
    using type = ::float4;
};

template <>
struct to_cuda_channel_t<covfie::vector::float4> {
    using type = ::float4;
};
}
