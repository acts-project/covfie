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

#include <covfie/core/definitions.hpp>

#if __cpp_concepts >= 201907L
namespace covfie::concepts {
template <typename T>
concept field_backend = requires
{
    typename T::contravariant_input_t;
    typename T::contravariant_output_t;
    typename T::covariant_input_t;
    typename T::covariant_output_t;
};

template <typename T>
concept output_scalar = true;

template <typename T>
concept input_scalar = true;

template <typename T>
concept integral_input_scalar = true;

template <typename T>
concept floating_point_input_scalar = true;

template <typename T>
concept layout = true;

template <typename T>
concept storage = true;

template <typename T>
concept input_vector = true;

template <typename T>
concept output_vector = true;
}
#endif
