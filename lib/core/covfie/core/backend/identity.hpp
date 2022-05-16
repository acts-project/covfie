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

#include <type_traits>

#include <covfie/core/backend/vector/input.hpp>
#include <covfie/core/backend/vector/output.hpp>
#include <covfie/core/concepts.hpp>

namespace covfie::backend {
template <
    CONSTRAINT(concepts::input_vector) _input_vector_t,
    CONSTRAINT(concepts::output_vector) _output_vector_t>
struct _identity {
    using input_vector_t = _input_vector_t;
    using output_vector_t = _output_vector_t;

    using coordinate_t = typename input_vector_t::vector_t;
    using output_t = typename output_vector_t::vector_t;

    static_assert(
        input_vector_t::dimensions == output_vector_t::dimensions,
        "Identity backend requires input and output to have identical "
        "dimensionality."
    );
    static_assert(
        std::is_constructible_v<
            typename input_vector_t::scalar_t,
            typename output_vector_t::scalar_t>,
        "Identity backend requires type of input to be convertible to type of "
        "output."
    );

    struct owning_data_t;

    struct configuration_data_t {
    };

    struct owning_data_t {
        template <typename... Args>
        owning_data_t(configuration_data_t)
        {
        }
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t &)
        {
        }

        output_t at(coordinate_t c) const
        {
            typename output_vector_t::vector_t rv;

            for (std::size_t i = 0ul; i < input_vector_t::dimensions; ++i) {
                rv[i] = c[i];
            }

            return rv;
        }
    };
};

template <
    CONSTRAINT(concepts::input_vector) _input_vector_t,
    CONSTRAINT(concepts::output_vector) _output_vector_t>
using identity = _identity<_input_vector_t, _output_vector_t>;
}
