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

#include <covfie/core/backend/vector/input.hpp>
#include <covfie/core/backend/vector/output.hpp>
#include <covfie/core/concepts.hpp>

namespace covfie::backend {
template <
    CONSTRAINT(concepts::input_vector) _input_vector_t,
    CONSTRAINT(concepts::output_vector) _output_vector_t>
struct _constant {
    using input_vector_t = _input_vector_t;
    using output_vector_t = _output_vector_t;

    static constexpr std::size_t coordinate_dimensions =
        input_vector_t::dimensions;

    using index_t = std::size_t;

    using coordinate_t = typename input_vector_t::vector_t;
    using integral_coordinate_t =
        std::array<std::size_t, coordinate_dimensions>;
    using output_scalar_t = typename output_vector_t::output_scalar_t;
    using output_t = typename output_vector_t::vector_t;

    struct owning_data_t;

    struct configuration_data_t {
        output_t m_value;
    };

    struct owning_data_t {
        template <typename... Args>
        owning_data_t(configuration_data_t conf)
            : m_value(conf.m_value)
        {
        }

        output_t m_value;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_value(o.m_value)
        {
        }

        output_t at(coordinate_t) const
        {
            return m_value;
        }

        output_t m_value;
    };
};

template <
    std::size_t input_dimensions,
    CONSTRAINT(concepts::output_vector) _output_vector_t>
using constant =
    _constant<vector::input_vector<float, input_dimensions>, _output_vector_t>;
}
