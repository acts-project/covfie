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

#include <covfie/core/backend/vector/output.hpp>
#include <covfie/core/concepts.hpp>

namespace covfie::backend {
template <
    std::size_t _input_dimensions,
    CONSTRAINT(concepts::input_scalar) _input_scalar_type,
    CONSTRAINT(concepts::output_vector) _output_vector_t>
struct _constant {
    using output_vector_t = _output_vector_t;

    static constexpr std::size_t coordinate_dimensions = _input_dimensions;

    using index_t = std::size_t;

    using coordinate_scalar_t = _input_scalar_type;
    using coordinate_t = std::array<_input_scalar_type, coordinate_dimensions>;
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
using constant = _constant<input_dimensions, float, _output_vector_t>;
}
