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

#include <covfie/core/backend/datatype/datatype.hpp>
#include <covfie/core/concepts.hpp>

namespace covfie::backend {
template <
    std::size_t _input_dimensions,
    CONSTRAINT(concepts::input_scalar) _input_scalar_type,
    CONSTRAINT(concepts::datatype) _datatype_t,
    template <typename, std::size_t>
    typename _array_tc>
struct _constant {
    using datatype_t = _datatype_t;

    static constexpr std::size_t coordinate_dimensions = _input_dimensions;
    static constexpr std::size_t output_dimensions = datatype_t::dimensions;

    static constexpr bool support_access_global = true;
    static constexpr bool support_access_integral = false;

    static constexpr bool support_host = true;

    using index_t = std::size_t;

    using coordinate_scalar_t = _input_scalar_type;
    using coordinate_t = _array_tc<_input_scalar_type, coordinate_dimensions>;
    using integral_coordinate_t = _array_tc<std::size_t, coordinate_dimensions>;
    using output_scalar_t = datatype_t::output_scalar_t;
    using output_t = _array_tc<output_scalar_t, output_dimensions>;

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

template <std::size_t input_dimensions, std::size_t _output_dimensions>
using constant = _constant<
    input_dimensions,
    float,
    datatype::datatype<float, _output_dimensions>,
    std::array>;
}
