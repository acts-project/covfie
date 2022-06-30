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
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    CONSTRAINT(concepts::input_vector) _input_vector_t,
    CONSTRAINT(concepts::output_vector) _output_vector_t>
struct constant {
    using contravariant_input_t =
        typename covfie::vector::array_vector_d<_input_vector_t>;
    using contravariant_output_t = std::tuple<>;
    using covariant_input_t = std::tuple<>;
    using covariant_output_t =
        typename covfie::vector::array_vector_d<_output_vector_t>;

    struct owning_data_t;

    struct configuration_data_t {
        typename covariant_output_t::vector_t m_value;
    };

    struct owning_data_t {
        template <typename... Args>
        owning_data_t(configuration_data_t conf)
            : m_value(conf.m_value)
        {
        }

        owning_data_t(std::ifstream & fs)
            : m_value(
                  utility::read_binary<typename covariant_output_t::vector_t>(fs
                  )
              )
        {
        }

        void dump(std::ofstream & fs) const
        {
            fs.write(
                reinterpret_cast<const char *>(&m_value),
                sizeof(decltype(m_value))
            );
        }

        typename covariant_output_t::vector_t m_value;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_value(o.m_value)
        {
        }

        typename covariant_output_t::vector_t
            at(typename contravariant_input_t::vector_t) const
        {
            return m_value;
        }

        typename covariant_output_t::vector_t m_value;
    };
};
}
