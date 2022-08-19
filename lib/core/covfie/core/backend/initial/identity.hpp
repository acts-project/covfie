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

#include <fstream>
#include <type_traits>
#include <variant>

#include <covfie/core/concepts.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <CONSTRAINT(concepts::vector_descriptor) _vector_t>
struct identity {
    using this_t = identity<_vector_t>;
    static constexpr bool is_initial = true;

    using contravariant_input_t = covfie::vector::array_vector_d<_vector_t>;
    using covariant_output_t = covfie::vector::array_vector_d<_vector_t>;

    static_assert(
        contravariant_input_t::dimensions == covariant_output_t::dimensions,
        "Identity backend requires input and output to have identical "
        "dimensionality."
    );
    static_assert(
        std::is_constructible_v<
            typename contravariant_input_t::scalar_t,
            typename covariant_output_t::scalar_t>,
        "Identity backend requires type of input to be convertible to type of "
        "output."
    );

    struct owning_data_t;

    using configuration_t = std::monostate;

    struct owning_data_t {
        using parent_t = this_t;

        template <typename... Args>
        explicit owning_data_t(configuration_t)
        {
        }

        explicit owning_data_t(std::ifstream &)
        {
        }

        configuration_t get_configuration() const
        {
            return {};
        }

        void dump(std::ofstream &) const
        {
        }
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t &)
        {
        }

        typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            typename covariant_output_t::vector_t rv;

            for (std::size_t i = 0ul; i < contravariant_input_t::dimensions;
                 ++i) {
                rv[i] = c[i];
            }

            return rv;
        }
    };
};
}
