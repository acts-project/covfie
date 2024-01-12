/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022-2023 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <variant>

#include <covfie/core/concepts.hpp>
#include <covfie/core/definitions.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    CONSTRAINT(concepts::field_backend) _backend_t,
    CONSTRAINT(concepts::vector_descriptor) _input_vector_d = covfie::vector::
        vector_d<float, _backend_t::contravariant_input_t::dimensions>>
struct linear {
    using this_t = linear<_backend_t, _input_vector_d>;
    static constexpr bool is_initial = false;

    using input_scalar_type = typename _input_vector_d::type;
    using backend_t = _backend_t;

    using contravariant_input_t =
        covfie::vector::array_vector_d<_input_vector_d>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t =
        covfie::vector::array_vector_d<typename covariant_input_t::vector_d>;

    static_assert(
        std::is_floating_point_v<typename _input_vector_d::type>,
        "Nearest neighbour interpolation contravariant input must have a "
        "floating point scalar type."
    );
    static_assert(
        _input_vector_d::size == backend_t::contravariant_input_t::dimensions,
        "Nearest neighbour interpolation contravariant input must have the "
        "same size as the backend contravariant input."
    );
    static_assert(
        std::is_object_v<typename covariant_output_t::vector_t>,
        "Covariant input type of linear interpolator must be an object type."
    );

    using configuration_t = std::monostate;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020005;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;

        template <typename... Args>
        explicit owning_data_t(configuration_t, Args... args)
            : m_backend(std::forward<Args>(args)...)
        {
        }

        template <
            typename T,
            std::enable_if_t<
                std::is_same_v<
                    typename T::parent_t::configuration_t,
                    configuration_t>,
                bool> = true>
        explicit owning_data_t(const T & o)
            : m_backend(o.m_backend)
        {
        }

        explicit owning_data_t(const typename backend_t::owning_data_t & o)
            : m_backend(o)
        {
        }

        explicit owning_data_t(
            const configuration_t &, typename backend_t::owning_data_t && b
        )
            : m_backend(std::forward<typename backend_t::owning_data_t>(b))
        {
        }

        typename backend_t::owning_data_t & get_backend(void)
        {
            return m_backend;
        }

        const typename backend_t::owning_data_t & get_backend(void) const
        {
            return m_backend;
        }

        configuration_t get_configuration(void) const
        {
            return {};
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            auto be = decltype(m_backend)::read_binary(fs);

            return owning_data_t(configuration_t{}, std::move(be));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            decltype(m_backend)::write_binary(fs, o.m_backend);
        }

        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & src)
            : m_backend(src.m_backend)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t coord) const
        {
            if constexpr (covariant_output_t::dimensions == 3) {
                std::size_t i = static_cast<std::size_t>(coord[0]);
                std::size_t j = static_cast<std::size_t>(coord[1]);
                std::size_t k = static_cast<std::size_t>(coord[2]);

                input_scalar_type a = coord[0] - std::floor(coord[0]);
                input_scalar_type b = coord[1] - std::floor(coord[1]);
                input_scalar_type c = coord[2] - std::floor(coord[2]);

                input_scalar_type ra = static_cast<input_scalar_type>(1.) - a;
                input_scalar_type rb = static_cast<input_scalar_type>(1.) - b;
                input_scalar_type rc = static_cast<input_scalar_type>(1.) - c;

                std::remove_reference_t<typename covariant_input_t::vector_t>
                    pc[8];

                for (std::size_t n = 0; n < 8; ++n) {
                    pc[n] = m_backend.at(
                        {static_cast<typename decltype(m_backend
                         )::parent_t::contravariant_input_t::scalar_t>(
                             i + ((n & 4) ? 1 : 0)
                         ),
                         static_cast<typename decltype(m_backend
                         )::parent_t::contravariant_input_t::scalar_t>(
                             j + ((n & 2) ? 1 : 0)
                         ),
                         static_cast<typename decltype(m_backend
                         )::parent_t::contravariant_input_t::scalar_t>(
                             k + ((n & 1) ? 1 : 0)
                         )}
                    );
                }

                typename covariant_output_t::vector_t rv;

                for (std::size_t q = 0; q < covariant_output_t::dimensions; ++q)
                {
                    rv[q] = ra * rb * rc * pc[0][q] + ra * rb * c * pc[1][q] +
                            ra * b * rc * pc[2][q] + ra * b * c * pc[3][q] +
                            a * rb * rc * pc[4][q] + a * rb * c * pc[5][q] +
                            a * b * rc * pc[6][q] + a * b * c * pc[7][q];
                }

                return rv;
            } else {
                return {};
            }
        }

        typename backend_t::non_owning_data_t & get_backend(void)
        {
            return m_backend;
        }

        const typename backend_t::non_owning_data_t & get_backend(void) const
        {
            return m_backend;
        }

        typename backend_t::non_owning_data_t m_backend;
    };
};
}
