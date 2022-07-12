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

#include <cstddef>
#include <fstream>
#include <type_traits>

#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend::transformer::interpolator {
template <
    CONSTRAINT(concepts::field_backend) _backend_t,
    typename _input_scalar_type = float>
struct linear {
    using this_t = linear<_backend_t, _input_scalar_type>;
    static constexpr bool is_initial = false;

    template <typename new_backend>
    using reapply = linear<new_backend, _input_scalar_type>;

    using input_scalar_type = _input_scalar_type;
    using backend_t = _backend_t;

    using contravariant_input_t =
        covfie::vector::array_vector_d<covfie::vector::vector_d<
            _input_scalar_type,
            backend_t::contravariant_input_t::dimensions>>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t =
        covfie::vector::array_vector_d<typename covariant_input_t::vector_d>;

    static_assert(
        std::is_object_v<typename covariant_output_t::vector_t>,
        "Covariant input type of linear interpolator must be an object type."
    );

    struct configuration_t {
    };

    struct owning_data_t {
        using parent_t = this_t;

        template <typename... Args>
        explicit owning_data_t(configuration_t, Args... args)
            : m_backend(std::forward<Args>(args)...)
        {
        }

        template <
            typename T,
            std::enable_if_t<
                std::is_same_v<
                    typename T::parent_t::template reapply<backend_t>,
                    this_t>,
                bool> = true>
        explicit owning_data_t(const T & o)
            : m_backend(o.m_backend)
        {
        }

        explicit owning_data_t(const typename backend_t::owning_data_t & o)
            : m_backend(o)
        {
        }

        explicit owning_data_t(std::ifstream & fs)
            : m_backend(fs)
        {
        }

        void dump(std::ofstream & fs) const
        {
            m_backend.dump(fs);
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
                std::size_t i = std::lround(std::floor(coord[0]));
                std::size_t j = std::lround(std::floor(coord[1]));
                std::size_t k = std::lround(std::floor(coord[2]));

                input_scalar_type a = std::fmod(coord[0], 1.f);
                input_scalar_type b = std::fmod(coord[1], 1.f);
                input_scalar_type c = std::fmod(coord[2], 1.f);

                input_scalar_type ra = static_cast<input_scalar_type>(1.) - a;
                input_scalar_type rb = static_cast<input_scalar_type>(1.) - b;
                input_scalar_type rc = static_cast<input_scalar_type>(1.) - c;

                std::remove_reference_t<typename covariant_input_t::vector_t>
                    pc[8];

                for (std::size_t n = 0; n < 8; ++n) {
                    pc[n] = m_backend.at(
                        {i + ((n & 4) ? 1 : 0),
                         j + ((n & 2) ? 1 : 0),
                         k + ((n & 1) ? 1 : 0)}
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
            }

            return {};
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
