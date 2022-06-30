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

#include <cmath>
#include <cstddef>
#include <fstream>
#include <type_traits>

#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend::transformer::interpolator {
template <
    CONSTRAINT(concepts::field_backend) _backend_t,
    CONSTRAINT(concepts::floating_point_input_scalar) _input_scalar_type =
        float>
struct _nearest_neighbour {
    using backend_t = _backend_t;

    using contravariant_input_t =
        covfie::vector::array_vector_d<covfie::vector::vector_d<
            _input_scalar_type,
            backend_t::contravariant_input_t::dimensions>>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t = covariant_input_t;

    struct configuration_data_t {
    };

    struct owning_data_t {
        template <typename... Args>
        owning_data_t(configuration_data_t, Args... args)
            : m_backend(std::forward<Args>(args)...)
        {
        }

        template <
            typename T,
            std::enable_if_t<
                std::is_convertible_v<
                    decltype(std::declval<T>().m_backend),
                    typename backend_t::owning_data_t>,
                bool> = true>
        owning_data_t(const T & o)
            : m_backend(o.m_backend)
        {
        }

        template <
            typename T,
            std::enable_if_t<
                std::is_convertible_v<
                    decltype(std::declval<T>()),
                    typename backend_t::owning_data_t>,
                bool> = true>
        owning_data_t(const T & o)
            : m_backend(o)
        {
        }

        owning_data_t(std::ifstream & fs)
            : m_backend(fs)
        {
        }

        void dump(std::ofstream & fs) const
        {
            m_backend.dump(fs);
        }

        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & src)
            : m_backend(src.m_backend)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            typename contravariant_output_t::vector_t nc;

            for (std::size_t i = 0; i < contravariant_output_t::dimensions; ++i)
            {
                nc[i] = std::lround(c[i]);
            }

            return m_backend.at(nc);
        }

        typename backend_t::non_owning_data_t m_backend;
    };
};

template <CONSTRAINT(concepts::field_backend) _backend_tc>
using nearest_neighbour = _nearest_neighbour<_backend_tc>;
}
