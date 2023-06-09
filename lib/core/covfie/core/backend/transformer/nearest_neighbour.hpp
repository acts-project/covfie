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
#include <iostream>
#include <type_traits>
#include <variant>

#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    CONSTRAINT(concepts::field_backend) _backend_t,
    typename _input_scalar_type = float>
struct _nearest_neighbour {
    using this_t = _nearest_neighbour<_backend_t, _input_scalar_type>;
    static constexpr bool is_initial = false;

    using backend_t = _backend_t;

    using contravariant_input_t =
        covfie::vector::array_vector_d<covfie::vector::vector_d<
            _input_scalar_type,
            backend_t::contravariant_input_t::dimensions>>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t = covariant_input_t;

    using configuration_t = std::monostate;

    struct owning_data_t {
        using parent_t = this_t;

        template <typename... Args>
        explicit owning_data_t(configuration_t, Args... args)
            : m_backend(std::forward<Args>(args)...)
        {
        }

        owning_data_t(const owning_data_t &) = default;

        template <
            typename T,
            typename... Args,
            std::enable_if_t<
                std::is_same_v<
                    typename T::parent_t::configuration_t,
                    std::monostate>,
                bool> = true>
        explicit owning_data_t(const T & o)
            : m_backend(o.get_backend())
        {
        }

        explicit owning_data_t(std::istream & fs)
            : m_backend(fs)
        {
        }

        void dump(std::ostream & fs) const
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
        at(typename contravariant_input_t::vector_t c) const
        {
            typename contravariant_output_t::vector_t nc;

            for (std::size_t i = 0; i < contravariant_output_t::dimensions; ++i)
            {
                nc[i] = std::lrintf(c[i]);
            }

            return m_backend.at(nc);
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

template <CONSTRAINT(concepts::field_backend) _backend_tc>
using nearest_neighbour = _nearest_neighbour<_backend_tc>;
}
