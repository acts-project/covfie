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
#include <iostream>

#include <covfie/core/algebra/affine.hpp>
#include <covfie/core/algebra/vector.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>

namespace covfie::backend {
template <CONSTRAINT(concepts::field_backend) _backend_t>
struct affine {
    using this_t = affine<_backend_t>;
    static constexpr bool is_initial = false;

    using backend_t = _backend_t;

    using contravariant_input_t = typename backend_t::contravariant_input_t;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t = typename backend_t::covariant_output_t;

    using matrix_t = algebra::affine<
        contravariant_input_t::dimensions,
        typename contravariant_input_t::scalar_t>;

    struct owning_data_t;

    using configuration_t = matrix_t;

    struct owning_data_t {
        using parent_t = this_t;

        template <typename... Args>
        explicit owning_data_t(configuration_t conf, Args... args)
            : m_transform(conf)
            , m_backend(std::forward<Args>(args)...)
        {
        }

        owning_data_t(const owning_data_t &) = default;

        template <
            typename T,
            std::enable_if_t<
                std::is_same_v<
                    typename T::parent_t::configuration_t,
                    configuration_t>,
                bool> = true>
        explicit owning_data_t(const T & o)
            : m_transform(o.m_transform)
            , m_backend(o.m_backend)
        {
        }

        explicit owning_data_t(std::istream & fs)
            : m_transform(utility::read_binary<decltype(m_transform)>(fs))
            , m_backend(fs)
        {
        }

        void dump(std::ostream & fs) const
        {
            fs.write(
                reinterpret_cast<const char *>(&m_transform),
                sizeof(decltype(m_transform))
            );

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
            return m_transform;
        }

        matrix_t m_transform;
        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & src)
            : m_transform(src.m_transform)
            , m_backend(src.m_backend)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            covfie::algebra::vector<
                contravariant_input_t::dimensions,
                typename contravariant_input_t::scalar_t>
                v;

            for (std::size_t i = 0; i < contravariant_output_t::dimensions; ++i)
            {
                v(i) = c[i];
            }

            covfie::algebra::vector<
                contravariant_input_t::dimensions,
                typename contravariant_input_t::scalar_t>
                nv = m_transform * v;

            typename contravariant_output_t::vector_t nc;

            for (std::size_t i = 0; i < contravariant_output_t::dimensions; ++i)
            {
                nc[i] = nv(i);
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

        matrix_t m_transform;
        typename backend_t::non_owning_data_t m_backend;
    };
};
}
