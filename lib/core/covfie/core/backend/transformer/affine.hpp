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

#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>

namespace covfie::backend::transformer {
template <CONSTRAINT(concepts::field_backend) _backend_t>
struct affine {
    using this_t = affine<_backend_t>;
    static constexpr bool is_initial = false;

    template <typename new_backend>
    using reapply = affine<new_backend>;

    using backend_t = _backend_t;

    using contravariant_input_t = typename backend_t::contravariant_input_t;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t = typename backend_t::covariant_output_t;

    struct owning_data_t;

    struct configuration_t {
        configuration_t(
            const typename contravariant_input_t::vector_t & offsets,
            const typename contravariant_input_t::vector_t & scales
        )
            : m_offsets(offsets)
            , m_scales(scales)
        {
        }

        template <typename T>
        configuration_t(const T & o)
            : m_offsets(o.m_offsets)
            , m_scales(o.m_scales)
        {
        }

        typename contravariant_input_t::vector_t m_offsets;
        typename contravariant_input_t::vector_t m_scales;
    };

    struct owning_data_t {
        using parent_t = this_t;

        template <typename... Args>
        explicit owning_data_t(configuration_t conf, Args... args)
            : m_offsets(conf.m_offsets)
            , m_scales(conf.m_scales)
            , m_backend(std::forward<Args>(args)...)
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
            : m_offsets(o.m_offsets)
            , m_scales(o.m_scales)
            , m_backend(o.m_backend)
        {
        }

        explicit owning_data_t(std::ifstream & fs)
            : m_offsets(utility::read_binary<decltype(m_offsets)>(fs))
            , m_scales(utility::read_binary<decltype(m_scales)>(fs))
            , m_backend(fs)
        {
        }

        void dump(std::ofstream & fs) const
        {
            for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i)
            {
                fs.write(
                    reinterpret_cast<const char *>(&m_offsets[i]),
                    sizeof(typename decltype(m_offsets)::value_type)
                );
            }

            for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i)
            {
                fs.write(
                    reinterpret_cast<const char *>(&m_scales[i]),
                    sizeof(typename decltype(m_scales)::value_type)
                );
            }

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

        typename contravariant_input_t::vector_t m_offsets;
        typename contravariant_input_t::vector_t m_scales;
        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & src)
            : m_offsets(src.m_offsets)
            , m_scales(src.m_scales)
            , m_backend(src.m_backend)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            typename contravariant_output_t::vector_t nc;

            for (std::size_t i = 0; i < contravariant_output_t::dimensions; ++i)
            {
                nc[i] = (c[i] - m_offsets[i]) / m_scales[i];
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

        typename contravariant_input_t::vector_t m_offsets;
        typename contravariant_input_t::vector_t m_scales;
        typename backend_t::non_owning_data_t m_backend;
    };
};
}
