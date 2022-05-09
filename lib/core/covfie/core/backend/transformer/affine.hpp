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
template <CONSTRAINT(concepts::field_backend) _backend_tc>
struct affine {
    using backend_t = _backend_tc;
    static constexpr std::size_t output_dimensions =
        backend_t::output_dimensions;
    static constexpr std::size_t coordinate_dimensions =
        backend_t::coordinate_dimensions;

    using coordinate_t = typename backend_t::coordinate_t;

    using output_t = typename backend_t::output_t;

    struct owning_data_t;

    struct configuration_data_t {
        coordinate_t m_offsets;
        coordinate_t m_scales;
    };

    struct owning_data_t {
        template <typename... Args>
        owning_data_t(configuration_data_t conf, Args... args)
            : m_offsets(conf.m_offsets)
            , m_scales(conf.m_scales)
            , m_backend(std::forward<Args>(args)...)
        {
        }

        // TODO: This needs SFINAE guards.
        template <typename T>
        owning_data_t(const T & o)
            : m_offsets(o.m_offsets)
            , m_scales(o.m_scales)
            , m_backend(o.m_backend)
        {
        }

        owning_data_t(std::ifstream & fs)
            : m_offsets(utility::read_binary<decltype(m_offsets)>(fs))
            , m_scales(utility::read_binary<decltype(m_scales)>(fs))
            , m_backend(fs)
        {
        }

        void dump(std::ofstream & fs) const
        {
            for (std::size_t i = 0; i < std::tuple_size<coordinate_t>::value;
                 ++i) {
                fs.write(
                    reinterpret_cast<const char *>(&m_offsets[i]),
                    sizeof(typename decltype(m_offsets)::value_type)
                );
            }

            for (std::size_t i = 0; i < std::tuple_size<coordinate_t>::value;
                 ++i) {
                fs.write(
                    reinterpret_cast<const char *>(&m_scales[i]),
                    sizeof(typename decltype(m_scales)::value_type)
                );
            }

            m_backend.dump(fs);
        }

        coordinate_t m_offsets;
        coordinate_t m_scales;
        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & src)
            : m_offsets(src.m_offsets)
            , m_scales(src.m_scales)
            , m_backend(src.m_backend)
        {
        }

        COVFIE_DEVICE output_t at(coordinate_t c) const
        {
            typename backend_t::coordinate_t nc;

            for (std::size_t i = 0; i < std::tuple_size<coordinate_t>::value;
                 ++i) {
                nc[i] = (c[i] - m_offsets[i]) / m_scales[i];
            }

            return m_backend.at(nc);
        }

        coordinate_t m_offsets;
        coordinate_t m_scales;
        typename backend_t::non_owning_data_t m_backend;
    };
};
}
