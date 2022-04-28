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

#include <covfie/core/backend/builder.hpp>
#include <covfie/core/qualifiers.hpp>

namespace covfie::backend::transformer::interpolator {
template <typename _backend_tc>
struct _nearest_neighbour {

    using backend_t = _backend_tc;
    static constexpr std::size_t output_dimensions =
        backend_t::output_dimensions;
    static constexpr std::size_t coordinate_dimensions =
        backend_t::coordinate_dimensions;

    using coordinate_t = std::array<float, coordinate_dimensions>;
    using integral_coordinate_t = typename backend_t::integral_coordinate_t;
    using coordinate_scalar_t = typename backend_t::coordinate_scalar_t;
    using output_t = typename backend_t::output_t;

    struct configuration_data_t {
    };

    struct owning_data_t {
        template <typename... Args>
        owning_data_t(configuration_data_t, Args... args)
            : m_backend(std::forward<Args>(args)...)
        {
        }

        // TODO: This needs SFINAE guards.
        template <typename T>
        owning_data_t(const T & o)
            : m_backend(o.m_backend)
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

        COVFIE_DEVICE output_t at(coordinate_t c) const
        {
            typename backend_t::integral_coordinate_t nc;

            for (std::size_t i = 0; i < coordinate_dimensions; ++i) {
                nc[i] = std::lround(c[i]);
            }

            return m_backend.at(nc);
        }

        typename backend_t::non_owning_data_t m_backend;
    };
};

template <typename _backend_tc>
using nearest_neighbour = _nearest_neighbour<_backend_tc>;
}
