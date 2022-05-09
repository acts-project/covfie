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
template <typename _backend_tc, typename _input_scalar_type = float>
struct linear {
    using input_scalar_type = _input_scalar_type;
    using backend_t = _backend_tc;
    static constexpr std::size_t output_dimensions =
        backend_t::output_dimensions;
    static constexpr std::size_t coordinate_dimensions =
        backend_t::coordinate_dimensions;

    using coordinate_t = std::array<
        input_scalar_type,
        std::tuple_size<typename backend_t::coordinate_t>::value>;
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

        COVFIE_DEVICE output_t at(coordinate_t coord) const
        {
            if constexpr (std::tuple_size_v<coordinate_t> == 3) {
                std::size_t i = std::lround(std::floor(coord[0]));
                std::size_t j = std::lround(std::floor(coord[1]));
                std::size_t k = std::lround(std::floor(coord[2]));

                input_scalar_type a = std::fmod(coord[0], 1.f);
                input_scalar_type b = std::fmod(coord[1], 1.f);
                input_scalar_type c = std::fmod(coord[2], 1.f);

                output_t rv;

                for (std::size_t q = 0; q < output_dimensions; ++q) {
                    rv[q] =
                        (1. - a) * (1. - b) * (1. - c) *
                            m_backend.at({i, j, k})[q] +
                        a * (1. - b) * (1. - c) *
                            m_backend.at({i + 1, j, k})[q] +
                        (1. - a) * (b) * (1. - c) *
                            m_backend.at({i, j + 1, k})[q] +
                        a * b * (1. - c) * m_backend.at({i + 1, j + 1, k})[q] +
                        (1. - a) * (1. - b) * c *
                            m_backend.at({i, j, k + 1})[q] +
                        a * (1. - b) * c * m_backend.at({i + 1, j, k + 1})[q] +
                        (1. - a) * b * c * m_backend.at({i, j + 1, k + 1})[q] +
                        a * b * c * m_backend.at({i + 1, j + 1, k + 1})[q];
                }

                return rv;
            }
        }

        typename backend_t::non_owning_data_t m_backend;
    };
};
}
