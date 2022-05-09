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

#include <array>
#include <fstream>
#include <memory>
#include <numeric>

#include <covfie/core/backend/vector/input.hpp>
#include <covfie/core/backend/vector/output.hpp>
#include <covfie/core/concepts.hpp>

namespace covfie::backend {
template <
    CONSTRAINT(concepts::input_vector) _input_vector_t,
    CONSTRAINT(concepts::output_vector) _output_vector_t>
struct _builder {
    using input_vector_t = _input_vector_t;
    using output_vector_t = _output_vector_t;

    static constexpr std::size_t coordinate_dimensions =
        input_vector_t::dimensions;

    using index_t = typename input_vector_t::scalar_t;
    using ndsize_t = std::array<index_t, coordinate_dimensions>;
    using output_scalar_t = typename output_vector_t::output_scalar_t;

    using coordinate_scalar_t = index_t;
    using value_t = output_scalar_t[output_vector_t::dimensions];

    using coordinate_t = std::array<index_t, coordinate_dimensions>;
    using output_t = std::add_lvalue_reference_t<value_t>;
    using integral_coordinate_t = std::array<index_t, coordinate_dimensions>;

    struct configuration_data_t {
        ndsize_t m_sizes;
    };

    struct owning_data_t {
        template <typename... Args>
        owning_data_t(configuration_data_t conf)
            : m_ptr(std::make_unique<value_t[]>(std::accumulate(
                  std::begin(conf.m_sizes),
                  std::end(conf.m_sizes),
                  1,
                  std::multiplies<std::size_t>()
              )))
            , m_sizes(conf.m_sizes)
        {
        }

        owning_data_t(std::ifstream & fs)
        {
            for (std::size_t i = 0; i < coordinate_dimensions; ++i) {
                fs.read(
                    reinterpret_cast<char *>(&m_sizes[i]),
                    sizeof(typename decltype(m_sizes)::value_type)
                );
            }

            std::size_t total_elements = std::accumulate(
                std::begin(m_sizes),
                std::end(m_sizes),
                1,
                std::multiplies<std::size_t>()
            );

            m_ptr = std::make_unique<value_t[]>(total_elements);

            fs.read(
                reinterpret_cast<char *>(m_ptr.get()),
                total_elements * sizeof(value_t)
            );
        }

        output_t at(integral_coordinate_t c) const
        {
            index_t idx = 0;

            for (std::size_t k = 0; k < coordinate_dimensions; ++k) {
                index_t tmp = c[k];

                for (std::size_t l = k + 1; l < coordinate_dimensions; ++l) {
                    tmp *= m_sizes[l];
                }

                idx += tmp;
            }

            return m_ptr[idx];
        }

        void dump(std::ofstream & fs) const
        {
            for (std::size_t i = 0; i < coordinate_dimensions; ++i) {
                fs.write(
                    reinterpret_cast<const char *>(&m_sizes[i]),
                    sizeof(typename decltype(m_sizes)::value_type)
                );
            }

            std::size_t total_elements = std::accumulate(
                std::begin(m_sizes),
                std::end(m_sizes),
                1,
                std::multiplies<std::size_t>()
            );

            fs.write(
                reinterpret_cast<const char *>(m_ptr.get()),
                total_elements * sizeof(value_t)
            );
        }

        std::unique_ptr<value_t[]> m_ptr;
        ndsize_t m_sizes;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr.get())
            , m_sizes(o.m_sizes)
        {
        }

        output_t at(integral_coordinate_t c) const
        {
            index_t idx = 0;

            for (std::size_t k = 0; k < coordinate_dimensions; ++k) {
                index_t tmp = c[k];

                for (std::size_t l = k + 1; l < coordinate_dimensions; ++l) {
                    tmp *= m_sizes[l];
                }

                idx += tmp;
            }

            return m_ptr[idx];
        }

        typename decltype(owning_data_t::m_ptr)::pointer m_ptr;
        ndsize_t m_sizes;
    };
};

template <
    std::size_t _input_dimensions,
    CONSTRAINT(concepts::output_vector) _output_vector_t,
    CONSTRAINT(concepts::integral_input_scalar) _input_scalar_type =
        std::size_t>
using builder = _builder<
    vector::input_vector<_input_scalar_type, _input_dimensions>,
    _output_vector_t>;
}
