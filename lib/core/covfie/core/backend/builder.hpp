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
    using contravariant_input_t = _input_vector_t;
    using contravariant_output_t = std::tuple<>;
    using covariant_input_t = std::tuple<>;
    using covariant_output_t = vector::output_reference_array_vector<
        typename _output_vector_t::scalar_t,
        _output_vector_t::dimensions>;

    using index_t = typename contravariant_input_t::scalar_t;
    using ndsize_t = typename contravariant_input_t::
        template vector_tc<index_t, contravariant_input_t::dimensions>;

    using output_t = covariant_output_t;
    using integral_coordinate_t = typename contravariant_input_t::
        template vector_tc<index_t, contravariant_input_t::dimensions>;

    struct configuration_data_t {
        ndsize_t m_sizes;
    };

    struct owning_data_t {
        template <typename... Args>
        owning_data_t(configuration_data_t conf)
            : m_ptr(std::make_unique<std::remove_reference_t<
                        typename covariant_output_t::vector_t>[]>(
                  std::accumulate(
                      std::begin(conf.m_sizes),
                      std::end(conf.m_sizes),
                      1,
                      std::multiplies<std::size_t>()
                  )
              ))
            , m_sizes(conf.m_sizes)
        {
        }

        owning_data_t(std::ifstream & fs)
        {
            for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i)
            {
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

            m_ptr = std::make_unique<std::remove_reference_t<
                typename covariant_output_t::vector_t>[]>(total_elements);

            fs.read(
                reinterpret_cast<char *>(m_ptr.get()),
                total_elements * sizeof(std::remove_reference_t<
                                        typename covariant_output_t::vector_t>)
            );
        }

        typename covariant_output_t::vector_t at(integral_coordinate_t c) const
        {
            index_t idx = 0;

            for (std::size_t k = 0; k < contravariant_input_t::dimensions; ++k)
            {
                index_t tmp = c[k];

                for (std::size_t l = k + 1;
                     l < contravariant_input_t::dimensions;
                     ++l) {
                    tmp *= m_sizes[l];
                }

                idx += tmp;
            }

            return m_ptr[idx];
        }

        void dump(std::ofstream & fs) const
        {
            for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i)
            {
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
                total_elements * sizeof(std::remove_reference_t<
                                        typename covariant_output_t::vector_t>)
            );
        }

        std::unique_ptr<
            std::remove_reference_t<typename covariant_output_t::vector_t>[]>
            m_ptr;
        ndsize_t m_sizes;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr.get())
            , m_sizes(o.m_sizes)
        {
        }

        typename covariant_output_t::vector_t at(integral_coordinate_t c) const
        {
            index_t idx = 0;

            for (std::size_t k = 0; k < contravariant_input_t::dimensions; ++k)
            {
                index_t tmp = c[k];

                for (std::size_t l = k + 1;
                     l < contravariant_input_t::dimensions;
                     ++l) {
                    tmp *= m_sizes[l];
                }

                idx += tmp;
            }

            typename covariant_output_t::vector_t r = m_ptr[idx];

            return r;
        }

        typename decltype(owning_data_t::m_ptr)::pointer m_ptr;
        ndsize_t m_sizes;
    };
};

template <
    CONSTRAINT(concepts::input_vector) _input_vector_t,
    CONSTRAINT(concepts::output_vector) _output_vector_t>
using builder = _builder<_input_vector_t, _output_vector_t>;
}
