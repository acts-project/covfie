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

#include <numeric>

#include <covfie/core/backend/storage/array.hpp>
#include <covfie/core/concepts.hpp>

namespace covfie::backend::layout {
template <
    CONSTRAINT(concepts::input_vector) _input_vector_t,
    CONSTRAINT(concepts::storage) _storage_t>
struct strided {
    using input_vector_t = _input_vector_t;
    static constexpr std::size_t coordinate_dimensions =
        input_vector_t::dimensions;

    using storage_t = _storage_t;
    using output_vector_t = typename storage_t::output_vector_t;
    using output_t = typename output_vector_t::vector_t;

    using index_t = typename input_vector_t::scalar_t;
    using ndsize_t = std::array<index_t, coordinate_dimensions>;
    using coordinate_t = std::array<index_t, coordinate_dimensions>;

    using array_t = storage::array<output_vector_t, std::size_t>;

    struct owning_data_t {
        template <typename T>
        static typename array_t::owning_data_t
        make_data(ndsize_t sizes, const T & other)
        {
            typename array_t::owning_data_t tmp(std::accumulate(
                std::begin(sizes),
                std::end(sizes),
                1,
                std::multiplies<std::size_t>()
            ));
            typename array_t::non_owning_data_t sv(tmp);

            if constexpr (coordinate_dimensions == 1) {
                for (std::size_t x = 0; x < sizes[0]; ++x) {
                    coordinate_t c{x};
                    index_t idx = 0;

                    for (std::size_t k = 0; k < coordinate_dimensions; ++k) {
                        index_t tmp = c[k];

                        for (std::size_t l = k + 1; l < coordinate_dimensions;
                             ++l) {
                            tmp *= sizes[l];
                        }

                        idx += tmp;
                    }

                    for (std::size_t i = 0;
                         i < std::tuple_size<output_t>::value;
                         ++i) {
                        sv[idx][i] = other.at({x})[i];
                    }
                }
            } else if constexpr (coordinate_dimensions == 2) {
                for (std::size_t x = 0; x < sizes[0]; ++x) {
                    for (std::size_t y = 0; y < sizes[1]; ++y) {
                        coordinate_t c{x, y};
                        index_t idx = 0;

                        for (std::size_t k = 0; k < coordinate_dimensions; ++k)
                        {
                            index_t tmp = c[k];

                            for (std::size_t l = k + 1;
                                 l < coordinate_dimensions;
                                 ++l) {
                                tmp *= sizes[l];
                            }

                            idx += tmp;
                        }

                        for (std::size_t i = 0;
                             i < std::tuple_size<output_t>::value;
                             ++i) {
                            sv[idx][i] = other.at({x, y})[i];
                        }
                    }
                }
            } else if constexpr (coordinate_dimensions == 3) {
                for (std::size_t x = 0; x < sizes[0]; ++x) {
                    for (std::size_t y = 0; y < sizes[1]; ++y) {
                        for (std::size_t z = 0; z < sizes[2]; ++z) {
                            coordinate_t c{x, y, z};
                            index_t idx = 0;

                            for (std::size_t k = 0; k < coordinate_dimensions;
                                 ++k) {
                                index_t tmp = c[k];

                                for (std::size_t l = k + 1;
                                     l < coordinate_dimensions;
                                     ++l) {
                                    tmp *= sizes[l];
                                }

                                idx += tmp;
                            }

                            for (std::size_t i = 0;
                                 i < std::tuple_size<output_t>::value;
                                 ++i) {
                                sv[idx][i] = other.at({x, y, z})[i];
                            }
                        }
                    }
                }
            }

            return tmp;
        }

        owning_data_t(ndsize_t sizes)
            : m_sizes(sizes)
            , m_storage(std::accumulate(
                  std::begin(m_sizes),
                  std::end(m_sizes),
                  1,
                  std::multiplies<std::size_t>()
              ))
        {
        }

        template <typename T>
        owning_data_t(const T & o)
            : m_sizes(o.m_sizes)
            , m_storage(make_data(m_sizes, o))
        {
        }

        ndsize_t m_sizes;
        typename storage_t::owning_data_t m_storage;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_sizes(o.m_sizes)
            , m_storage(o.m_storage)
        {
        }

        COVFIE_DEVICE output_t at(coordinate_t c) const
        {
            index_t idx = 0;

            for (std::size_t k = 0; k < coordinate_dimensions; ++k) {
                index_t tmp = c[k];

                for (std::size_t l = k + 1; l < coordinate_dimensions; ++l) {
                    tmp *= m_sizes[l];
                }

                idx += tmp;
            }

            typename storage_t::value_t & res = m_storage[idx];
            output_t rv;

            for (std::size_t i = 0; i < output_vector_t::dimensions; ++i) {
                rv[i] = res[i];
            }

            return rv;
        }

        ndsize_t m_sizes;
        typename storage_t::non_owning_data_t m_storage;
    };
};
}
