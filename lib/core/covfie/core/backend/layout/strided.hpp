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
#include <type_traits>

#include <covfie/core/backend/storage/array.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/utility/nd_map.hpp>
#include <covfie/core/utility/tuple.hpp>

namespace covfie::backend::layout {
template <
    CONSTRAINT(concepts::input_vector) _input_vector_t,
    CONSTRAINT(concepts::field_backend) _storage_t>
struct strided {
    using storage_t = _storage_t;

    using contravariant_input_t = _input_vector_t;
    using contravariant_output_t = typename storage_t::contravariant_input_t;
    using covariant_input_t = typename storage_t::covariant_output_t;
    using covariant_output_t = covariant_input_t;

    using ndsize_t = typename contravariant_input_t::vector_t;
    using coordinate_t = typename contravariant_input_t::vector_t;
    using array_t = storage_t;

    struct configuration_data_t {
        ndsize_t m_sizes;
    };

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

            using tuple_t = decltype(std::tuple_cat(
                std::declval<
                    std::array<std::size_t, contravariant_input_t::dimensions>>(
                )
            ));

            utility::nd_map<tuple_t>(
                [&sizes, &sv, &other](tuple_t t) {
                    coordinate_t c = utility::to_array(t);
                    typename contravariant_input_t::scalar_t idx = 0;

                    for (std::size_t k = 0;
                         k < contravariant_input_t::dimensions;
                         ++k) {
                        typename contravariant_input_t::scalar_t tmp = c[k];

                        for (std::size_t l = k + 1;
                             l < contravariant_input_t::dimensions;
                             ++l) {
                            tmp *= sizes[l];
                        }

                        idx += tmp;
                    }

                    for (std::size_t i = 0; i < covariant_output_t::dimensions;
                         ++i) {
                        sv[idx][i] = other.at(c)[i];
                    }
                },
                std::tuple_cat(sizes)
            );

            return tmp;
        }

        owning_data_t(configuration_data_t conf)
            : m_sizes(conf.m_sizes)
            , m_storage(std::accumulate(
                  std::begin(m_sizes),
                  std::end(m_sizes),
                  1,
                  std::multiplies<std::size_t>()
              ))
        {
        }

        template <
            typename T,
            std::enable_if_t<
                std::is_convertible_v<
                    decltype(std::declval<T>().m_sizes),
                    ndsize_t>,
                bool> = true>
        owning_data_t(const T & o)
            : m_sizes(o.m_sizes)
            , m_storage(make_data(m_sizes, o))
        {
        }

        owning_data_t(std::ifstream & fs)
            : m_sizes(utility::read_binary<decltype(m_sizes)>(fs))
            , m_storage(fs)
        {
        }

        void dump(std::ofstream & fs) const
        {
            fs.write(
                reinterpret_cast<const char *>(&m_sizes),
                sizeof(decltype(m_sizes))
            );

            m_storage.dump(fs);
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

        COVFIE_DEVICE typename covariant_output_t::vector_t at(coordinate_t c
        ) const
        {
            typename contravariant_input_t::scalar_t idx = 0;

            for (std::size_t k = 0; k < contravariant_input_t::dimensions; ++k)
            {
                typename contravariant_input_t::scalar_t tmp = c[k];

                for (std::size_t l = k + 1;
                     l < contravariant_input_t::dimensions;
                     ++l) {
                    tmp *= m_sizes[l];
                }

                idx += tmp;
            }

            return m_storage[idx];
        }

        ndsize_t m_sizes;
        typename storage_t::non_owning_data_t m_storage;
    };
};
}
