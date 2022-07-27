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

#include <algorithm>
#include <memory>
#include <numeric>
#include <type_traits>

#include <x86intrin.h>

#include <covfie/core/backend/initial/array.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/utility/nd_map.hpp>
#include <covfie/core/utility/nd_size.hpp>
#include <covfie/core/utility/numeric.hpp>
#include <covfie/core/utility/tuple.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend::layout {
template <
    CONSTRAINT(concepts::vector_descriptor) _input_vector_t,
    CONSTRAINT(concepts::field_backend) _storage_t,
    bool use_bmi2 = true>
struct morton {
    using this_t = morton<_input_vector_t, _storage_t>;
    static constexpr bool is_initial = false;

    using backend_t = _storage_t;

    using contravariant_input_t =
        covfie::vector::array_vector_d<_input_vector_t>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t = covariant_input_t;

    using coordinate_t = typename contravariant_input_t::vector_t;
    using array_t = backend_t;

    using configuration_t = utility::nd_size<contravariant_input_t::dimensions>;

    template <typename T>
    static std::unique_ptr<
        std::decay_t<typename backend_t::covariant_output_t::vector_t>[]>
    make_morton_copy(const T & other) {
        configuration_t sizes = other.get_configuration();
        std::unique_ptr<
            std::decay_t<typename backend_t::covariant_output_t::vector_t>[]>
            res = std::make_unique<std::decay_t<
                typename backend_t::covariant_output_t::vector_t>[]>(
                utility::ipow(
                    utility::round_pow2(
                        *std::max_element(sizes.begin(), sizes.end())
                    ),
                    contravariant_input_t::dimensions
                )
            );
        typename T::parent_t::non_owning_data_t nother(other);

        using tuple_t = decltype(std::tuple_cat(
            std::declval<
                std::array<std::size_t, contravariant_input_t::dimensions>>()
        ));

        utility::nd_map<tuple_t>(
            [&nother, &res](tuple_t t) {
                coordinate_t c = utility::to_array(t);
                std::size_t idx = 0;

                for (std::size_t i = 0;
                     i < (64 / contravariant_input_t::dimensions);
                     ++i) {
                    for (std::size_t j = 0;
                         j < contravariant_input_t::dimensions;
                         ++j) {
                        idx |= (c[j] << (i + j)) &
                               (1UL
                                << (contravariant_input_t::dimensions * i + j));
                    }
                }

                for (std::size_t i = 0; i < covariant_output_t::dimensions; ++i)
                {
                    res[idx][i] = nother.at(c)[i];
                }
            },
            std::tuple_cat(sizes)
        );

        return res;
    }

    struct owning_data_t {
        using parent_t = this_t;

        template <
            typename T,
            typename B = backend_t,
            std::enable_if_t<
                std::is_same_v<
                    typename T::parent_t::configuration_t,
                    configuration_t>,
                bool> = true,
            std::enable_if_t<
                std::is_constructible_v<
                    typename B::owning_data_t,
                    std::size_t,
                    std::add_rvalue_reference_t<std::unique_ptr<std::decay_t<
                        typename B::covariant_output_t::vector_t>[]>>>,
                bool> = true>
        explicit owning_data_t(const T & o)
            : m_sizes(o.get_configuration())
            , m_storage(
                  utility::ipow(
                      utility::round_pow2(
                          *std::max_element(m_sizes.begin(), m_sizes.end())
                      ),
                      contravariant_input_t::dimensions
                  ),
                  make_morton_copy(o)
              )
        {
        }

        template <
            typename B = backend_t,
            std::enable_if_t<
                std::is_constructible_v<typename B::owning_data_t, std::size_t>,
                bool> = true>
        explicit owning_data_t(configuration_t conf)
            : m_sizes(conf)
            , m_storage(std::accumulate(
                  std::begin(m_sizes),
                  std::end(m_sizes),
                  1,
                  std::multiplies<std::size_t>()
              ))
        {
        }

        template <
            typename... Args,
            typename B = backend_t,
            std::enable_if_t<
                !std::
                    is_constructible_v<typename B::owning_data_t, std::size_t>,
                bool> = true>
        explicit owning_data_t(configuration_t conf, Args... args)
            : m_sizes(conf)
            , m_storage(std::forward<Args>(args)...)
        {
        }

        explicit owning_data_t(std::ifstream & fs)
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

        typename backend_t::owning_data_t & get_backend(void)
        {
            return m_storage;
        }

        const typename backend_t::owning_data_t & get_backend(void) const
        {
            return m_storage;
        }

        configuration_t get_configuration(void) const
        {
            return m_sizes;
        }

        configuration_t m_sizes;
        typename backend_t::owning_data_t m_storage;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & o)
            : m_sizes(o.m_sizes)
            , m_storage(o.m_storage)
        {
        }

        static constexpr std::size_t get_mask(std::size_t i)
        {
            std::size_t r = 0;

            for (std::size_t i = 0; i < 64; ++i) {
                r |=
                    (i % contravariant_input_t::dimensions == 0 ? (1UL << i)
                                                                : 0UL);
            }

            return r << i;
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t at(coordinate_t c
        ) const
        {
            std::size_t idx = 0;

            if constexpr (use_bmi2) {
                for (std::size_t i = 0; i < contravariant_input_t::dimensions;
                     ++i) {
                    idx |= _pdep_u64(c[i], get_mask(i));
                }
            } else {
                for (std::size_t i = 0;
                     i < ((CHAR_BIT *
                           sizeof(typename contravariant_output_t::scalar_t)) /
                          contravariant_input_t::dimensions);
                     ++i)
                {
                    for (std::size_t j = 0;
                         j < contravariant_input_t::dimensions;
                         ++j) {
                        idx |= (c[j] << (i + j)) &
                               (1UL
                                << (contravariant_input_t::dimensions * i + j));
                    }
                }
            }

            return m_storage.at({idx});
        }

        typename backend_t::non_owning_data_t & get_backend(void)
        {
            return m_storage;
        }

        const typename backend_t::non_owning_data_t & get_backend(void) const
        {
            return m_storage;
        }

        configuration_t m_sizes;
        typename backend_t::non_owning_data_t m_storage;
    };
};
}
