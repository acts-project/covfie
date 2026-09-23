/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/utility/checked_size.hpp>
#include <covfie/core/utility/nd_map.hpp>
#include <covfie/core/utility/nd_size.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    concepts::vector_descriptor _input_vector_t,
    concepts::field_backend _storage_t>
struct strided {
    using this_t = strided<_input_vector_t, _storage_t>;
    static constexpr bool is_initial = false;

    using backend_t = _storage_t;

    using contravariant_input_t =
        covfie::vector::array_vector_d<_input_vector_t>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t = covariant_input_t;

    using coordinate_t = typename contravariant_input_t::vector_t;
    using array_t = backend_t;

    using configuration_t = utility::nd_size<
        contravariant_input_t::dimensions,
        typename contravariant_input_t::scalar_t>;

    // Keep the binary dimensions independent of the in-memory index type.
    using io_configuration_t =
        utility::nd_size<contravariant_input_t::dimensions, std::size_t>;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020010;

    static std::size_t configuration_size(const configuration_t & sizes)
    {
        // An empty grid has no offsets, regardless of the other extents.
        for (auto size : sizes) {
            if (size == 0) {
                return 0;
            }
        }

        std::size_t count = 1;
        for (auto size : sizes) {
            const auto extent = utility::checked_size<std::size_t>(size);
            if (count > std::numeric_limits<std::size_t>::max() / extent) {
                throw std::overflow_error(
                    "Strided field volume overflows size_t."
                );
            }
            count *= extent;
        }
        return count;
    }

    template <
        concepts::is_nd_size_of_dim<contravariant_input_t::dimensions> config_t>
    static configuration_t checked_configuration(const config_t & conf)
    {
        configuration_t sizes;
        for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i) {
            sizes[i] =
                utility::checked_size<typename contravariant_input_t::scalar_t>(
                    conf[i]
                );
        }

        const auto count = configuration_size(sizes);
        if (count != 0) {
            // Coordinate arithmetic and the child index must both represent
            // every flattened offset. The count itself can be one larger.
            utility::checked_size<typename contravariant_input_t::scalar_t>(
                count - 1
            );
            utility::checked_size<typename contravariant_output_t::scalar_t>(
                count - 1
            );
        }
        if constexpr (concepts::is_nd_size_of_dim<
                          typename backend_t::configuration_t,
                          1>)
        {
            // Check before implicitly narrowing a one-dimensional child
            // configuration, which would hide overflow from its constructor.
            utility::checked_size<
                typename backend_t::configuration_t::value_type>(count);
        }
        return sizes;
    }

    template <typename T>
    static std::unique_ptr<
        std::decay_t<typename backend_t::covariant_output_t::vector_t>[]>
    make_strided_copy(const T & other)
    {
        configuration_t sizes =
            checked_configuration(other.get_configuration());
        std::unique_ptr<
            std::decay_t<typename backend_t::covariant_output_t::vector_t>[]>
            res = std::make_unique<std::decay_t<
                typename backend_t::covariant_output_t::vector_t>[]>(
                configuration_size(sizes)
            );
        typename T::parent_t::non_owning_data_t nother(other);

        utility::nd_map<decltype(sizes)>(
            [&sizes, &nother, &res](decltype(sizes) t) {
                typename contravariant_input_t::scalar_t idx = 0;

                for (std::size_t k = 0; k < contravariant_input_t::dimensions;
                     ++k) {
                    typename contravariant_input_t::scalar_t tmp = t[k];

                    for (std::size_t l = k + 1;
                         l < contravariant_input_t::dimensions;
                         ++l)
                    {
                        tmp *= sizes[l];
                    }

                    idx += tmp;
                }

                for (std::size_t i = 0; i < covariant_output_t::dimensions; ++i)
                {
                    if constexpr (covariant_output_t::dimensions == 1 && !std::unsigned_integral<std::decay_t<decltype(t)>>)
                    {
                        res[idx][i] = nother.at(t[0])[i];
                    } else {
                        res[idx][i] = nother.at(t)[i];
                    }
                }
            },
            sizes
        );

        return res;
    }

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t()
            : owning_data_t(configuration_t{})
        {
        }

        owning_data_t(const owning_data_t &) = default;
        owning_data_t(owning_data_t &&) = default;
        owning_data_t & operator=(const owning_data_t &) = default;
        owning_data_t & operator=(owning_data_t &&) = default;

        template <typename T>
        requires(std::convertible_to<
                 typename T::parent_t::configuration_t,
                 configuration_t> &&
                     std::constructible_from<
                         typename backend_t::owning_data_t,
                         std::size_t,
                         std::add_rvalue_reference_t<std::unique_ptr<std::decay_t<
                             typename backend_t::covariant_output_t::
                                 vector_t>[]>>>) explicit owning_data_t(const T &
                                                                            o)
            : m_sizes(checked_configuration(o.get_configuration()))
            , m_storage(configuration_size(m_sizes), make_strided_copy(o))
        {
        }

        // Intercept other scalar types before implicit array conversion can
        // truncate a dimension on the way into the configuration overloads.
        template <concepts::is_nd_size_of_dim<contravariant_input_t::dimensions>
                      config_t>
        requires(std::constructible_from<typename backend_t::owning_data_t, std::size_t> || std::constructible_from<typename backend_t::owning_data_t, utility::nd_size<1, std::size_t>>) explicit owning_data_t(
            const config_t & conf
        )
            : owning_data_t(checked_configuration(conf))
        {
        }

        explicit owning_data_t(configuration_t conf
        ) requires(std::constructible_from<typename backend_t::owning_data_t, std::size_t> && !std::constructible_from<typename backend_t::owning_data_t, utility::nd_size<1, std::size_t>>)
            : m_sizes(checked_configuration(conf))
            , m_storage(configuration_size(m_sizes))
        {
        }

        explicit owning_data_t(configuration_t conf)
            requires(std::constructible_from<
                     typename backend_t::owning_data_t,
                     utility::nd_size<1, std::size_t>>)
            : m_sizes(checked_configuration(conf))
            , m_storage(utility::nd_size<1, std::size_t>{
                  configuration_size(m_sizes)})
        {
        }

        template <concepts::is_nd_size_of_dim<contravariant_input_t::dimensions>
                      config_t>
        explicit owning_data_t(
            const config_t & c, typename backend_t::owning_data_t && b
        )
            : owning_data_t(checked_configuration(c), std::move(b))
        {
        }

        explicit owning_data_t(
            const configuration_t & c, typename backend_t::owning_data_t && b
        )
            : m_sizes(checked_configuration(c))
            , m_storage(std::forward<typename backend_t::owning_data_t>(b))
        {
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

        static owning_data_t read_binary(std::istream & fs)
        {
            utility::read_io_header(fs, IO_MAGIC_HEADER);

            auto sizes = checked_configuration(
                utility::read_binary<io_configuration_t>(fs)
            );
            auto be = backend_t::owning_data_t::read_binary(fs);

            utility::read_io_footer(fs, IO_MAGIC_HEADER);

            return owning_data_t(sizes, std::move(be));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            utility::write_io_header(fs, IO_MAGIC_HEADER);

            const io_configuration_t sizes = o.m_sizes;
            fs.write(reinterpret_cast<const char *>(&sizes), sizeof(sizes));

            backend_t::owning_data_t::write_binary(fs, o.m_storage);

            utility::write_io_footer(fs, IO_MAGIC_HEADER);
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

        COVFIE_HOST_DEVICE typename covariant_output_t::vector_t
        at(coordinate_t c) const
        {
            typename contravariant_input_t::scalar_t idx = 0;

#ifndef NDEBUG
            for (std::size_t i = 0; i < contravariant_input_t::dimensions; ++i)
            {
                assert(c[i] < m_sizes[i]);
            }
#endif

            for (std::size_t k = 0; k < contravariant_input_t::dimensions; ++k)
            {
                typename contravariant_input_t::scalar_t tmp = c[k];

                for (std::size_t l = k + 1;
                     l < contravariant_input_t::dimensions;
                     ++l)
                {
                    tmp *=
                        static_cast<typename contravariant_input_t::scalar_t>(
                            m_sizes[l]
                        );
                }

                idx += tmp;
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
