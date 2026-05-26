/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>
#include <variant>

#include <covfie/core/concepts.hpp>
#include <covfie/core/definitions.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    concepts::field_backend _backend_t,
    concepts::vector_descriptor _input_vector_d = covfie::vector::
        vector_d<float, _backend_t::contravariant_input_t::dimensions>>
struct linear {
    using this_t = linear<_backend_t, _input_vector_d>;
    static constexpr bool is_initial = false;

    using input_scalar_type = typename _input_vector_d::type;
    using backend_t = _backend_t;

    using contravariant_input_t =
        covfie::vector::array_vector_d<_input_vector_d>;
    using contravariant_output_t = typename backend_t::contravariant_input_t;
    using covariant_input_t = typename backend_t::covariant_output_t;
    using covariant_output_t =
        covfie::vector::array_vector_d<typename covariant_input_t::vector_d>;

    static_assert(
        std::is_floating_point_v<typename _input_vector_d::type>,
        "Linear interpolation contravariant input must have a "
        "floating point scalar type."
    );
    static_assert(
        std::is_floating_point_v<typename covariant_input_t::scalar_t>,
        "Linear interpolation covariant input must have a "
        "floating point scalar type."
    );
    static_assert(
        _input_vector_d::size == backend_t::contravariant_input_t::dimensions,
        "Linear interpolation contravariant input must have the "
        "same size as the backend contravariant input."
    );
    static_assert(
        std::is_object_v<typename covariant_output_t::vector_t>,
        "Covariant input type of linear interpolator must be an object type."
    );

    using configuration_t = std::monostate;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020005;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;

        template <typename... Args>
        explicit owning_data_t(configuration_t, Args... args)
            : m_backend(std::forward<Args>(args)...)
        {
        }

        template <typename T>
        requires(std::same_as<
                 typename T::parent_t::configuration_t,
                 configuration_t>) explicit owning_data_t(const T & o)
            : m_backend(o.m_backend)
        {
        }

        explicit owning_data_t(const typename backend_t::owning_data_t & o)
            : m_backend(o)
        {
        }

        explicit owning_data_t(
            const configuration_t &, typename backend_t::owning_data_t && b
        )
            : m_backend(std::forward<typename backend_t::owning_data_t>(b))
        {
        }

        template <typename... Args>
        explicit owning_data_t(parameter_pack<configuration_t, Args...> && args)
            : m_backend(std::move(args.xs))
        {
        }

        explicit owning_data_t(parameter_pack<owning_data_t> && conf)
            : owning_data_t(std::move(conf.x))
        {
        }

        typename backend_t::owning_data_t & get_backend(void)
        {
            return m_backend;
        }

        const typename backend_t::owning_data_t & get_backend(void) const
        {
            return m_backend;
        }

        configuration_t get_configuration(void) const
        {
            return {};
        }

        static owning_data_t read_binary(std::istream & fs)
        {
            auto be = decltype(m_backend)::read_binary(fs);

            return owning_data_t(configuration_t{}, std::move(be));
        }

        static void write_binary(std::ostream & fs, const owning_data_t & o)
        {
            decltype(m_backend)::write_binary(fs, o.m_backend);
        }

        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & src)
            : m_backend(src.m_backend)
        {
        }

        COVFIE_HOST_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t coord) const
        {
            typename contravariant_output_t::scalar_t
                is[contravariant_output_t::dimensions];
            input_scalar_type ls[contravariant_output_t::dimensions],
                rs[contravariant_output_t::dimensions];
            typename covariant_output_t::vector_t rv;

            for (unsigned int i = 0; i < contravariant_output_t::dimensions;
                 ++i)
            {
                is[contravariant_output_t::dimensions - (i + 1)] =
                    static_cast<typename contravariant_output_t::scalar_t>(
                        coord[i]
                    );
                ls[contravariant_output_t::dimensions - (i + 1)] =
                    coord[i] - std::trunc(coord[i]);
                rs[contravariant_output_t::dimensions - (i + 1)] =
                    static_cast<input_scalar_type>(1.) -
                    ls[contravariant_output_t::dimensions - (i + 1)];
            }

            for (unsigned int i = 0; i < covariant_output_t::dimensions; ++i) {
                rv[i] = static_cast<typename covariant_output_t::scalar_t>(0.f);
            }

            for (std::size_t n = 0;
                 n < (1u << contravariant_output_t::dimensions);
                 ++n)
            {
                decltype(auto) pc = underlying_getter(n, is);
                const auto ifac = weight_helper(n, ls, rs);

                for (std::size_t q = 0; q < covariant_output_t::dimensions; ++q)
                {
                    rv[q] += static_cast<covariant_output_t::scalar_t>(
                        ifac * static_cast<input_scalar_type>(pc[q])
                    );
                }
            }

            return rv;
        }

        typename backend_t::non_owning_data_t & get_backend(void)
        {
            return m_backend;
        }

        const typename backend_t::non_owning_data_t & get_backend(void) const
        {
            return m_backend;
        }

        typename backend_t::non_owning_data_t m_backend;

    private:
        template <std::size_t... Is>
        COVFIE_HOST_DEVICE input_scalar_type
        weight_helper_impl(std::size_t n, const input_scalar_type (&ls)[contravariant_output_t::dimensions], const input_scalar_type (&rs)[contravariant_output_t::dimensions], std::index_sequence<Is...>)
            const
        {
            return (((n & (1u << Is)) ? ls[Is] : rs[Is]) * ...);
        }

        COVFIE_HOST_DEVICE input_scalar_type weight_helper(
            std::size_t n,
            const input_scalar_type (&ls)[contravariant_output_t::dimensions],
            const input_scalar_type (&rs)[contravariant_output_t::dimensions]
        ) const
        {
            return weight_helper_impl(
                n,
                ls,
                rs,
                std::make_index_sequence<contravariant_output_t::dimensions>()
            );
        }

        template <std::size_t... Is>
        COVFIE_HOST_DEVICE decltype(auto)
        underlying_getter_impl(std::size_t n, const typename contravariant_output_t::scalar_t (&is)[contravariant_output_t::dimensions], std::index_sequence<Is...>)
            const
        {
            return m_backend.at({(static_cast<typename decltype(m_backend
                                  )::parent_t::contravariant_input_t::scalar_t>(
                is[contravariant_output_t::dimensions - (Is + 1)] +
                ((n & (1u << (contravariant_output_t::dimensions - (Is + 1))))
                     ? 1
                     : 0)
            ))...});
        }

        COVFIE_HOST_DEVICE decltype(auto) underlying_getter(
            std::size_t n,
            const typename contravariant_output_t::scalar_t (&is
            )[contravariant_output_t::dimensions]
        ) const
        {
            return underlying_getter_impl(
                n,
                is,
                std::make_index_sequence<contravariant_output_t::dimensions>()
            );
        }
    };
};
}
