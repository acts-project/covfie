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

#include <cstring>
#include <iostream>
#include <memory>
#include <tuple>
#include <utility>

#include <covfie/core/concepts.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>
#include <covfie/core/utility/nd_size.hpp>
#include <covfie/core/vector.hpp>

namespace covfie::backend {
template <
    CONSTRAINT(concepts::vector_descriptor) _output_vector_t,
    typename _index_t = std::size_t>
struct array {
    using this_t = array<_output_vector_t, _index_t>;
    static constexpr bool is_initial = true;

    using contravariant_input_t =
        covfie::vector::scalar_d<covfie::vector::vector_d<_index_t, 1>>;
    using covariant_output_t =
        covfie::vector::array_reference_vector_d<_output_vector_t>;

    using vector_t = std::decay_t<typename covariant_output_t::vector_t>;

    using configuration_t = utility::nd_size<contravariant_input_t::dimensions>;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t()
            : m_size(0)
            , m_ptr({})
        {
        }

        explicit owning_data_t(owning_data_t && o)
            : m_size(o.m_size)
            , m_ptr(std::move(o.m_ptr))
        {
        }

        explicit owning_data_t(std::size_t n)
            : m_size(n)
            , m_ptr(std::make_unique<vector_t[]>(n))
        {
        }

        explicit owning_data_t(configuration_t conf)
            : owning_data_t(conf[0])
        {
        }

        explicit owning_data_t(parameter_pack<configuration_t> && conf)
            : owning_data_t(conf.x[0])
        {
        }

        explicit owning_data_t(
            std::size_t size, std::unique_ptr<vector_t[]> && ptr
        )
            : m_size(size)
            , m_ptr(std::move(ptr))
        {
        }

        explicit owning_data_t(std::istream & fs)
            : m_size(utility::read_binary<decltype(m_size)>(fs))
            , m_ptr(utility::read_binary_array<vector_t>(fs, m_size))
        {
        }

        owning_data_t(const owning_data_t & o)
            : m_size(o.m_size)
            , m_ptr(std::make_unique<vector_t[]>(m_size))
        {
            std::memcpy(m_ptr.get(), o.m_ptr.get(), m_size * sizeof(vector_t));
        }

        owning_data_t & operator=(const owning_data_t & o)
        {
            m_size = o.m_size;
            m_ptr = std::make_unique<vector_t[]>(m_size);

            std::memcpy(m_ptr.get(), o.m_ptr.get(), m_size * sizeof(vector_t));
        }

        configuration_t get_configuration() const
        {
            return {m_size};
        }

        void dump(std::ostream & fs) const
        {
            fs.write(
                reinterpret_cast<const char *>(&m_size),
                sizeof(decltype(m_size))
            );

            fs.write(
                reinterpret_cast<const char *>(m_ptr.get()),
                m_size * sizeof(vector_t)
            );
        }

        std::size_t m_size;
        std::unique_ptr<vector_t[]> m_ptr;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr.get())
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t i) const
        {
            return m_ptr[i];
        }

        typename decltype(owning_data_t::m_ptr)::pointer m_ptr;
    };
};
}
