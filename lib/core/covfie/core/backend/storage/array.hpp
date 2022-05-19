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
#include <memory>
#include <tuple>
#include <utility>

#include <covfie/core/backend/vector/input.hpp>
#include <covfie/core/backend/vector/transformer.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/qualifiers.hpp>
#include <covfie/core/utility/binary_io.hpp>

namespace covfie::backend::storage {
template <
    CONSTRAINT(concepts::output_vector) _output_vector_t,
    typename _index_t = std::size_t>
struct array {
    using contravariant_input_t = vector::input_scalar<_index_t>;
    using contravariant_output_t = std::tuple<>;
    using covariant_input_t = std::tuple<>;
    using covariant_output_t = vector::add_lvalue_reference<_output_vector_t>;

    using vector_t = std::decay_t<typename _output_vector_t::vector_t>;

    struct owning_data_t {
        using parent_t = array<_output_vector_t, _index_t>;

        owning_data_t(owning_data_t && o)
            : m_size(o.m_size)
            , m_ptr(std::move(o.m_ptr))
        {
        }

        owning_data_t(std::size_t n)
            : m_size(n)
            , m_ptr(std::make_unique<vector_t[]>(n))
        {
        }

        owning_data_t(std::ifstream & fs)
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

        void dump(std::ofstream & fs) const
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
        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr.get())
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        operator[](typename contravariant_input_t::vector_t i) const
        {
            return m_ptr[i];
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
