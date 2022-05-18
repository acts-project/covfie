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

#include <memory>
#include <tuple>
#include <utility>

#include <covfie/core/backend/vector/input.hpp>
#include <covfie/core/concepts.hpp>

namespace covfie::backend::storage {
template <
    CONSTRAINT(concepts::output_vector) _output_vector_t,
    typename _index_t = std::size_t>
struct array {
    using contravariant_input_t = vector::input_scalar<_index_t>;
    using contravariant_output_t = std::tuple<>;
    using covariant_input_t = std::tuple<>;
    using covariant_output_t = _output_vector_t;

    using vector_t = typename covariant_output_t::vector_t;

    struct owning_data_t {
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

        std::size_t m_size;
        std::unique_ptr<vector_t[]> m_ptr;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr.get())
        {
        }

        // TODO: Remove this reference operator.
        vector_t & operator[](typename contravariant_input_t::vector_t i) const
        {
            return m_ptr[i];
        }

        typename decltype(owning_data_t::m_ptr)::pointer m_ptr;
    };
};
}
