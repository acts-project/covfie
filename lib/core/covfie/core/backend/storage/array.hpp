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
#include <utility>

namespace covfie::backend::storage {
template <typename _value_t, std::size_t _dims, typename _index_t = std::size_t>
struct array {
    static constexpr std::size_t dims = _dims;

    using value_t = _value_t[_dims];
    using index_t = _index_t;

    struct owning_data_t {
        owning_data_t(owning_data_t && o)
            : m_size(o.m_size)
            , m_ptr(std::move(o.m_ptr))
        {
        }

        owning_data_t(std::size_t n)
            : m_size(n)
            , m_ptr(std::make_unique<value_t[]>(n))
        {
        }

        std::size_t m_size;
        std::unique_ptr<value_t[]> m_ptr;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr.get())
        {
        }

        value_t & operator[](index_t i) const
        {
            return m_ptr[i];
        }

        typename decltype(owning_data_t::m_ptr)::pointer m_ptr;
    };
};
}
