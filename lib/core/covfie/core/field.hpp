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

#include <iostream>

#include <covfie/core/concepts.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/core/parameter_pack.hpp>

namespace covfie {
template <CONSTRAINT(concepts::field_backend) _backend_t>
class field
{
public:
    using backend_t = _backend_t;
    using view_t = field_view<backend_t>;
    using storage_t = typename backend_t::owning_data_t;
    using output_t = typename backend_t::covariant_output_t::vector_t;
    using coordinate_t = typename backend_t::contravariant_input_t::vector_t;

    field(field &) = default;
    field(const field &) = default;
    field(field &&) = default;

    template <CONSTRAINT(concepts::field_backend) other_backend>
    explicit field(field<other_backend> & other)
        : m_backend(other.m_backend)
    {
    }

    template <CONSTRAINT(concepts::field_backend) other_backend>
    explicit field(field<other_backend> && other)
        : m_backend(std::forward<decltype(other.m_backend)>(other.m_backend))
    {
    }

    template <CONSTRAINT(concepts::field_backend) other_backend>
    explicit field(const field<other_backend> & other)
        : m_backend(other.m_backend)
    {
    }

    template <typename... Args>
    explicit field(parameter_pack<Args...> && args)
        : m_backend(std::forward<parameter_pack<Args...>>(args))
    {
    }

    explicit field(std::istream & fs)
        : m_backend(fs)
    {
    }

    field & operator=(const field &) = default;

    field & operator=(field &&) = default;

    const storage_t & backend(void) const
    {
        return m_backend;
    }

    void dump(std::ostream & fs) const
    {
        m_backend.dump(fs);
    }

private:
    storage_t m_backend;

    friend class field_view<_backend_t>;

    template <CONSTRAINT(concepts::field_backend) other_backend>
    friend class field;
};
}
