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

#include <fstream>

#include <covfie/core/concepts.hpp>
#include <covfie/core/field_view.hpp>

namespace covfie {
template <CONSTRAINT(concepts::field_backend) _backend_t>
class field
{
public:
    using backend_t = _backend_t;
    using view_t = field_view<backend_t>;
    using storage_t = typename backend_t::owning_data_t;
    using output_t = typename backend_t::output_t;
    using coordinate_t = typename backend_t::coordinate_t;

    template <typename other_backend>
    field(const field<other_backend> & other)
        : m_backend(other.m_backend)
    {
    }

    template <typename... Args>
    field(Args... args)
        : m_backend(args...)
    {
    }

    field(std::ifstream & fs)
        : m_backend(fs)
    {
    }

    const storage_t & backend(void)
    {
        return m_backend;
    }

    void dump(std::ofstream & fs) const
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
