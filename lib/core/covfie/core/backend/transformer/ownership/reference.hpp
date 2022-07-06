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

#include <cstddef>
#include <fstream>

#include <covfie/core/concepts.hpp>

namespace covfie::backend::transformer::ownership {
template <CONSTRAINT(concepts::field_backend) _backend_t>
struct reference {
    static constexpr bool is_initial = false;

    using backend_t = _backend_t;

    using contravariant_input_t = typename backend_t::contravariant_input_t;
    using covariant_output_t = typename backend_t::covariant_output_t;

    struct owning_data_t {
        template <typename... Args>
        owning_data_t(const typename backend_t::owning_data_t & o)
            : m_backend(o)
        {
        }

        void dump(std::ofstream & fs) const
        {
            m_backend.dump(fs);
        }

        typename backend_t::owning_data_t & get_backend(void)
        {
            return m_backend;
        }

        const typename backend_t::owning_data_t & get_backend(void) const
        {
            return m_backend;
        }

        typename std::reference_wrapper<const typename backend_t::owning_data_t>
            m_backend;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & src)
            : m_backend(src.m_backend)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            return m_backend.at(c);
        }

        typename backend_t::non_owning_data_t m_backend;
    };
};
}
