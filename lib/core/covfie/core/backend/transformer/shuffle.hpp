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
#include <variant>

namespace covfie::backend {
template <CONSTRAINT(concepts::field_backend) _backend_t, typename _shuffle>
struct shuffle {
    using this_t = shuffle<_backend_t, _shuffle>;
    static constexpr bool is_initial = false;

    using backend_t = _backend_t;

    using contravariant_input_t = typename backend_t::contravariant_input_t;
    using covariant_output_t = typename backend_t::covariant_output_t;

    using configuration_t = std::monostate;
    using indices = _shuffle;

    static constexpr uint32_t IO_MAGIC_HEADER = 0xAB020009;

    struct owning_data_t {
        using parent_t = this_t;

        owning_data_t() = default;

        template <typename... Args>
        explicit owning_data_t(configuration_t, Args... args)
            : m_backend(std::forward<Args>(args)...)
        {
        }

        explicit owning_data_t(std::istream & fs)
            : m_backend(utility::read_io_header(fs, IO_MAGIC_HEADER))
        {
            utility::read_io_footer(fs, IO_MAGIC_HEADER);
        }

        void dump(std::ostream & fs) const
        {
            utility::write_io_header(fs, IO_MAGIC_HEADER);

            m_backend.dump(fs);

            utility::write_io_footer(fs, IO_MAGIC_HEADER);
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

        typename backend_t::owning_data_t m_backend;
    };

    struct non_owning_data_t {
        using parent_t = this_t;

        non_owning_data_t(const owning_data_t & src)
            : m_backend(src.m_backend)
        {
        }

        template <std::size_t... Is>
        COVFIE_DEVICE typename contravariant_input_t::vector_t
        shuffle(typename contravariant_input_t::vector_t c, std::index_sequence<Is...>)
            const
        {
            return {std::get<Is>(c)...};
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t c) const
        {
            return m_backend.at(shuffle(c, indices{}));
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
    };
};
}
