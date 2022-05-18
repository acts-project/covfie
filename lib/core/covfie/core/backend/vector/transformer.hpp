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

#include <covfie/core/concepts.hpp>

namespace covfie::backend::vector {
template <CONSTRAINT(concepts::output_vector) _output_vector_t>
struct add_lvalue_reference {
    static constexpr std::size_t dimensions = _output_vector_t::dimensions;

    template <typename T>
    using reapply =
        add_lvalue_reference<typename _output_vector_t::template reapply<T>>;
    using vector_t = std::conditional_t<
        std::is_lvalue_reference_v<typename _output_vector_t::vector_t>,
        typename _output_vector_t::vector_t,
        std::add_lvalue_reference_t<typename _output_vector_t::vector_t>>;
};

template <CONSTRAINT(concepts::output_vector) _output_vector_t>
struct remove_lvalue_reference {
    static constexpr std::size_t dimensions = _output_vector_t::dimensions;

    template <typename T>
    using reapply =
        remove_lvalue_reference<typename _output_vector_t::template reapply<T>>;
    using vector_t =
        std::remove_reference_t<typename _output_vector_t::vector_t>;
};
}
