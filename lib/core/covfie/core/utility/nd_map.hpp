/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022-2023 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <array>
#include <functional>
#include <tuple>

namespace covfie::utility {
template <std::size_t... Ns, typename... Ts>
auto tail_impl(std::index_sequence<Ns...>, [[maybe_unused]] std::tuple<Ts...> t)
{
    return std::make_tuple(std::get<Ns + 1u>(t)...);
}

template <typename... Ts>
auto tail(std::tuple<Ts...> t)
{
    return tail_impl(std::make_index_sequence<sizeof...(Ts) - 1u>(), t);
}

template <typename Tuple>
void nd_map(std::function<void(Tuple)> f, Tuple s)
{
    if constexpr (std::tuple_size<Tuple>::value == 0u) {
        f({});
    } else {
        using head_t = typename std::tuple_element<0u, Tuple>::type;
        using tail_t = decltype(tail(std::declval<Tuple>()));

        for (head_t i = static_cast<head_t>(0); i < std::get<0u>(s); ++i) {
            nd_map<tail_t>(
                [f, i](tail_t r) {
                    f(std::tuple_cat(std::tuple<head_t>{i}, r));
                },
                tail(s)
            );
        }
    }
}
}
