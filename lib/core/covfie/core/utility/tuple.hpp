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

#include <array>
#include <functional>

namespace covfie::utility {
template <typename T, typename... Ts>
std::array<T, sizeof...(Ts) + 1u> to_array(std::tuple<T, Ts...> i)
{
    return std::apply(
        [](T t, Ts... ts) {
            return std::array<T, sizeof...(Ts) + 1u>{t, ts...};
        },
        i
    );
}
}
