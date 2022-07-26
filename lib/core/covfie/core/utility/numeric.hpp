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

#include <type_traits>

namespace covfie::utility {
template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
T round_pow2(T i)
{
    T j = 1;
    for (; j < i; j *= 2)
        ;
    return j;
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
T ipow(T i, T p)
{
    T r = 1;

    for (; p; p >>= 1) {
        if (p & 1) {
            r *= i;
        }

        i *= i;
    }

    return r;
}
}
