/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace covfie::utility {
template <std::integral T>
T round_pow2(T i)
{
    T j = 1;
    for (; j < i; j *= 2)
        ;
    return j;
}

template <std::integral T>
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
