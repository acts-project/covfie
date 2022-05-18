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
#include <initializer_list>
#include <type_traits>
#include <utility>

#include <covfie/core/algebra/matrix.hpp>

namespace covfie::algebra {
template <std::size_t N, typename T = float, typename I = std::size_t>
struct vector : public matrix<N, 1, T, I> {
    vector()
        : matrix<N, 1, T, I>()
    {
    }

    vector(std::array<T, N> l)
        : matrix<N, 1, T, I>()
    {
        for (I i = 0; i < N; ++i) {
            matrix<N, 1, T, I>::operator()(i, 0) = l[i];
        }
    }

    vector(const matrix<N, 1, T, I> & o)
        : matrix<N, 1, T, I>(o)
    {
    }

    template <
        typename... Args,
        std::enable_if_t<
            (std::is_scalar_v<Args> && ...) &&
                (std::is_convertible_v<Args, T> && ...) && sizeof...(Args) == N,
            bool> = true>
    vector(Args... args)
        : vector(std::array<T, N>{std::forward<Args>(args)...})
    {
    }

    T operator()(const I & i) const
    {
        return matrix<N, 1, T, I>::operator()(i, 0);
    }

    T & operator()(const I & i)
    {
        return matrix<N, 1, T, I>::operator()(i, 0);
    }
};
}
