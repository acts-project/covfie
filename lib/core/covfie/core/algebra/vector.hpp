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
#include <covfie/core/array.hpp>
#include <covfie/core/qualifiers.hpp>

namespace covfie::algebra {
template <std::size_t N, typename T = float, typename I = std::size_t>
struct vector : public matrix<N, 1, T, I> {
    COVFIE_DEVICE vector()
        : matrix<N, 1, T, I>()
    {
    }

    COVFIE_DEVICE vector(array::array<T, N> l)
        : matrix<N, 1, T, I>()
    {
        for (I i = 0; i < N; ++i) {
            matrix<N, 1, T, I>::operator()(i, 0) = l[i];
        }
    }

    COVFIE_DEVICE vector(const matrix<N, 1, T, I> & o)
        : matrix<N, 1, T, I>(o)
    {
    }

    template <
        typename... Args,
        std::enable_if_t<
            std::conjunction_v<std::is_scalar<Args>...> &&
                std::conjunction_v<std::is_convertible<Args, T>...> &&
                sizeof...(Args) == N,
            bool> = true>
    COVFIE_DEVICE vector(Args... args)
        : vector(array::array<T, N>{std::forward<Args>(args)...})
    {
    }

    COVFIE_DEVICE T operator()(const I & i) const
    {
        return matrix<N, 1, T, I>::operator()(i, 0);
    }

    COVFIE_DEVICE T & operator()(const I & i)
    {
        return matrix<N, 1, T, I>::operator()(i, 0);
    }
};
}
