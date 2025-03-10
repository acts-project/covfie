/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
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
    COVFIE_HOST_DEVICE vector()
        : matrix<N, 1, T, I>()
    {
    }

    COVFIE_HOST_DEVICE vector(array::array<T, N> l)
        : matrix<N, 1, T, I>()
    {
        for (I i = 0; i < N; ++i) {
            matrix<N, 1, T, I>::operator()(i, 0) = l[i];
        }
    }

    COVFIE_HOST_DEVICE vector(const matrix<N, 1, T, I> & o)
        : matrix<N, 1, T, I>(o)
    {
    }

    template <typename... Args>
    requires(
        (std::is_scalar_v<Args> && ...) &&
        (std::is_convertible_v<Args, T> && ...) && sizeof...(Args) == N
    ) COVFIE_HOST_DEVICE vector(Args... args)
        : vector(array::array<T, N>{std::forward<Args>(args)...})
    {
    }

    COVFIE_HOST_DEVICE T operator()(const I & i) const
    {
        return matrix<N, 1, T, I>::operator()(i, 0);
    }

    COVFIE_HOST_DEVICE T & operator()(const I & i)
    {
        return matrix<N, 1, T, I>::operator()(i, 0);
    }
};
}
