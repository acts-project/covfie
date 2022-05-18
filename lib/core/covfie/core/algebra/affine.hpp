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

#include <covfie/core/algebra/matrix.hpp>
#include <covfie/core/algebra/vector.hpp>

namespace covfie::algebra {
template <std::size_t N, typename T = float, typename I = std::size_t>
struct affine : public matrix<N, N + 1, T, I> {
    affine(std::array<std::array<T, N + 1>, N> l)
        : matrix<N, N + 1, T, I>(l)
    {
    }

    vector<N, T, I> operator*(const vector<N, T, I> & v)
    {
        vector<N + 1, T, I> r;

        for (I i = 0; i < N; ++i) {
            r(i) = v(i);
        }

        r(N) = static_cast<T>(1.);

        return matrix<N, N + 1, T, I>::operator*(r);
    }
};
}
