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
#include <cstddef>

namespace covfie::algebra {
template <
    std::size_t N,
    std::size_t M,
    typename T = float,
    typename I = std::size_t>
struct matrix {
    matrix()
    {
    }

    matrix(std::array<std::array<T, M>, N> l)
    {
        for (I i = 0; i < l.size(); ++i) {
            for (I j = 0; j < l[i].size(); ++j) {
                m_elems[i][j] = l[i][j];
            }
        }
    }

    T operator()(const I i, const I j) const
    {
        return m_elems[i][j];
    }

    T & operator()(const I i, const I j)
    {
        return m_elems[i][j];
    }

    template <std::size_t P>
    matrix<N, P, T, I> operator*(const matrix<M, P, T, I> & o)
    {
        matrix<N, P, T, I> r;

        for (I i = 0; i < N; ++i) {
            for (I j = 0; j < P; ++j) {
                T t = static_cast<T>(0.);

                for (I k = 0; k < M; ++k) {
                    t += this->operator()(i, k) * o(k, j);
                }

                r(i, j) = t;
            }
        }

        return r;
    }

private:
    T m_elems[N][M];
};
}
