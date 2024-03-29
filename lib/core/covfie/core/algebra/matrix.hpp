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

#include <covfie/core/qualifiers.hpp>

namespace covfie::algebra {
template <
    std::size_t N,
    std::size_t M,
    typename T = float,
    typename I = std::size_t>
struct matrix {
    COVFIE_DEVICE matrix()
    {
    }

    COVFIE_DEVICE matrix(std::array<std::array<T, M>, N> l)
    {
        for (I i = 0; i < l.size(); ++i) {
            for (I j = 0; j < l[i].size(); ++j) {
                m_elems[i][j] = l[i][j];
            }
        }
    }

    matrix(const matrix<N, M, T, I> &) = default;

    COVFIE_DEVICE T operator()(const I i, const I j) const
    {
        return m_elems[i][j];
    }

    COVFIE_DEVICE T & operator()(const I i, const I j)
    {
        return m_elems[i][j];
    }

    template <std::size_t P>
    COVFIE_DEVICE matrix<N, P, T, I> operator*(const matrix<M, P, T, I> & o
    ) const
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

    COVFIE_DEVICE static matrix<N, M, T, I> identity()
    {
        matrix<N, M, T, I> result;

        for (I i = 0; i < N; ++i) {
            for (I j = 0; j < M; ++j) {
                result(i, j) = (i == j) ? static_cast<T>(1) : static_cast<T>(0);
            }
        }

        return result;
    }

private:
    T m_elems[N][M];
};
}
