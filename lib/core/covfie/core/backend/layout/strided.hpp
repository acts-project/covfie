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

#include <numeric>

#include <covfie/core/concepts.hpp>

namespace covfie::backend::layout {
template <
    std::size_t _dims,
    CONSTRAINT(concepts::integral_input_scalar) _index_t = std::size_t,
    template <typename, std::size_t> typename _array_tc = std::array>
struct strided {
    static constexpr std::size_t dims = _dims;

    using index_t = _index_t;
    using ndsize_t = _array_tc<index_t, dims>;
    using coordinate_t = _array_tc<index_t, dims>;

    struct owning_data_t {
        owning_data_t(ndsize_t sizes)
            : m_sizes(sizes)
        {
        }

        index_t required_size() const
        {
            return std::accumulate(
                std::begin(m_sizes),
                std::end(m_sizes),
                1,
                std::multiplies<std::size_t>()
            );
        }

        ndsize_t m_sizes;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_sizes(o.m_sizes)
        {
        }

        index_t required_size() const
        {
            return std::accumulate(
                std::begin(m_sizes),
                std::end(m_sizes),
                1,
                std::multiplies<std::size_t>()
            );
        }

        COVFIE_DEVICE index_t operator()(coordinate_t c) const
        {
            index_t idx = 0;

            for (std::size_t k = 0; k < dims; ++k) {
                index_t tmp = c[k];

                for (std::size_t l = k + 1; l < dims; ++l) {
                    tmp *= m_sizes[l];
                }

                idx += tmp;
            }

            return idx;
        }

        ndsize_t m_sizes;
    };
};
}
