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

#include <cmath>

#include <covfie/core/backend/builder.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/utility/nd_map.hpp>

namespace covfie::backend {
template <
    template <typename, std::size_t>
    typename _array_t,
    CONSTRAINT(concepts::layout) _layout_t,
    template <typename, std::size_t, typename>
    typename _storage_tc,
    CONSTRAINT(concepts::datatype) _datatype_t>
struct _modular {
    using datatype_t = _datatype_t;
    static constexpr std::size_t output_dimensions = datatype_t::dimensions;
    using index_t = std::size_t;
    using layout_t = _layout_t;
    static constexpr std::size_t coordinate_dimensions = layout_t::dims;
    using output_scalar_t = typename datatype_t::output_scalar_t;
    using storage_t =
        _storage_tc<output_scalar_t, output_dimensions, std::size_t>;
    using value_t = output_scalar_t[output_dimensions];

    using builder_t = builder<coordinate_dimensions, output_dimensions>;

    using coordinate_t = typename layout_t::coordinate_t;
    using output_t = _array_t<output_scalar_t, output_dimensions>;

    struct owning_data_t {
        static std::unique_ptr<value_t[]> make_data(
            typename layout_t::non_owning_data_t layout_view,
            const typename builder_t::owning_data_t & builder
        )
        {
            auto bsizes = builder.m_sizes;

            typename builder_t::non_owning_data_t builder_view(builder);

            std::unique_ptr<value_t[]> rv =
                std::make_unique<value_t[]>(layout_view.required_size());

            // todo: Make more elegant!
            if constexpr (coordinate_dimensions == 1) {
                utility::nd_map(
                    std::function<void(std::size_t)>(
                        [&rv, &builder_view, &layout_view](auto... args) {
                            typename builder_t::output_t bv =
                                builder_view.at({args...});

                            coordinate_t c{args...};

                            for (std::size_t j = 0; j < output_dimensions; ++j)
                            {
                                rv[layout_view(c)][j] = bv[j];
                            }
                        }
                    ),
                    bsizes[0]
                );
            } else if constexpr (coordinate_dimensions == 2) {
                utility::nd_map(
                    std::function<void(std::size_t, std::size_t)>(
                        [&rv, &builder_view, &layout_view](auto... args) {
                            typename builder_t::output_t bv =
                                builder_view.at({args...});

                            coordinate_t c{args...};

                            for (std::size_t j = 0; j < output_dimensions; ++j)
                            {
                                rv[layout_view(c)][j] = bv[j];
                            }
                        }
                    ),
                    bsizes[0],
                    bsizes[1]
                );
            } else if constexpr (coordinate_dimensions == 3) {
                utility::nd_map(
                    std::function<void(std::size_t, std::size_t, std::size_t)>(
                        [&rv, &builder_view, &layout_view](auto... args) {
                            typename builder_t::output_t bv =
                                builder_view.at({args...});

                            coordinate_t c{args...};

                            for (std::size_t j = 0; j < output_dimensions; ++j)
                            {
                                rv[layout_view(c)][j] = bv[j];
                            }
                        }
                    ),
                    bsizes[0],
                    bsizes[1],
                    bsizes[2]
                );
            }

            return rv;
        }

        owning_data_t(const typename builder_t::owning_data_t & builder)
            : layout(builder.m_sizes)
            , storage(make_data(layout, builder), layout.required_size())
        {
        }

        typename layout_t::owning_data_t layout;
        typename storage_t::owning_data_t storage;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & src)
            : layout(src.layout)
            , storage(src.storage)
        {
        }

        COVFIE_DEVICE output_t at(coordinate_t c) const
        {
            std::size_t idx = layout(c);
            value_t & res = storage[idx];
            output_t rv;

            for (std::size_t i = 0; i < output_dimensions; ++i) {
                rv[i] = res[i];
            }

            return rv;
        }

        typename layout_t::non_owning_data_t layout;
        typename storage_t::non_owning_data_t storage;
    };
};

template <
    template <typename, std::size_t>
    typename _array_tc,
    CONSTRAINT(concepts::layout) _layout_tc,
    template <typename, std::size_t, typename>
    typename _storage_tc,
    CONSTRAINT(concepts::datatype) _datatype_t>
using modular = _modular<_array_tc, _layout_tc, _storage_tc, _datatype_t>;
}
