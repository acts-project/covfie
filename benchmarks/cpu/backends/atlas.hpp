/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <covfie/benchmark/atlas.hpp>
#include <covfie/core/backend/initial/constant.hpp>
#include <covfie/core/backend/transformer/interpolator/linear.hpp>
#include <covfie/core/backend/transformer/interpolator/nearest_neighbour.hpp>
#include <covfie/core/backend/transformer/layout/morton.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

struct AtlasConstant {
    using backend_t = covfie::backend::constant<
        covfie::vector::vector_d<float, 3>,
        covfie::vector::vector_d<float, 3>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(typename backend_t::configuration_t{
            0.f, 0.f, 2.f});
    }
};

template <typename T>
using InterpolateNN =
    covfie::backend::transformer::interpolator::nearest_neighbour<T>;

template <typename T>
using InterpolateLin = covfie::backend::transformer::interpolator::linear<T>;

template <typename V, typename T>
using LayoutStride = covfie::backend::layout::strided<V, T>;

template <typename V, typename T>
using LayoutMortonBMI2 = covfie::backend::layout::morton<V, T, true>;

template <typename V, typename T>
using LayoutMortonNaive = covfie::backend::layout::morton<V, T, false>;

template <
    template <typename>
    typename Interpolator,
    template <typename, typename>
    typename Layout>
struct Atlas {
    using backend_t = covfie::backend::transformer::affine<Interpolator<Layout<
        covfie::vector::ulong3,
        covfie::backend::storage::array<covfie::vector::float3>>>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(get_atlas_field());
    }
};

struct AtlasIntBase {
    using backend_t = covfie::backend::layout::strided<
        covfie::vector::ulong3,
        covfie::backend::storage::array<covfie::vector::float3>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(
            get_atlas_field().backend().get_backend().get_backend()
        );
    }
};

struct AtlasIntMorton {
    using backend_t = covfie::backend::layout::morton<
        covfie::vector::ulong3,
        covfie::backend::storage::array<covfie::vector::float3>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(
            get_atlas_field().backend().get_backend().get_backend()
        );
    }
};
