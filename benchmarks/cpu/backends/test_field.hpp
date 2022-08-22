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

#include <covfie/benchmark/test_field.hpp>
#include <covfie/core/backend/initial/constant.hpp>
#include <covfie/core/backend/transformer/interpolator/linear.hpp>
#include <covfie/core/backend/transformer/interpolator/nearest_neighbour.hpp>
#include <covfie/core/backend/transformer/layout/morton.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

struct FieldConstant {
    using backend_t = covfie::backend::constant<
        covfie::vector::vector_d<float, 3>,
        covfie::vector::vector_d<float, 3>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(typename backend_t::configuration_t{
            0.f, 0.f, 2.f});
    }
};

struct InterpolateNN {
    template <typename T>
    using apply =
        covfie::backend::transformer::interpolator::nearest_neighbour<T>;
};

struct InterpolateLin {
    template <typename T>
    using apply = covfie::backend::transformer::interpolator::linear<T>;
};

struct LayoutStride {
    template <typename T>
    using apply = covfie::backend::layout::strided<covfie::vector::ulong3, T>;
};

struct LayoutMortonBMI2 {
    template <typename T>
    using apply =
        covfie::backend::layout::morton<covfie::vector::ulong3, T, true>;
};

struct LayoutMortonNaive {
    template <typename T>
    using apply =
        covfie::backend::layout::morton<covfie::vector::ulong3, T, false>;
};

template <typename Interpolator, typename Layout>
struct Field {
    using backend_t = covfie::backend::transformer::affine<
        typename Interpolator::template apply<typename Layout::template apply<

            covfie::backend::storage::array<covfie::vector::float3>>>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(get_test_field());
    }
};

struct FieldIntBase {
    using backend_t = covfie::backend::layout::strided<
        covfie::vector::ulong3,
        covfie::backend::storage::array<covfie::vector::float3>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(
            get_test_field().backend().get_backend().get_backend()
        );
    }
};

struct FieldIntMorton {
    using backend_t = covfie::backend::layout::morton<
        covfie::vector::ulong3,
        covfie::backend::storage::array<covfie::vector::float3>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(
            get_test_field().backend().get_backend().get_backend()
        );
    }
};
