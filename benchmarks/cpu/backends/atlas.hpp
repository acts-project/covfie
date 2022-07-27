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

struct AtlasBaseNN {
    using backend_t = covfie::backend::transformer::affine<
        covfie::backend::transformer::interpolator::nearest_neighbour<
            covfie::backend::layout::strided<
                covfie::vector::ulong3,
                covfie::backend::storage::array<covfie::vector::float3>>>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(get_atlas_field());
    }
};

struct AtlasMortonNN {
    using backend_t = covfie::backend::transformer::affine<
        covfie::backend::transformer::interpolator::nearest_neighbour<
            covfie::backend::layout::morton<
                covfie::vector::ulong3,
                covfie::backend::storage::array<covfie::vector::float3>>>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(get_atlas_field());
    }
};

struct AtlasApproxNN {
    using backend_t = covfie::backend::transformer::affine<
        covfie::backend::transformer::interpolator::nearest_neighbour<
            covfie::backend::layout::strided<
                covfie::vector::ulong3,
                covfie::backend::
                    constant<covfie::vector::ulong1, covfie::vector::float3>>>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(
            get_atlas_field().backend().get_configuration(),
            std::monostate{},
            get_atlas_field()
                .backend()
                .get_backend()
                .get_backend()
                .get_configuration(),
            backend_t::backend_t::backend_t::backend_t::configuration_t{
                0.f, 0.f, 2.f}
        );
    }
};

struct AtlasBaseLin {
    using backend_t = covfie::backend::transformer::affine<
        covfie::backend::transformer::interpolator::linear<
            covfie::backend::layout::strided<
                covfie::vector::ulong3,
                covfie::backend::storage::array<covfie::vector::float3>>>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(get_atlas_field());
    }
};

struct AtlasMortonLin {
    using backend_t = covfie::backend::transformer::affine<
        covfie::backend::transformer::interpolator::linear<
            covfie::backend::layout::morton<
                covfie::vector::ulong3,
                covfie::backend::storage::array<covfie::vector::float3>>>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(get_atlas_field());
    }
};

struct AtlasApproxLin {
    using backend_t = covfie::backend::transformer::affine<
        covfie::backend::transformer::interpolator::linear<
            covfie::backend::layout::strided<
                covfie::vector::ulong3,
                covfie::backend::
                    constant<covfie::vector::ulong1, covfie::vector::float3>>>>;

    static covfie::field<backend_t> get_field()
    {
        return covfie::field<backend_t>(
            get_atlas_field().backend().get_configuration(),
            std::monostate{},
            get_atlas_field()
                .backend()
                .get_backend()
                .get_backend()
                .get_configuration(),
            backend_t::backend_t::backend_t::backend_t::configuration_t{
                0.f, 0.f, 2.f}
        );
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
