/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cmath>
#include <cstddef>

#include <gtest/gtest.h>

#include <covfie/core/backend/initial/identity.hpp>
#include <covfie/core/backend/transformer/interpolator/nearest_neighbour.hpp>
#include <covfie/core/field.hpp>

TEST(TestNearestNeighbourInterpolator, Identity1Nto1F)
{
    using field_t = covfie::field<
        covfie::backend::transformer::interpolator::nearest_neighbour<
            covfie::backend::
                identity<covfie::vector::uint1, covfie::vector::float1>>>;

    field_t f(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    );
    field_t::view_t fv(f);

    EXPECT_EQ(fv.at(1.f)[0], 1.f);
    EXPECT_EQ(fv.at(5.f)[0], 5.f);
    EXPECT_EQ(fv.at(5.4f)[0], 5.f);
    EXPECT_EQ(fv.at(5.6f)[0], 6.f);

    for (float x = 0.f; x < 10.f; x += 0.1f) {
        EXPECT_EQ(fv.at(x)[0], std::round(x));
    }
}

TEST(TestNearestNeighbourInterpolator, Identity2Nto2F)
{
    using field_t = covfie::field<
        covfie::backend::transformer::interpolator::nearest_neighbour<
            covfie::backend::
                identity<covfie::vector::uint2, covfie::vector::float2>>>;

    field_t f(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    );
    field_t::view_t fv(f);

    for (float x = 0.f; x < 10.f; x += 0.1f) {
        for (float y = 0.f; y < 10.f; y += 0.1f) {
            EXPECT_EQ(fv.at(x, y)[0], std::round(x));
            EXPECT_EQ(fv.at(x, y)[1], std::round(y));
        }
    }
}

TEST(TestNearestNeighbourInterpolator, Identity3Nto3F)
{
    using field_t = covfie::field<
        covfie::backend::transformer::interpolator::nearest_neighbour<
            covfie::backend::
                identity<covfie::vector::uint3, covfie::vector::float3>>>;

    field_t f(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    );
    field_t::view_t fv(f);

    for (float x = 0.f; x < 3.f; x += 0.1f) {
        for (float y = 0.f; y < 3.f; y += 0.1f) {
            for (float z = 0.f; z < 3.f; z += 0.1f) {
                EXPECT_EQ(fv.at(x, y, z)[0], std::round(x));
                EXPECT_EQ(fv.at(x, y, z)[1], std::round(y));
                EXPECT_EQ(fv.at(x, y, z)[2], std::round(z));
            }
        }
    }
}
