/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cstddef>

#include <gtest/gtest.h>

#include <covfie/core/backend/initial/identity.hpp>
#include <covfie/core/field.hpp>

TEST(TestIdentityBackend, 1Fto1F)
{
    using field_t = covfie::field<covfie::backend::identity<
        covfie::vector::float1,
        covfie::vector::float1>>;

    field_t f(field_t::backend_t::configuration_data_t({}));

    field_t::view_t fv(f);

    for (float i = -10.f; i < 10.f; i += 1.f) {
        EXPECT_EQ(fv.at(i)[0], i);
    }
}

TEST(TestIdentityBackend, 2Fto2F)
{
    using field_t = covfie::field<covfie::backend::identity<
        covfie::vector::float2,
        covfie::vector::float2>>;

    field_t f(field_t::backend_t::configuration_data_t({}));

    field_t::view_t fv(f);

    for (float x = -10.f; x < 10.f; x += 1.f) {
        for (float y = -10.f; y < 10.f; y += 1.f) {
            EXPECT_EQ(fv.at(x, y)[0], x);
            EXPECT_EQ(fv.at(x, y)[1], y);
        }
    }
}

TEST(TestIdentityBackend, 3Fto3F)
{
    using field_t = covfie::field<covfie::backend::identity<
        covfie::vector::float3,
        covfie::vector::float3>>;

    field_t f(field_t::backend_t::configuration_data_t({}));

    field_t::view_t fv(f);

    for (float x = -10.f; x < 10.f; x += 1.f) {
        for (float y = -10.f; y < 10.f; y += 1.f) {
            for (float z = -10.f; z < 10.f; z += 1.f) {
                EXPECT_EQ(fv.at(x, y, z)[0], x);
                EXPECT_EQ(fv.at(x, y, z)[1], y);
                EXPECT_EQ(fv.at(x, y, z)[2], z);
            }
        }
    }
}
