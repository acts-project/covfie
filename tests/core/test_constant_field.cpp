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

#include <covfie/core/backend/constant.hpp>
#include <covfie/core/field.hpp>
#include <gtest/gtest.h>

TEST(TestConstantField, Constant1Dto1D)
{
    using field_t = covfie::field<covfie::backend::constant<
        covfie::backend::vector::input::float1,
        covfie::backend::vector::output::float1>>;

    field_t f(field_t::backend_t::configuration_data_t({5.f}));

    field_t::view_t fv(f);

    for (float i = -10.f; i < 10.f; i += 1.f) {
        EXPECT_EQ(fv.at(i)[0], 5.f);
    }
}

TEST(TestConstantField, Constant1Dto3D)
{
    using field_t = covfie::field<covfie::backend::constant<
        covfie::backend::vector::input::float1,
        covfie::backend::vector::output::float3>>;

    field_t f(field_t::backend_t::configuration_data_t({5.f, 2.f, 8.f}));

    field_t::view_t fv(f);

    for (float i = -10.f; i < 10.f; i += 1.f) {
        EXPECT_EQ(fv.at(i)[0], 5.f);
        EXPECT_EQ(fv.at(i)[1], 2.f);
        EXPECT_EQ(fv.at(i)[2], 8.f);
    }
}

TEST(TestConstantField, Constant3Dto1D)
{
    using field_t = covfie::field<covfie::backend::constant<
        covfie::backend::vector::input::float3,
        covfie::backend::vector::output::float1>>;

    field_t f(field_t::backend_t::configuration_data_t({5.f}));

    field_t::view_t fv(f);

    for (float x = -10.f; x < 10.f; x += 1.f) {
        for (float y = -10.f; y < 10.f; y += 1.f) {
            for (float z = -10.f; z < 10.f; z += 1.f) {
                EXPECT_EQ(fv.at(x, y, z)[0], 5.f);
            }
        }
    }
}

TEST(TestConstantField, Constant3Dto3D)
{
    using field_t = covfie::field<covfie::backend::constant<
        covfie::backend::vector::input::float3,
        covfie::backend::vector::output::float3>>;

    field_t f(field_t::backend_t::configuration_data_t({5.f, 2.f, 8.f}));

    field_t::view_t fv(f);

    for (float x = -10.f; x < 10.f; x += 1.f) {
        for (float y = -10.f; y < 10.f; y += 1.f) {
            for (float z = -10.f; z < 10.f; z += 1.f) {
                EXPECT_EQ(fv.at(x, y, z)[0], 5.f);
                EXPECT_EQ(fv.at(x, y, z)[1], 2.f);
                EXPECT_EQ(fv.at(x, y, z)[2], 8.f);
            }
        }
    }
}
