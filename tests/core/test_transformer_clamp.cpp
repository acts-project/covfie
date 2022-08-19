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
#include <covfie/core/backend/transformer/boundary/clamp.hpp>
#include <covfie/core/field.hpp>

TEST(TestTransformerClamp, ClampIdentityInt1D)
{
    using field_t = covfie::field<covfie::backend::transformer::boundary::clamp<
        covfie::backend::identity<covfie::vector::int1>>>;

    field_t f(
        field_t::backend_t::configuration_t{{0}, {5}},
        field_t::backend_t::backend_t::configuration_t{}
    );

    field_t::view_t fv(f);

    EXPECT_EQ(fv.at(-10)[0], 0);
    EXPECT_EQ(fv.at(-1)[0], 0);
    EXPECT_EQ(fv.at(0)[0], 0);
    EXPECT_EQ(fv.at(1)[0], 1);
    EXPECT_EQ(fv.at(3)[0], 3);
    EXPECT_EQ(fv.at(5)[0], 5);
    EXPECT_EQ(fv.at(6)[0], 5);
    EXPECT_EQ(fv.at(10)[0], 5);
}

TEST(TestTransformerClamp, ClampIdentityInt2D)
{
    using field_t = covfie::field<covfie::backend::transformer::boundary::clamp<
        covfie::backend::identity<covfie::vector::int2>>>;

    field_t f(
        field_t::backend_t::configuration_t{{0, 4}, {5, 8}},
        field_t::backend_t::backend_t::configuration_t{}
    );

    field_t::view_t fv(f);

    EXPECT_EQ(fv.at(-10, -10)[0], 0);
    EXPECT_EQ(fv.at(-10, -10)[1], 4);
    EXPECT_EQ(fv.at(-1, -10)[0], 0);
    EXPECT_EQ(fv.at(-1, -10)[1], 4);
    EXPECT_EQ(fv.at(0, -10)[0], 0);
    EXPECT_EQ(fv.at(0, -10)[1], 4);
    EXPECT_EQ(fv.at(1, -10)[0], 1);
    EXPECT_EQ(fv.at(1, -10)[1], 4);
    EXPECT_EQ(fv.at(2, 5)[0], 2);
    EXPECT_EQ(fv.at(2, 5)[1], 5);
}
