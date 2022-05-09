/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <array>
#include <cstddef>

#include <covfie/core/backend/builder.hpp>
#include <covfie/core/field.hpp>
#include <gtest/gtest.h>

TEST(TestFieldBuilder, ConstructSingleFloat)
{
    using field_t = covfie::field<
        covfie::backend::builder<3, covfie::backend::vector::output::float1>>;
    field_t f(field_t::backend_t::configuration_data_t{5u, 7u, 2u});

    std::array<std::size_t, 3> s = f.backend().m_sizes;

    EXPECT_EQ(s[0], 5u);
    EXPECT_EQ(s[1], 7u);
    EXPECT_EQ(s[2], 2u);
}

TEST(TestFieldBuilder, ConstructArrayFloat)
{
    using field_t = covfie::field<
        covfie::backend::builder<3, covfie::backend::vector::output::float3>>;
    field_t f(field_t::backend_t::configuration_data_t{5u, 7u, 2u});

    std::array<std::size_t, 3> s = f.backend().m_sizes;

    EXPECT_EQ(s[0], 5u);
    EXPECT_EQ(s[1], 7u);
    EXPECT_EQ(s[2], 2u);
}

TEST(TestFieldBuilder, WriteReadSingleFloat)
{
    using field_t = covfie::field<
        covfie::backend::builder<3, covfie::backend::vector::output::float1>>;
    field_t f(field_t::backend_t::configuration_data_t{5u, 7u, 2u});

    field_t::view_t fv(f);

    fv.at(1u, 2u, 1u)[0] = 5.f;
    fv.at(2u, 0u, 0u)[0] = 7.f;
    fv.at(4u, 6u, 1u)[0] = 11.f;

    EXPECT_EQ(fv.at(1u, 2u, 1u)[0], 5.f);
    EXPECT_EQ(fv.at(2u, 0u, 0u)[0], 7.f);
    EXPECT_EQ(fv.at(4u, 6u, 1u)[0], 11.f);
}

TEST(TestFieldBuilder, WriteReadArrayFloat)
{
    using field_t = covfie::field<
        covfie::backend::builder<3, covfie::backend::vector::output::float3>>;
    field_t f(field_t::backend_t::configuration_data_t{5u, 7u, 2u});

    field_t::view_t fv(f);

    float(&p1)[3] = fv.at(1u, 2u, 1u);
    float(&p2)[3] = fv.at(2u, 0u, 0u);

    p1[0] = 5.f;
    p1[1] = 6.f;
    p1[2] = 7.f;
    p2[0] = 1.f;
    p2[1] = 2.f;
    p2[2] = 3.f;

    EXPECT_EQ(fv.at(1u, 2u, 1u)[0], 5.f);
    EXPECT_EQ(fv.at(1u, 2u, 1u)[1], 6.f);
    EXPECT_EQ(fv.at(1u, 2u, 1u)[2], 7.f);
    EXPECT_EQ(fv.at(2u, 0u, 0u)[0], 1.f);
    EXPECT_EQ(fv.at(2u, 0u, 0u)[1], 2.f);
    EXPECT_EQ(fv.at(2u, 0u, 0u)[2], 3.f);
}
