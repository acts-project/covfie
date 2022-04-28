/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <covfie/core/utility/nd_map.hpp>
#include <gtest/gtest.h>

TEST(TestNDMap, Simple1DAdd)
{
    std::size_t i = 0;

    covfie::utility::nd_map<std::size_t>(
        std::function([&i](std::size_t x) { i += x; }), 10
    );

    EXPECT_EQ(i, 45);
}

TEST(TestNDMap, Simple2DAdd)
{
    std::size_t i = 0;

    covfie::utility::nd_map<std::size_t, std::size_t>(
        std::function([&i](std::size_t x, std::size_t y) { i += (x + y); }),
        5,
        10
    );

    EXPECT_EQ(i, 325);
}
