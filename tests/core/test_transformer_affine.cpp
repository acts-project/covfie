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
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/field.hpp>
#include <gtest/gtest.h>

TEST(TestAffineTransformer, AffineConstant1Dto1D)
{
    using field_t = covfie::field<covfie::backend::transformer::affine<
        covfie::backend::constant<1, covfie::backend::vector::output::float1>>>;

    field_t f(
        field_t::backend_t::configuration_data_t({{5.f}, {5.f}}),
        field_t::backend_t::backend_t::configuration_data_t({5.f})
    );

    field_t::view_t fv(f);

    for (float i = -10.f; i < 10.f; i += 1.f) {
        EXPECT_EQ(fv.at(i)[0], 5.f);
    }
}
