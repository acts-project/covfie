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

#include <covfie/core/backend/primitive/constant.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/field.hpp>

TEST(TestAffineTransformer, AffineConstant1Dto1D)
{
    using field_t = covfie::field<covfie::backend::affine<
        covfie::backend::
            constant<covfie::vector::float1, covfie::vector::float1>>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t(covfie::algebra::affine<1>(
            std::array<std::array<float, 2>, 1>({{{0.f, 5.f}}})
        )),
        field_t::backend_t::backend_t::configuration_t({5.f})
    ));

    field_t::view_t fv(f);

    for (float i = -10.f; i < 10.f; i += 1.f) {
        EXPECT_EQ(fv.at(i)[0], 5.f);
    }
}
