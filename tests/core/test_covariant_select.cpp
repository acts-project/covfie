/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/identity.hpp>
#include <covfie/core/backend/transformer/covariant_select.hpp>
#include <covfie/core/field.hpp>

TEST(TestCovariantSelect, Identity3DSelect1)
{
    using field_t = covfie::field<covfie::backend::covariant_select<
        covfie::backend::identity<covfie::vector::int3>,
        1>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            for (int z = -2; z < 2; z++) {
                EXPECT_EQ(fv.at(x, y, z)[0], y);
            }
        }
    }
}

TEST(TestCovariantSelect, Identity3DSelect20)
{
    using field_t = covfie::field<covfie::backend::covariant_select<
        covfie::backend::identity<covfie::vector::int3>,
        2,
        0>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            for (int z = -2; z < 2; z++) {
                EXPECT_EQ(fv.at(x, y, z)[0], z);
                EXPECT_EQ(fv.at(x, y, z)[1], x);
            }
        }
    }
}

TEST(TestCovariantSelect, Identity3DSelectAll)
{
    using field_t = covfie::field<covfie::backend::covariant_select<
        covfie::backend::identity<covfie::vector::int3>,
        0,
        1,
        2>>;

    field_t f(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t({}),
        field_t::backend_t::backend_t::configuration_t({})
    ));
    field_t::view_t fv(f);

    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            for (int z = -2; z < 2; z++) {
                EXPECT_EQ(fv.at(x, y, z)[0], x);
                EXPECT_EQ(fv.at(x, y, z)[1], y);
                EXPECT_EQ(fv.at(x, y, z)[2], z);
            }
        }
    }
}
