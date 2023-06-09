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

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/vector.hpp>

TEST(TestFieldViewCPUArrayBackend, WriteRead1DSingleFloat)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::ulong1,
        covfie::backend::array<covfie::vector::float1>>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        5u}));
    field_t::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t j = 0; j < 1; ++j) {
            fv.at(x)[j] = 1000. * x + 1. * j;
        }
    }

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t j = 0; j < 1; ++j) {
            EXPECT_EQ(fv.at(x)[j], 1000. * x + 1. * j);
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead1DArrayFloat)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::ulong1,
        covfie::backend::array<covfie::vector::float3>>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        5u}));
    field_t::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t j = 0; j < 3; ++j) {
            fv.at(x)[j] = 1000. * x + 1. * j;
        }
    }

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(fv.at(x)[j], 1000. * x + 1. * j);
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead2DSingleFloat)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::ulong2,
        covfie::backend::array<covfie::vector::float1>>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        5u, 7u}));
    field_t::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t j = 0; j < 1; ++j) {
                fv.at(x, y)[j] = 1000. * x + 100. * y + 1. * j;
            }
        }
    }

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t j = 0; j < 1; ++j) {
                EXPECT_EQ(fv.at(x, y)[j], 1000. * x + 100. * y + 1. * j);
            }
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead2DArrayFloat)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::ulong2,
        covfie::backend::array<covfie::vector::float3>>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        5u, 7u}));
    field_t::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t j = 0; j < 3; ++j) {
                fv.at(x, y)[j] = 1000. * x + 100. * y + 1. * j;
            }
        }
    }

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t j = 0; j < 3; ++j) {
                EXPECT_EQ(fv.at(x, y)[j], 1000. * x + 100. * y + 1. * j);
            }
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead3DSingleFloat)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::ulong3,
        covfie::backend::array<covfie::vector::float1>>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        5u, 7u, 2u}));
    field_t::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t z = 0; z < 2; ++z) {
                for (std::size_t j = 0; j < 1; ++j) {
                    fv.at(x, y, z)[j] = 1000. * x + 100. * y + 10. * z + 1. * j;
                }
            }
        }
    }

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t z = 0; z < 2; ++z) {
                for (std::size_t j = 0; j < 1; ++j) {
                    EXPECT_EQ(
                        fv.at(x, y, z)[j],
                        1000. * x + 100. * y + 10. * z + 1. * j
                    );
                }
            }
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead3DArrayFloat)
{
    using field_t = covfie::field<covfie::backend::strided<
        covfie::vector::ulong3,
        covfie::backend::array<covfie::vector::float3>>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        5u, 7u, 2u}));
    field_t::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t z = 0; z < 2; ++z) {
                for (std::size_t j = 0; j < 3; ++j) {
                    fv.at(x, y, z)[j] = 1000. * x + 100. * y + 10. * z + 1. * j;
                }
            }
        }
    }

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t z = 0; z < 2; ++z) {
                for (std::size_t j = 0; j < 3; ++j) {
                    EXPECT_EQ(
                        fv.at(x, y, z)[j],
                        1000. * x + 100. * y + 10. * z + 1. * j
                    );
                }
            }
        }
    }
}
