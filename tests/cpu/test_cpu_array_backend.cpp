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
#include <covfie/core/backend/vector/input.hpp>
#include <covfie/core/backend/vector/output.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/cpu/backend/cpu_array.hpp>
#include <gtest/gtest.h>

TEST(TestFieldViewCPUArrayBackend, WriteRead1DSingleFloat)
{
    using field_t1 = covfie::field<
        covfie::backend::builder<1, covfie::backend::vector::output::float1>>;
    using field_t2 = covfie::field<
        covfie::backend::cpu_array<1, covfie::backend::vector::output::float1>>;

    field_t1 f(field_t1::backend_t::configuration_data_t{5u});
    field_t1::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t j = 0; j < 1; ++j) {
            fv.at(x)[j] = 1000. * x + 1. * j;
        }
    }

    field_t2 nf(f);
    field_t2::view_t nfv(nf);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t j = 0; j < 1; ++j) {
            EXPECT_EQ(nfv.at(x)[j], 1000. * x + 1. * j);
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead1DArrayFloat)
{
    using field_t1 = covfie::field<
        covfie::backend::builder<1, covfie::backend::vector::output::float3>>;
    using field_t2 = covfie::field<
        covfie::backend::cpu_array<1, covfie::backend::vector::output::float3>>;

    field_t1 f(field_t1::backend_t::configuration_data_t{5u});
    field_t1::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t j = 0; j < 3; ++j) {
            fv.at(x)[j] = 1000. * x + 1. * j;
        }
    }

    field_t2 nf(f);
    field_t2::view_t nfv(nf);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_EQ(nfv.at(x)[j], 1000. * x + 1. * j);
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead2DSingleFloat)
{
    using field_t1 = covfie::field<
        covfie::backend::builder<2, covfie::backend::vector::output::float1>>;
    using field_t2 = covfie::field<
        covfie::backend::cpu_array<2, covfie::backend::vector::output::float1>>;

    field_t1 f(field_t1::backend_t::configuration_data_t{5u, 7u});
    field_t1::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t j = 0; j < 1; ++j) {
                fv.at(x, y)[j] = 1000. * x + 100. * y + 1. * j;
            }
        }
    }

    field_t2 nf(f);
    field_t2::view_t nfv(nf);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t j = 0; j < 1; ++j) {
                EXPECT_EQ(nfv.at(x, y)[j], 1000. * x + 100. * y + 1. * j);
            }
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead2DArrayFloat)
{
    using field_t1 = covfie::field<
        covfie::backend::builder<2, covfie::backend::vector::output::float3>>;
    using field_t2 = covfie::field<
        covfie::backend::cpu_array<2, covfie::backend::vector::output::float3>>;

    field_t1 f(field_t1::backend_t::configuration_data_t{5u, 7u});
    field_t1::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t j = 0; j < 3; ++j) {
                fv.at(x, y)[j] = 1000. * x + 100. * y + 1. * j;
            }
        }
    }

    field_t2 nf(f);
    field_t2::view_t nfv(nf);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t j = 0; j < 3; ++j) {
                EXPECT_EQ(nfv.at(x, y)[j], 1000. * x + 100. * y + 1. * j);
            }
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead3DSingleFloat)
{
    using field_t1 = covfie::field<
        covfie::backend::builder<3, covfie::backend::vector::output::float1>>;
    using field_t2 = covfie::field<
        covfie::backend::cpu_array<3, covfie::backend::vector::output::float1>>;

    field_t1 f(field_t1::backend_t::configuration_data_t{5u, 7u, 2u});
    field_t1::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t z = 0; z < 2; ++z) {
                for (std::size_t j = 0; j < 1; ++j) {
                    fv.at(x, y, z)[j] = 1000. * x + 100. * y + 10. * z + 1. * j;
                }
            }
        }
    }

    field_t2 nf(f);
    field_t2::view_t nfv(nf);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t z = 0; z < 2; ++z) {
                for (std::size_t j = 0; j < 1; ++j) {
                    EXPECT_EQ(
                        nfv.at(x, y, z)[j],
                        1000. * x + 100. * y + 10. * z + 1. * j
                    );
                }
            }
        }
    }
}

TEST(TestFieldViewCPUArrayBackend, WriteRead3DArrayFloat)
{
    using field_t1 = covfie::field<
        covfie::backend::builder<3, covfie::backend::vector::output::float3>>;
    using field_t2 = covfie::field<
        covfie::backend::cpu_array<3, covfie::backend::vector::output::float3>>;

    field_t1 f(field_t1::backend_t::configuration_data_t{5u, 7u, 2u});
    field_t1::view_t fv(f);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t z = 0; z < 2; ++z) {
                for (std::size_t j = 0; j < 3; ++j) {
                    fv.at(x, y, z)[j] = 1000. * x + 100. * y + 10. * z + 1. * j;
                }
            }
        }
    }

    field_t2 nf(f);
    field_t2::view_t nfv(nf);

    for (std::size_t x = 0; x < 5; ++x) {
        for (std::size_t y = 0; y < 7; ++y) {
            for (std::size_t z = 0; z < 2; ++z) {
                for (std::size_t j = 0; j < 3; ++j) {
                    EXPECT_EQ(
                        nfv.at(x, y, z)[j],
                        1000. * x + 100. * y + 10. * z + 1. * j
                    );
                }
            }
        }
    }
}
