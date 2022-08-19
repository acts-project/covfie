/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <optional>

#include <gtest/gtest.h>

#include <covfie/core/backend/initial/array.hpp>
#include <covfie/core/backend/transformer/layout/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/core/vector.hpp>
#include <covfie/cuda/backend/storage/cuda_device_array.hpp>

#include "retrieve_vector.hpp"

template <typename B>
class TestCudaLookupIntegerIndexed3D : public ::testing::Test
{
protected:
    void SetUp() override
    {
        using canonical_backend_t = covfie::backend::layout::strided<
            covfie::vector::ulong3,
            covfie::backend::storage::array<covfie::vector::ulong3>>;

        std::array<std::size_t, 3> sizes;
        sizes.fill(10);

        covfie::field<canonical_backend_t> f(
            canonical_backend_t::configuration_t{sizes}
        );
        covfie::field_view<canonical_backend_t> fv(f);

        for (std::size_t x = 0; x < sizes[0]; ++x) {
            for (std::size_t y = 0; y < sizes[1]; ++y) {
                for (std::size_t z = 0; z < sizes[2]; ++z) {
                    fv.at(x, y, z) = {x, y, z};
                }
            }
        }

        m_field = covfie::field<B>(f);
    }

    std::optional<covfie::field<B>> m_field;
};

using Backends = ::testing::Types<covfie::backend::layout::strided<
    covfie::vector::ulong3,
    covfie::backend::storage::cuda_device_array<covfie::vector::ulong3>>>;
TYPED_TEST_SUITE(TestCudaLookupIntegerIndexed3D, Backends);

TYPED_TEST(TestCudaLookupIntegerIndexed3D, LookUp)
{
    for (std::size_t x = 0; x < 10; ++x) {
        for (std::size_t y = 0; y < 10; ++y) {
            for (std::size_t z = 0; z < 10; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field), {x, y, z}
                    );

                EXPECT_EQ(o[0], x);
                EXPECT_EQ(o[1], y);
                EXPECT_EQ(o[2], z);
            }
        }
    }
}
