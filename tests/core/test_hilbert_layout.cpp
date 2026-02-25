/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstddef>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/hilbert.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/vector.hpp>

TEST(TestHilbertLayout, RoundTrip2D)
{
    // Build a 4x4 strided field backed by a writable array
    using strided_t = covfie::field<covfie::backend::strided<
        covfie::vector::size2,
        covfie::backend::array<covfie::vector::float1>>>;

    strided_t src(covfie::make_parameter_pack(
        strided_t::backend_t::configuration_t{4u, 4u}
    ));
    strided_t::view_t sv(src);

    for (std::size_t x = 0; x < 4; ++x) {
        for (std::size_t y = 0; y < 4; ++y) {
            sv.at(x, y)[0] = static_cast<float>(x * 4 + y);
        }
    }

    // Copy into a hilbert field - exercises make_hilbert_copy
    using hilbert_t = covfie::field<covfie::backend::hilbert<
        covfie::vector::size2,
        covfie::backend::array<covfie::vector::float1>>>;

    hilbert_t dst(src);

    // Read back via at() - exercises non_owning_data_t::at()
    hilbert_t::view_t hv(dst);

    for (std::size_t x = 0; x < 4; ++x) {
        for (std::size_t y = 0; y < 4; ++y) {
            EXPECT_EQ(hv.at(x, y)[0], static_cast<float>(x * 4 + y));
        }
    }
}
