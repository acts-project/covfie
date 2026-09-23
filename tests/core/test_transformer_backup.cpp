/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstddef>
#include <sstream>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/backup.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>

namespace {
using inner_backend_t = covfie::backend::strided<
    covfie::vector::size2,
    covfie::backend::array<covfie::vector::float2>>;
using field_t = covfie::field<covfie::backend::backup<inner_backend_t>>;

/*
 * Build a 4x4 field of f(x, y) = (x, y), guarded by a backup layer which
 * derives its bounds from the size of the underlying field.
 */
field_t make_field()
{
    using inner_field_t = covfie::field<inner_backend_t>;

    inner_field_t inner(
        covfie::make_parameter_pack(inner_backend_t::configuration_t{4ul, 4ul})
    );
    inner_field_t::view_t inner_view(inner);

    for (std::size_t x = 0ul; x < 4ul; ++x) {
        for (std::size_t y = 0ul; y < 4ul; ++y) {
            inner_view.at(x, y)[0] = static_cast<float>(x);
            inner_view.at(x, y)[1] = static_cast<float>(y);
        }
    }

    return field_t(field_t::storage_t(inner.backend()));
}
}

TEST(TestTransformerBackup, ConfigurationFromBackendSize)
{
    field_t f = make_field();

    field_t::backend_t::configuration_t conf = f.backend().get_configuration();

    /*
     * A 4x4 field has valid indices zero through three, and the bounds of the
     * backup layer are inclusive, so the maximum must be three and not four.
     */
    for (std::size_t i = 0ul; i < 2ul; ++i) {
        EXPECT_EQ(conf.min[i], 0ul);
        EXPECT_EQ(conf.max[i], 3ul);
        EXPECT_EQ(conf.default_value[i], 0.f);
    }
}

TEST(TestTransformerBackup, UnsignedIndexConfigurationFromBackendSize)
{
    using inner_t = covfie::backend::strided<
        covfie::vector::uint2,
        covfie::backend::array<covfie::vector::float2>>;
    using backup_t = covfie::backend::backup<inner_t>;
    backup_t::owning_data_t data(inner_t::configuration_t{2u, 3u});
    auto conf = data.get_configuration();
    for (std::size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(conf.min[i], 0u);
        EXPECT_EQ(conf.max[i], i + 1);
        EXPECT_EQ(conf.default_value[i], 0.f);
    }
    backup_t::non_owning_data_t view(data);
    auto outside = view.at({2u, 3u});
    EXPECT_EQ(outside[0], 0.f);
    EXPECT_EQ(outside[1], 0.f);
}

TEST(TestTransformerBackup, InRangeLookup)
{
    field_t f = make_field();
    field_t::view_t fv(f);

    for (std::size_t x = 0ul; x < 4ul; ++x) {
        for (std::size_t y = 0ul; y < 4ul; ++y) {
            EXPECT_EQ(fv.at(x, y)[0], static_cast<float>(x));
            EXPECT_EQ(fv.at(x, y)[1], static_cast<float>(y));
        }
    }
}

TEST(TestTransformerBackup, OutOfRangeLookupGivesDefault)
{
    field_t f = make_field();
    field_t::view_t fv(f);

    /*
     * Index four lies one past the end of the 4x4 field in both dimensions,
     * so the backup layer must return the default value instead of reading
     * out of bounds.
     */
    for (std::size_t i = 0ul; i < 5ul; ++i) {
        EXPECT_EQ(fv.at(4ul, i)[0], 0.f);
        EXPECT_EQ(fv.at(4ul, i)[1], 0.f);
        EXPECT_EQ(fv.at(i, 4ul)[0], 0.f);
        EXPECT_EQ(fv.at(i, 4ul)[1], 0.f);
    }
}

TEST(TestTransformerBackup, WriteRead)
{
    field_t f = make_field();

    std::stringstream ss;

    f.dump(ss);

    field_t nf(ss);
    field_t::view_t nfv(nf);

    /*
     * The bounds and the default value are stored in the file, so the
     * deserialized field must guard its range in the same way.
     */
    for (std::size_t x = 0ul; x < 4ul; ++x) {
        for (std::size_t y = 0ul; y < 4ul; ++y) {
            EXPECT_EQ(nfv.at(x, y)[0], static_cast<float>(x));
            EXPECT_EQ(nfv.at(x, y)[1], static_cast<float>(y));
        }
    }

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        EXPECT_EQ(nfv.at(4ul, i)[0], 0.f);
        EXPECT_EQ(nfv.at(i, 4ul)[1], 0.f);
    }
}
