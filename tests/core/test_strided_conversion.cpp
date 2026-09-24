/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2026 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstddef>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <covfie/core/array.hpp>
#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/clamp.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

TEST(TestStridedConversion, ArrayScalarConversion)
{
    using source_t = covfie::array::array<std::size_t, 1>;
    using target_t = covfie::array::array<unsigned int, 1>;
    static_assert(std::is_convertible_v<source_t, target_t>);
    static_assert(!std::is_convertible_v<
                  source_t,
                  covfie::array::array<unsigned int, 2>>);

    source_t source{std::size_t{7}};
    target_t target = source;
    source_t round_trip = target;
    EXPECT_EQ(target[0], 7u);
    EXPECT_EQ(round_trip[0], source[0]);
}

TEST(TestStridedConversion, ConfigurationAndStorage)
{
    using storage_t = covfie::backend::array<covfie::vector::float3>;
    using target_t = covfie::backend::strided<covfie::vector::uint3, storage_t>;
    storage_t::owning_data_t storage(24);
    storage_t::non_owning_data_t storage_view(storage);
    storage_view.at(23)[0] = 42.f;
    auto * original = &storage_view.at(23);

    target_t::owning_data_t braced({2u, 3u, 4u}, std::move(storage));
    target_t::non_owning_data_t braced_view(braced);
    EXPECT_EQ(&braced_view.at({1u, 2u, 3u}), original);

    covfie::utility::nd_size<3, std::size_t> sizes{
        std::size_t{2}, std::size_t{3}, std::size_t{4}};
    target_t::owning_data_t converted(sizes, std::move(braced.get_backend()));
    target_t::non_owning_data_t converted_view(converted);
    EXPECT_EQ(&converted_view.at({1u, 2u, 3u}), original);
    EXPECT_EQ(converted_view.at({1u, 2u, 3u})[0], 42.f);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(converted.get_configuration()[i], sizes[i]);
    }
}

TEST(TestStridedConversion, IndexConversionAndClampInsertion)
{
    using storage_t = covfie::backend::array<covfie::vector::float3>;
    using source_t = covfie::backend::strided<covfie::vector::size3, storage_t>;
    using target_t = covfie::backend::strided<covfie::vector::uint3, storage_t>;
    using io_t = covfie::backend::affine<covfie::backend::linear<source_t>>;
    using runtime_t = covfie::backend::affine<
        covfie::backend::linear<covfie::backend::clamp<target_t>>>;

    covfie::field<source_t> source(
        covfie::make_parameter_pack(source_t::configuration_t{
            std::size_t{2}, std::size_t{3}, std::size_t{4}})
    );
    covfie::field_view<source_t> source_view(source);
    for (unsigned int x = 0; x < 2; ++x) {
        for (unsigned int y = 0; y < 3; ++y) {
            for (unsigned int z = 0; z < 4; ++z) {
                auto & value = source_view.at(x, y, z);
                value[0] = static_cast<float>(x);
                value[1] = static_cast<float>(y);
                value[2] = static_cast<float>(z);
            }
        }
    }

    covfie::field<io_t> io(covfie::make_parameter_pack(
        io_t::configuration_t{},
        io_t::backend_t::configuration_t{},
        source.backend()
    ));
    covfie::field<runtime_t> converted(io);
    const auto & clamped = converted.backend().get_backend().get_backend();
    const auto & strided = clamped.get_backend();
    target_t::non_owning_data_t view(strided);
    auto sizes = strided.get_configuration();
    auto bounds = clamped.get_configuration();
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(sizes[i], source.backend().get_configuration()[i]);
        EXPECT_EQ(bounds.min[i], 0u);
        EXPECT_EQ(bounds.max[i], sizes[i] - 1);
    }
    for (unsigned int x = 0; x < 2; ++x) {
        for (unsigned int y = 0; y < 3; ++y) {
            for (unsigned int z = 0; z < 4; ++z) {
                auto value = view.at({x, y, z});
                EXPECT_EQ(value[0], static_cast<float>(x));
                EXPECT_EQ(value[1], static_cast<float>(y));
                EXPECT_EQ(value[2], static_cast<float>(z));
            }
        }
    }
}
