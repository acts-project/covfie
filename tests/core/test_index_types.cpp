/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2026 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/hilbert.hpp>
#include <covfie/core/backend/transformer/morton.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

namespace {
template <typename Layout>
void check_layout_conversion()
{
    using strided_t = covfie::backend::strided<
        typename Layout::contravariant_input_t::vector_d,
        typename Layout::backend_t>;
    using index_t = typename Layout::contravariant_input_t::scalar_t;
    // 256 elements also verifies that allocation arithmetic does not wrap
    // when the coordinate scalar is uint8_t.
    covfie::field<strided_t> source(covfie::make_parameter_pack(
        typename strided_t::configuration_t{index_t{16}, index_t{16}}
    ));
    covfie::field_view<strided_t> source_view(source);
    for (unsigned int x = 0; x < 16; ++x) {
        for (unsigned int y = 0; y < 16; ++y) {
            source_view.at(x, y)[0] = static_cast<float>(16 * x + y);
            source_view.at(x, y)[1] = static_cast<float>(y);
        }
    }

    covfie::field<Layout> converted(source);
    EXPECT_EQ(converted.backend().get_backend().get_configuration()[0], 256u);
    covfie::field_view<Layout> view(converted);
    for (unsigned int x = 0; x < 16; ++x) {
        for (unsigned int y = 0; y < 16; ++y) {
            EXPECT_EQ(view.at(x, y)[0], static_cast<float>(16 * x + y));
            EXPECT_EQ(view.at(x, y)[1], static_cast<float>(y));
        }
    }
}

using storage_t = covfie::backend::array<covfie::vector::float2>;
using byte2 = covfie::vector::vector_d<std::uint8_t, 2>;
}

TEST(TestIndexTypes, UnsignedArrayConfiguration)
{
    using array_t =
        covfie::backend::array<covfie::vector::float2, unsigned int>;
    array_t::owning_data_t data(17);
    EXPECT_EQ(data.get_configuration()[0], 17u);
}

TEST(TestIndexTypes, ArraySizeUsesIndexType)
{
    using array_t =
        covfie::backend::array<covfie::vector::float2, std::uint8_t>;
    static_assert(std::is_same_v<
                  decltype(array_t::owning_data_t::m_size),
                  std::uint8_t>);
    static_assert(std::is_same_v<
                  decltype(array_t::non_owning_data_t::m_size),
                  std::uint8_t>);
    array_t::owning_data_t empty(0);
    EXPECT_EQ(empty.get_size(), 0u);
    array_t::owning_data_t maximum(255);
    EXPECT_EQ(maximum.get_size(), 255u);
    EXPECT_EQ(maximum.get_configuration()[0], 255u);
    EXPECT_THROW(array_t::owning_data_t(256), std::overflow_error);
    EXPECT_THROW(
        (array_t::owning_data_t(std::numeric_limits<std::size_t>::max())),
        std::overflow_error
    );

    auto storage = std::make_unique<array_t::vector_t[]>(256);
    auto * original = storage.get();
    EXPECT_THROW(
        array_t::owning_data_t(256, std::move(storage)), std::overflow_error
    );
    EXPECT_EQ(storage.get(), original);
}

TEST(TestIndexTypes, ArraySizeBinaryCompatibility)
{
    using wide_t = covfie::backend::array<covfie::vector::float2>;
    using narrow_t =
        covfie::backend::array<covfie::vector::float2, std::uint8_t>;
    wide_t::owning_data_t wide(17);
    wide_t::non_owning_data_t view(wide);
    for (std::size_t i = 0; i < 17; ++i) {
        view.at(i)[0] = static_cast<float>(i);
        view.at(i)[1] = -static_cast<float>(i);
    }
    std::stringstream wide_stream;
    wide_t::owning_data_t::write_binary(wide_stream, wide);
    auto narrow = narrow_t::owning_data_t::read_binary(wide_stream);
    EXPECT_EQ(narrow.get_size(), 17u);
    std::stringstream narrow_stream;
    narrow_t::owning_data_t::write_binary(narrow_stream, narrow);
    EXPECT_EQ(narrow_stream.str(), wide_stream.str());
    // Two header words, a float-width word, the legacy uint64_t count,
    // the payload, and two footer words.
    EXPECT_EQ(
        narrow_stream.str().size(),
        5 * sizeof(std::uint32_t) + sizeof(std::uint64_t) + 34 * sizeof(float)
    );
    auto restored = wide_t::owning_data_t::read_binary(narrow_stream);
    wide_t::non_owning_data_t restored_view(restored);
    for (std::size_t i = 0; i < 17; ++i) {
        EXPECT_EQ(restored_view.at(i)[0], view.at(i)[0]);
        EXPECT_EQ(restored_view.at(i)[1], view.at(i)[1]);
    }

    wide_t::owning_data_t too_large(256);
    std::stringstream oversized_stream;
    wide_t::owning_data_t::write_binary(oversized_stream, too_large);
    EXPECT_THROW(
        narrow_t::owning_data_t::read_binary(oversized_stream),
        std::overflow_error
    );
}

TEST(TestIndexTypes, MortonAllocation)
{
    check_layout_conversion<
        covfie::backend::morton<covfie::vector::uint2, storage_t, false>>();
    check_layout_conversion<covfie::backend::morton<byte2, storage_t, false>>();
}

TEST(TestIndexTypes, HilbertAllocation)
{
    check_layout_conversion<
        covfie::backend::hilbert<covfie::vector::uint2, storage_t>>();
    check_layout_conversion<covfie::backend::hilbert<byte2, storage_t>>();
}
