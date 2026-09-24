/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2026 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

namespace {
using storage_t = covfie::backend::array<covfie::vector::float2>;
using small_storage_t =
    covfie::backend::array<covfie::vector::float2, std::uint8_t>;
using small_coordinates_t = covfie::vector::vector_d<std::uint8_t, 2>;
using wide_t = covfie::backend::strided<covfie::vector::size2, storage_t>;
using small_t = covfie::backend::strided<small_coordinates_t, storage_t>;
using small_count_t =
    covfie::backend::strided<covfie::vector::size2, small_storage_t>;

template <typename Backend>
void check_rejected_binary_shape(typename Backend::io_configuration_t sizes)
{
    std::stringstream stream;
    covfie::utility::write_io_header(stream, Backend::IO_MAGIC_HEADER);
    stream.write(reinterpret_cast<const char *>(&sizes), sizeof(sizes));
    const auto end = stream.tellp();

    // No child payload: invalid metadata must be rejected before reading or
    // allocating the underlying array, rather than failing on a later header.
    EXPECT_THROW(
        Backend::owning_data_t::read_binary(stream), std::overflow_error
    );
    EXPECT_TRUE(stream.good());
    EXPECT_EQ(stream.tellg(), end);
}
}

TEST(TestStridedLimits, RejectDimensionNarrowing)
{
    // Offset 255 fits, but the dimension 256 itself does not.
    wide_t::configuration_t sizes{std::size_t{256}, std::size_t{1}};
    EXPECT_THROW(small_t::owning_data_t{sizes}, std::overflow_error);
    EXPECT_THROW(
        covfie::field<small_t>{covfie::make_parameter_pack(sizes)},
        std::overflow_error
    );

    storage_t::owning_data_t storage(256);
    auto * original = storage.m_ptr.get();
    EXPECT_THROW(
        (small_t::owning_data_t{sizes, std::move(storage)}), std::overflow_error
    );
    EXPECT_EQ(storage.m_ptr.get(), original);

    wide_t::owning_data_t source(sizes);
    EXPECT_THROW(small_t::owning_data_t{source}, std::overflow_error);
    EXPECT_THROW(small_t::make_strided_copy(source), std::overflow_error);
}

TEST(TestStridedLimits, RejectFlattenedIndexOverflow)
{
    small_t::configuration_t sizes{std::uint8_t{17}, std::uint8_t{17}};
    EXPECT_THROW(small_t::owning_data_t{sizes}, std::overflow_error);

    storage_t::owning_data_t storage(289);
    auto * original = storage.m_ptr.get();
    EXPECT_THROW(
        (small_t::owning_data_t{sizes, std::move(storage)}), std::overflow_error
    );
    EXPECT_EQ(storage.m_ptr.get(), original);

    wide_t::owning_data_t source(wide_t::configuration_t{
        std::size_t{17}, std::size_t{17}});
    EXPECT_THROW(small_t::owning_data_t{source}, std::overflow_error);
    EXPECT_THROW(small_t::make_strided_copy(source), std::overflow_error);
}

TEST(TestStridedLimits, RejectChildCountNarrowing)
{
    // Every offset fits in uint8_t, but the child's element count does not.
    small_count_t::configuration_t sizes{std::size_t{16}, std::size_t{16}};
    EXPECT_THROW(small_count_t::owning_data_t{sizes}, std::overflow_error);
    wide_t::owning_data_t source(sizes);
    EXPECT_THROW(small_count_t::owning_data_t{source}, std::overflow_error);
    EXPECT_THROW(small_count_t::make_strided_copy(source), std::overflow_error);
}

TEST(TestStridedLimits, RejectVolumeOverflow)
{
    wide_t::configuration_t sizes{
        std::numeric_limits<std::size_t>::max() / 2 + 1, std::size_t{2}};
    EXPECT_THROW(wide_t::owning_data_t{sizes}, std::overflow_error);
    check_rejected_binary_shape<wide_t>(sizes);
}

TEST(TestStridedLimits, RejectNegativeDimensions)
{
    using signed_t = covfie::backend::strided<covfie::vector::int2, storage_t>;
    signed_t::configuration_t sizes{-1, 0};
    EXPECT_THROW(signed_t::owning_data_t{sizes}, std::overflow_error);
    EXPECT_THROW(small_t::owning_data_t{sizes}, std::overflow_error);
}

TEST(TestStridedLimits, RejectBinaryShapeBeforeReadingStorage)
{
    check_rejected_binary_shape<small_t>({std::size_t{256}, std::size_t{1}});
    check_rejected_binary_shape<small_t>({std::size_t{257}, std::size_t{1}});
    check_rejected_binary_shape<small_t>({std::size_t{17}, std::size_t{17}});
    check_rejected_binary_shape<small_count_t>(
        {std::size_t{16}, std::size_t{16}}
    );
}

TEST(TestStridedLimits, AcceptLargestFlattenedIndex)
{
    // The coordinate type holds offsets 0..255; a size_t child holds count 256.
    small_t::owning_data_t data(small_t::configuration_t{
        std::uint8_t{16}, std::uint8_t{16}});
    EXPECT_EQ(data.get_backend().get_size(), 256u);
    small_t::non_owning_data_t view(data);
    view.at({std::uint8_t{15}, std::uint8_t{15}})[0] = 42.f;
    EXPECT_EQ(data.get_backend().m_ptr[255][0], 42.f);

    std::stringstream stream;
    small_t::owning_data_t::write_binary(stream, data);
    auto restored = small_t::owning_data_t::read_binary(stream);
    small_t::non_owning_data_t restored_view(restored);
    EXPECT_EQ(restored_view.at({std::uint8_t{15}, std::uint8_t{15}})[0], 42.f);
}

TEST(TestStridedLimits, AcceptLargestDimensionAndChildCount)
{
    using narrow_t =
        covfie::backend::strided<small_coordinates_t, small_storage_t>;
    wide_t::configuration_t sizes{std::size_t{255}, std::size_t{1}};
    narrow_t::owning_data_t data(sizes);
    EXPECT_EQ(data.get_configuration()[0], 255u);
    EXPECT_EQ(data.get_backend().get_size(), 255u);
    narrow_t::non_owning_data_t view(data);
    view.at({std::uint8_t{254}, std::uint8_t{0}})[0] = 42.f;
    EXPECT_EQ(data.get_backend().m_ptr[254][0], 42.f);
}

TEST(TestStridedLimits, AcceptEmptyGridWithoutSpuriousProductOverflow)
{
    using backend_t =
        covfie::backend::strided<covfie::vector::size3, storage_t>;
    backend_t::owning_data_t data(backend_t::configuration_t{
        std::numeric_limits<std::size_t>::max(),
        std::numeric_limits<std::size_t>::max(),
        std::size_t{0}});
    EXPECT_EQ(data.get_backend().get_size(), 0u);
    small_t::owning_data_t empty;
    EXPECT_EQ(empty.get_backend().get_size(), 0u);
}
