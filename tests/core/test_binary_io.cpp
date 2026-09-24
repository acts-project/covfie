/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <utility>

#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <tmp_file.hpp>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/dereference.hpp>
#include <covfie/core/backend/transformer/hilbert.hpp>
#include <covfie/core/backend/transformer/morton.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

namespace {
template <typename Wide, typename Narrow>
void check_index_independent_dimensions()
{
    using storage_t = typename Wide::backend_t;
    typename storage_t::owning_data_t storage(16);
    typename storage_t::non_owning_data_t storage_view(storage);
    for (std::size_t i = 0; i < 16; ++i) {
        storage_view.at(i)[0] = static_cast<float>(i);
        storage_view.at(i)[1] = -static_cast<float>(i);
    }

    covfie::field<Wide> wide(typename Wide::owning_data_t(
        typename Wide::configuration_t{std::size_t{4}, std::size_t{4}},
        std::move(storage)
    ));
    std::stringstream wide_stream;
    wide.dump(wide_stream);

    // The field and layout headers are followed by two size_t dimensions,
    // matching the format used before dimensions had a configurable type.
    std::size_t dimensions[2]{};
    const auto bytes = wide_stream.str();
    std::memcpy(
        dimensions, bytes.data() + 4 * sizeof(uint32_t), sizeof(dimensions)
    );
    EXPECT_EQ(dimensions[0], 4u);
    EXPECT_EQ(dimensions[1], 4u);

    covfie::field<Narrow> narrow(wide_stream);
    std::stringstream narrow_stream;
    narrow.dump(narrow_stream);
    EXPECT_EQ(narrow_stream.str(), bytes);

    covfie::field<Wide> round_trip(narrow_stream);
    covfie::field_view<Wide> wide_view(wide);
    covfie::field_view<Narrow> narrow_view(narrow);
    covfie::field_view<Wide> round_trip_view(round_trip);
    for (unsigned int x = 0; x < 4; ++x) {
        for (unsigned int y = 0; y < 4; ++y) {
            for (std::size_t component = 0; component < 2; ++component) {
                EXPECT_EQ(
                    narrow_view.at(x, y)[component],
                    wide_view.at(x, y)[component]
                );
                EXPECT_EQ(
                    round_trip_view.at(x, y)[component],
                    wide_view.at(x, y)[component]
                );
            }
        }
    }
}
}

TEST(TestBinaryIO, StridedDimensionsUseSizeT)
{
    using storage_t = covfie::backend::array<covfie::vector::float2>;
    check_index_independent_dimensions<
        covfie::backend::strided<covfie::vector::size2, storage_t>,
        covfie::backend::strided<covfie::vector::uint2, storage_t>>();
}

TEST(TestBinaryIO, MortonDimensionsUseSizeT)
{
    using storage_t = covfie::backend::array<covfie::vector::float2>;
    check_index_independent_dimensions<
        covfie::backend::morton<covfie::vector::size2, storage_t, false>,
        covfie::backend::morton<covfie::vector::uint2, storage_t, false>>();
}

TEST(TestBinaryIO, HilbertDimensionsUseSizeT)
{
    using storage_t = covfie::backend::array<covfie::vector::float2>;
    check_index_independent_dimensions<
        covfie::backend::hilbert<covfie::vector::size2, storage_t>,
        covfie::backend::hilbert<covfie::vector::uint2, storage_t>>();
}

TEST(TestBinaryIO, WriteRead1DSingleFloatBuilder)
{
    using field_t =
        covfie::field<covfie::backend::array<covfie::vector::float1>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        5ul}));

    field_t::view_t fv(f);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t::output_t & p = fv.at(i);
        p[0] = static_cast<float>(i);
    }

    boost::filesystem::path ofile = get_tmp_file();

    std::ofstream ofs(ofile.native(), std::ofstream::binary);

    if (!ofs.good()) {
        throw std::runtime_error("Invalid file was somehow opened!");
    }

    f.dump(ofs);
    ofs.close();

    std::ifstream ifs(ofile.native(), std::ifstream::binary);

    if (!ifs.good()) {
        throw std::runtime_error("Invalid file was somehow opened!");
    }

    field_t nf(ifs);
    ifs.close();

    field_t::view_t nfv(nf);

    for (std::size_t i = 0ul; i < 5ul; ++i) {
        field_t::output_t & p = nfv.at(i);
        EXPECT_EQ(p[0], static_cast<float>(i));
    }
}

// TEST(TestBinaryIO, WriteRead1DArrayFloat)
// {
//     covfie::field_builder<float, 1, float, 3> fb(5u);

//     for (std::size_t i = 0; i < 5u; ++i) {
//         decltype(fb)::output_type & p = fb.at_integral(i);
//         p[0] = static_cast<float>(3 * i);
//         p[1] = static_cast<float>(3 * i + 1);
//         p[2] = static_cast<float>(3 * i + 2);
//     }

//     boost::filesystem::path ofile =
//         boost::filesystem::temp_directory_path() /
//         boost::filesystem::unique_path("covfie_test_%%%%_%%%%_%%%%_%%%%.covfie"
//         );

//     covfie::field<float, 1, float, 3, covfie::backend::file_writer>(
//         fb, ofile.native()
//     );

//     covfie::field_builder<float, 1, float, 3> nfb =
//         covfie::io::read_binary_file<float, 1, float, 3>(ofile.native());

//     EXPECT_EQ(nfb.at_integral(0u)[0], 0.0f);
//     EXPECT_EQ(nfb.at_integral(0u)[1], 1.0f);
//     EXPECT_EQ(nfb.at_integral(0u)[2], 2.0f);

//     EXPECT_EQ(nfb.at_integral(1u)[0], 3.0f);
//     EXPECT_EQ(nfb.at_integral(1u)[1], 4.0f);
//     EXPECT_EQ(nfb.at_integral(1u)[2], 5.0f);

//     EXPECT_EQ(nfb.at_integral(2u)[0], 6.0f);
//     EXPECT_EQ(nfb.at_integral(2u)[1], 7.0f);
//     EXPECT_EQ(nfb.at_integral(2u)[2], 8.0f);

//     EXPECT_EQ(nfb.at_integral(3u)[0], 9.0f);
//     EXPECT_EQ(nfb.at_integral(3u)[1], 10.0f);
//     EXPECT_EQ(nfb.at_integral(3u)[2], 11.0f);

//     EXPECT_EQ(nfb.at_integral(4u)[0], 12.0f);
//     EXPECT_EQ(nfb.at_integral(4u)[1], 13.0f);
//     EXPECT_EQ(nfb.at_integral(4u)[2], 14.0f);
// }

// TEST(TestBinaryIO, WriteRead3DSingleFloat)
// {
//     covfie::field_builder<float, 3, float, 1> fb(3u, 4u, 5u);

//     for (std::size_t x = 0; x < 3u; ++x) {
//         for (std::size_t y = 0; y < 4u; ++y) {
//             for (std::size_t z = 0; z < 5u; ++z) {
//                 decltype(fb)::output_type & p = fb.at_integral(x, y, z);
//                 p[0] = static_cast<float>(100 * x + 10 * y + z);
//             }
//         }
//     }

//     boost::filesystem::path ofile =
//         boost::filesystem::temp_directory_path() /
//         boost::filesystem::unique_path("covfie_test_%%%%_%%%%_%%%%_%%%%.covfie"
//         );

//     covfie::field<float, 3, float, 1, covfie::backend::file_writer>(
//         fb, ofile.native()
//     );

//     covfie::field_builder<float, 3, float, 1> nfb =
//         covfie::io::read_binary_file<float, 3, float, 1>(ofile.native());

//     EXPECT_EQ(nfb.at_integral(0u, 0u, 0u)[0], 0.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 0u, 1u)[0], 1.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 0u, 2u)[0], 2.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 0u, 3u)[0], 3.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 0u, 4u)[0], 4.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 0u)[0], 10.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 1u)[0], 11.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 2u)[0], 12.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 3u)[0], 13.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 1u, 4u)[0], 14.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 0u)[0], 20.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 1u)[0], 21.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 2u)[0], 22.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 3u)[0], 23.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 2u, 4u)[0], 24.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 0u)[0], 30.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 1u)[0], 31.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 2u)[0], 32.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 3u)[0], 33.0f);
//     EXPECT_EQ(nfb.at_integral(0u, 3u, 4u)[0], 34.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 0u)[0], 100.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 1u)[0], 101.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 2u)[0], 102.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 3u)[0], 103.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 0u, 4u)[0], 104.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 0u)[0], 110.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 1u)[0], 111.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 2u)[0], 112.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 3u)[0], 113.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 1u, 4u)[0], 114.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 0u)[0], 120.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 1u)[0], 121.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 2u)[0], 122.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 3u)[0], 123.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 2u, 4u)[0], 124.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 0u)[0], 130.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 1u)[0], 131.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 2u)[0], 132.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 3u)[0], 133.0f);
//     EXPECT_EQ(nfb.at_integral(1u, 3u, 4u)[0], 134.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 0u)[0], 200.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 1u)[0], 201.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 2u)[0], 202.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 3u)[0], 203.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 0u, 4u)[0], 204.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 0u)[0], 210.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 1u)[0], 211.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 2u)[0], 212.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 3u)[0], 213.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 1u, 4u)[0], 214.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 0u)[0], 220.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 1u)[0], 221.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 2u)[0], 222.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 3u)[0], 223.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 2u, 4u)[0], 224.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 0u)[0], 230.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 1u)[0], 231.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 2u)[0], 232.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 3u)[0], 233.0f);
//     EXPECT_EQ(nfb.at_integral(2u, 3u, 4u)[0], 234.0f);
// }

TEST(TestBinaryIO, WriteReadDereference2D)
{
    using field_t =
        covfie::field<covfie::backend::dereference<covfie::backend::strided<
            covfie::vector::size2,
            covfie::backend::array<covfie::vector::float2>>>>;
    using inner_field_t = covfie::field<typename field_t::backend_t::backend_t>;

    inner_field_t inner(covfie::make_parameter_pack(
        inner_field_t::backend_t::configuration_t{3ul, 4ul}
    ));
    inner_field_t::view_t inner_view(inner);

    for (std::size_t x = 0ul; x < 3ul; ++x) {
        for (std::size_t y = 0ul; y < 4ul; ++y) {
            inner_view.at(x, y)[0] = static_cast<float>(x);
            inner_view.at(x, y)[1] = static_cast<float>(y);
        }
    }

    field_t f(field_t::storage_t(
        field_t::backend_t::configuration_t{}, inner.backend()
    ));

    std::stringstream ss;

    f.dump(ss);

    field_t nf(ss);
    field_t::view_t nfv(nf);

    for (std::size_t x = 0ul; x < 3ul; ++x) {
        for (std::size_t y = 0ul; y < 4ul; ++y) {
            EXPECT_EQ(nfv.at(x, y)[0], static_cast<float>(x));
            EXPECT_EQ(nfv.at(x, y)[1], static_cast<float>(y));
        }
    }
}

TEST(TestBinaryIO, ReadWrongBackendHeaderReportsItsValue)
{
    using field_t =
        covfie::field<covfie::backend::array<covfie::vector::float1>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        2ul}));

    std::stringstream ss;

    f.dump(ss);

    /*
     * The backend header is the second word of the stream. Corrupt it, and
     * check that the error names the value that was actually found rather
     * than the global header that preceded it.
     */
    std::string data = ss.str();
    const std::uint32_t bad = 0xDEADBEEF;
    std::memcpy(data.data() + sizeof(std::uint32_t), &bad, sizeof(bad));

    std::stringstream bs(data);

    try {
        field_t nf(bs);
        FAIL() << "Deserialization of a corrupt header must throw.";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("DEADBEEF"), std::string::npos)
            << "Error message should report the header that was found: "
            << e.what();
    }
}

TEST(TestBinaryIO, ReadWrongGlobalFooterReportsItsValue)
{
    using field_t =
        covfie::field<covfie::backend::array<covfie::vector::float1>>;

    field_t f(covfie::make_parameter_pack(field_t::backend_t::configuration_t{
        2ul}));

    std::stringstream ss;

    f.dump(ss);

    /*
     * The global footer of the field is the second to last word of the
     * stream. Corrupt it, and check that the error names it.
     */
    std::string data = ss.str();
    const std::uint32_t bad = 0xFEEDFACE;
    std::memcpy(
        data.data() + data.size() - 2 * sizeof(std::uint32_t), &bad, sizeof(bad)
    );

    std::stringstream bs(data);

    try {
        field_t nf(bs);
        FAIL() << "Deserialization of a corrupt footer must throw.";
    } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find("footer"), std::string::npos)
            << e.what();
        EXPECT_NE(std::string(e.what()).find("FEEDFACE"), std::string::npos)
            << "Error message should report the footer that was found: "
            << e.what();
    }
}
