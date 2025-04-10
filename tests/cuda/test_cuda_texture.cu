/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#include <optional>
#include <utility>

#include <gtest/gtest.h>

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/field_view.hpp>
#include <covfie/core/utility/nd_map.hpp>
#include <covfie/core/vector.hpp>
#include <covfie/cuda/backend/primitive/cuda_texture.hpp>
#include <covfie/cuda/utility/copy.hpp>

#include "retrieve_vector.hpp"

namespace {
class NameGenerator
{
public:
    template <typename B>
    static std::string GetName(int)
    {
        using scalar_t = typename B::covariant_output_t::scalar_t;
        constexpr std::size_t dims = B::covariant_output_t::dimensions;

        if constexpr (std::is_same_v<scalar_t, float>) {
            return "Float[" + std::to_string(dims) + "]";
        }
    }
};

template <std::size_t N, std::size_t... Is>
covfie::array::array<std::size_t, N>
make_size_array_helper(std::index_sequence<Is...> &&)
{
    covfie::array::array<std::size_t, N> rv;
    ((rv[Is] = 10UL), ...);

    return rv;
}

template <std::size_t N>
covfie::array::array<std::size_t, N> make_size_array()
{
    return make_size_array_helper<N>(std::make_index_sequence<N>());
}

template <std::size_t N, typename T, typename U, std::size_t... Is>
covfie::array::array<U, N>
convert_array_helper(const covfie::array::array<T, N> & a, std::index_sequence<Is...> &&)
{
    covfie::array::array<U, N> rv;
    ((rv[Is] = static_cast<U>(a[Is])), ...);
    return rv;
}

template <std::size_t N, typename T, typename U>
covfie::array::array<U, N> convert_array(const covfie::array::array<T, N> & a)
{
    return convert_array_helper<N, T, U>(a, std::make_index_sequence<N>());
}

covfie::array::array<float, 3>
convert_2d_to_3d(const covfie::array::array<std::size_t, 2> & a)
{
    return covfie::array::array<float, 3>{
        static_cast<float>(a[0]),
        static_cast<float>(a[1]),
        static_cast<float>(a[0] + a[1])};
}
}

template <std::size_t N, typename B>
class TestLookupGeneric : public ::testing::Test
{
protected:
    void SetUp() override
    {
        using scalar_t = typename B::covariant_output_t::scalar_t;

        using canonical_backend_t = covfie::backend::strided<
            covfie::vector::vector_d<std::size_t, N>,
            covfie::backend::array<covfie::vector::vector_d<scalar_t, 3>>>;

        m_sizes = make_size_array<N>();

        covfie::field<canonical_backend_t> f(covfie::make_parameter_pack(
            typename canonical_backend_t::configuration_t(m_sizes)
        ));
        covfie::field_view<canonical_backend_t> fv(f);

        covfie::utility::nd_map<decltype(m_sizes)>(
            [&fv](decltype(m_sizes) t) {
                if constexpr (N == 3) {
                    fv.at(t) = convert_array<N, std::size_t, scalar_t>(t);
                } else if constexpr (N == 2) {
                    fv.at(t) = convert_2d_to_3d(t);
                }
            },
            m_sizes
        );

        cudaStream_t stream;
        cudaErrorCheck(cudaStreamCreate(&stream));

        m_field = covfie::field<B>(f);
        m_field_stream =
            covfie::utility::cuda::copy_field_with_stream<covfie::field<B>>(
                f, stream
            );
        m_field_copied = *m_field;
        m_field_stream_copied =
            covfie::utility::cuda::copy_field_with_stream<covfie::field<B>>(
                f, stream
            );
        m_field_assigned = covfie::field<B>();
        m_field_move_assigned = covfie::field<B>();
        *m_field_assigned = *m_field;
        covfie::field<B> tmp = *m_field;
        *m_field_move_assigned = std::move(tmp);
    }

    covfie::array::array<std::size_t, N> m_sizes;
    std::optional<covfie::field<B>> m_field;
    std::optional<covfie::field<B>> m_field_stream;
    std::optional<covfie::field<B>> m_field_copied;
    std::optional<covfie::field<B>> m_field_stream_copied;
    std::optional<covfie::field<B>> m_field_assigned;
    std::optional<covfie::field<B>> m_field_move_assigned;
};

// 2D tests
template <typename B>
using TestCudaLerpBackend2D = TestLookupGeneric<2, B>;
using Backends2D = ::testing::Types<covfie::backend::cuda_texture<
    covfie::vector::float2,
    covfie::vector::float3>>;
TYPED_TEST_SUITE(TestCudaLerpBackend2D, Backends2D, NameGenerator);
TYPED_TEST(TestCudaLerpBackend2D, LookUpMoveConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field),
                    {static_cast<float>(x), static_cast<float>(y)}
                );

            EXPECT_EQ(o[0], static_cast<float>(x));
            EXPECT_EQ(o[1], static_cast<float>(y));
            EXPECT_EQ(o[2], static_cast<float>(x + y));
        }
    }
}

TYPED_TEST(TestCudaLerpBackend2D, LookUpMoveStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_stream),
                    {static_cast<float>(x), static_cast<float>(y)}
                );

            EXPECT_EQ(o[0], static_cast<float>(x));
            EXPECT_EQ(o[1], static_cast<float>(y));
            EXPECT_EQ(o[2], static_cast<float>(x + y));
        }
    }
}

TYPED_TEST(TestCudaLerpBackend2D, LookUpCopyConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_copied),
                    {static_cast<float>(x), static_cast<float>(y)}
                );

            EXPECT_EQ(o[0], static_cast<float>(x));
            EXPECT_EQ(o[1], static_cast<float>(y));
            EXPECT_EQ(o[2], static_cast<float>(x + y));
        }
    }
}

TYPED_TEST(TestCudaLerpBackend2D, LookUpCopyStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_stream_copied),
                    {static_cast<float>(x), static_cast<float>(y)}
                );

            EXPECT_EQ(o[0], static_cast<float>(x));
            EXPECT_EQ(o[1], static_cast<float>(y));
            EXPECT_EQ(o[2], static_cast<float>(x + y));
        }
    }
}

TYPED_TEST(TestCudaLerpBackend2D, LookUpCopyAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_assigned),
                    {static_cast<float>(x), static_cast<float>(y)}
                );

            EXPECT_EQ(o[0], static_cast<float>(x));
            EXPECT_EQ(o[1], static_cast<float>(y));
            EXPECT_EQ(o[2], static_cast<float>(x + y));
        }
    }
}

TYPED_TEST(TestCudaLerpBackend2D, LookUpMoveAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_move_assigned),
                    {static_cast<float>(x), static_cast<float>(y)}
                );

            EXPECT_EQ(o[0], static_cast<float>(x));
            EXPECT_EQ(o[1], static_cast<float>(y));
            EXPECT_EQ(o[2], static_cast<float>(x + y));
        }
    }
}

TYPED_TEST(TestCudaLerpBackend2D, LookUpInterpolated)
{
    for (std::size_t x = 0; x < this->m_sizes[0] - 1; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1] - 1; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field),
                    {static_cast<float>(x) + 0.25f,
                     static_cast<float>(y) + 0.50f}
                );

            EXPECT_EQ(o[0], static_cast<float>(x) + 0.25f);
            EXPECT_EQ(o[1], static_cast<float>(y) + 0.50f);
        }
    }
}

// 3D tests
template <typename B>
using TestCudaLerpBackend3D = TestLookupGeneric<3, B>;
using Backends3D = ::testing::Types<covfie::backend::cuda_texture<
    covfie::vector::float3,
    covfie::vector::float3>>;
TYPED_TEST_SUITE(TestCudaLerpBackend3D, Backends3D, NameGenerator);
TYPED_TEST(TestCudaLerpBackend3D, LookUpMoveConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field),
                        {static_cast<float>(x),
                         static_cast<float>(y),
                         static_cast<float>(z)}
                    );

                EXPECT_EQ(o[0], static_cast<float>(x));
                EXPECT_EQ(o[1], static_cast<float>(y));
                EXPECT_EQ(o[2], static_cast<float>(z));
            }
        }
    }
}

TYPED_TEST(TestCudaLerpBackend3D, LookUpMoveStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_stream),
                        {static_cast<float>(x),
                         static_cast<float>(y),
                         static_cast<float>(z)}
                    );

                EXPECT_EQ(o[0], static_cast<float>(x));
                EXPECT_EQ(o[1], static_cast<float>(y));
                EXPECT_EQ(o[2], static_cast<float>(z));
            }
        }
    }
}

TYPED_TEST(TestCudaLerpBackend3D, LookUpCopyConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_copied),
                        {static_cast<float>(x),
                         static_cast<float>(y),
                         static_cast<float>(z)}
                    );

                EXPECT_EQ(o[0], static_cast<float>(x));
                EXPECT_EQ(o[1], static_cast<float>(y));
                EXPECT_EQ(o[2], static_cast<float>(z));
            }
        }
    }
}

TYPED_TEST(TestCudaLerpBackend3D, LookUpCopyStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_stream_copied),
                        {static_cast<float>(x),
                         static_cast<float>(y),
                         static_cast<float>(z)}
                    );

                EXPECT_EQ(o[0], static_cast<float>(x));
                EXPECT_EQ(o[1], static_cast<float>(y));
                EXPECT_EQ(o[2], static_cast<float>(z));
            }
        }
    }
}

TYPED_TEST(TestCudaLerpBackend3D, LookUpCopyAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_assigned),
                        {static_cast<float>(x),
                         static_cast<float>(y),
                         static_cast<float>(z)}
                    );

                EXPECT_EQ(o[0], static_cast<float>(x));
                EXPECT_EQ(o[1], static_cast<float>(y));
                EXPECT_EQ(o[2], static_cast<float>(z));
            }
        }
    }
}

TYPED_TEST(TestCudaLerpBackend3D, LookUpMoveAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_move_assigned),
                        {static_cast<float>(x),
                         static_cast<float>(y),
                         static_cast<float>(z)}
                    );

                EXPECT_EQ(o[0], static_cast<float>(x));
                EXPECT_EQ(o[1], static_cast<float>(y));
                EXPECT_EQ(o[2], static_cast<float>(z));
            }
        }
    }
}

TYPED_TEST(TestCudaLerpBackend3D, LookUpInterpolated)
{
    for (std::size_t x = 0; x < this->m_sizes[0] - 1; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1] - 1; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2] - 1; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field),
                        {static_cast<float>(x) + 0.25f,
                         static_cast<float>(y) + 0.50f,
                         static_cast<float>(z) + 0.75f}
                    );

                EXPECT_EQ(o[0], static_cast<float>(x) + 0.25f);
                EXPECT_EQ(o[1], static_cast<float>(y) + 0.50f);
                EXPECT_EQ(o[2], static_cast<float>(z) + 0.75f);
            }
        }
    }
}
