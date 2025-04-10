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
#include <covfie/cuda/backend/primitive/cuda_device_array.hpp>
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
        } else if constexpr (std::is_same_v<scalar_t, std::size_t>) {
            return "UInt64[" + std::to_string(dims) + "]";
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
            covfie::backend::array<covfie::vector::vector_d<scalar_t, N>>>;

        m_sizes = make_size_array<N>();

        covfie::field<canonical_backend_t> f(covfie::make_parameter_pack(
            typename canonical_backend_t::configuration_t(m_sizes)
        ));
        covfie::field_view<canonical_backend_t> fv(f);

        covfie::utility::nd_map<decltype(m_sizes)>(
            [&fv](decltype(m_sizes) t) {
                fv.at(t) = convert_array<N, std::size_t, scalar_t>(t);
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

template <std::size_t N, typename T>
using CudaStridedArrayBackend = covfie::backend::strided<
    covfie::vector::vector_d<std::size_t, N>,
    covfie::backend::cuda_device_array<covfie::vector::vector_d<T, N>>>;

// 1D tests
template <typename B>
using TestCudaStrided1D = TestLookupGeneric<1, B>;
using Backends1D = ::testing::Types<
    CudaStridedArrayBackend<1, std::size_t>,
    CudaStridedArrayBackend<1, float>>;
TYPED_TEST_SUITE(TestCudaStrided1D, Backends1D, NameGenerator);
TYPED_TEST(TestCudaStrided1D, LookUpMoveConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(*(this->m_field), {x});

        EXPECT_EQ(o[0], x);
    }
}

TYPED_TEST(TestCudaStrided1D, LookUpMoveStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_stream), {x}
            );

        EXPECT_EQ(o[0], x);
    }
}

TYPED_TEST(TestCudaStrided1D, LookUpCopyConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_copied), {x}
            );

        EXPECT_EQ(o[0], x);
    }
}

TYPED_TEST(TestCudaStrided1D, LookUpCopyStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_stream_copied), {x}
            );

        EXPECT_EQ(o[0], x);
    }
}

TYPED_TEST(TestCudaStrided1D, LookUpMoveAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_move_assigned), {x}
            );

        EXPECT_EQ(o[0], x);
    }
}

TYPED_TEST(TestCudaStrided1D, LookUpCopyAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_assigned), {x}
            );

        EXPECT_EQ(o[0], x);
    }
}

// 2D tests
template <typename B>
using TestCudaStrided2D = TestLookupGeneric<2, B>;
using Backends2D = ::testing::Types<
    CudaStridedArrayBackend<2, std::size_t>,
    CudaStridedArrayBackend<2, float>>;
TYPED_TEST_SUITE(TestCudaStrided2D, Backends2D, NameGenerator);
TYPED_TEST(TestCudaStrided2D, LookUpMoveConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field), {x, y}
                );

            EXPECT_EQ(o[0], x);
            EXPECT_EQ(o[1], y);
        }
    }
}

TYPED_TEST(TestCudaStrided2D, LookUpMoveStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_stream), {x, y}
                );

            EXPECT_EQ(o[0], x);
            EXPECT_EQ(o[1], y);
        }
    }
}

TYPED_TEST(TestCudaStrided2D, LookUpCopyConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_copied), {x, y}
                );

            EXPECT_EQ(o[0], x);
            EXPECT_EQ(o[1], y);
        }
    }
}

TYPED_TEST(TestCudaStrided2D, LookUpCopyStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_stream_copied), {x, y}
                );

            EXPECT_EQ(o[0], x);
            EXPECT_EQ(o[1], y);
        }
    }
}

TYPED_TEST(TestCudaStrided2D, LookUpMoveAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_move_assigned), {x, y}
                );

            EXPECT_EQ(o[0], x);
            EXPECT_EQ(o[1], y);
        }
    }
}

TYPED_TEST(TestCudaStrided2D, LookUpCopyAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                retrieve_vector<covfie::field<TypeParam>>(
                    *(this->m_field_assigned), {x, y}
                );

            EXPECT_EQ(o[0], x);
            EXPECT_EQ(o[1], y);
        }
    }
}

template <typename B>
using TestCudaStrided3D = TestLookupGeneric<3, B>;
using Backends3D = ::testing::Types<
    CudaStridedArrayBackend<3, std::size_t>,
    CudaStridedArrayBackend<3, float>>;
TYPED_TEST_SUITE(TestCudaStrided3D, Backends3D, NameGenerator);
TYPED_TEST(TestCudaStrided3D, LookUpMoveConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
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

TYPED_TEST(TestCudaStrided3D, LookUpMoveStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_stream), {x, y, z}
                    );

                EXPECT_EQ(o[0], x);
                EXPECT_EQ(o[1], y);
                EXPECT_EQ(o[2], z);
            }
        }
    }
}

TYPED_TEST(TestCudaStrided3D, LookUpCopyConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_copied), {x, y, z}
                    );

                EXPECT_EQ(o[0], x);
                EXPECT_EQ(o[1], y);
                EXPECT_EQ(o[2], z);
            }
        }
    }
}

TYPED_TEST(TestCudaStrided3D, LookUpCopyStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_stream_copied), {x, y, z}
                    );

                EXPECT_EQ(o[0], x);
                EXPECT_EQ(o[1], y);
                EXPECT_EQ(o[2], z);
            }
        }
    }
}

TYPED_TEST(TestCudaStrided3D, LookUpMoveAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_move_assigned), {x, y, z}
                    );

                EXPECT_EQ(o[0], x);
                EXPECT_EQ(o[1], y);
                EXPECT_EQ(o[2], z);
            }
        }
    }
}

TYPED_TEST(TestCudaStrided3D, LookUpCopyAssigned)
{
    for (std::size_t x = 0; x < this->m_sizes[0]; ++x) {
        for (std::size_t y = 0; y < this->m_sizes[1]; ++y) {
            for (std::size_t z = 0; z < this->m_sizes[2]; ++z) {
                std::decay_t<typename covfie::field<TypeParam>::output_t> o =
                    retrieve_vector<covfie::field<TypeParam>>(
                        *(this->m_field_assigned), {x, y, z}
                    );

                EXPECT_EQ(o[0], x);
                EXPECT_EQ(o[1], y);
                EXPECT_EQ(o[2], z);
            }
        }
    }
}
