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
template <std::size_t N, typename T, std::size_t... Is>
covfie::array::array<T, N>
create_test_array_helper(std::size_t i, std::index_sequence<Is...> &&)
{
    covfie::array::array<T, N> rv;
    ((rv[Is] = static_cast<T>(i + Is)), ...);
    return rv;
}

template <std::size_t N, typename T>
covfie::array::array<T, N> create_test_array(std::size_t i)
{
    return create_test_array_helper<N, T>(i, std::make_index_sequence<N>());
}

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
        } else if constexpr (std::is_same_v<scalar_t, int>) {
            return "Int32[" + std::to_string(dims) + "]";
        } else if constexpr (std::is_same_v<scalar_t, double>) {
            return "Double[" + std::to_string(dims) + "]";
        }
    }
};
}

template <typename B>
class TestLookupGeneric : public ::testing::Test
{
protected:
    void SetUp() override
    {
        using vector_t = typename B::covariant_output_t::vector_d;
        using scalar_t = typename B::covariant_output_t::scalar_t;

        using canonical_backend_t = covfie::backend::array<vector_t>;

        m_size = 1000UL;

        covfie::field<canonical_backend_t> f(covfie::make_parameter_pack(
            typename canonical_backend_t::configuration_t(m_size)
        ));

        covfie::field_view<canonical_backend_t> fv(f);

        for (std::size_t i = 0; i < m_size; ++i) {
            fv.at(i) =
                create_test_array<B::covariant_output_t::dimensions, scalar_t>(i
                );
            ;
        }

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

    std::size_t m_size;
    std::optional<covfie::field<B>> m_field;
    std::optional<covfie::field<B>> m_field_stream;
    std::optional<covfie::field<B>> m_field_copied;
    std::optional<covfie::field<B>> m_field_stream_copied;
    std::optional<covfie::field<B>> m_field_assigned;
    std::optional<covfie::field<B>> m_field_move_assigned;
};

template <std::size_t N, typename T>
using CudaArrayBackend =
    covfie::backend::cuda_device_array<covfie::vector::vector_d<T, N>>;

// 1D tests
template <typename B>
using TestCudaArray = TestLookupGeneric<B>;
using BackendsInteger1D = ::testing::Types<
    CudaArrayBackend<1, std::size_t>,
    CudaArrayBackend<1, int>,
    CudaArrayBackend<1, float>,
    CudaArrayBackend<1, double>,
    CudaArrayBackend<2, std::size_t>,
    CudaArrayBackend<2, int>,
    CudaArrayBackend<2, float>,
    CudaArrayBackend<2, double>,
    CudaArrayBackend<3, std::size_t>,
    CudaArrayBackend<3, int>,
    CudaArrayBackend<3, float>,
    CudaArrayBackend<3, double>>;
TYPED_TEST_SUITE(TestCudaArray, BackendsInteger1D, NameGenerator);
TYPED_TEST(TestCudaArray, LookUpMoveConstructed)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(*(this->m_field), {x});

        EXPECT_EQ(o[0], x);
        if constexpr (TypeParam::covariant_output_t::dimensions >= 2) {
            EXPECT_EQ(o[1], x + 1);
        }
        if constexpr (TypeParam::covariant_output_t::dimensions >= 3) {
            EXPECT_EQ(o[2], x + 2);
        }
    }
}

TYPED_TEST(TestCudaArray, LookUpMoveStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_stream), {x}
            );

        EXPECT_EQ(o[0], x);
        if constexpr (TypeParam::covariant_output_t::dimensions >= 2) {
            EXPECT_EQ(o[1], x + 1);
        }
        if constexpr (TypeParam::covariant_output_t::dimensions >= 3) {
            EXPECT_EQ(o[2], x + 2);
        }
    }
}

TYPED_TEST(TestCudaArray, LookUpCopyConstructed)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_stream), {x}
            );

        EXPECT_EQ(o[0], x);
        if constexpr (TypeParam::covariant_output_t::dimensions >= 2) {
            EXPECT_EQ(o[1], x + 1);
        }
        if constexpr (TypeParam::covariant_output_t::dimensions >= 3) {
            EXPECT_EQ(o[2], x + 2);
        }
    }

    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_copied), {x}
            );

        EXPECT_EQ(o[0], x);
        if constexpr (TypeParam::covariant_output_t::dimensions >= 2) {
            EXPECT_EQ(o[1], x + 1);
        }
        if constexpr (TypeParam::covariant_output_t::dimensions >= 3) {
            EXPECT_EQ(o[2], x + 2);
        }
    }
}

TYPED_TEST(TestCudaArray, LookUpCopyStreamConstructed)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_stream_copied), {x}
            );

        EXPECT_EQ(o[0], x);
        if constexpr (TypeParam::covariant_output_t::dimensions >= 2) {
            EXPECT_EQ(o[1], x + 1);
        }
        if constexpr (TypeParam::covariant_output_t::dimensions >= 3) {
            EXPECT_EQ(o[2], x + 2);
        }
    }

    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_copied), {x}
            );

        EXPECT_EQ(o[0], x);
        if constexpr (TypeParam::covariant_output_t::dimensions >= 2) {
            EXPECT_EQ(o[1], x + 1);
        }
        if constexpr (TypeParam::covariant_output_t::dimensions >= 3) {
            EXPECT_EQ(o[2], x + 2);
        }
    }
}

TYPED_TEST(TestCudaArray, LookUpCopyAssigned)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_stream_copied), {x}
            );

        EXPECT_EQ(o[0], x);
        if constexpr (TypeParam::covariant_output_t::dimensions >= 2) {
            EXPECT_EQ(o[1], x + 1);
        }
        if constexpr (TypeParam::covariant_output_t::dimensions >= 3) {
            EXPECT_EQ(o[2], x + 2);
        }
    }

    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_assigned), {x}
            );

        EXPECT_EQ(o[0], x);
        if constexpr (TypeParam::covariant_output_t::dimensions >= 2) {
            EXPECT_EQ(o[1], x + 1);
        }
        if constexpr (TypeParam::covariant_output_t::dimensions >= 3) {
            EXPECT_EQ(o[2], x + 2);
        }
    }
}

TYPED_TEST(TestCudaArray, LookUpMoveAssigned)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_move_assigned), {x}
            );

        EXPECT_EQ(o[0], x);
        if constexpr (TypeParam::covariant_output_t::dimensions >= 2) {
            EXPECT_EQ(o[1], x + 1);
        }
        if constexpr (TypeParam::covariant_output_t::dimensions >= 3) {
            EXPECT_EQ(o[2], x + 2);
        }
    }
}
