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
#include <covfie/sycl/backend/primitive/sycl_device_array.hpp>
#include <covfie/sycl/utility/copy.hpp>

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

        m_queue = ::sycl::queue(::sycl::default_selector_v);

        m_field =
            covfie::utility::sycl::copy_field_with_queue<covfie::field<B>>(
                f, m_queue
            );
        m_field_copied = *m_field;
        m_field_assigned = *m_field;
        m_field_move_assigned = *m_field;
        *m_field_assigned = *m_field;
        covfie::field<B> tmp = *m_field;
        *m_field_move_assigned = std::move(tmp);
    }

    ::sycl::queue m_queue;
    std::size_t m_size;
    std::optional<covfie::field<B>> m_field;
    std::optional<covfie::field<B>> m_field_copied;
    std::optional<covfie::field<B>> m_field_assigned;
    std::optional<covfie::field<B>> m_field_move_assigned;
};

template <std::size_t N, typename T>
using SyclArrayBackend =
    covfie::backend::sycl_device_array<covfie::vector::vector_d<T, N>>;

// 1D tests
template <typename B>
using TestSyclArray = TestLookupGeneric<B>;
using BackendsInteger1D = ::testing::Types<
    SyclArrayBackend<1, std::size_t>,
    SyclArrayBackend<1, int>,
    SyclArrayBackend<1, float>,
    SyclArrayBackend<1, double>,
    SyclArrayBackend<2, std::size_t>,
    SyclArrayBackend<2, int>,
    SyclArrayBackend<2, float>,
    SyclArrayBackend<2, double>,
    SyclArrayBackend<3, std::size_t>,
    SyclArrayBackend<3, int>,
    SyclArrayBackend<3, float>,
    SyclArrayBackend<3, double>>;
TYPED_TEST_SUITE(TestSyclArray, BackendsInteger1D, NameGenerator);
TYPED_TEST(TestSyclArray, LookUpMoveConstructed)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field), {x}, this->m_queue
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

TYPED_TEST(TestSyclArray, LookUpCopyConstructed)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_copied), {x}, this->m_queue
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

TYPED_TEST(TestSyclArray, LookUpCopyAssigned)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_assigned), {x}, this->m_queue
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

TYPED_TEST(TestSyclArray, LookUpMoveAssigned)
{
    for (std::size_t x = 0; x < this->m_size; ++x) {
        std::decay_t<typename covfie::field<TypeParam>::output_t> o =
            retrieve_vector<covfie::field<TypeParam>>(
                *(this->m_field_move_assigned), {x}, this->m_queue
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
