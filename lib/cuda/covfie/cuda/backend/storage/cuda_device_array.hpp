/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <memory>

#include <covfie/core/backend/storage/array.hpp>
#include <covfie/core/backend/vector/input.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/cuda/error_check.hpp>
#include <cuda_runtime.h>

namespace covfie::backend::storage {
template <
    CONSTRAINT(concepts::output_vector) _output_vector_t,
    typename _index_t = std::size_t>
struct cuda_device_array {
    using contravariant_input_t = vector::input_scalar<_index_t>;
    using contravariant_output_t = std::tuple<>;
    using covariant_input_t = std::tuple<>;
    using covariant_output_t = vector::add_lvalue_reference<_output_vector_t>;

    using output_vector_t = _output_vector_t;
    static constexpr std::size_t dimensions = output_vector_t::dimensions;

    using value_t = typename output_vector_t::scalar_t[dimensions];
    using vector_t = std::decay_t<typename _output_vector_t::vector_t>;

    struct owning_data_t {
        owning_data_t(
            typename array<output_vector_t, _index_t>::owning_data_t && o
        )
            : m_size(o.m_size)
            , m_ptr(nullptr)
        {
            cudaErrorCheck(cudaMalloc(
                reinterpret_cast<void **>(&m_ptr), m_size * sizeof(vector_t)
            ));
            cudaErrorCheck(cudaMemcpy(
                m_ptr,
                o.m_ptr.get(),
                m_size * sizeof(vector_t),
                cudaMemcpyHostToDevice
            ));
        }

        ~owning_data_t()
        {
            cudaErrorCheck(cudaFree(m_ptr));
        }

        std::size_t m_size;
        vector_t * m_ptr;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr)
        {
        }

        COVFIE_DEVICE typename covariant_output_t::vector_t
        operator[](typename contravariant_input_t::vector_t i) const
        {
            return m_ptr[i];
        }

        vector_t * m_ptr;
    };
};
}
