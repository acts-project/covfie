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

#include <cuda_runtime.h>

#include <covfie/core/backend/initial/array.hpp>
#include <covfie/core/concepts.hpp>
#include <covfie/core/vector.hpp>
#include <covfie/cuda/error_check.hpp>

namespace covfie::backend::storage {
template <
    CONSTRAINT(concepts::vector_descriptor) _output_vector_t,
    typename _index_t = std::size_t>
struct cuda_device_array {
    using contravariant_input_t =
        covfie::vector::scalar_d<covfie::vector::vector_d<_index_t, 1>>;
    using covariant_output_t =
        covfie::vector::array_reference_vector_d<_output_vector_t>;

    using output_vector_t = _output_vector_t;
    static constexpr std::size_t size = output_vector_t::size;

    using value_t = typename output_vector_t::type[size];
    using vector_t = std::decay_t<typename covariant_output_t::vector_t>;

    using configuration_t = utility::nd_size<1>;

    struct owning_data_t {
        explicit owning_data_t(
            std::size_t size, std::unique_ptr<vector_t[]> && ptr
        )
            : m_size(size)
            , m_ptr(nullptr)
        {
            cudaErrorCheck(cudaMalloc(
                reinterpret_cast<void **>(&m_ptr), m_size * sizeof(vector_t)
            ));
            cudaErrorCheck(cudaMemcpy(
                m_ptr,
                ptr.get(),
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

        COVFIE_DEVICE typename covariant_output_t::vector_t
        at(typename contravariant_input_t::vector_t i) const
        {
            return m_ptr[i];
        }

        vector_t * m_ptr;
    };
};
}
