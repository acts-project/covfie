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

#include <covfie/cuda/error_check.hpp>
#include <cuda_runtime.h>

namespace covfie::backend::storage {
template <typename _value_t, std::size_t _dims, typename _index_t = std::size_t>
struct cuda_device_array {
    static constexpr std::size_t dims = _dims;

    using value_t = _value_t[_dims];
    using index_t = _index_t;

    struct owning_data_t {
        owning_data_t(std::unique_ptr<value_t[]> && ptr, std::size_t n)
            : m_ptr(nullptr)
        {
            cudaErrorCheck(cudaMalloc(
                reinterpret_cast<void **>(&m_ptr), n * sizeof(value_t)
            ));
            cudaErrorCheck(cudaMemcpy(
                m_ptr, ptr.get(), n * sizeof(value_t), cudaMemcpyHostToDevice
            ));
        }

        ~owning_data_t()
        {
            cudaErrorCheck(cudaFree(m_ptr));
        }

        value_t * m_ptr;
    };

    struct non_owning_data_t {
        non_owning_data_t(const owning_data_t & o)
            : m_ptr(o.m_ptr)
        {
        }

        COVFIE_DEVICE value_t & operator[](index_t i) const
        {
            return m_ptr[i];
        }

        value_t * m_ptr;
    };
};
}
