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

#include <covfie/cuda/error_check.hpp>
#include <covfie/cuda/utility/unique_ptr.hpp>

namespace covfie::utility::cuda {
template <typename T>
unique_device_ptr<T> device_allocate()
{
    static_assert(
        !(std::is_array_v<T> && std::extent_v<T> == 0),
        "Allocation pointer type cannot be an unbounded array."
    );

    using pointer_t =
        std::conditional_t<std::is_array_v<T>, std::decay_t<T>, T *>;

    pointer_t p;

    cudaErrorCheck(cudaMalloc(&p, sizeof(T)));

    return unique_device_ptr<T>(p);
}

template <typename T>
unique_device_ptr<T> device_allocate(std::size_t n)
{
    static_assert(
        std::is_array_v<T>, "Allocation pointer type must be an array type."
    );
    static_assert(
        std::extent_v<T> == 0, "Allocation pointer type must be unbounded."
    );

    using pointer_t =
        std::conditional_t<std::is_array_v<T>, std::decay_t<T>, T *>;

    pointer_t p;

    cudaErrorCheck(cudaMalloc(&p, n * sizeof(std::remove_extent_t<T>)));

    return unique_device_ptr<T>(p);
}

template <typename T>
unique_device_ptr<T> device_copy(std::unique_ptr<T> && h)
{
    unique_device_ptr<T> r = device_allocate<T>();

    cudaErrorCheck(
        cudaMemcpy(r.get(), h.get(), sizeof(T), cudaMemcpyHostToDevice)
    );

    return r;
}

template <typename T>
unique_device_ptr<T> device_copy(std::unique_ptr<T> && h, std::size_t n)
{
    unique_device_ptr<T> r = device_allocate<T>(n);

    cudaErrorCheck(cudaMemcpy(
        r.get(),
        h.get(),
        n * sizeof(std::remove_extent_t<T>),
        cudaMemcpyHostToDevice
    ));

    return r;
}
}
