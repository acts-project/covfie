/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>
#include <optional>

#include <hip/hip_runtime.h>

#include <covfie/hip/error_check.hpp>
#include <covfie/hip/utility/unique_ptr.hpp>

namespace covfie::utility::hip {
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

    hipErrorCheck(hipMalloc(&p, sizeof(T)));

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

    hipErrorCheck(hipMalloc(&p, n * sizeof(std::remove_extent_t<T>)));

    return unique_device_ptr<T>(p);
}

template <typename T>
unique_device_ptr<T[]>
device_copy_h2d(const T * h, std::optional<hipStream_t> stream = std::nullopt)
{
    unique_device_ptr<T[]> r = device_allocate<T[]>();

    if (stream.has_value()) {
        hipErrorCheck(hipMemcpyAsync(
            r.get(), h, sizeof(T), hipMemcpyHostToDevice, *stream
        ));
        hipErrorCheck(hipStreamSynchronize(*stream));
    } else {
        hipErrorCheck(hipMemcpy(r.get(), h, sizeof(T), hipMemcpyHostToDevice));
    }

    return r;
}

template <typename T>
unique_device_ptr<T[]> device_copy_h2d(
    const T * h, std::size_t n, std::optional<hipStream_t> stream = std::nullopt
)
{
    unique_device_ptr<T[]> r = device_allocate<T[]>(n);

    if (stream.has_value()) {
        hipErrorCheck(hipMemcpyAsync(
            r.get(),
            h,
            n * sizeof(std::remove_extent_t<T>),
            hipMemcpyHostToDevice,
            *stream
        ));
        hipErrorCheck(hipStreamSynchronize(*stream));
    } else {
        hipErrorCheck(hipMemcpy(
            r.get(),
            h,
            n * sizeof(std::remove_extent_t<T>),
            hipMemcpyHostToDevice
        ));
    }

    return r;
}

template <typename T>
unique_device_ptr<T[]>
device_copy_d2d(const T * h, std::optional<hipStream_t> stream = std::nullopt)
{
    unique_device_ptr<T[]> r = device_allocate<T[]>();

    if (stream.has_value()) {
        hipErrorCheck(hipMemcpyAsync(
            r.get(), h, sizeof(T), hipMemcpyDeviceToDevice, *stream
        ));
        hipErrorCheck(hipStreamSynchronize(*stream));
    } else {
        hipErrorCheck(hipMemcpy(r.get(), h, sizeof(T), hipMemcpyDeviceToDevice)
        );
    }

    return r;
}

template <typename T>
unique_device_ptr<T[]> device_copy_d2d(
    const T * h, std::size_t n, std::optional<hipStream_t> stream = std::nullopt
)
{
    unique_device_ptr<T[]> r = device_allocate<T[]>(n);

    if (stream.has_value()) {
        hipErrorCheck(hipMemcpyAsync(
            r.get(),
            h,
            n * sizeof(std::remove_extent_t<T>),
            hipMemcpyDeviceToDevice,
            *stream
        ));
        hipErrorCheck(hipStreamSynchronize(*stream));
    } else {
        hipErrorCheck(hipMemcpy(
            r.get(),
            h,
            n * sizeof(std::remove_extent_t<T>),
            hipMemcpyDeviceToDevice
        ));
    }

    return r;
}
}
