/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>

#include <sycl/sycl.hpp>

#include <covfie/sycl/utility/unique_ptr.hpp>

namespace covfie::utility::sycl {
template <typename T>
unique_device_ptr<T> device_allocate(::sycl::queue & queue)
{
    static_assert(
        !(std::is_array_v<T> && std::extent_v<T> == 0),
        "Allocation pointer type cannot be an unbounded array."
    );

    return unique_device_ptr<T>(
        ::sycl::malloc_device<T>(1, queue), device_deleter(queue)
    );
}

template <typename T>
unique_device_ptr<T> device_allocate(std::size_t n, ::sycl::queue & queue)
{
    static_assert(
        std::is_array_v<T>, "Allocation pointer type must be an array type."
    );
    static_assert(
        std::extent_v<T> == 0, "Allocation pointer type must be unbounded."
    );

    return unique_device_ptr<T>(
        ::sycl::malloc_device<std::remove_extent_t<T>>(n, queue),
        device_deleter(queue)
    );
}

template <typename T>
unique_device_ptr<T[]> device_copy_h2d(const T * h, ::sycl::queue & queue)
{
    unique_device_ptr<T[]> r = device_allocate<T[]>(queue);

    queue.memcpy(r.get(), h, sizeof(T)).wait();

    return r;
}

template <typename T>
unique_device_ptr<T[]>
device_copy_h2d(const T * h, std::size_t n, ::sycl::queue & queue)
{
    unique_device_ptr<T[]> r = device_allocate<T[]>(n, queue);

    queue.memcpy(r.get(), h, n * sizeof(T)).wait();

    return r;
}

template <typename T>
unique_device_ptr<T[]> device_copy_d2d(const T * h, ::sycl::queue & queue)
{
    unique_device_ptr<T[]> r = device_allocate<T[]>(queue);

    queue.memcpy(r.get(), h, sizeof(T)).wait();

    return r;
}

template <typename T>
unique_device_ptr<T[]>
device_copy_d2d(const T * h, std::size_t n, ::sycl::queue & queue)
{
    unique_device_ptr<T[]> r = device_allocate<T[]>(n, queue);

    queue.memcpy(r.get(), h, n * sizeof(std::remove_extent_t<T>)).wait();

    return r;
}
}
