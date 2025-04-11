/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>

#include <sycl/sycl.hpp>

namespace covfie::utility::sycl {
struct device_deleter {
    device_deleter(const ::sycl::queue & q)
        : queue(q)
    {
    }

    device_deleter(device_deleter &&) = default;

    device_deleter & operator=(device_deleter &&) = default;

    void operator()(void * p) const
    {
        ::sycl::free(p, queue);
    }

private:
    ::sycl::queue queue;
};

template <typename T>
using unique_device_ptr = std::unique_ptr<T, device_deleter>;
}
