/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

template <typename F>
std::decay_t<typename F::output_t> retrieve_vector(
    typename F::view_t v, typename F::coordinate_t c, ::sycl::queue & queue
)
{
    std::decay_t<typename F::output_t> * rv_d;
    std::decay_t<typename F::output_t> rv_h;

    rv_d = ::sycl::malloc_device<std::decay_t<typename F::output_t>>(1, queue);

    queue
        .submit([&](sycl::handler & h) {
            h.single_task([v, c, rv_d]() { *rv_d = v.at(c); });
        })
        .wait();

    queue.memcpy(&rv_h, rv_d, sizeof(std::decay_t<typename F::output_t>))
        .wait();
    ::sycl::free(rv_d, queue);

    return rv_h;
}
