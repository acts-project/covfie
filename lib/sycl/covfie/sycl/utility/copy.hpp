/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <sycl/sycl.hpp>

#include <covfie/core/concepts.hpp>

#include "covfie/core/parameter_pack.hpp"

namespace covfie::utility::sycl {
template <typename T, typename U>
requires(concepts::field_backend<T> &&
             concepts::field_backend<typename std::decay_t<U>::parent_t>)
    typename T::owning_data_t
    copy_backend_with_queue(U && backend, ::sycl::queue & queue)
{
    constexpr bool can_construct_with_queue =
        std::constructible_from<typename T::owning_data_t, U, ::sycl::queue &>;

    static_assert(
        can_construct_with_queue ||
        (!std::decay_t<T>::is_initial && !std::decay_t<U>::parent_t::is_initial)
    );

    if constexpr (can_construct_with_queue) {
        return typename T::owning_data_t(std::forward<U>(backend), queue);
    } else {
        auto new_backend = copy_backend_with_queue<typename T::backend_t>(
            backend.get_backend(), queue
        );

        if constexpr (std::constructible_from<
                          typename T::owning_data_t,
                          typename T::backend_t::owning_data_t &&>)
        {
            return typename T::owning_data_t(std::move(new_backend));
        } else if constexpr (std::constructible_from<
                                 typename T::owning_data_t,
                                 decltype(backend.get_configuration()),
                                 typename T::backend_t::owning_data_t &&>)
        {
            return typename T::owning_data_t(
                backend.get_configuration(), std::move(new_backend)
            );
        } else {
            return typename T::owning_data_t(make_parameter_pack(
                backend.get_configuration(), std::move(new_backend)
            ));
        }
    }
}

template <typename T, typename U>
requires(concepts::field_backend<typename T::backend_t> &&
             concepts::field_backend<typename std::decay_t<U>::backend_t>) T
    copy_field_with_queue(U && field, ::sycl::queue & queue)
{
    return T(
        copy_backend_with_queue<typename T::backend_t>(field.backend(), queue)
    );
}
}
