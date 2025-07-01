/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <covfie/core/concepts.hpp>

#include "covfie/core/parameter_pack.hpp"

namespace covfie::utility::cuda {
template <typename T, typename U>
requires(concepts::field_backend<T> &&
             concepts::field_backend<typename std::decay_t<U>::parent_t>)
    typename T::owning_data_t
    copy_backend_with_stream(U && backend, const cudaStream_t & stream)
{
    constexpr bool can_construct_with_stream = std::
        constructible_from<typename T::owning_data_t, U, const cudaStream_t &>;

    static_assert(
        can_construct_with_stream ||
        (!std::decay_t<T>::is_initial && !std::decay_t<U>::parent_t::is_initial)
    );

    if constexpr (can_construct_with_stream) {
        return typename T::owning_data_t(std::forward<U>(backend), stream);
    } else {
        auto new_backend = copy_backend_with_stream<typename T::backend_t>(
            backend.get_backend(), stream
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
    copy_field_with_stream(U && field, const cudaStream_t & stream)
{
    return T(
        copy_backend_with_stream<typename T::backend_t>(field.backend(), stream)
    );
}
}
