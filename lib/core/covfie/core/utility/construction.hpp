/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <utility>

#include <covfie/core/parameter_pack.hpp>

namespace covfie::utility {
/**
 * @brief Build the owning data of a backend from a parameter pack.
 *
 * The head of the pack configures the outermost layer. If a tail remains, it
 * builds the child of that layer, which the layer then takes ownership of. If
 * no tail remains, the layer builds its own child from its own configuration.
 */
template <typename B, typename C, typename... Cs>
typename B::owning_data_t construct(parameter_pack<C, Cs...> && p)
{
    if constexpr (sizeof...(Cs) == 0) {
        return typename B::owning_data_t(std::move(p.x));
    } else {
        return typename B::owning_data_t(
            std::move(p.x), construct<typename B::backend_t>(std::move(p.xs))
        );
    }
}
}
