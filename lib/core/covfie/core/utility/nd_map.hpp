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

#include <array>
#include <functional>

namespace covfie::utility {
template <typename T, typename... Args>
void nd_map(std::function<void(T, Args...)> f, T t, Args... args)
{
    for (T i = 0; i < t; ++i) {
        if constexpr (sizeof...(Args) > 0) {
            nd_map<Args...>(
                std::function<void(Args...)>([f, i](Args... args) {
                    f(i, args...);
                }),
                args...
            );
        } else {
            f(i);
        }
    }
}
}
