/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2023 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace covfie {
template <typename... Ts>
class parameter_pack
{
};

template <>
class parameter_pack<>
{
public:
    parameter_pack()
    {
    }
};

template <typename T, typename... Ts>
class parameter_pack<T, Ts...>
{
public:
    parameter_pack(T && _x, Ts &&... _xs)
        : x(std::forward<T>(_x))
        , xs(std::forward<Ts>(_xs)...)
    {
    }

    T x;
    parameter_pack<Ts...> xs;
};

template <typename... Ts>
parameter_pack<Ts...> make_parameter_pack(Ts &&... args)
{
    return parameter_pack<Ts...>(std::forward<Ts>(args)...);
}
}
