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

#include <iostream>
#include <memory>

namespace covfie::utility {
template <typename T>
T read_binary(std::istream & fs)
{
    std::byte mem[sizeof(T)];

    fs.read(reinterpret_cast<char *>(&mem), sizeof(T));

    return *(reinterpret_cast<T *>(&mem));
}

template <typename T>
std::unique_ptr<T[]> read_binary_array(std::istream & fs, std::size_t n)
{
    std::unique_ptr<T[]> r = std::make_unique<T[]>(n);

    fs.read(reinterpret_cast<char *>(r.get()), n * sizeof(T));

    return r;
}
}
