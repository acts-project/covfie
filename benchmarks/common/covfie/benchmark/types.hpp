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

#include <benchmark/benchmark.h>

namespace covfie::benchmark {
struct counters {
    std::size_t access_per_it;
    std::size_t bytes_per_it;
    std::size_t flops_per_it;
};

template <std::size_t N>
struct lorentz_agent {
    float pos[N];
    float mom[N];
};
}
