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

#include <string>

#include <benchmark/benchmark.h>

#include <covfie/core/field.hpp>

namespace covfie::benchmark {
template <typename Bd>
struct named_field {
    covfie::field<typename Bd::backend_t> field;
    std::string name;
};

template <template <typename> typename F, typename Bd>
auto register_benchmark(const named_field<Bd> & fv)
{
    using bm = F<typename Bd::backend_t>;

    std::string bm_name = std::string(bm::name) + "/" + std::string(Bd::name) +
                          "/Field:" + fv.name;

    return ::benchmark::RegisterBenchmark(
        bm_name.c_str(), [&fv](::benchmark::State & s) { bm::run(s, fv.field); }
    );
}
}
