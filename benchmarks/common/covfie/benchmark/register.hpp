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
template <typename Pattern, typename Backend>
void register_bm()
{
    using Benchmark =
        typename Pattern::template Bench<typename Backend::backend_t>;

    std::vector<std::vector<int64_t>> parameter_ranges =
        Pattern::get_parameter_ranges();

    ::benchmark::RegisterBenchmark(
        (std::string(Pattern::name) + "/" + std::string(Backend::get_name()))
            .c_str(),
        [](::benchmark::State & state) {
            Benchmark::execute(state, Backend::get_field());
        }
    )
        ->ArgNames(
            {Pattern::parameter_names.begin(), Pattern::parameter_names.end()}
        )
        ->ArgsProduct({parameter_ranges.begin(), parameter_ranges.end()});
}
}
