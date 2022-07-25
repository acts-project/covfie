/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <benchmark/benchmark.h>

#include <covfie/benchmark/register.hpp>

#include "backends/constant.hpp"
#include "patterns/lorentz_euler.hpp"
#include "patterns/sequential.hpp"

void register_benchmarks(void)
{
    covfie::benchmark::register_bm<Sequential1D, Constant<float, 1, 1>>();
    covfie::benchmark::register_bm<Sequential2D, Constant<float, 2, 1>>();
    covfie::benchmark::register_bm<Sequential2D, Constant<float, 2, 2>>();
    covfie::benchmark::register_bm<LorentzEulerWide, Constant<float, 3, 3>>();
}

int main(int argc, char ** argv)
{
    register_benchmarks();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
