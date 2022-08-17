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
#include <boost/mp11.hpp>

#include <covfie/benchmark/register.hpp>

#include "backends/atlas.hpp"
#include "patterns/lorentz_euler.hpp"

void register_benchmarks(void)
{
    covfie::benchmark::register_product_bm<
        boost::mp11::mp_list<Lorentz<Euler>>,
        boost::mp11::mp_list<
            AtlasConstant,
            AtlasTex<TexInterpolateLin>,
            AtlasTex<TexInterpolateNN>,
            Atlas<InterpolateNN, LayoutStride>,
            Atlas<InterpolateNN, LayoutMortonNaive>,
            Atlas<InterpolateLin, LayoutStride>,
            Atlas<InterpolateLin, LayoutMortonNaive>>>();
}

int main(int argc, char ** argv)
{
    register_benchmarks();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}