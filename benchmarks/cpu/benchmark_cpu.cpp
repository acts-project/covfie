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

#include "backends/constant.hpp"
#include "backends/test_field.hpp"
#include "patterns/lorentz.hpp"
#include "patterns/random.hpp"
#include "patterns/sequential.hpp"

void register_benchmarks(void)
{
    covfie::benchmark::register_bm<Sequential1D, Constant<float, 1, 1>>();
    covfie::benchmark::register_bm<Sequential2D, Constant<float, 2, 1>>();
    covfie::benchmark::register_bm<Sequential2D, Constant<float, 2, 2>>();
    covfie::benchmark::register_product_bm<
        boost::mp11::
            mp_list<Lorentz<Euler, Deep>, Lorentz<Euler, Wide>, RandomFloat>,
        boost::mp11::mp_list<
            FieldConstant,
            Field<InterpolateNN, LayoutStride>,
            Field<InterpolateNN, LayoutMortonNaive>,
            Field<InterpolateNN, LayoutMortonBMI2>,
            Field<InterpolateLin, LayoutStride>,
            Field<InterpolateLin, LayoutMortonNaive>,
            Field<InterpolateLin, LayoutMortonBMI2>>>();
    covfie::benchmark::register_product_bm<
        boost::mp11::mp_list<RandomIntegral, Sequential3D, Sequential3DZYX>,
        boost::mp11::mp_list<FieldIntBase, FieldIntMorton>>();
}

int main(int argc, char ** argv)
{
    register_benchmarks();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
