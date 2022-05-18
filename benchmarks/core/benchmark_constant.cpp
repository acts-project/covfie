/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <array>

#include <benchmark/benchmark.h>
#include <covfie/core/backend/constant.hpp>
#include <covfie/core/field.hpp>

template <typename backend_t>
void Sequential1D(benchmark::State & state)
{
    using field_t = typename covfie::field<backend_t>;

    field_t f(typename field_t::backend_t::configuration_data_t{5.f});
    typename field_t::view_t fv(f);

    unsigned long xr = state.range(0);

    for (auto _ : state) {
        for (unsigned long x = 0; x < xr; ++x) {
            benchmark::DoNotOptimize(fv.at(static_cast<float>(x)));
        }
    }
}

template <typename backend_t>
void Sequential2D(benchmark::State & state)
{
    using field_t = typename covfie::field<backend_t>;

    field_t f(typename field_t::backend_t::configuration_data_t{5.f, -2.f});
    typename field_t::view_t fv(f);

    unsigned long xr = state.range(0);
    unsigned long yr = state.range(1);

    for (auto _ : state) {
        for (unsigned long x = 0; x < xr; ++x) {
            for (unsigned long y = 0; y < yr; ++y) {
                benchmark::DoNotOptimize(
                    fv.at(static_cast<float>(x), static_cast<float>(y))
                );
            }
        }
    }
}

template <typename backend_t>
void Sequential3D(benchmark::State & state)
{
    using field_t = typename covfie::field<backend_t>;

    field_t f(typename field_t::backend_t::configuration_data_t{5.f, -2.f, 6.f}
    );
    typename field_t::view_t fv(f);

    unsigned long xr = state.range(0);
    unsigned long yr = state.range(1);
    unsigned long zr = state.range(2);

    for (auto _ : state) {
        for (unsigned long x = 0; x < xr; ++x) {
            for (unsigned long y = 0; y < yr; ++y) {
                for (unsigned long z = 0; z < zr; ++z) {
                    benchmark::DoNotOptimize(fv.at(
                        static_cast<float>(x),
                        static_cast<float>(y),
                        static_cast<float>(z)
                    ));
                }
            }
        }
    }
}

BENCHMARK(Sequential1D<covfie::backend::constant<
              covfie::backend::vector::input::float1,
              covfie::backend::vector::output::float1>>)
    ->Name("Sequential/Constant/float1/float1")
    ->ArgNames({"X"})
    ->Ranges({{1, 4096}});
BENCHMARK(Sequential2D<covfie::backend::constant<
              covfie::backend::vector::input::float2,
              covfie::backend::vector::output::float2>>)
    ->Name("Sequential/Constant/float2/float2")
    ->ArgNames({"X", "Y"})
    ->Ranges({{1, 4096}, {1, 4096}});
BENCHMARK(Sequential3D<covfie::backend::constant<
              covfie::backend::vector::input::float3,
              covfie::backend::vector::output::float3>>)
    ->Name("Sequential/Constant/float3/float3")
    ->ArgNames({"X", "Y", "Z"})
    ->Ranges({{1, 1024}, {1, 1024}, {1, 1024}});
