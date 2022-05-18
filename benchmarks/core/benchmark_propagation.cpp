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
#include <cmath>
#include <iostream>
#include <random>

#include <benchmark/benchmark.h>
#include <covfie/core/backend/constant.hpp>
#include <covfie/core/field.hpp>

namespace {
template <std::size_t N>
struct object {
    float pos[N];
    float mom[N];
};
}

template <typename backend_t>
void PropagateWide3D(benchmark::State & state)
{
    using field_t = typename covfie::field<backend_t>;
    field_t f(typename field_t::backend_t::configuration_data_t{0.f, 0.f, 2.f});
    typename field_t::view_t fv(f);

    std::random_device rd;
    std::mt19937 e(rd());

    std::uniform_real_distribution<> phi_dist(0.f, 2.f * 3.1415927f);
    std::uniform_real_distribution<> costheta_dist(-1.f, 1.f);

    unsigned long np = state.range(0);
    unsigned long ns = state.range(1);
    float ss = state.range(2) / 1000000.f;

    std::vector<object<3>> objs(np);

    for (std::size_t i = 0; i < np; ++i) {
        float theta = std::acos(costheta_dist(e));
        float phi = phi_dist(e);

        objs[i].pos[0] = 0.f;
        objs[i].pos[1] = 0.f;
        objs[i].pos[2] = 0.f;

        objs[i].mom[0] = std::sin(theta) * std::cos(phi);
        objs[i].mom[1] = std::sin(theta) * std::sin(phi);
        objs[i].mom[2] = std::cos(theta);
    }

    for (auto _ : state) {
        state.PauseTiming();

        std::vector<object<3>> tmp_objs = objs;

        state.ResumeTiming();

        for (std::size_t s = 0; s < ns; ++s) {
            for (std::size_t i = 0; i < np; ++i) {
                object<3> & o = tmp_objs[i];
                typename field_t::output_t b =
                    fv.at(o.pos[0], o.pos[1], o.pos[2]);
                float f[3] = {
                    o.mom[1] * b[2] - o.mom[2] * b[1],
                    o.mom[2] * b[0] - o.mom[0] * b[2],
                    o.mom[0] * b[1] - o.mom[1] * b[0]};

                o.pos[0] += o.mom[0] * ss;
                o.pos[1] += o.mom[1] * ss;
                o.pos[2] += o.mom[2] * ss;

                o.mom[0] += f[0] * ss;
                o.mom[1] += f[1] * ss;
                o.mom[2] += f[2] * ss;
            }
        }
    }
}

BENCHMARK(PropagateWide3D<covfie::backend::constant<
              covfie::backend::vector::input::float3,
              covfie::backend::vector::output::float3>>)
    ->Name("PropagateWide/Constant/float3/float3")
    ->ArgNames({"N", "S", "L"})
    ->ArgsProduct(
        {benchmark::CreateRange(1, 65536, 8),
         benchmark::CreateRange(1, 65536, 8),
         {1, 50, 100, 500, 1000}}
    );
