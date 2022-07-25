/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <random>

#include <covfie/benchmark/pattern.hpp>
#include <covfie/core/field_view.hpp>

struct LorentzEulerWide : covfie::benchmark::AccessPattern<LorentzEulerWide> {
    struct parameters {
        std::size_t particles, steps;
    };

    static constexpr std::array<std::string_view, 2> parameter_names = {
        "N", "S"};
    static constexpr std::string_view name = "LorentzEulerWide";

    template <typename backend_t>
    static void iteration(
        const parameters & p,
        const covfie::field_view<backend_t> & f,
        ::benchmark::State & state
    )
    {
        state.PauseTiming();

        std::random_device rd;
        std::mt19937 e(rd());

        std::uniform_real_distribution<> phi_dist(0.f, 2.f * 3.1415927f);
        std::uniform_real_distribution<> costheta_dist(-1.f, 1.f);

        float ss = 1.f / 1000000.f;

        std::vector<covfie::benchmark::lorentz_agent<3>> objs(p.particles);

        for (std::size_t i = 0; i < p.particles; ++i) {
            float theta = std::acos(costheta_dist(e));
            float phi = phi_dist(e);

            objs[i].pos[0] = 0.f;
            objs[i].pos[1] = 0.f;
            objs[i].pos[2] = 0.f;

            objs[i].mom[0] = std::sin(theta) * std::cos(phi);
            objs[i].mom[1] = std::sin(theta) * std::sin(phi);
            objs[i].mom[2] = std::cos(theta);
        }

        state.ResumeTiming();

        for (std::size_t s = 0; s < p.steps; ++s) {
            for (std::size_t i = 0; i < p.particles; ++i) {
                covfie::benchmark::lorentz_agent<3> & o = objs[i];
                typename std::decay_t<decltype(f)>::output_t b =
                    f.at(o.pos[0], o.pos[1], o.pos[2]);
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

    static std::vector<std::vector<int64_t>> get_parameter_ranges()
    {
        return {{1, 8, 64, 512, 4096}, {1, 8, 64, 512, 4096}};
    }

    static parameters get_parameters(benchmark::State & state)
    {
        return {
            static_cast<std::size_t>(state.range(0)),
            static_cast<std::size_t>(state.range(1))};
    }

    template <typename S, std::size_t N>
    static covfie::benchmark::counters get_counters(const parameters & p)
    {
        return {
            p.particles * p.steps,
            p.particles * p.steps * N * sizeof(S),
            p.particles * p.steps * 21};
    }
};
