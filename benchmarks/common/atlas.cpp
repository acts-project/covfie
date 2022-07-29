/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cstdlib>

#include <covfie/benchmark/atlas.hpp>

std::unique_ptr<atlas_field_t> ATLAS_FIELD;

const atlas_field_t & get_atlas_field()
{
    if (!ATLAS_FIELD) {
        char * file_name = std::getenv("COVFIE_BENCHMARK_ATLAS_FIELD");

        if (file_name) {
            std::ifstream ifs(file_name, std::ifstream::binary);
            if (!ifs.good()) {
                throw std::runtime_error(
                    "Failed to open ATLAS magnetic field file!"
                );
            }
            ATLAS_FIELD = std::make_unique<atlas_field_t>(ifs);
        } else {
            throw std::runtime_error(
                "Environment variable \"COVFIE_BENCHMARK_ATLAS_FIELD\" is not "
                "set!"
            );
        }
    }

    return *ATLAS_FIELD.get();
}
