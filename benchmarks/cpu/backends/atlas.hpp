/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <covfie/benchmark/atlas.hpp>
#include <covfie/core/backend/initial/constant.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

struct AtlasBase {
    using backend_t = atlas_field_t::backend_t;

    static atlas_field_t get_field()
    {
        return atlas_field_t(get_atlas_field());
    }

    static std::string get_name()
    {
        return "AtlasBase";
    }
};
