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

#include <memory>

#include <covfie/core/backend/initial/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/interpolator/nearest_neighbour.hpp>
#include <covfie/core/backend/transformer/layout/strided.hpp>
#include <covfie/core/field.hpp>

using data_field_t = covfie::field<covfie::backend::transformer::affine<
    covfie::backend::transformer::interpolator::nearest_neighbour<
        covfie::backend::layout::strided<
            covfie::vector::ulong3,
            covfie::backend::storage::array<covfie::vector::float3>>>>>;

extern std::unique_ptr<data_field_t> TEST_FIELD;

const data_field_t & get_test_field();
