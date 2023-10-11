/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/dereference.hpp>
#include <covfie/core/backend/transformer/shuffle.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field_view.hpp>

using backend_base = covfie::backend::strided<
    covfie::vector::size3,
    covfie::backend::dereference<
        covfie::backend::array<covfie::vector::size3>>>;
using backend_shuffle =
    covfie::backend::shuffle<backend_base, std::index_sequence<1, 2, 0>>;

template covfie::field_view<backend_base>::output_t
    covfie::field_view<backend_base>::at<std::size_t, std::size_t, std::size_t>(
        std::size_t, std::size_t, std::size_t
    ) const;

template covfie::field_view<backend_shuffle>::output_t covfie::
    field_view<backend_shuffle>::at<std::size_t, std::size_t, std::size_t>(
        std::size_t, std::size_t, std::size_t
    ) const;
