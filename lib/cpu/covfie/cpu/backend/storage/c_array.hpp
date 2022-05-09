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

#include <covfie/core/backend/storage/array.hpp>

namespace covfie::backend::storage {
template <typename _value_t, std::size_t _dims, typename _index_t = std::size_t>
using c_array = array<_value_t, _dims, _index_t>;
}
