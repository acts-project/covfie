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

#ifdef _MSC_VER
#define UNLIKELY(x) x
#else
#define UNLIKELY(x) __builtin_expect(x, false)
#endif

#ifdef _MSC_VER
#define LIKELY(x) x
#else
#define LIKELY(x) __builtin_expect(x, true)
#endif
