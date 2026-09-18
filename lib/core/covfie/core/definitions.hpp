/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef _MSC_VER
#define COVFIE_UNLIKELY(x) x
#else
#define COVFIE_UNLIKELY(x) __builtin_expect(x, false)
#endif

#ifdef _MSC_VER
#define COVFIE_LIKELY(x) x
#else
#define COVFIE_LIKELY(x) __builtin_expect(x, true)
#endif
