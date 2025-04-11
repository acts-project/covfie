/*
 * SPDX-PackageName: "covfie, a part of the ACTS project"
 * SPDX-FileCopyrightText: 2022 CERN
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <sstream>

#include <hip/hip_runtime.h>

#define hipErrorCheck(r)                                                       \
    {                                                                          \
        _hipErrorCheck((r), __FILE__, __LINE__);                               \
    }

inline void _hipErrorCheck(hipError_t code, const char * file, int line)
{
    if (code != hipSuccess) {
        std::stringstream ss;

        ss << "[" << file << ":" << line
           << "] HIP error: " << hipGetErrorString(code);

        throw std::runtime_error(ss.str());
    }
}
