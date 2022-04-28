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

#define cudaErrorCheck(r)                                                      \
    {                                                                          \
        _cudaErrorCheck((r), __FILE__, __LINE__);                              \
    }

void _cudaErrorCheck(cudaError_t code, const char * file, int line)
{
    if (code != cudaSuccess) {
        std::stringstream ss;

        ss << "[" << file << ":" << line
           << "] CUDA error: " << cudaGetErrorString(code);

        throw std::runtime_error(ss.str());
    }
}
