# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

cmake_minimum_required(VERSION 3.18)

project("covfie")

# Declare options which control the parts of the code being built.
option(
    COVFIE_BUILD_BENCHMARKS
    "Build benchmark executables."
)
option(
    COVFIE_BUILD_TESTS
    "Build test executables."
)
option(
    COVFIE_BUILD_EXAMPLES
    "Build example executables."
)

# Declare options for the different platforms that we wish to support.
option(
    COVFIE_PLATFORM_CPU
    "Enable building of CPU code."
    On
)
option(
    COVFIE_PLATFORM_CUDA
    "Enable building of CUDA code."
)

# The core library should always be built.
add_subdirectory(lib)

# The benchmarks should be build only if requested...
if(COVFIE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ...the same goes for the tests...
if(COVFIE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ...and the examples.
if(COVFIE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
