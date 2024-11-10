# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Load the OpenMP platform.
find_package(OpenMP COMPONENTS CXX REQUIRED)

# Create the benchmark executable from the individual files.
add_executable(covfie_benchmark_openmp benchmark_openmp.cpp)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    covfie_benchmark_openmp
    PUBLIC
        covfie::core
        covfie::cpu
        covfie_benchmark
        OpenMP::OpenMP_CXX
        benchmark::benchmark
        Boost::headers
)
