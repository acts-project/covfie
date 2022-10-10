# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

enable_language(CUDA)

# Create the benchmark executable from the individual files.
add_executable(
    benchmark_cuda

    benchmark_cuda.cu
)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    benchmark_cuda

    PUBLIC
    core
    cuda
    benchmark
    benchmark::benchmark
    Boost::headers
)
