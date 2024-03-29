# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022-2023 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Enable the CUDA language!
enable_language(CUDA)

# Set up the CUDA compiler flags for the tests.
include(covfie-compiler-options-cuda)

# Create the test executable from the individual test groups.
add_executable(
    test_cuda

    test_cuda_array.cu
)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    test_cuda

    PUBLIC
    core
    cuda
    GTest::gtest
    GTest::gtest_main
)
