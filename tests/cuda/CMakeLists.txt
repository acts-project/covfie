# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# Enable the CUDA language!
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

# Set up the CUDA compiler flags for the tests.
include(covfie-compiler-options-cuda)

# Create the test executable from the individual test groups.
add_executable(
    test_cuda
    test_cuda_array.cu
    test_cuda_strided_array.cu
    test_cuda_texture.cu
)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    test_cuda
    PUBLIC
        covfie_core
        covfie_cuda
        CUDA::cudart
        GTest::gtest
        GTest::gtest_main
)
