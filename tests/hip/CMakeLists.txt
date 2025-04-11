# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

# Enable the CUDA language!
enable_language(HIP)

find_package(hip REQUIRED)

# Create the test executable from the individual test groups.
add_executable(test_hip test_hip_array.hip)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    test_hip
    PUBLIC
        covfie_core
        covfie_hip
        GTest::gtest
        GTest::gtest_main
)
