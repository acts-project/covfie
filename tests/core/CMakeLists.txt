# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# We use the Boost filesystem for the creation of some temporary files, so
# we'll load that as well.
find_package(
    Boost
    1.71.0
    REQUIRED
    COMPONENTS
    filesystem
)

# Create the test executable from the individual test groups.
add_executable(
    test_core

    test_field_builder.cpp
    test_utility.cpp
    test_constant_field.cpp
    test_transformer_affine.cpp
    test_binary_io.cpp
)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    test_core

    PUBLIC
    covfie_core
    GTest::gtest
    GTest::gtest_main
    Boost::filesystem
)
