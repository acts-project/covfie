# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

find_package(
    Boost
    1.71.0
    REQUIRED
    COMPONENTS
    filesystem
)

add_library(
    testing_utils

    tmp_file.cpp
)

# Ensure that the utils.
target_link_libraries(
    testing_utils

    PUBLIC
    Boost::filesystem
)

target_include_directories(
    testing_utils
    PUBLIC

    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
