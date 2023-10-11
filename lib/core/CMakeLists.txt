# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022-2023 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

add_library(core INTERFACE)

target_include_directories(
    core
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(core INTERFACE cxx_std_17)

if(COVFIE_REQUIRE_CXX20)
    target_compile_definitions(
        core
        INTERFACE
        COVFIE_REQUIRE_CXX20
    )
endif()

if(COVFIE_QUIET)
    target_compile_definitions(
        core
        INTERFACE
        COVFIE_QUIET
    )
endif()

# Logic to ensure that the core module can be installed properly.
install(
    TARGETS core
    EXPORT ${PROJECT_NAME}Targets
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/covfie
    DESTINATION include
)

# Hack for people using the disgusting mal-practice of pullling in external
# projects via "add_subdirectory"...
add_library(covfie::core ALIAS core)

# Test the public headers of covfie::core.
if(COVFIE_TEST_HEADERS)
    include(covfie-functions)

    file(
        GLOB_RECURSE
        public_headers
        RELATIVE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/covfie/*.hpp"
    )

    covfie_test_public_headers(
        core
        "${public_headers}"
    )
endif()
