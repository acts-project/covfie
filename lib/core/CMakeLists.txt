# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

add_library(covfie_core INTERFACE)

target_include_directories(
    covfie_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(covfie_core INTERFACE cxx_std_20)

if(COVFIE_QUIET)
    target_compile_definitions(covfie_core INTERFACE COVFIE_QUIET)
endif()

# Logic to ensure that the core module can be installed properly.
set_target_properties(
    covfie_core
    PROPERTIES
        EXPORT_NAME
            core
)

install(TARGETS covfie_core EXPORT ${PROJECT_NAME}Targets)

install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/covfie
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Hack for people using the disgusting mal-practice of pullling in external
# projects via "add_subdirectory"...
add_library(covfie::core ALIAS covfie_core)

# Test the public headers of covfie::core.
if(COVFIE_TEST_HEADERS)
    include(covfie-functions)

    file(
        GLOB_RECURSE public_headers
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/covfie/*.hpp"
    )

    covfie_test_public_headers(
        covfie_core
        "${public_headers}"
    )
endif()
