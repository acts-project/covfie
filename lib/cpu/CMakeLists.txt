# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022-2023 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

add_library(cpu INTERFACE)

target_include_directories(
    cpu
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(cpu INTERFACE cxx_std_17)

# Logic to ensure that the CPU module can be installed properly.
install(
    TARGETS cpu
    EXPORT ${PROJECT_NAME}Targets
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/covfie
    DESTINATION include
)

target_link_libraries(
    cpu
    INTERFACE
    covfie::core
)

# Hack for compatibility
add_library(covfie::cpu ALIAS cpu)

# Test the public headers of covfie::cpu.
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
        cpu
        "${public_headers}"
    )
endif()
