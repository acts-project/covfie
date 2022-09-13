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

# Remove this line!
target_compile_options(covfie_core INTERFACE "-Wall" "-Wextra")

target_compile_features(covfie_core INTERFACE cxx_std_17)

if(COVFIE_REQUIRE_CXX20)
    target_compile_definitions(
        covfie_core
        INTERFACE
        COVFIE_REQUIRE_CXX20
    )
endif()

if(COVFIE_QUIET)
    target_compile_definitions(
        covfie_core
        INTERFACE
        COVFIE_QUIET
    )
endif()

# Logic to ensure that the core module can be installed properly.
install(
    TARGETS covfie_core
    EXPORT ${PROJECT_NAME}Targets
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/covfie
    DESTINATION include
)
