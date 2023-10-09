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

# Hack for compatibility
add_library(covfie::cpu ALIAS cpu)
