# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022-2023 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

find_package(CUDAToolkit REQUIRED)

add_library(cuda INTERFACE)

target_include_directories(
    cuda
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(cuda INTERFACE cxx_std_20)

target_link_libraries(
    cuda
    INTERFACE
        CUDA::cudart
        covfie::core
)

# Logic to ensure that the CUDA module can be installed properly.
install(TARGETS cuda EXPORT ${PROJECT_NAME}Targets)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/covfie DESTINATION include)

# Hack for compatibility
add_library(covfie::cuda ALIAS cuda)

# Test the public headers of covfie::cuda.
if(COVFIE_TEST_HEADERS)
    include(covfie-functions)

    file(
        GLOB_RECURSE public_headers
        RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/covfie/*.hpp"
    )

    covfie_test_public_headers(
        cuda
        "${public_headers}"
    )
endif()
