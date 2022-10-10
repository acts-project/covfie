# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

add_library(cuda INTERFACE)

target_include_directories(
    cuda
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Remove this line!
target_compile_options(
    cuda
    INTERFACE
    -Wall -Wextra
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

target_compile_features(cuda INTERFACE cxx_std_17)

# Logic to ensure that the CUDA module can be installed properly.
install(
    TARGETS cuda
    EXPORT ${PROJECT_NAME}Targets
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/covfie
    DESTINATION include
)
