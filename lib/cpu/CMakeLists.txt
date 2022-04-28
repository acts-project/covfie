# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

add_library(covfie_cpu INTERFACE)

target_include_directories(covfie_cpu INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Remove this line!
target_compile_options(covfie_cpu INTERFACE "-Wall" "-Wextra")

target_compile_features(covfie_cpu INTERFACE cxx_std_17)
