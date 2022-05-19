# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Create the benchmark executable from the individual files.
add_executable(
    benchmark_core

    benchmark_core.cpp
)

# Ensure that the tests are linked against the required libraries.
target_link_libraries(
    benchmark_core

    PUBLIC
    covfie_core
    benchmark::benchmark
)
