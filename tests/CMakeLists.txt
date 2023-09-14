# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# All the tests here will require Google Test, so we will go ahead and find it.
find_package(GTest CONFIG REQUIRED)

# The testing utils are always built.
add_subdirectory(utils)

# The core library tests are always built (if tests are built at all).
add_subdirectory(core)

# The CPU tests are only built if requested.
if(COVFIE_PLATFORM_CPU)
    add_subdirectory(cpu)
endif()

# The CUDA tests are only built if requested.
if(COVFIE_PLATFORM_CUDA)
    add_subdirectory(cuda)
endif()
