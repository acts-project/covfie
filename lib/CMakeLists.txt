# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

add_subdirectory(core)

if(COVFIE_PLATFORM_CPU)
    add_subdirectory(cpu)
endif()

if(COVFIE_PLATFORM_CUDA)
    add_subdirectory(cuda)
endif()
