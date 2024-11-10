# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Add the 3D field slice rendering tool.
add_executable(covfie_render_slice_cpu render_slice.cpp)

target_link_libraries(
    covfie_render_slice_cpu
    PRIVATE
        covfie::core
        covfie::cpu
        covfie_bitmap
        Boost::log
        Boost::log_setup
        Boost::program_options
)

add_executable(covfie_render_image_cpu render_image.cpp)

target_link_libraries(
    covfie_render_image_cpu
    PRIVATE
        covfie::core
        covfie::cpu
        covfie_bitmap
        Boost::log
        Boost::log_setup
        Boost::program_options
)
