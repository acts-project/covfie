# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# First, we need to make sure that we have a CUDA compiler, and that all the
# necessary tooling is set up.
enable_language(CUDA)

# Set up the CUDA compiler flags for the examples.
include(covfie-compiler-options-cuda)

# Add the 3D field slice rendering tool based on CUDA.
add_executable(render_slice_cuda render_slice.cu)

target_link_libraries(
    render_slice_cuda
    PRIVATE
        core
        cuda
        bitmap
        Boost::log
        Boost::log_setup
        Boost::program_options
)

# Add the 3D field slice rendering tool based on CUDA textures.
add_executable(render_slice_texture_cuda render_slice_texture.cu)

target_link_libraries(
    render_slice_texture_cuda
    PRIVATE
        core
        cuda
        bitmap
        Boost::log
        Boost::log_setup
        Boost::program_options
)
