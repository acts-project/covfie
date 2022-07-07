# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Add the conversion of the ATLAS B-field to our native format.
add_executable(
    convert_atlas_bfield

    convert_atlas_bfield.cpp
)

target_link_libraries(
    convert_atlas_bfield

    PRIVATE
    covfie_core
    Boost::log
    Boost::log_setup
    Boost::program_options
)

# Add the first example that can be found in the README.
add_executable(
    readme_example_1

    readme_example_1.cpp
)

target_link_libraries(
    readme_example_1

    PRIVATE
    covfie_core
)

# Add the second example that can be found in the README.
add_executable(
    readme_example_2

    readme_example_2.cpp
)

target_link_libraries(
    readme_example_2

    PRIVATE
    covfie_core
)

# Add the an executable to create 2D slices from 3D fields.
add_executable(
    slice3dto2d

    slice3dto2d.cpp
)

target_link_libraries(
    slice3dto2d

    PRIVATE
    covfie_core
    Boost::log
    Boost::log_setup
    Boost::program_options
)
