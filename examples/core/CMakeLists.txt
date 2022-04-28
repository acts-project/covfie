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
