# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Add the conversion of B-fields to our native format.
add_executable(covfie_convert_bfield convert_bfield.cpp)

target_link_libraries(
    covfie_convert_bfield
    PRIVATE
        covfie::core
        Boost::log
        Boost::log_setup
        Boost::program_options
)

# Add the conversion of B-fields from CSV to our native format.
add_executable(covfie_convert_bfield_csv convert_bfield_csv.cpp)

target_link_libraries(
    covfie_convert_bfield_csv
    PRIVATE
        covfie::core
        Boost::log
        Boost::log_setup
        Boost::program_options
)

# Add the first example that can be found in the README.
add_executable(covfie_readme_example_1 readme_example_1.cpp)

target_link_libraries(covfie_readme_example_1 PRIVATE covfie::core)

# Add the second example that can be found in the README.
add_executable(covfie_readme_example_2 readme_example_2.cpp)

target_link_libraries(covfie_readme_example_2 PRIVATE covfie::core)

# Add the an executable to create 2D slices from 3D fields.
add_executable(covfie_slice3dto2d slice3dto2d.cpp)

target_link_libraries(
    covfie_slice3dto2d
    PRIVATE
        covfie::core
        Boost::log
        Boost::log_setup
        Boost::program_options
)

# Add the an executable to scale up B-fields.
add_executable(covfie_scaleup_bfield scaleup_bfield.cpp)

target_link_libraries(
    covfie_scaleup_bfield
    PRIVATE
        covfie::core
        Boost::log
        Boost::log_setup
        Boost::program_options
)

add_library(covfie_asm asm.cpp)
target_link_libraries(covfie_asm PRIVATE covfie::core)

# Some libraries which are designed to inspected by the user.
add_library(covfie_shuffle_asm shuffle_asm.cpp)
target_link_libraries(covfie_shuffle_asm PRIVATE covfie::core)

# Executable for generating an testable field.
add_executable(covfie_generate_test_field generate_test_field.cpp)

target_link_libraries(
    covfie_generate_test_field
    PRIVATE
        covfie::core
        Boost::log
        Boost::log_setup
        Boost::program_options
)
