# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

add_library(covfie_core INTERFACE)

target_include_directories(
    covfie_core
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(covfie_core INTERFACE cxx_std_20)

# Logic to ensure that the core module can be installed properly.
set_target_properties(
    covfie_core
    PROPERTIES
        EXPORT_NAME
            core
)

install(TARGETS covfie_core EXPORT ${PROJECT_NAME}Targets)

install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/covfie
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Hack for people using the disgusting mal-practice of pullling in external
# projects via "add_subdirectory"...
if(NOT PROJECT_IS_TOP_LEVEL)
    add_library(covfie::core ALIAS covfie_core)
endif()
