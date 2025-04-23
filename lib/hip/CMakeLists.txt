# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

add_library(covfie_hip INTERFACE)

target_include_directories(
    covfie_hip
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(
    covfie_hip
    INTERFACE
        cxx_std_20
        hip_std_20
)

target_link_libraries(covfie_hip INTERFACE covfie_core)

# Logic to ensure that the HIP module can be installed properly.
set_target_properties(
    covfie_hip
    PROPERTIES
        EXPORT_NAME
            hip
)

install(TARGETS covfie_hip EXPORT ${PROJECT_NAME}Targets)

install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/covfie
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Hack for compatibility
if(NOT PROJECT_IS_TOP_LEVEL)
    add_library(covfie::hip ALIAS covfie_hip)
endif()
