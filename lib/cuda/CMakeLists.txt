# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
# SPDX-License-Identifier: MPL-2.0

add_library(covfie_cuda INTERFACE)

target_include_directories(
    covfie_cuda
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(
    covfie_cuda
    INTERFACE
        cxx_std_20
        cuda_std_20
)

target_link_libraries(covfie_cuda INTERFACE covfie_core)

# Logic to ensure that the CUDA module can be installed properly.
set_target_properties(
    covfie_cuda
    PROPERTIES
        EXPORT_NAME
            cuda
)

install(TARGETS covfie_cuda EXPORT ${PROJECT_NAME}Targets)

install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/covfie
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Hack for compatibility
if(NOT PROJECT_IS_TOP_LEVEL)
    add_library(covfie::cuda ALIAS covfie_cuda)
endif()
