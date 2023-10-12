# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2023 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# CMake include(s).
cmake_minimum_required( VERSION 3.11 )
include( FetchContent )

# Silence FetchContent warnings with CMake >=3.24.
if(POLICY CMP0135)
   cmake_policy(SET CMP0135 NEW)
endif()

# Tell the user what's happening.
message(STATUS "Building GoogleTest as part of the Covfie project")

# Declare where to get GoogleTest from.
FetchContent_Declare(GoogleTest
   URL "https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"
   URL_MD5 "c8340a482851ef6a3fe618a082304cfc")

# Options used in the build of GoogleTest.
set(BUILD_GMOCK FALSE CACHE BOOL "Turn off the build of GMock")
set(INSTALL_GTEST FALSE CACHE BOOL "Turn off the installation of GoogleTest")
if( WIN32 )
   set(gtest_force_shared_crt TRUE CACHE BOOL
       "Use shared (DLL) run-time library, even with static libraries")
endif()

# Silence some warnings with modern versions of CMake on macOS.
set(CMAKE_MACOSX_RPATH TRUE)

# Get it into the current directory.
FetchContent_Populate( GoogleTest )
add_subdirectory("${googletest_SOURCE_DIR}" "${googletest_BINARY_DIR}"
   EXCLUDE_FROM_ALL)

# Set up aliases for the GTest targets with the same name that they have
# when we find GTest pre-installed.
add_library(GTest::gtest ALIAS gtest)
add_library(GTest::gtest_main ALIAS gtest_main)
