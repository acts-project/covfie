# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2023 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# FindCUDAToolkit needs at least CMake 3.17.
cmake_minimum_required( VERSION 3.17 )

# Include the helper function(s).
include( covfie-functions )

# Figure out the properties of CUDA being used.
find_package( CUDAToolkit REQUIRED )

# Set the architecture to build code for.
set( CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING
   "CUDA architectures to build device code for" )

# Turn on the correct setting for the __cplusplus macro with MSVC.
if( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )
   covfie_add_flag( CMAKE_CUDA_FLAGS "-Xcompiler /Zc:__cplusplus" )
endif()

if( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA" )
   # Make CUDA generate debug symbols for the device code as well in a debug
   # build.
   covfie_add_flag( CMAKE_CUDA_FLAGS_DEBUG "-G" )
   # Allow to use functions in device code that are constexpr, even if they are
   # not marked with __device__.
   covfie_add_flag( CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr" )
endif()

# Fail on warnings, if asked for that behaviour.
if( COVFIE_FAIL_ON_WARNINGS )
   if( ( "${CUDAToolkit_VERSION}" VERSION_GREATER_EQUAL "10.2" ) AND
       ( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA" ) )
      covfie_add_flag( CMAKE_CUDA_FLAGS "-Werror all-warnings" )
   elseif( "${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang" )
      covfie_add_flag( CMAKE_CUDA_FLAGS "-Werror" )
   endif()
endif()
