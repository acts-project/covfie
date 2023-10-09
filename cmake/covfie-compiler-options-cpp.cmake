# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2023 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Include the helper function(s).
include( covfie-functions )

# Turn on a number of warnings for the "known compilers".
if( ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" ) OR
    ( "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" ) )

   # Basic flags for all build modes.
   covfie_add_flag( CMAKE_CXX_FLAGS "-Wall" )
   covfie_add_flag( CMAKE_CXX_FLAGS "-Wextra" )
   covfie_add_flag( CMAKE_CXX_FLAGS "-Wshadow" )
   covfie_add_flag( CMAKE_CXX_FLAGS "-Wunused-local-typedefs" )
   covfie_add_flag( CMAKE_CXX_FLAGS "-pedantic" )

   # Fail on warnings, if asked for that behaviour.
   if( COVFIE_FAIL_ON_WARNINGS )
      covfie_add_flag( CMAKE_CXX_FLAGS "-Werror" )
   endif()

elseif( "${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC" )

   # Basic flags for all build modes.
   string( REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS}" )
   covfie_add_flag( CMAKE_CXX_FLAGS "/W4" )

   # Fail on warnings, if asked for that behaviour.
   if( COVFIE_FAIL_ON_WARNINGS )
      covfie_add_flag( CMAKE_CXX_FLAGS "/WX" )
   endif()

   # Turn on the correct setting for the __cplusplus macro with MSVC.
   covfie_add_flag( CMAKE_CXX_FLAGS "/Zc:__cplusplus" )

endif()
