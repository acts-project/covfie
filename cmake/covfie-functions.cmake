# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2023 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

# Helper function for adding individual flags to "flag variables".
#
# Usage: covfie_add_flag( CMAKE_CXX_FLAGS "-Wall" )
#
function( covfie_add_flag name value )

   # Escape special characters in the value:
   set( matchedValue "${value}" )
   foreach( c "*" "." "^" "$" "+" "?" )
      string( REPLACE "${c}" "\\${c}" matchedValue "${matchedValue}" )
   endforeach()

   # Check if the variable already has this value in it:
   if( "${${name}}" MATCHES "${matchedValue}" )
      return()
   endif()

   # If not, then let's add it now:
   set( ${name} "${${name}} ${value}" PARENT_SCOPE )

endfunction( covfie_add_flag )
