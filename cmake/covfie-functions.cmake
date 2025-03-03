# SPDX-PackageName: "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
#
# SPDX-License-Identifier: MPL-2.0

# Helper function for adding individual flags to "flag variables".
#
# Usage: covfie_add_flag( CMAKE_CXX_FLAGS "-Wall" )
#
function(covfie_add_flag name value)
    # Escape special characters in the value:
    set(matchedValue "${value}")
    foreach(
        c
        "*"
        "."
        "^"
        "$"
        "+"
        "?"
    )
        string(REPLACE "${c}" "\\${c}" matchedValue "${matchedValue}")
    endforeach()

    # Check if the variable already has this value in it:
    if("${${name}}" MATCHES "${matchedValue}")
        return()
    endif()

    # If not, then let's add it now:
    set(${name} "${${name}} ${value}" PARENT_SCOPE)
endfunction(covfie_add_flag)
