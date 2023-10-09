/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022-2023 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

// Local include(s).
#include <tmp_file.hpp>

// System include(s).
#include <cstdio>

std::filesystem::path get_tmp_file()
{
    char fname[L_tmpnam];
    char * dummy = std::tmpnam(fname);
    (void)dummy;
    return std::filesystem::temp_directory_path() /
           std::filesystem::path(fname);
}
