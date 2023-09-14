/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <boost/filesystem.hpp>
#include <tmp_file.hpp>

boost::filesystem::path get_tmp_file()
{
    return boost::filesystem::temp_directory_path() /
           boost::filesystem::unique_path(
               "covfie_test_%%%%_%%%%_%%%%_%%%%.covfie"
           );
}
