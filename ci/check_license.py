#!/usr/bin/env python3
# SPDX-PackageName = "covfie, a part of the ACTS project"
# SPDX-FileCopyrightText: 2022 CERN
#
# SPDX-License-Identifier: MPL-2.0

import argparse
import os
import sys
from subprocess import check_output
import re
import difflib
from datetime import datetime
from fnmatch import fnmatch

EXCLUDE = []

def get_licence_format(file):
    if file.endswith("CMakeLists.txt") or file.endswith(".cmake"):
        return """# This file is part of covfie, a part of the ACTS project
#
# Copyright (c) 2022 CERN
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
"""
    elif any(file.endswith(x) for x in [".cpp",".hpp",".ipp",".cu",".cuh"]):
        return """/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */
"""
    else:
        raise ValueError("No license known for file `{}`".format(file))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="+")
    p.add_argument("--exclude", "-e", action="append", default=EXCLUDE)

    args = p.parse_args()

    srcs = args.input

    exit = 0
    srcs = list(srcs)
    nsrcs = len(srcs)
    step = max(int(nsrcs / 20), 1)
    # Iterate over all files
    for i, src in enumerate(srcs):
        if any([fnmatch(src, e) for e in args.exclude]):
            continue

        # Read the header
        with open(src, "r+") as f:
            # License could not be found in header
            if not f.read().startswith(get_licence_format(src)):
                print("Invalid / missing license in " + src + "")
                exit = 1
                continue

    sys.exit(exit)


if "__main__" == __name__:
    main()
