#!/usr/bin/env python3
import argparse
import os
import sys
from subprocess import check_output
import re
import difflib
from datetime import datetime
from fnmatch import fnmatch

EXCLUDE = []


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


CROSS_SYMBOL = "\u2717"


def err(string):
    if sys.stdout.isatty():
        return bcolors.FAIL + bcolors.BOLD + string + bcolors.ENDC
    else:
        return string


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="+")
    p.add_argument("--exclude", "-e", action="append", default=EXCLUDE)

    args = p.parse_args()
    extensions = ["cpp", "hpp", "ipp", "cuh", "cu", "C", "h"]

    if len(args.input) == 1 and os.path.isdir(args.input[0]):
        find_command = ["find", args.input[0]]
        for ext in extensions:
            find_command.extend(["-iname", f"*.{ext}", "-or"])
        # Remove the last "-or" for a valid command
        find_command = find_command[:-1]

        srcs = (
            str(
                check_output(find_command),
                "utf-8",
            )
            .strip()
            .split("\n")
        )
        srcs = filter(lambda p: not p.startswith("./build"), srcs)
    else:
        srcs = args.input

    raw = """/*
 * This file is part of covfie, a part of the ACTS project
 *
 * Copyright (c) 2022 CERN
 *
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 */"""

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
            if not f.read().startswith(raw):
                print("Invalid / missing license in " + src + "")
                exit = 1
                continue

    sys.exit(exit)


if "__main__" == __name__:
    main()
