# cineAlign -- aligning film scans in position and colour
#
# Copyright (C) 2026 Dr. R. Henkel
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version. See <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Commercial licences are available for use cases the GPL does not cover.
# Enquiries: license@pixelcircus.com

import os

def listDirectory(directory, extension):

    wanted = set()
    for e in extension:
        e = e.lower()
        if not e.startswith('.'):
            e = '.' + e
        wanted.add(e)

    fl = []
    with os.scandir(directory) as it:
        for entry in it:
            if os.path.splitext(entry.name)[1].lower() in wanted and entry.is_file():
                fl.append(entry.path)
    return fl

def listSubDirectories(directory, exclude=None):

    excludeAbs = os.path.abspath(exclude) if exclude else None

    dl = []
    for f in sorted(os.listdir(directory)):
        full = os.path.join(directory, f)
        if not os.path.isdir(full):
            continue
        if excludeAbs and os.path.abspath(full) == excludeAbs:
            continue
        dl.append(full)
    return dl
