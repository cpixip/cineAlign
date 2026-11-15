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
import re

def getIndex(fileName):
    stem = os.path.splitext(os.path.basename(fileName))[0]

    m = re.search(r'(\d+)\s*$', stem)
    if not m:
        return 0

    return int(m.group(1))
