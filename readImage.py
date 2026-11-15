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

import cv2
import numpy as np

def readImage(fileName, rgb=True):
    inputImage = cv2.imdecode(np.fromfile(fileName, dtype=np.uint8),
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if inputImage is None:
        raise IOError('cannot decode %s' % fileName)
    if rgb:
        return cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
    return inputImage
