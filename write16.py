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

def writeTIF16(fileName, img):
    imgOut = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".tif", imgOut,
                                   [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    if not success:
        raise IOError('cannot encode %s' % fileName)
    buffer.tofile(fileName)
