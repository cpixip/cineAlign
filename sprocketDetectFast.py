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

import numpy as np

def sprocketDetectFast(img,
                       roi         = (0.01, 0.06, 0.1, 0.9),
                       thresholds  = (0.35, 0.5),
                       filterSize  = 25,
                       refFrac     = 0.96,
                       nominalSize = 0.23,
                       sizeTol     = 0.06,
                       colStep     = 2,
                       maxGap      = 0.012,
                       horizontal  = False,
                       hRef        = None,
                       hSearch     = 2.0,
                       hBand       = 1.5,
                       info        = None):

    dy, dx = img.shape[0], img.shape[1]
    if info is not None:
        info.clear()
        info.update(found=False, route='none')

    x0 = int(roi[0]*dx); x1 = int(roi[1]*dx)
    y0 = int(roi[2]*dy); y1 = int(roi[3]*dy)

    strip = img[y0:y1, x0:x1:colStep]
    if strip.ndim == 3:
        strip = strip.mean(axis=2)
    strip = np.ascontiguousarray(strip.astype(np.float32))

    rowmed = np.median(strip, axis=1)
    if rowmed.size == 0 or rowmed.max() <= 0:
        return (0, 0)

    peak = rowmed.max()
    near = rowmed > (0.90 * peak)
    m = np.empty(near.size + 2, np.int8); m[0]=0; m[-1]=0; m[1:-1]=near
    d = np.diff(m)
    s = np.flatnonzero(d ==  1)
    e = np.flatnonzero(d == -1)
    if s.size == 0:
        return (0, 0)
    kb = int(np.argmax(e - s))
    b_lo, b_hi = s[kb], e[kb]

    c0 = b_lo + (b_hi - b_lo)//3
    c1 = b_hi - (b_hi - b_lo)//3
    if c1 <= c0:
        c0, c1 = b_lo, b_hi
    reference = float(np.median(rowmed[c0:c1]))
    if reference <= 0:
        return (0, 0)

    thr  = refFrac * reference
    mask = rowmed > thr
    m = np.empty(mask.size + 2, np.int8); m[0]=0; m[-1]=0; m[1:-1]=mask
    d = np.diff(m)
    starts = np.flatnonzero(d ==  1)
    ends   = np.flatnonzero(d == -1)
    if starts.size == 0:
        return (0, 0)

    k  = int(np.argmax(ends - starts))
    lo = starts[k] + y0
    hi = ends[k]   + y0
    sprocketSize = hi - lo

    nomPx  = int(nominalSize*dy)
    loBand = int((nominalSize - sizeTol)*dy)
    hiBand = int((nominalSize + sizeTol)*dy)

    gapPx = int(maxGap*dy)
    bridged = 0
    if gapPx > 0 and starts.size > 1:
        keep = (starts[1:] - ends[:-1]) > gapPx
        s2 = starts[np.concatenate(([True], keep))]
        e2 = ends[np.concatenate((keep, [True]))]
        k2 = int(np.argmax(e2 - s2))
        size2 = e2[k2] - s2[k2]
        if loBand < size2 < hiBand and (not (loBand < sprocketSize < hiBand)
                                        or abs(size2 - nomPx) < abs(sprocketSize - nomPx)):
            lo = s2[k2] + y0
            hi = e2[k2] + y0
            sprocketSize = hi - lo
            bridged = int(np.count_nonzero((starts >= s2[k2]) & (ends <= e2[k2]))) - 1

    if loBand < sprocketSize < hiBand:
        sprocketCenter = (lo + hi)//2
        route = 'clean'
    elif sprocketSize >= hiBand:
        sprocketCenter = hi - nomPx//2
        route = 'lower edge'
    else:
        sprocketCenter = dy//2
        sprocketSize   = 0
        route = 'too small'

    if info is not None:
        info.update(found=(route != 'too small'), route=route, lo=int(lo), hi=int(hi),
                    center=int(sprocketCenter), size=int(hi - lo), nominal=nomPx,
                    bridged=bridged)

    xShift = 0
    if horizontal:
        wroi   = x1 - x0
        edge   = None
        hroute = 'none'
        hrows  = None
        if sprocketSize > 0:
            rx1 = min(dx, x1 + int(round(hSearch*wroi)))
            ry  = max(2, min(int(round(hBand*wroi)), int(0.8*min(sprocketSize, nomPx))))
            ry0 = max(0,  sprocketCenter - ry//2)
            ry1 = min(dy, sprocketCenter + ry//2)
            hrows = (int(ry0), int(ry1))

            hstrip = img[ry0:ry1, x0:rx1]
            if hstrip.ndim == 3:
                hstrip = hstrip.mean(axis=2)
            hstrip = np.ascontiguousarray(hstrip.astype(np.float32))

            if hstrip.size:
                colmed = np.median(hstrip, axis=0)
                hthr   = refFrac * reference
                hmask  = colmed > hthr
                past   = np.flatnonzero(~hmask[wroi:])
                if past.size:
                    i = wroi + int(past[0])
                    a = float(colmed[i - 1]) if i > 0 else hthr
                    b = float(colmed[i])
                    frac = (a - hthr)/(a - b) if a > hthr and a > b else 1.0
                    edge   = round(x0 + (i - 1) + frac, 2)
                    hroute = 'found'
                    if hRef is None:
                        xShift = int(round(x1 - edge))
                    else:
                        off = float(hRef) - edge
                        if abs(off) > wroi:
                            hroute = 'discarded'
                        else:
                            xShift = int(round(off))
        if info is not None:
            info.update(edge=edge, hroute=hroute, hlimit=int(wroi), hrows=hrows)

    return (xShift, dy//2 - sprocketCenter)
