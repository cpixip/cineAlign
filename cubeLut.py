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

import hashlib
import os
import threading

import numpy as np

class CubeLut:

    def __init__(self, path):
        self.path = os.path.abspath(path)
        with open(path, 'rb') as fh:
            raw = fh.read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.film = ''
        self.title = ''
        size = None
        lo = np.zeros(3)
        hi = np.ones(3)
        rows = []
        for line in raw.decode('utf-8', errors='replace').splitlines():
            t = line.strip()
            if not t:
                continue
            if t.startswith('#'):
                c = t[1:].strip()
                if c.lower().startswith('film:') and not self.film:
                    self.film = c[5:].strip()
                continue
            key = t.split()[0].upper()
            if key == 'TITLE':
                self.title = t[5:].strip().strip('"')
            elif key == 'LUT_3D_SIZE':
                raise ValueError('%s: 3D LUT -- only 1D LUTs are supported' % path)
            elif key == 'LUT_1D_SIZE':
                size = int(t.split()[1])
            elif key == 'LUT_1D_INPUT_RANGE':
                a, b = (float(v) for v in t.split()[1:3])
                lo[:], hi[:] = a, b
            elif key == 'DOMAIN_MIN':
                lo = np.array([float(v) for v in t.split()[1:4]])
            elif key == 'DOMAIN_MAX':
                hi = np.array([float(v) for v in t.split()[1:4]])
            else:
                rows.append([float(v) for v in t.split()[:3]])
        if size is None:
            raise ValueError('%s: no LUT_1D_SIZE' % path)
        data = np.array(rows, dtype=np.float64)
        if data.shape != (size, 3):
            raise ValueError('%s: %d data rows, LUT_1D_SIZE says %d'
                             % (path, len(rows), size))
        self.size = size
        self.lo, self.hi, self.data = lo, hi, data
        self._tables = {}
        self._lock = threading.Lock()

    def describe(self):
        name = self.film or self.title or os.path.basename(self.path)
        return '%s  (%s, %d entries, sha256 %s...)' % (
            name, os.path.basename(self.path), self.size, self.sha256[:12])

    def record(self):
        return {'path': self.path, 'film': self.film, 'title': self.title,
                'size': self.size, 'sha256': self.sha256}

    def _table(self, dtype):
        dtype = np.dtype(dtype)
        with self._lock:
            tab = self._tables.get(dtype)
            if tab is None:
                if dtype == np.uint16:
                    top = 65535
                elif dtype == np.uint8:
                    top = 255
                else:
                    raise TypeError('LUT: only 8/16-bit integer images, got %s' % dtype)
                x = np.arange(top + 1, dtype=np.float64) / top
                tab = np.empty((3, top + 1), dtype=dtype)
                for c in range(3):
                    grid = np.linspace(self.lo[c], self.hi[c], self.size)
                    v = np.interp(x, grid, self.data[:, c])
                    tab[c] = np.clip(np.round(v * top), 0, top).astype(dtype)
                self._tables[dtype] = tab
            return tab

    def apply(self, img):
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError('LUT: RGB image expected, got shape %s' % (img.shape,))
        tab = self._table(img.dtype)
        out = np.empty_like(img)
        for c in range(3):
            out[..., c] = tab[c][img[..., c]]
        if img.shape[2] > 3:
            out[..., 3:] = img[..., 3:]
        return out
