#!/usr/bin/env python3
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

import json
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool

import numpy as np

try:
    from PyQt5.QtCore import Qt, QTimer, QPoint, QPointF, QRectF, QThread, QEvent, pyqtSignal
    from PyQt5.QtGui import QCursor, QImage, QPixmap, QPainter, QPalette, QColor, QPen
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
        QVBoxLayout, QHBoxLayout, QGridLayout, QRadioButton, QButtonGroup,
        QDoubleSpinBox, QSlider, QFileDialog, QGroupBox, QCheckBox,
        QSizePolicy, QMessageBox, QProgressBar, QPlainTextEdit, QScrollArea, QFrame)
except ImportError:
    sys.exit("\ncineAlignQt: PyQt5 is not installed.\n         pip install PyQt5\n")

import cineAlign as ca
from sprocketDetectFast import sprocketDetectFast
from shiftImg import shiftImg
from readImage import readImage
from getIndex import getIndex

import inspect
_old = []
if not all(hasattr(ca, n) for n in ("loadRoi", "saveRoi", "summaryLines", "progressLine", "ROI_DEFAULT",
                                        "SHIFT_LOG", "IGNORED_DIRS", "loadAlign", "measureEdgeRef", "EDGE_METHOD")) \
        or tuple(int(v) for v in str(getattr(ca, "VERSION", "0")).split(".")[:2]) < (2, 5):
    _old.append("cineAlign.py (the version that came with cineAlignQt is needed, found %s)"
                % getattr(ca, "VERSION", "?"))
if not all(n in inspect.signature(sprocketDetectFast).parameters for n in ("info", "hRef", "hBand")):
    _old.append("sprocketDetectFast.py (the version that came with cineAlignQt is needed)")
if _old:
    sys.exit("\ncineAlignQt: older files in this folder -- please replace them with the ones "
             "delivered together with cineAlignQt.py:\n         " + "\n         ".join(_old) + "\n")

VERSION = ca.VERSION
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "cineAlignQt_settings.json")
PLAY_FPS = 18.0
CACHE_N = 12
SHADOW_MAX = 4.0
ROI_MIN_W, ROI_MIN_H = 0.004, 0.05
GRAB = 7

COL_ROI = QColor(60, 200, 230)
COL_EDGE = QColor(255, 220, 0)
COL_CENTER = QColor(90, 220, 90)
COL_TORN = QColor(255, 150, 40)
COL_NONE = QColor(235, 70, 70)
COL_H = QColor(110, 180, 255)
COL_HBAD = QColor(230, 90, 230)
OK_ROUTES = ("clean", "lower edge")

def inv709(v):
    v = np.asarray(v, np.float64)
    return np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45))

def oetf709(L):
    L = np.asarray(L, np.float64)
    return np.where(L < 0.018, 4.5 * L, 1.099 * np.maximum(L, 0.0) ** 0.45 - 0.099)

def display_table(dtype, ev):
    top = 65535 if np.dtype(dtype) == np.uint16 else 255
    v = np.arange(top + 1, dtype=np.float64) / top
    if ev > 0:
        a = 2.0 ** ev
        L = inv709(v)
        v = oetf709(a * L / (1 + (a - 1) * L))
    return np.clip(v * 255 + 0.5, 0, 255).astype(np.uint8)

class Canvas(QWidget):
    roiChanged = pyqtSignal(object)
    roiReleased = pyqtSignal(object)
    viewChanged = pyqtSignal()
    ZOOM_MAX = 8.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self._pix = None
        self._size = None
        self.z = None
        self.pan = QPoint(0, 0)
        self.roi = None
        self.roi_edit = True
        self.show_roi = True
        self.show_center = False
        self.det = None
        self.href = None
        self.message = "drop an image folder here (flat, or one level of scene folders)"
        self._drag = None

    def set_image(self, rgb8, size):
        if rgb8 is None:
            self._pix, self._size = None, None
        else:
            h, w = rgb8.shape[:2]
            buf = np.ascontiguousarray(rgb8)
            self._pix = QPixmap.fromImage(QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888))
            self._size = size
        self.update()

    def fit_scale(self):
        if self._size is None:
            return 1.0
        return min(self.width() / self._size[0], self.height() / self._size[1])

    def scale(self):
        return self.z if self.z is not None else self.fit_scale()

    def origin(self):
        s = self.scale()
        w, h = self._size[0] * s, self._size[1] * s
        x = (self.width() - w) / 2 + (self.pan.x() if self.z is not None else 0)
        y = (self.height() - h) / 2 + (self.pan.y() if self.z is not None else 0)
        return x, y, s

    def to_image(self, sx, sy):
        x0, y0, s = self.origin()
        return (sx - x0) / s, (sy - y0) / s

    def to_screen(self, ix, iy):
        x0, y0, s = self.origin()
        return x0 + ix * s, y0 + iy * s

    def zoom_at(self, factor, sx, sy):
        if self._size is None:
            return
        p = self.to_image(sx, sy)
        z = min(self.ZOOM_MAX, self.scale() * factor)
        if z <= self.fit_scale() * 1.001:
            self.z, self.pan = None, QPoint(0, 0)
        else:
            self.z = z
            w, h = self._size[0] * z, self._size[1] * z
            self.pan = QPoint(int(round(sx - p[0] * z - (self.width() - w) / 2)),
                              int(round(sy - p[1] * z - (self.height() - h) / 2)))
        self.update()
        self.viewChanged.emit()

    def _roi_screen(self):
        W, H = self._size
        l, r, t, b = self.roi
        x0, y0 = self.to_screen(l * W, t * H)
        x1, y1 = self.to_screen(r * W, b * H)
        return x0, y0, x1, y1

    def _edge_at(self, pos):
        if self._size is None or self.roi is None or not (self.roi_edit and self.show_roi):
            return None
        x0, y0, x1, y1 = self._roi_screen()
        x, y = pos.x(), pos.y()
        best = None
        if y0 - GRAB <= y <= y1 + GRAB:
            for k, ex in ((0, x0), (1, x1)):
                d = abs(x - ex)
                if d <= GRAB and (best is None or d < best[0]):
                    best = (d, k)
        if x0 - GRAB <= x <= x1 + GRAB:
            for k, ey in ((2, y0), (3, y1)):
                d = abs(y - ey)
                if d <= GRAB and (best is None or d < best[0]):
                    best = (d, k)
        return best[1] if best else None

    def _inside(self, pos):
        if self._size is None or self.roi is None or not (self.roi_edit and self.show_roi):
            return False
        x0, y0, x1, y1 = self._roi_screen()
        return x0 < pos.x() < x1 and y0 < pos.y() < y1

    def _move_roi(self, start, roi0, pos):
        W, H = self._size
        ix, iy = self.to_image(pos.x(), pos.y())
        l, r, t, b = roi0
        dx = float(np.clip((ix - start[0]) / W, -l, 1.0 - r))
        dy = float(np.clip((iy - start[1]) / H, -t, 1.0 - b))
        self.roi = (l + dx, r + dx, t + dy, b + dy)
        self.update()
        self.roiChanged.emit(self.roi)

    def _move_edge(self, k, pos):
        W, H = self._size
        ix, iy = self.to_image(pos.x(), pos.y())
        l, r, t, b = self.roi
        if k == 0:
            l = float(np.clip(ix / W, 0.0, r - ROI_MIN_W))
        elif k == 1:
            r = float(np.clip(ix / W, l + ROI_MIN_W, 1.0))
        elif k == 2:
            t = float(np.clip(iy / H, 0.0, b - ROI_MIN_H))
        else:
            b = float(np.clip(iy / H, t + ROI_MIN_H, 1.0))
        self.roi = (l, r, t, b)
        self.update()
        self.roiChanged.emit(self.roi)

    def wheelEvent(self, ev):
        self.zoom_at(1.25 if ev.angleDelta().y() > 0 else 0.8, ev.x(), ev.y())

    def mousePressEvent(self, ev):
        self.setFocus()
        if ev.button() != Qt.LeftButton or self._size is None:
            return
        k = self._edge_at(ev.pos())
        if k is not None:
            self._drag = ("edge", k)
        elif self._inside(ev.pos()):
            self._drag = ("move", self.to_image(ev.x(), ev.y()), self.roi)
        else:
            self._drag = ("pan", ev.pos() - self.pan)

    def mouseMoveEvent(self, ev):
        if self._drag is None:
            k = self._edge_at(ev.pos())
            self.setCursor(Qt.SizeHorCursor if k in (0, 1) else
                           Qt.SizeVerCursor if k in (2, 3) else
                           Qt.SizeAllCursor if self._inside(ev.pos()) else Qt.ArrowCursor)
            return
        if self._drag[0] == "edge":
            self._move_edge(self._drag[1], ev.pos())
        elif self._drag[0] == "move":
            self._move_roi(self._drag[1], self._drag[2], ev.pos())
        elif self.z is not None:
            self.pan = ev.pos() - self._drag[1]
            self.update()
            self.viewChanged.emit()

    def mouseReleaseEvent(self, ev):
        if self._drag is not None and self._drag[0] in ("edge", "move"):
            self.roiReleased.emit(self.roi)
        self._drag = None

    def mouseDoubleClickEvent(self, ev):
        if self._size is None or self._edge_at(ev.pos()) is not None or self._inside(ev.pos()):
            return
        if self.z is None:
            self.zoom_at(1.0 / self.fit_scale(), ev.x(), ev.y())
        else:
            self.z, self.pan = None, QPoint(0, 0)
            self.update()
            self.viewChanged.emit()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.black)
        if self._pix is None or self._size is None:
            p.setPen(Qt.gray)
            p.drawText(self.rect(), Qt.AlignCenter, self.message)
            return
        x0, y0, s = self.origin()
        W, H = self._size
        p.setRenderHint(QPainter.SmoothPixmapTransform, s < 1.0)
        p.drawPixmap(QRectF(x0, y0, W * s, H * s), self._pix, QRectF(self._pix.rect()))
        p.setRenderHint(QPainter.Antialiasing, False)
        d = self.det or {}
        if self.roi is not None and self.show_roi:
            rx0, ry0, rx1, ry1 = self._roi_screen()
            found = bool(d.get("found"))
            col = COL_ROI if (found or not d) else COL_NONE
            for c, w in ((QColor(0, 0, 0), 3.0), (col, 1.4)):
                p.setPen(QPen(c, w))
                p.drawRect(QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0))
            if found:
                ext = max(40.0, 0.06 * W * s)
                edge = COL_TORN if d.get("route") == "lower edge" else COL_EDGE
                for row, c, st in ((d["lo"], edge, Qt.SolidLine), (d["hi"], edge, Qt.SolidLine),
                                   (d["center"], COL_CENTER, Qt.SolidLine)):
                    _x, sy = self.to_screen(0, row)
                    for cc, w in ((QColor(0, 0, 0), 3.0), (c, 1.4)):
                        p.setPen(QPen(cc, w, st))
                        p.drawLine(QPointF(rx0, sy), QPointF(rx1 + ext, sy))
                if d.get("edge") is not None:
                    ra, rb = d.get("hrows") or (d["lo"], d["hi"])
                    sx, sya = self.to_screen(d["edge"] + 0.5, ra)
                    _x, syb = self.to_screen(d["edge"] + 0.5, rb)
                    c = COL_H if d.get("hroute") == "found" else COL_HBAD
                    for cc, w in ((QColor(0, 0, 0), 3.0), (c, 1.4)):
                        p.setPen(QPen(cc, w))
                        p.drawLine(QPointF(sx, sya), QPointF(sx, syb))
        if self.show_center or self.href is not None:
            p.setPen(QPen(QColor(255, 255, 255), 1.0, Qt.DashLine))
            if self.show_center:
                _x, sy = self.to_screen(0, H // 2)
                sx0, _y = self.to_screen(0, 0)
                p.drawLine(QPointF(sx0, sy), QPointF(sx0 + max(60.0, 0.12 * W * s), sy))
            if self.href is not None:
                if self.show_roi and d.get("found"):
                    ra, rb = d["lo"] - 0.03 * H, d["hi"] + 0.03 * H
                else:
                    ra, rb = 0.38 * H, 0.62 * H
                sx, sya = self.to_screen(self.href + 0.5, ra)
                _x, syb = self.to_screen(self.href + 0.5, rb)
                p.drawLine(QPointF(sx, sya), QPointF(sx, syb))

class NavBar(QWidget):
    seek = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.n, self.idx = 0, 0
        self.rng = None
        self.cuts = []
        self.setToolTip("Click or drag: go to a frame. Blue: the range that will be processed\n"
                        "(i / o set its start / end). Ticks: scene folders.")

    def _x(self, i):
        m = 6
        return m + (self.width() - 2 * m) * (i / max(1, self.n - 1))

    def paintEvent(self, ev):
        p = QPainter(self)
        h = self.height()
        p.fillRect(QRectF(6, h / 2 - 4, self.width() - 12, 8), QColor(52, 54, 60))
        if self.n:
            a, b = self.rng if self.rng is not None else (0, self.n - 1)
            xa, xb = self._x(a), self._x(b)
            p.fillRect(QRectF(xa, h / 2 - 4, max(2.0, xb - xa), 8), QColor(78, 132, 190))
            p.setPen(QPen(QColor(150, 155, 162), 1))
            for c in self.cuts:
                x = self._x(c)
                p.drawLine(QPointF(x, h / 2 - 7), QPointF(x, h / 2 + 7))
            x = self._x(self.idx)
            p.fillRect(QRectF(x - 1.5, 2, 3, h - 4), QColor(255, 220, 0))

    def _seek(self, ev):
        if self.n:
            m = 6
            f = (ev.x() - m) / max(1, self.width() - 2 * m)
            self.seek.emit(int(round(np.clip(f, 0, 1) * (self.n - 1))))

    def mousePressEvent(self, ev):
        self._seek(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self._seek(ev)

class ShiftPlot(QWidget):
    seek = pyqtSignal(int)
    M = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.n, self.idx = 0, 0
        self.rng = None
        self.data = {}
        self.setToolTip("Vertical shift found by the detection [px], over the whole film -- filled\n"
                        "in as frames are looked at and as the run proceeds.\n"
                        "Green: clean hole. Orange: hole too large or torn (centre from the lower\n"
                        "edge). Red tick: nothing found, the frame stays unshifted.\n"
                        "With horizontal alignment also the horizontal shift (blue). Magenta tick\n"
                        "at the top: edge not found or value discarded, no horizontal shift.\n"
                        "The scale follows the values (1 to 99 %); outliers sit at the border.\n"
                        "A new search area clears the plot. Click or drag: go to a frame.")

    def _x(self, i):
        return self.M + (self.width() - 2 * self.M) * (i / max(1, self.n - 1))

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(22, 22, 24))
        if not self.n:
            return
        if self.rng is not None:
            xa, xb = self._x(self.rng[0]), self._x(self.rng[1])
            p.fillRect(QRectF(xa, 0, max(2.0, xb - xa), h), QColor(40, 52, 68))
        vals = np.array([v for v, r, _hx, _hr in self.data.values() if r in OK_ROUTES and v is not None]
                        + [hx for _v, _r, hx, hr in self.data.values() if hr == "found"], float)
        if len(vals):
            lo, hi = np.percentile(vals, [1, 99])
        else:
            lo, hi = -5.0, 5.0
        span = max(10.0, (hi - lo) * 1.2)
        mid = (lo + hi) / 2
        lo, hi = mid - span / 2, mid + span / 2
        pad = 4
        y = lambda v: float(np.clip(h - pad - (v - lo) / (hi - lo) * (h - 2 * pad), pad, h - pad))
        if lo < 0 < hi:
            p.setPen(QPen(QColor(90, 94, 102), 1))
            p.drawLine(QPointF(self.M, y(0)), QPointF(w - self.M, y(0)))
        size = 1.0 if self.n > (w - 2 * self.M) else 2.0
        cols = {"clean": QColor(90, 220, 90), "lower edge": QColor(255, 150, 40)}
        p.setPen(Qt.NoPen)
        for i, (v, r, hx, hr) in self.data.items():
            x = self._x(i)
            if hr == "found":
                p.setBrush(COL_H)
                p.drawRect(QRectF(x - size / 2, y(hx) - size / 2, size, size))
            elif hr is not None:
                p.setBrush(COL_HBAD)
                p.drawRect(QRectF(x - 0.5, 0, 1, 7))
            if r in cols:
                if v is not None:
                    p.setBrush(cols[r])
                    p.drawRect(QRectF(x - size / 2, y(v) - size / 2, size, size))
            else:
                p.setBrush(QColor(235, 70, 70))
                p.drawRect(QRectF(x - 0.5, h - 7, 1, 7))
        p.setPen(QPen(QColor(255, 220, 0), 1))
        x = self._x(self.idx)
        p.drawLine(QPointF(x, 0), QPointF(x, h))
        p.setPen(QColor(150, 155, 162))
        f = p.font(); f.setPointSizeF(max(6.5, f.pointSizeF() * 0.8)); p.setFont(f)
        p.drawText(QRectF(self.M + 2, 1, 120, 14), Qt.AlignLeft | Qt.AlignTop, f"{hi:+.0f} px")
        p.drawText(QRectF(self.M + 2, h - 15, 120, 14), Qt.AlignLeft | Qt.AlignBottom, f"{lo:+.0f} px")
        if not self.data:
            p.drawText(self.rect(), Qt.AlignCenter, "shift per frame -- appears as frames are looked at or processed")

    def _seek(self, ev):
        if self.n:
            f = (ev.x() - self.M) / max(1, self.width() - 2 * self.M)
            self.seek.emit(int(round(np.clip(f, 0, 1) * (self.n - 1))))

    def mousePressEvent(self, ev):
        self._seek(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self._seek(ev)

class ShiftZoom(QWidget):
    HALVES = (10, 20, 30, 50, 100, 200)

    def __init__(self, data_of, parent=None):
        super().__init__(parent)
        self.setFixedHeight(64)
        self.setMinimumWidth(168)
        self.data_of = data_of
        self.half_i = 2
        self.setToolTip("The shift around the current frame, enlarged: lines from frame to frame.\n"
                        "Orange: centre from the lower edge. Red tick: nothing found. A gap in the\n"
                        "line: frame not known yet. Blue: horizontal shift; magenta tick at the\n"
                        "top: edge not found or value discarded. Mouse wheel: narrower / wider.")

    def wheelEvent(self, ev):
        d = -1 if ev.angleDelta().y() > 0 else 1
        self.half_i = max(0, min(len(self.HALVES) - 1, self.half_i + d))
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(22, 22, 24))
        p.setPen(QPen(QColor(58, 58, 62), 1))
        p.drawRect(QRectF(0.5, 0.5, w - 1, h - 1))
        data, n, idx = self.data_of()
        half = self.HALVES[self.half_i]
        if not n:
            return
        m = 4
        x = lambda i: m + (w - 2 * m) * ((i - idx + half) / (2.0 * half))
        pts = [(i, data[i]) for i in range(max(0, idx - half), min(n - 1, idx + half) + 1) if i in data]
        vals = ([d[0] for _i, d in pts if d[1] in OK_ROUTES and d[0] is not None]
                + [d[2] for _i, d in pts if d[3] == "found"])
        if vals:
            lo, hi = min(vals), max(vals)
        else:
            lo, hi = -3.0, 3.0
        span = max(6.0, (hi - lo) * 1.25)
        mid = (lo + hi) / 2
        lo, hi = mid - span / 2, mid + span / 2
        pad = 5
        y = lambda v: h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
        if lo < 0 < hi:
            p.setPen(QPen(QColor(90, 94, 102), 1))
            p.drawLine(QPointF(m, y(0)), QPointF(w - m, y(0)))
        p.setPen(QPen(QColor(255, 220, 0, 150), 1))
        p.drawLine(QPointF(x(idx), 0), QPointF(x(idx), h))
        p.setRenderHint(QPainter.Antialiasing, True)
        prev = None
        for i, d in pts:
            if d[3] != "found":
                if d[3] is not None:
                    p.setPen(QPen(COL_HBAD, 1.5))
                    p.drawLine(QPointF(x(i), 1), QPointF(x(i), 8))
                prev = None
                continue
            if prev is not None and prev[0] == i - 1:
                p.setPen(QPen(COL_H, 1.2))
                p.drawLine(QPointF(x(prev[0]), y(prev[1])), QPointF(x(i), y(d[2])))
            prev = (i, d[2])
        prev = None
        for i, d in pts:
            v, r = d[0], d[1]
            if r not in OK_ROUTES:
                p.setPen(QPen(COL_NONE, 1.5))
                p.drawLine(QPointF(x(i), h - 8), QPointF(x(i), h - 1))
                prev = None
                continue
            if v is None:
                prev = None
                continue
            if prev is not None and prev[0] == i - 1:
                torn = r == "lower edge" or prev[2] == "lower edge"
                p.setPen(QPen(COL_TORN if torn else COL_CENTER, 1.2))
                p.drawLine(QPointF(x(prev[0]), y(prev[1])), QPointF(x(i), y(v)))
            prev = (i, v, r)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.setPen(QColor(150, 155, 162))
        f = p.font(); f.setPointSizeF(max(6.5, f.pointSizeF() * 0.8)); p.setFont(f)
        p.drawText(QRectF(m + 1, 1, 80, 13), Qt.AlignLeft | Qt.AlignTop, f"{hi:+.0f} px")
        p.drawText(QRectF(m + 1, h - 14, 80, 13), Qt.AlignLeft | Qt.AlignBottom, f"{lo:+.0f} px")
        p.drawText(QRectF(w - 62, 1, 58, 13), Qt.AlignRight | Qt.AlignTop, f"\u00b1{half} fr.")
        cur = data.get(idx)
        right = w - m
        if cur is not None and cur[1] in OK_ROUTES and cur[0] is not None:
            t = f"{cur[0]:+d}"
            tw = p.fontMetrics().horizontalAdvance(t)
            p.setPen(COL_EDGE)
            p.drawText(QRectF(right - tw, h - 14, tw, 13), Qt.AlignRight | Qt.AlignBottom, t)
            right -= tw + 6
        if cur is not None and cur[3] == "found":
            t = f"{cur[2]:+d}"
            tw = p.fontMetrics().horizontalAdvance(t)
            p.setPen(COL_H)
            p.drawText(QRectF(right - tw, h - 14, tw, 13), Qt.AlignRight | Qt.AlignBottom, t)

class RunWorker(QThread):
    progress = pyqtSignal(int, int, float)
    finished_run = pyqtSignal(int, object, float, bool)

    def __init__(self, tasks):
        super().__init__()
        self.tasks = tasks
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        pool = ThreadPool()
        errors, done, total = [], 0, len(self.tasks)
        t0 = time.time()
        try:
            for ok, imgPath, msg in pool.imap_unordered(ca.processImage, self.tasks):
                done += 1
                if not ok:
                    errors.append((imgPath, msg))
                self.progress.emit(done, total, time.time() - t0)
                if self._cancel:
                    break
        finally:
            if self._cancel:
                pool.terminate()
            else:
                pool.close()
            pool.join()
        self.finished_run.emit(done, errors, time.time() - t0, self._cancel)

class RefWorker(QThread):
    done = pyqtSignal(object)

    def __init__(self, paths, roi):
        super().__init__()
        self.paths, self.roi = paths, roi
        self.stop = False

    def run(self):
        try:
            res = ca.measureEdgeRef(self.paths, self.roi, cancelled=lambda: self.stop)
        except Exception:
            res = (None, 0, 0)
        self.done.emit(res)

class ScanWorker(QThread):
    step = pyqtSignal(str)
    done = pyqtSignal(object)

    def __init__(self, path, out):
        super().__init__()
        self.path, self.out = path, out

    def run(self):
        try:
            res = ca.collectTasks(self.path, self.out, makeDirs=False, progress=self.step.emit)
        except Exception as e:
            res = e
        self.done.emit(res)

class Main(QMainWindow):
    NAV_STEP = {Qt.NoModifier: 1, Qt.ShiftModifier: 10, Qt.ControlModifier: 100}
    PLAY_TXT = {"rev": "◀", "stop": "■", "fwd": "▶"}

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"cineAlignQt {VERSION}")
        self.setAcceptDrops(True)
        self.settings = self._load_settings()
        self.input = None
        self.tasks = []
        self.scenes, self.mode = [], "flat"
        self.idx = 0
        self.roi = tuple(ca.ROI_DEFAULT)
        self.rng = None
        self._cache = OrderedDict()
        self._pf = ThreadPoolExecutor(max_workers=3)
        self._pf_jobs = {}
        self._tabs = {}
        self._play_dir, self._play_speed = 0, 1
        self._worker = None
        self._scan = None
        self.href = None
        self._ref_roi = None
        self._ref = None
        self._ref_again = False
        self._hstat = {}
        self._run_info = None
        self._extra = ([], [])
        self._path_idx = {}
        self._build()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)
        self._apply_settings()

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        self.canvas = Canvas()
        self.canvas.roiChanged.connect(self._roi_dragged)
        self.canvas.roiReleased.connect(lambda r: self._set_roi(r, save=True))
        self.canvas.viewChanged.connect(self._view_changed)

        top = QGridLayout()
        top.addWidget(QLabel("Input"), 0, 0)
        self.lbl_in = QLabel("drop an image folder on the window, or choose one")
        self.lbl_in.setStyleSheet("color: gray")
        top.addWidget(self.lbl_in, 0, 1)
        b = QPushButton("…"); b.setFixedWidth(28); b.setToolTip("choose the input folder")
        b.clicked.connect(self._browse_in)
        top.addWidget(b, 0, 2)
        top.addWidget(QLabel("Output"), 1, 0)
        self.ed_out = QLineEdit()
        self.ed_out.setToolTip("Output folder. A relative name is a folder inside the input folder,\n"
                               "as in cineAlign. Scene folders are mirrored into it.\n"
                               "Enter hands the keyboard back to the picture.")
        self.ed_out.editingFinished.connect(self._out_edited)
        top.addWidget(self.ed_out, 1, 1)
        b = QPushButton("…"); b.setFixedWidth(28); b.setToolTip("choose the output folder")
        b.clicked.connect(self._browse_out)
        top.addWidget(b, 1, 2)
        top.setColumnStretch(1, 1)
        v.addLayout(top)

        mid = QHBoxLayout()
        mid.addWidget(self.canvas, 1)
        side = QVBoxLayout()

        g = QGroupBox("Search area")
        gl = QGridLayout(g)
        self.sp_roi = []
        for k, (name, tip) in enumerate((("left", "left edge"), ("right", "right edge"),
                                         ("top", "top edge"), ("bottom", "bottom edge"))):
            gl.addWidget(QLabel(name), k // 2, 2 * (k % 2))
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1.0); sp.setDecimals(3); sp.setSingleStep(0.005)
            sp.setToolTip(f"{tip.capitalize()} of the strip in which the sprocket hole is searched,\n"
                          "as a fraction of the image " + ("width." if k < 2 else "height.")
                          + "\nIn the picture: drag an edge, or drag inside the area to move it.")
            sp.valueChanged.connect(self._roi_typed)
            sp.editingFinished.connect(self.canvas.setFocus)
            gl.addWidget(sp, k // 2, 2 * (k % 2) + 1)
            self.sp_roi.append(sp)
        self.btn_roi_def = QPushButton("Default")
        self.btn_roi_def.setToolTip("Back to the default strip in the left overscan "
                                    f"{tuple(ca.ROI_DEFAULT)}.")
        self.btn_roi_def.clicked.connect(lambda: self._set_roi(ca.ROI_DEFAULT, save=True))
        gl.addWidget(self.btn_roi_def, 2, 0, 1, 2)
        self.lbl_det = QLabel("–")
        self.lbl_det.setWordWrap(True)
        self.lbl_det.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.lbl_det.setMinimumHeight(6 * self.lbl_det.fontMetrics().lineSpacing())
        self.lbl_det.setToolTip("What the detection finds in this frame.\n"
                                "clean: hole of the expected height, centre between its edges.\n"
                                "lower edge: hole too large or torn -- centre from the lower edge\n"
                                "and the nominal height.\n"
                                "not found: the frame would stay unshifted.\n"
                                "With horizontal alignment: the right hole edge, its normal position and\n"
                                "the horizontal shift; discarded: further off than the search area is wide.")
        gl.addWidget(self.lbl_det, 3, 0, 1, 4)
        side.addWidget(g)

        g = QGroupBox("View")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        self.view_group = QButtonGroup(self)
        for k, (name, tip) in enumerate((
                ("Original", "The frame as read, with the search area and what is found in it (1)."),
                ("Result", "The frame as it will be written: with its LUT and shifted (2).\n"
                           "v switches between the two."))):
            rb = QRadioButton(name)
            rb.setToolTip(tip)
            self.view_group.addButton(rb, k)
            row.addWidget(rb)
        self.view_group.button(0).setChecked(True)
        self.view_group.buttonClicked.connect(lambda _b: self._redraw())
        row.addStretch(1)
        gl.addLayout(row)
        row = QHBoxLayout()
        self.lbl_shadow = QLabel("Shadows +0.0 EV")
        self.lbl_shadow.setMinimumWidth(118)
        row.addWidget(self.lbl_shadow)
        self.sl_shadow = QSlider(Qt.Horizontal)
        self.sl_shadow.setRange(0, int(SHADOW_MAX * 10))
        self.sl_shadow.setToolTip("Display only: lift the dark parts of the picture, white stays white.\n"
                                  "Detection, LUT and output are not affected. Double-click: reset.")
        self.sl_shadow.valueChanged.connect(self._shadow_changed)
        row.addWidget(self.sl_shadow, 1)
        gl.addLayout(row)
        for wdg in (self.lbl_shadow, self.sl_shadow):
            wdg.installEventFilter(self)
        side.addWidget(g)

        g = QGroupBox("Range")
        gl = QVBoxLayout(g)
        self.lbl_rng = QLabel("–")
        self.lbl_rng.setWordWrap(True)
        gl.addWidget(self.lbl_rng)
        row = QHBoxLayout()
        for text, tip, fn in (("Start (i)", "Range starts at the current frame.", lambda: self._set_range("in")),
                              ("End (o)", "Range ends at the current frame.", lambda: self._set_range("out")),
                              ("Whole film", "Process all frames.", lambda: self._set_range("all"))):
            b = QPushButton(text); b.setToolTip(tip); b.clicked.connect(fn)
            row.addWidget(b)
        gl.addLayout(row)
        side.addWidget(g)

        g = QGroupBox("Run")
        gl = QVBoxLayout(g)
        self.cb_align = QCheckBox("align vertically")
        self.cb_align.setToolTip("Shift every frame so that the centre of the sprocket hole sits at the\n"
                                 "centre of the image. The two alignments are independent; with both off,\n"
                                 "only the LUTs are applied.\n"
                                 "Both switches stay as they are when other material is loaded, and are\n"
                                 "remembered between sessions. The search area is stored with the material\n"
                                 "(_cineAlign/cineAlign.json in the input folder).")
        self.cb_align.toggled.connect(self._align_toggled)
        gl.addWidget(self.cb_align)
        self.cb_horiz = QCheckBox("align horizontally")
        self.cb_horiz.setToolTip(
            "Align sideways, to the right edge of the sprocket hole (the one towards the\n"
            "picture) -- with or without the vertical alignment. The search area has to lie inside the hole, its right border left of\n"
            "that edge; the edge is searched up to two widths of the search area further right,\n"
            "in a band around the hole centre 1.5 widths high (clear of the rounded corners).\n"
            "Reference is the normal position of the edge, measured once on 50 frames from the\n"
            "middle of the film (again after every change of the search area): a frame in the\n"
            "normal position is not moved. Edge not found, or further off than the search area\n"
            "is wide: no horizontal shift for that frame.")
        self.cb_horiz.toggled.connect(self._horiz_toggled)
        gl.addWidget(self.cb_horiz)
        self.btn_run = QPushButton("Start")
        self.btn_run.setToolTip("Write the frames of the range, every one anew (LUT, then shift).\n"
                                "Before anything is written, the output folder is shown for confirmation.")
        self.btn_run.clicked.connect(self._run_or_cancel)
        gl.addWidget(self.btn_run)
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("%v / %m")
        self.progress.hide()
        gl.addWidget(self.progress)
        side.addWidget(g)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log.setStyleSheet("font-family: Consolas, 'DejaVu Sans Mono', monospace;")
        side.addWidget(self.log, 1)

        w = QWidget(); w.setLayout(side)
        side.setContentsMargins(0, 0, 0, 0)
        self.log.setMinimumHeight(90)
        sc = QScrollArea(); sc.setWidget(w); sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setFixedWidth(340)
        mid.addWidget(sc)
        v.addLayout(mid, 1)

        nav = QHBoxLayout()
        row = QHBoxLayout(); row.setSpacing(4)
        self.btn_play = {}
        for key, tip in (("rev", "Play backwards (j); again: 2x, 4x"), ("stop", "Stop (k)"),
                         ("fwd", "Play forwards (l, space); again: 2x, 4x")):
            b = QPushButton(self.PLAY_TXT[key]); b.setFixedWidth(52); b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, k=key: self._play_cmd(k))
            row.addWidget(b)
            self.btn_play[key] = b
        left = QVBoxLayout()
        left.addStretch(1)
        left.addLayout(row)
        nav.addLayout(left)
        col = QVBoxLayout(); col.setSpacing(2)
        self.plot = ShiftPlot()
        self.plot.seek.connect(self._seek)
        col.addWidget(self.plot)
        self.bar = NavBar()
        self.bar.seek.connect(self._seek)
        col.addWidget(self.bar)
        nav.addLayout(col, 1)
        right = QVBoxLayout(); right.setSpacing(2)
        self.zoom = ShiftZoom(lambda: (self.plot.data, self.plot.n, self.idx))
        right.addWidget(self.zoom)
        self.lbl_frame = QLabel("–")
        self.lbl_frame.setMinimumWidth(260)
        right.addWidget(self.lbl_frame)
        nav.addLayout(right)
        v.addLayout(nav)

        hint = QLabel("←/→ frame · Shift ±10 · Ctrl ±100 · Home/End "
                      "· Up/Down: scene · j/k/l: play back / stop / play (again: 2×, 4×) "
                      "· space: play / pause · i/o: range start / end · 1 / 2: Original / Result (v: switch) "
                      "· wheel: zoom · double-click: fit / 1:1 · Esc: cancel run")
        hint.setStyleSheet("color: #8a9099")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.status = self.statusBar()

        for wdg in root.findChildren((QPushButton, QRadioButton, QCheckBox, QSlider)):
            wdg.setFocusPolicy(Qt.NoFocus)
        for sp in self.sp_roi:
            sp.setFocusPolicy(Qt.ClickFocus)
        self.ed_out.setFocusPolicy(Qt.ClickFocus)
        self.log.setFocusPolicy(Qt.ClickFocus)
        self._edit_widgets = self.sp_roi + [self.btn_roi_def, self.cb_align, self.cb_horiz, self.ed_out]
        self.canvas.setFocus()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.MouseButtonDblClick and obj in (self.lbl_shadow, self.sl_shadow):
            self.sl_shadow.setValue(0)
            return True
        return super().eventFilter(obj, ev)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        for url in ev.mimeData().urls():
            p = url.toLocalFile()
            if p:
                self.open_folder(p if os.path.isdir(p) else os.path.dirname(p))
                break

    def _browse_in(self):
        p = QFileDialog.getExistingDirectory(self, "Input folder", self.input or
                                             self.settings.get("input", ""))
        if p:
            self.open_folder(p)
        self.canvas.setFocus()

    def _browse_out(self):
        p = QFileDialog.getExistingDirectory(self, "Output folder", self._out_dir() or "")
        if p:
            self.ed_out.setText(os.path.normpath(p))
            self._out_edited()
        self.canvas.setFocus()

    def _out_dir(self):
        t = self.ed_out.text().strip()
        if not t or not self.input:
            return None
        return os.path.join(self.input, t)

    def _out_edited(self):
        if self.ed_out.text().strip() == self.settings.get("output"):
            self.canvas.setFocus()
            return
        self.settings["output"] = self.ed_out.text().strip()
        if self.input and not self._running():
            self.open_folder(self.input, keep=True)
        self.canvas.setFocus()

    def open_folder(self, path, keep=False):
        if self._running() or self._scan is not None:
            return
        self._play_stop()
        self._ref_cancel()
        path = os.path.normpath(path)
        if not self.ed_out.text().strip():
            QMessageBox.information(self, "cineAlignQt", "Please name an output folder first.")
            return
        out = os.path.join(path, self.ed_out.text().strip())
        self._scan = ScanWorker(path, out)
        self._scan.keep = keep
        self._scan.step.connect(self._scan_step)
        self._scan.done.connect(self._opened)
        self.progress.setRange(0, 0)
        self.progress.show()
        self.btn_run.setEnabled(False)
        self.status.showMessage(f"reading {path} \u2026")
        if not self.tasks:
            self.canvas.message = f"reading {path} \u2026"
            self.canvas.update()
        self._scan.start()

    def _scan_step(self, text):
        self.status.showMessage(text + " \u2026")

    def _opened(self, res):
        scan, self._scan = self._scan, None
        scan.wait()
        path, keep = scan.path, scan.keep
        self.progress.hide()
        self.progress.setRange(0, 1)
        self.btn_run.setEnabled(True)
        self.canvas.message = "drop an image folder here (flat, or one level of scene folders)"
        self.canvas.update()
        if isinstance(res, Exception):
            self.status.showMessage("")
            QMessageBox.warning(self, "cineAlignQt", str(res))
            return
        tasks, scenes, empty, stale, mode = res
        if not tasks:
            self.status.showMessage("")
            QMessageBox.information(
                self, "cineAlignQt",
                f"No images found in\n{path}\n\nLooked for: "
                + ", ".join("." + e for e in ca.extensionList)
                + " -- directly in the folder, or in its subfolders (one level).")
            return
        same = keep and self.input == path and len(tasks) == len(self.tasks)
        self.input = path
        self.tasks, self.scenes, self.mode = tasks, scenes, mode
        self._extra = (empty, stale)
        self._cache.clear()
        for f in self._pf_jobs.values():
            f.cancel()
        self._pf_jobs.clear()
        if not same:
            self.idx, self.rng = 0, None
            self.canvas.z, self.canvas.pan = None, QPoint(0, 0)
            st = ca.loadAlign(path)
            self.href = st["edge_ref"]
            self._set_roi(st["roi"] or ca.ROI_DEFAULT, save=False, redraw=False)
            self._ref_roi = self.roi if self.href is not None else None
            self._roi_widgets()
        self.lbl_in.setText(path)
        self.lbl_in.setStyleSheet("")
        self.settings["input"] = path
        self._path_idx = {os.path.normcase(os.path.abspath(t[0])): i for i, t in enumerate(tasks)}
        self.plot.n = len(tasks)
        if not same:
            self.plot.data = {}
        self.bar.n = len(tasks)
        self.bar.cuts = [i for i in range(1, len(tasks))
                         if os.path.dirname(tasks[i][0]) != os.path.dirname(tasks[i - 1][0])]
        self._range_changed()
        self._show_plan()
        self._redraw()
        self._ref_check()
        self.setWindowTitle(f"cineAlignQt {VERSION} -- {os.path.basename(path)}  "
                            f"({len(tasks)} frames, "
                            f"{'scene folders' if mode == 'mirror' else 'flat'})")

    def _frame_range(self):
        if self.rng is None:
            return None
        a, b = self.rng
        return (getIndex(self.tasks[a][0]), getIndex(self.tasks[b][0]), b - a + 1)

    def _show_plan(self):
        empty, stale = self._extra
        lines = ca.summaryLines(self.tasks, self.scenes, empty, stale, self.mode,
                                self.cb_align.isChecked(), roi=self.roi,
                                frameRange=self._frame_range(),
                                horizontal=self._horiz_on(), edgeRef=self.href)
        self.log.setPlainText("\n".join(lines))

    def _frame(self, i):
        img = self._cache.get(i)
        if img is not None:
            self._cache.move_to_end(i)
            return img
        job = self._pf_jobs.pop(i, None)
        try:
            img = job.result() if job is not None else readImage(self.tasks[i][0])
        except Exception as e:
            self.status.showMessage(f"cannot read {self.tasks[i][0]}: {e}")
            return None
        self._cache[i] = img
        while len(self._cache) > CACHE_N:
            self._cache.popitem(last=False)
        return img

    def _table(self, dtype):
        key = (np.dtype(dtype).str, self.sl_shadow.value())
        tab = self._tabs.get(key)
        if tab is None:
            if len(self._tabs) > 8:
                self._tabs.clear()
            tab = self._tabs[key] = display_table(dtype, self.sl_shadow.value() / 10.0)
        return tab

    def _result_view(self):
        return self.view_group.checkedId() == 1

    def _redraw(self, *_):
        if not self.tasks:
            self.canvas.set_image(None, None)
            return
        img = self._frame(self.idx)
        path, _out, lut = self.tasks[self.idx]
        if img is None:
            self.canvas.message = f"cannot read {os.path.basename(path)}"
            self.canvas.set_image(None, None)
            return
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        H, W = img.shape[:2]
        align = self.cb_align.isChecked()
        search = align or self._horiz_on()
        det, shift = {}, (0, 0)
        hz = self._href_now()
        if search:
            sx, sy = sprocketDetectFast(img, roi=self.roi, horizontal=hz is not None, hRef=hz, info=det)
            shift = (sx, sy if align else 0)
            self.plot.data[self.idx] = (int(sy) if align else None, det.get("route", "none"),
                                        int(shift[0]) if hz is not None else None,
                                        det.get("hroute", "none") if hz is not None else None)
        result = self._result_view()
        shown = img
        if result:
            if lut is not None:
                try:
                    shown = lut.apply(img)
                except Exception as e:
                    self.status.showMessage(f"LUT: {e}")
            if search:
                shown = shiftImg(shown, shift[0], shift[1])
        s = self.canvas.scale() if self.canvas._size == (W, H) else min(
            self.canvas.width() / W, self.canvas.height() / H)
        step = 4 if s <= 0.25 else 2 if s <= 0.5 else 1
        disp = self._table(shown.dtype)[shown[::step, ::step, :3]]
        self.canvas.roi = self.roi
        self.canvas.show_roi = search and not result
        self.canvas.roi_edit = not self._running()
        self.canvas.det = det if search else None
        self.canvas.href = hz
        self.canvas.show_center = align
        self._disp_step = step
        self.canvas.set_image(disp, (W, H))
        self._det_text(det, shift, H, align, hz, search)
        n = len(self.tasks)
        scene = os.path.basename(os.path.dirname(path)) if self.mode == "mirror" else ""
        self.lbl_frame.setText(f"frame {getIndex(path)}  ·  {self.idx + 1} / {n}"
                               + (f"  ·  {scene}" if scene else ""))
        self.bar.idx = self.plot.idx = self.idx
        self.bar.update()
        self.plot.update()
        self.zoom.update()
        if not self._running():
            self.status.showMessage(os.path.basename(path) + ("" if lut is None else
                                    f"  ·  LUT {os.path.basename(lut.path)}"))

    def _det_text(self, d, shift, H, align, hz, search):
        if not search:
            self.lbl_det.setText("Alignment is off -- no search.")
            self.lbl_det.setStyleSheet("color: #8a9099")
            return
        if not d.get("found"):
            self.lbl_det.setText("Hole not found in this frame -- it would stay unshifted."
                                 + (f" (bright run of {d['size']} px, expected about {d['nominal']})"
                                    if d.get("size") is not None else ""))
            self.lbl_det.setStyleSheet("color: #eb4646")
            return
        torn = d["route"] == "lower edge"
        self.lbl_det.setText(
            f"Hole rows {d['lo']}–{d['hi']} ({d['size']} px, nominal {d['nominal']})"
            f"{' -- too large, centre from the lower edge' if torn else ''}\n"
            + (f"centre {d['center']}, image centre {H // 2}: shift {shift[1]:+d} px" if align
               else f"centre {d['center']} -- no vertical shift (switched off)")
            + self._h_text(d, shift, hz))
        self.lbl_det.setStyleSheet("color: #ff9628" if torn else "")

    def _h_text(self, d, shift, hz):
        if not self._horiz_on():
            return ""
        if hz is None:
            return ("\nhorizontal: measuring the normal edge position \u2026" if self._ref is not None
                    else "\nhorizontal: normal edge position not known -- no horizontal shift")
        if d.get("edge") is None:
            return f"\nright edge not found (normal x {hz:.1f}): no horizontal shift"
        if d.get("hroute") == "discarded":
            return (f"\nright edge x {d['edge']:.1f}, {hz - d['edge']:+.1f} px from normal {hz:.1f}: "
                    f"discarded (limit {d['hlimit']} px)")
        return f"\nright edge x {d['edge']:.1f}, normal {hz:.1f}: shift {shift[0]:+d} px"

    def _view_changed(self):
        if not self.tasks:
            return
        s = self.canvas.scale()
        step = 4 if s <= 0.25 else 2 if s <= 0.5 else 1
        if step != getattr(self, "_disp_step", None):
            self._redraw()

    def _shadow_changed(self, v):
        self.lbl_shadow.setText(f"Shadows +{v / 10:.1f} EV")
        self.lbl_shadow.setStyleSheet("color: #ff8c28" if v else "")
        self.settings["shadows"] = v
        self._redraw()

    def _roi_dragged(self, roi):
        self._set_roi(roi, save=False)

    def _roi_typed(self, *_):
        l, r, t, b = (sp.value() for sp in self.sp_roi)
        r = max(r, l + ROI_MIN_W)
        b = max(b, t + ROI_MIN_H)
        if r > 1.0 or b > 1.0:
            return
        self._set_roi((l, r, t, b), save=True)

    def _set_roi(self, roi, save, redraw=True):
        new = tuple(round(float(x), 4) for x in roi)
        if new != self.roi:
            self.plot.data = {}
        self.roi = new
        for sp, x in zip(self.sp_roi, self.roi):
            sp.blockSignals(True)
            sp.setValue(x)
            sp.blockSignals(False)
        if save and self.input:
            if self._ref_roi != self.roi:
                self.href, self._ref_roi = None, None
            self._save_align()
            self._show_plan()
        if redraw:
            self._redraw()
        if save and self.input:
            self._ref_check()

    def _horiz_on(self):
        return self.cb_horiz.isChecked()

    def _href_now(self):
        return self.href if self._horiz_on() else None

    def _save_align(self):
        try:
            ca.saveRoi(self.input, self.roi, self.cb_horiz.isChecked(), self.href,
                       vertical=self.cb_align.isChecked())
        except (OSError, ValueError) as e:
            self.status.showMessage(f"search area not saved: {e}")

    def _horiz_toggled(self, on):
        self.settings["horizontal"] = bool(on)
        self.plot.data = {}
        self._roi_widgets()
        if self.input:
            self._save_align()
        if self.tasks:
            self._show_plan()
            self._redraw()
            self._ref_check()

    def _ref_check(self):
        if not self.tasks or not self._horiz_on() or self._running():
            return
        if self._ref is not None:
            if self._ref.roi != self.roi:
                self._ref.stop = True
                self._ref_again = True
            return
        if self.href is not None and self._ref_roi == self.roi:
            return
        self._ref = RefWorker([t[0] for t in self.tasks], self.roi)
        self._ref.done.connect(self._ref_done)
        self.btn_run.setEnabled(False)
        self.status.showMessage("measuring the normal position of the hole edge \u2026")
        self._ref.start()
        self._redraw()

    def _ref_done(self, res):
        worker, self._ref = self._ref, None
        worker.wait()
        self.btn_run.setEnabled(True)
        if self._ref_again or worker.roi != self.roi:
            self._ref_again = False
            self._ref_check()
            return
        ref, found, tried = res
        self.href, self._ref_roi = ref, (self.roi if ref is not None else None)
        self.plot.data = {}
        if self.input:
            self._save_align()
        self._show_plan()
        self._redraw()
        if ref is None:
            self.status.showMessage(f"horizontal: edge found in {found} of {tried} frames only -- "
                                    "normal position not known, no horizontal alignment")
        else:
            self.status.showMessage(f"horizontal: normal position of the hole edge x = {ref:.1f} px "
                                    f"(median of {found} frames from the middle of the film)")

    def _ref_cancel(self):
        if self._ref is not None:
            self._ref.stop = True
            self._ref.done.disconnect()
            self._ref.wait()
            self._ref = None
            self._ref_again = False
            self.btn_run.setEnabled(True)

    def _roi_widgets(self):
        on = (self.cb_align.isChecked() or self._horiz_on()) and not self._running()
        for wdg in self.sp_roi + [self.btn_roi_def]:
            wdg.setEnabled(on)

    def _align_toggled(self, on):
        self.settings["align"] = bool(on)
        if self.input:
            self._save_align()
        self.plot.data = {}
        self.plot.update()
        self.zoom.update()
        self._roi_widgets()
        if self.tasks:
            self._show_plan()
            self._redraw()
            self._ref_check()

    def _set_range(self, what):
        if not self.tasks or self._running():
            return
        n = len(self.tasks)
        a, b = self.rng if self.rng is not None else (0, n - 1)
        if what == "all":
            self.rng = None
        elif what == "in":
            a = self.idx
            self.rng = (a, max(a, b))
        else:
            b = self.idx
            self.rng = (min(a, b), b)
        if self.rng == (0, n - 1):
            self.rng = None
        self._range_changed()
        self._show_plan()

    def _range_changed(self):
        self.bar.rng = self.plot.rng = self.rng
        self.bar.update()
        self.plot.update()
        if not self.tasks:
            self.lbl_rng.setText("–")
            return
        fr = self._frame_range()
        if fr is None:
            self.lbl_rng.setText(f"Whole film: {len(self.tasks)} frames.")
            self.lbl_rng.setStyleSheet("")
        else:
            self.lbl_rng.setText(f"Frames {fr[0]}–{fr[1]}: {fr[2]} of {len(self.tasks)}. "
                                 "The others are not written.")
            self.lbl_rng.setStyleSheet("color: #7fb2e6")

    def _goto(self, i):
        if not self.tasks:
            return
        i = max(0, min(len(self.tasks) - 1, int(i)))
        if i != self.idx:
            self.idx = i
            self._redraw()

    def _seek(self, i):
        self._play_stop()
        self._goto(i)

    def _scene_step(self, d):
        starts = [0] + self.bar.cuts
        if d > 0:
            nxt = [s for s in starts if s > self.idx]
            self._goto(nxt[0] if nxt else len(self.tasks) - 1)
        else:
            prv = [s for s in starts if s < self.idx]
            self._goto(prv[-1] if prv else 0)

    def _play_cmd(self, cmd):
        if not self.tasks:
            return
        if cmd == "stop":
            self._play_stop()
        elif cmd == "toggle":
            if self._play_dir:
                self._play_stop()
            else:
                self._play_start(1, 1)
        else:
            d = 1 if cmd == "fwd" else -1
            self._play_start(d, min(4, self._play_speed * 2) if self._play_dir == d else 1)

    def _play_show(self):
        for key, b in self.btn_play.items():
            on = (key == "fwd" and self._play_dir > 0) or (key == "rev" and self._play_dir < 0)
            sp = f" {self._play_speed}×" if on and self._play_speed > 1 else ""
            b.setText(self.PLAY_TXT[key] + sp)
            b.setStyleSheet("color: #ffb000; font-weight: bold;" if on else "")

    def _play_start(self, d, speed):
        self._play_dir, self._play_speed = d, speed
        self._play_show()
        self._timer.start(0)

    def _play_stop(self):
        if not self._play_dir:
            return
        self._play_dir, self._play_speed = 0, 1
        self._timer.stop()
        for f in self._pf_jobs.values():
            f.cancel()
        self._pf_jobs.clear()
        self._play_show()

    def _tick(self):
        if not self._play_dir or not self.tasks:
            return
        t0 = time.perf_counter()
        n = len(self.tasks)
        step = self._play_dir * self._play_speed
        nxt = self.idx + step
        if not (0 <= nxt < n):
            self._goto(n - 1 if step > 0 else 0)
            self._play_stop()
            return
        for k in range(1, 4):
            q = nxt + k * step
            if 0 <= q < n and q not in self._cache and q not in self._pf_jobs:
                self._pf_jobs[q] = self._pf.submit(readImage, self.tasks[q][0])
        for q in [q for q in self._pf_jobs if (q - nxt) * self._play_dir < 0]:
            self._pf_jobs.pop(q).cancel()
        self._goto(nxt)
        fps = float(self.settings.get("play_fps", PLAY_FPS))
        wait = 1000.0 / max(1.0, fps) - (time.perf_counter() - t0) * 1000.0
        self._timer.start(max(0, int(wait)))

    def keyPressEvent(self, ev):
        k = ev.key()
        if k == Qt.Key_Escape and self._running():
            self._cancel_run()
            return
        if isinstance(QApplication.focusWidget(), (QDoubleSpinBox, QLineEdit, QPlainTextEdit)):
            super().keyPressEvent(ev)
            return
        if k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Home, Qt.Key_End, Qt.Key_Up, Qt.Key_Down,
                 Qt.Key_PageUp, Qt.Key_PageDown):
            self._play_stop()
        if k in (Qt.Key_Left, Qt.Key_Right):
            mods = ev.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier)
            step = self.NAV_STEP.get(mods, 1)
            self._goto(self.idx + (step if k == Qt.Key_Right else -step))
        elif k == Qt.Key_Home:
            self._goto(0)
        elif k == Qt.Key_End:
            self._goto(len(self.tasks) - 1)
        elif k == Qt.Key_PageUp:
            self._goto(self.idx - 10)
        elif k == Qt.Key_PageDown:
            self._goto(self.idx + 10)
        elif k in (Qt.Key_Up, Qt.Key_Down):
            if self.tasks:
                self._scene_step(1 if k == Qt.Key_Down else -1)
        elif k == Qt.Key_Space:
            self._play_cmd("toggle")
        elif k == Qt.Key_L:
            self._play_cmd("fwd")
        elif k == Qt.Key_J:
            self._play_cmd("rev")
        elif k == Qt.Key_K:
            self._play_cmd("stop")
        elif k == Qt.Key_I:
            self._set_range("in")
        elif k == Qt.Key_O:
            self._set_range("out")
        elif k == Qt.Key_V:
            self.view_group.button(0 if self._result_view() else 1).setChecked(True)
            self._redraw()
        elif k in (Qt.Key_1, Qt.Key_2):
            self.view_group.button(0 if k == Qt.Key_1 else 1).setChecked(True)
            self._redraw()
        else:
            super().keyPressEvent(ev)

    def _running(self):
        return self._worker is not None

    def _run_or_cancel(self):
        if self._running():
            self._cancel_run()
        else:
            self._start_run()

    def _cancel_run(self):
        if self._worker is not None:
            self._worker.cancel()
            self.btn_run.setEnabled(False)
            self.btn_run.setText("Cancelling …")

    def _confirm_output(self, out, todo):
        have = {}
        exist = 0
        for t in todo:
            d = os.path.dirname(t[1])
            if d not in have:
                try:
                    with os.scandir(d) as it:
                        have[d] = {e.name.lower() for e in it}
                except OSError:
                    have[d] = set()
            exist += os.path.basename(t[1]).lower() in have[d]
        box = QMessageBox(self)
        box.setWindowTitle("cineAlignQt")
        box.setIcon(QMessageBox.Question)
        box.setText(f"Write {len(todo)} frame(s) to\n\n{out}\n")
        box.setInformativeText(
            (f"{exist} of them exist there already and will be overwritten."
             if exist else "None of them exists there yet.")
            + ("" if os.path.isdir(out) else "\nThe folder does not exist yet; it will be created."))
        b_start = box.addButton("Start", QMessageBox.AcceptRole)
        b_change = box.addButton("Other folder \u2026", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(b_start)
        box.exec_()
        hit = box.clickedButton()
        return "start" if hit is b_start else "change" if hit is b_change else "cancel"

    def _start_run(self):
        if not self.tasks or not self.input or self._scan is not None or self._ref is not None:
            return
        self._play_stop()
        align = self.cb_align.isChecked()
        horiz = self._horiz_on()
        hz = self._href_now()
        self._save_align()
        a, b = self.rng if self.rng is not None else (0, len(self.tasks) - 1)
        wanted = set(os.path.normcase(os.path.abspath(t[0])) for t in self.tasks[a:b + 1])
        while True:
            out = self._out_dir()
            self.status.showMessage("reading the input folder \u2026")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                tasks, scenes, empty, stale, mode = ca.collectTasks(self.input, out, makeDirs=False)
            except (RuntimeError, ValueError, OSError) as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self, "cineAlignQt", str(e))
                return
            QApplication.restoreOverrideCursor()
            todo = [t + [align, self.roi, hz] for t in tasks
                    if os.path.normcase(os.path.abspath(t[0])) in wanted]
            if not todo:
                QMessageBox.information(self, "cineAlignQt", "No frame of the range is left in the "
                                                             "input folder.")
                return
            answer = self._confirm_output(out, todo)
            if answer == "start":
                break
            if answer == "cancel":
                self.status.showMessage("not started")
                return
            p = QFileDialog.getExistingDirectory(self, "Output folder", out)
            if not p:
                self.status.showMessage("not started")
                return
            self.ed_out.setText(os.path.normpath(p))
            self.settings["output"] = self.ed_out.text().strip()
        try:
            for d in sorted({os.path.dirname(t[1]) for t in todo}):
                os.makedirs(d, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, "cineAlignQt", f"Cannot create the output folder:\n{e}")
            return
        fr = self._frame_range()
        lines = ca.summaryLines(tasks, scenes, empty, stale, mode, align, roi=self.roi, frameRange=fr,
                                horizontal=horiz, edgeRef=hz)
        self.log.setPlainText("\n".join(lines))
        print("***** cineAlignQt %s" % VERSION)
        for line in lines:
            print(line)
        print()
        self._run_info = dict(out=out, mode=mode, align=align, scenes=scenes, stale=stale,
                              roi=self.roi, fr=fr, total=len(todo), href=hz)
        self._hstat = {}
        self.progress.setRange(0, len(todo))
        self.progress.setValue(0)
        self.progress.show()
        self.btn_run.setText("Cancel  (Esc)")
        for wdg in self._edit_widgets:
            wdg.setEnabled(False)
        ca.SHIFT_LOG = {} if (align or hz is not None) else None
        self._worker = RunWorker(todo)
        self._worker.progress.connect(self._run_progress)
        self._worker.finished_run.connect(self._run_done)
        self.status.showMessage(ca.progressLine(0, len(todo), 0))
        self.canvas.roi_edit = False
        self._worker.start()

    def _take_shifts(self):
        log = ca.SHIFT_LOG
        if not log:
            return
        while log:
            try:
                path, v = log.popitem()
            except KeyError:
                break
            i = self._path_idx.get(os.path.normcase(os.path.abspath(path)))
            if i is not None:
                self.plot.data[i] = v
            if v[3] is not None:
                self._hstat[v[3]] = self._hstat.get(v[3], 0) + 1
                if v[2]:
                    self._hstat["moved"] = self._hstat.get("moved", 0) + 1
        self.plot.update()
        self.zoom.update()

    def _run_progress(self, done, total, seconds):
        self.progress.setValue(done)
        self._take_shifts()
        if seconds > 0:
            line = ca.progressLine(done, total, seconds)
            self.status.showMessage(line)
            print(line + "   \r", end="")

    def _run_done(self, done, errors, seconds, cancelled):
        info = self._run_info
        self._worker.wait()
        self._worker = None
        self._take_shifts()
        ca.SHIFT_LOG = None
        print()
        print()
        head = ("***** cancelled: %d of %d written, %d failed -- no run log"
                % (done - len(errors), info["total"], len(errors)) if cancelled else
                "***** done: %d processed, %d failed" % (done - len(errors), len(errors)))
        print(head)
        self.log.appendPlainText("")
        self.log.appendPlainText(head)
        if info["href"] is not None:
            hs = self._hstat
            line = (".... horizontal: %d frame(s) shifted sideways; edge not found in %d, "
                    "value discarded in %d" % (hs.get("moved", 0), hs.get("none", 0),
                                               hs.get("discarded", 0)))
            print(line)
            self.log.appendPlainText(line)
        if not cancelled:
            ca.writeRunLog(info["out"], self.input, info["mode"], info["align"], info["scenes"],
                           done, errors, info["stale"], seconds, roi=info["roi"],
                           frameRange=info["fr"], edgeRef=info["href"])
            self.log.appendPlainText(".... run log   : %s" % os.path.join(info["out"], ca.RUN_LOG))
        if errors:
            self.log.appendPlainText("***** failed files:")
            print("***** failed files:")
            for p, m in errors:
                self.log.appendPlainText("      %s  ->  %s" % (p, m))
                print("      %s  ->  %s" % (p, m))
        self.progress.hide()
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Start")
        for wdg in self._edit_widgets:
            wdg.setEnabled(True)
        self._roi_widgets()
        self._redraw()
        self.status.showMessage(head.strip("* "))

    def _load_settings(self):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def _save_settings(self):
        s = self.settings
        s["output"] = self.ed_out.text().strip()
        s["align"] = self.cb_align.isChecked()
        s["horizontal"] = self.cb_horiz.isChecked()
        s["shadows"] = self.sl_shadow.value()
        s.setdefault("play_fps", PLAY_FPS)
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as fh:
                json.dump(s, fh, indent=2)
        except Exception:
            pass

    def _apply_settings(self):
        s = self.settings
        self.ed_out.setText(s.get("output", ca.saveDir))
        self.cb_align.blockSignals(True)
        self.cb_align.setChecked(bool(s.get("align", ca.alignSprocket)))
        self.cb_align.blockSignals(False)
        self.cb_horiz.blockSignals(True)
        self.cb_horiz.setChecked(bool(s.get("horizontal", False)))
        self.cb_horiz.blockSignals(False)
        self.sl_shadow.setValue(int(s.get("shadows", 0)))
        self._set_roi(self.roi, save=False, redraw=False)
        self._align_toggled(self.cb_align.isChecked())
        p = s.get("input")
        if p and os.path.isdir(p):
            self.open_folder(p)

    def closeEvent(self, ev):
        if self._running():
            self._worker.cancel()
            self._worker.wait()
        if self._scan is not None:
            self._scan.done.disconnect()
            self._scan.wait()
        self._ref_cancel()
        self._play_stop()
        self._pf.shutdown(wait=False)
        self._save_settings()
        super().closeEvent(ev)

def _apply_dark(app):
    app.setStyle("Fusion")
    p = QPalette()
    bg, base, alt = QColor(30, 30, 32), QColor(22, 22, 24), QColor(38, 38, 41)
    text, dim, accent = QColor(210, 210, 214), QColor(130, 130, 136), QColor(78, 132, 190)
    p.setColor(QPalette.Window, bg)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, alt)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, QColor(33, 34, 38))
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.ToolTipBase, base)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Highlight, accent)
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        p.setColor(QPalette.Disabled, role, dim)
    app.setPalette(p)
    app.setStyleSheet("""
        QDialog, QFileDialog { background: #1e1e20; }
        QGroupBox { border: 1px solid #3a3a3e; border-radius: 4px; margin-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #8a9099; }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 14px; height: 14px; border: 1px solid #8a9099; background: #161618; }
        QCheckBox::indicator { border-radius: 3px; }
        QRadioButton::indicator { border-radius: 8px; }
        QCheckBox::indicator:checked, QRadioButton::indicator:checked {
            background: #4e84be; border-color: #8fb8d8; }
        QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #d2d2d6; }
        QPlainTextEdit { background: #161618; border: 1px solid #3a3a3e; }
    """)

def _scale_font(app):
    scr = app.primaryScreen()
    if scr is None:
        return
    h = scr.geometry().height()
    f = app.font()
    f.setPointSize(12 if h >= 2000 else (10 if h >= 1400 else 9))
    app.setFont(f)

def _place(w, app):
    scr = app.screenAt(QCursor.pos()) or app.primaryScreen()
    if scr is None:
        w.resize(1500, 950)
        return
    ag = scr.availableGeometry()
    w.resize(min(1500, ag.width() - 20), min(950, ag.height() - 50))
    w.move(ag.x() + max(0, (ag.width() - w.width()) // 2),
           ag.y() + max(0, (ag.height() - w.height() - 40) // 2))

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    _scale_font(app)
    _apply_dark(app)
    w = Main()
    _place(w, app)
    w.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main())
