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

import datetime
import json
import os
import re
import time

from multiprocessing.pool import ThreadPool

time_msec = lambda: int(round(time.time() * 1000))

from sprocketDetectFast import sprocketDetectFast
from shiftImg           import shiftImg
from listDirectory      import listDirectory
from getIndex           import getIndex
from readImage          import readImage
from write16            import writeTIF16
from cubeLut            import CubeLut

VERSION          = '2.6'

defaultDir       = '.'
saveDir          = '02_SPR'

extensionList    = ["tif", "tiff", "png", "jpg", "jpeg"]

outputPrefix     = 'SPR'

alignSprocket    = True

LUT_NAME         = 'csample.cube'
LUT_RANGE        = re.compile(r'^csample_(\d+)-(\d+)(?:_.*)?\.cube$', re.IGNORECASE)
LUT_SUBDIR       = '_LUT'

def lutDirs(*dirs):
    out = []
    for d in dirs:
        out += [os.path.join(d, LUT_SUBDIR), d]
    return out

RUN_LOG          = 'cineAlign_run.json'

ROI_DEFAULT      = (0.01, 0.06, 0.1, 0.9)
WORK_SUBDIR      = '_cineAlign'
ROI_FILE         = 'cineAlign.json'

def roiPath(inputDir):
    return os.path.join(inputDir, WORK_SUBDIR, ROI_FILE)

def checkRoi(roi):
    r = tuple(float(v) for v in roi)
    if len(r) != 4 or not (0.0 <= r[0] < r[1] <= 1.0 and 0.0 <= r[2] < r[3] <= 1.0):
        raise ValueError('search area %s is not (left, right, top, bottom) within 0..1' % (roi,))
    return r

def loadRoi(inputDir):
    try:
        with open(roiPath(inputDir), encoding='utf-8') as fh:
            return checkRoi(json.load(fh)['roi'])
    except (OSError, ValueError, KeyError, TypeError):
        return None

def loadAlign(inputDir):
    out = {'roi': None, 'vertical': None, 'horizontal': False, 'edge_ref': None}
    try:
        with open(roiPath(inputDir), encoding='utf-8') as fh:
            doc = json.load(fh)
        out['roi'] = checkRoi(doc['roi'])
        if doc.get('vertical') is not None:
            out['vertical'] = bool(doc['vertical'])
        out['horizontal'] = bool(doc.get('horizontal', False))
        ref = doc.get('edge_ref')
        if ref is not None and doc.get('edge_method') == EDGE_METHOD:
            out['edge_ref'] = float(ref)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return out

def saveRoi(inputDir, roi, horizontal=False, edgeRef=None, vertical=True):
    path = roiPath(inputDir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'vertical': bool(vertical),
                   'horizontal': bool(horizontal),
                   'roi': [round(float(v), 4) for v in checkRoi(roi)],
                   'edge_ref': round(float(edgeRef), 2) if edgeRef is not None else None,
                   'edge_method': EDGE_METHOD}, fh, indent=1)
        fh.write('\n')

EDGE_SAMPLES = 50
EDGE_METHOD  = 2

def edgeSamplePaths(imgPaths, samples=EDGE_SAMPLES):
    n = len(imgPaths)
    a, b = n // 3, max(n // 3 + 1, n - n // 3)
    mid = imgPaths[a:b]
    if len(mid) <= samples:
        return list(mid)
    return [mid[int(round(k * (len(mid) - 1) / (samples - 1)))] for k in range(samples)]

def measureEdgeRef(imgPaths, roi=None, samples=EDGE_SAMPLES, cancelled=None):
    roi = roi if roi is not None else ROI_DEFAULT
    paths = edgeSamplePaths(imgPaths, samples)

    def one(p):
        if cancelled is not None and cancelled():
            return None
        try:
            info = {}
            sprocketDetectFast(readImage(p), roi=roi, horizontal=True, info=info)
            return info.get('edge')
        except Exception:
            return None

    pool = ThreadPool()
    try:
        edges = pool.map(one, paths)
    finally:
        pool.close()
        pool.join()
    if cancelled is not None and cancelled():
        return (None, 0, 0)
    edges = sorted(e for e in edges if e is not None)
    if not edges or 2 * len(edges) < len(paths):
        return (None, len(edges), len(paths))
    return (float(edges[len(edges) // 2]), len(edges), len(paths))

SHIFT_LOG = None

PREVIEW = None

def processImage(para):
    imgPath, outputName, lut, align = para[:4]
    roi = para[4] if len(para) > 4 and para[4] is not None else ROI_DEFAULT
    hRef = para[5] if len(para) > 5 else None
    horiz = hRef is not None
    search = align or horiz

    try:
        inputImg = readImage(imgPath)

        if search:
            log = SHIFT_LOG
            info = {} if log is not None else None
            shiftX, shiftY = sprocketDetectFast(inputImg, roi=roi, horizontal=horiz, hRef=hRef,
                                                info=info)
            if not align:
                shiftY = 0
            if log is not None:
                log[imgPath] = (int(shiftY) if align else None, info.get('route', 'none'),
                                int(shiftX) if horiz else None,
                                info.get('hroute', 'none') if horiz else None)

        if lut is not None:
            inputImg = lut.apply(inputImg)

        outputImage = shiftImg(inputImg, shiftX, shiftY) if search else inputImg

        writeTIF16(outputName, outputImage)

        preview = PREVIEW
        if preview is not None:
            preview['last'] = (imgPath, outputImage)

    except Exception as e:
        return (False, imgPath, '%s: %s' % (type(e).__name__, e))

    return (True, imgPath, '')

def makeOutputName(outputDir, imgPath):
    return os.path.join(outputDir, '%s_%08d.tif' % (outputPrefix, getIndex(imgPath)))

def checkIndexCollisions(pairs):
    seen      = {}
    collided  = []
    for imgPath, outName in pairs:
        key = os.path.normcase(os.path.abspath(outName))
        if key in seen:
            collided.append((seen[key], imgPath, outName))
        else:
            seen[key] = imgPath
    return collided

def lowestIndex(imgNames):
    best = None
    for f in imgNames:
        try:
            idx = getIndex(f)
        except Exception:
            continue
        if best is None or idx < best:
            best = idx
    return best if best is not None else 0

def findLuts(dirs, cache):
    def load(p):
        p = os.path.abspath(p)
        if p not in cache:
            cache[p] = CubeLut(p)
        return cache[p]

    for d in dirs:
        if not os.path.isdir(d):
            continue
        base, ranges = None, []
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                continue
            if f.lower() == LUT_NAME:
                base = load(p)
                continue
            m = LUT_RANGE.match(f)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a > b:
                    raise ValueError('LUT %s: the range starts behind its end' % p)
                ranges.append([a, b, load(p)])
        if base is None and not ranges:
            continue
        ranges.sort(key=lambda r: r[0])
        for r0, r1 in zip(ranges, ranges[1:]):
            if r1[0] <= r0[1]:
                raise ValueError(
                    'range LUTs overlap:\n'
                    '          %s\n          %s' % (r0[2].path, r1[2].path))
        return base, ranges
    return None, []

def lutFor(idx, base, ranges):
    for a, b, lut in ranges:
        if a <= idx <= b:
            return lut
    return base

def sceneEntry(name, imgs, base, ranges):
    per = []
    counts = {}
    for imgPath in imgs:
        lut = lutFor(getIndex(imgPath), base, ranges)
        per.append(lut)
        counts[id(lut)] = counts.get(id(lut), 0) + 1
    scene = {'name': name, 'frames': len(imgs),
             'base': base, 'baseFrames': counts.get(id(base), 0),
             'ranges': [(a, b, lut, counts.get(id(lut), 0)) for a, b, lut in ranges]}
    return scene, per

def staleOutputs(targetDir, planned):
    if not os.path.isdir(targetDir):
        return []
    keep = set(os.path.normcase(os.path.abspath(p)) for p in planned)
    out = []
    for f in listDirectory(targetDir, ['tif']):
        if os.path.basename(f).startswith(outputPrefix + '_') \
                and os.path.normcase(os.path.abspath(f)) not in keep:
            out.append(f)
    return out

IGNORED_DIRS = []

def collectTasks(inputDir, outputDir, makeDirs=True, progress=None):
    say = progress or (lambda text: None)
    say('reading %s' % inputDir)
    topImages = listDirectory(inputDir, extensionList)

    outAbs  = os.path.abspath(outputDir)
    subDirs = []
    for name in sorted(os.listdir(inputDir)):
        full = os.path.join(inputDir, name)
        if not os.path.isdir(full):
            continue
        if os.path.abspath(full) == outAbs:
            continue
        if name.startswith('_'):
            continue
        subDirs.append(full)

    del IGNORED_DIRS[:]
    if topImages and subDirs:
        IGNORED_DIRS.extend(subDirs)
        subDirs = []

    lutCache   = {}
    pairs      = []
    luts       = []
    scenes     = []
    emptyDirs  = []
    dirsToMake = []
    planned    = {}

    if subDirs:
        mode = 'mirror'
        found = []
        for d in subDirs:
            say('reading %s' % d)
            imgs = listDirectory(d, extensionList)
            imgs.sort()
            found.append((lowestIndex(imgs), d, imgs))
        found.sort(key=lambda s: s[0])

        for _, d, imgs in found:
            targetDir = os.path.join(outputDir, os.path.basename(d))
            dirsToMake.append(targetDir)

            if not imgs:
                emptyDirs.append(d)
                continue

            base, ranges = findLuts(lutDirs(d, inputDir), lutCache)
            scene, per = sceneEntry(os.path.basename(d), imgs, base, ranges)
            scenes.append(scene)
            for imgPath, lut in zip(imgs, per):
                out = makeOutputName(targetDir, imgPath)
                pairs.append((imgPath, out))
                luts.append(lut)
                planned.setdefault(targetDir, []).append(out)
    else:
        mode = 'flat'
        topImages.sort()
        dirsToMake.append(outputDir)
        base, ranges = findLuts(lutDirs(inputDir), lutCache)
        scene, per = sceneEntry('', topImages, base, ranges)
        scenes.append(scene)

        for imgPath, lut in zip(topImages, per):
            out = makeOutputName(outputDir, imgPath)
            pairs.append((imgPath, out))
            luts.append(lut)
            planned.setdefault(outputDir, []).append(out)

    say('checking %d frames' % len(pairs))
    collided = checkIndexCollisions(pairs)
    if collided:
        first = collided[0]
        raise RuntimeError(
            'output names are not unique - getIndex() returns the same\n'
            '        number for different input files.\n'
            '        %d collisions, the first:\n'
            '          %s\n'
            '          %s\n'
            '        -> both would be written to %s.'
            % (len(collided), first[0], first[1], os.path.basename(first[2])))

    stale = []
    for d in dirsToMake:
        stale += staleOutputs(d, planned.get(d, []))

    if makeDirs:
        for d in dirsToMake:
            os.makedirs(d, exist_ok=True)

    tasks = [[imgPath, outName, lut] for (imgPath, outName), lut in zip(pairs, luts)]
    return tasks, scenes, emptyDirs, stale, mode

def summaryLines(tasks, scenes, emptyDirs, stale, mode, align, roi=None, frameRange=None,
                 horizontal=False, edgeRef=None):
    hz = horizontal and edgeRef is not None
    what = ('align to sprocket hole, vertically and horizontally' if align and hz
            else 'align to sprocket hole, vertically' if align
            else 'align to sprocket hole, horizontally only' if hz
            else 'no alignment -- colour correction only')
    out = ['.... mode      : %s' % ('scene folders' if mode == 'mirror' else 'flat directory'),
           '.... sprocket  : %s' % what]
    if align or horizontal:
        r = roi if roi is not None else ROI_DEFAULT
        out.append('.... search    : left %.3f right %.3f top %.3f bottom %.3f%s'
                   % (r[0], r[1], r[2], r[3], '' if roi is not None else '  (default)'))
        if horizontal and edgeRef is not None:
            out.append('.... horizontal: right hole edge to its normal position x = %.1f px;' % edgeRef)
            out.append('                 not found, or further off than the width of the search area:')
            out.append('                 no horizontal shift')
        elif horizontal:
            out.append('.... horizontal: OFF -- the normal position of the hole edge is not known')
    if frameRange is None:
        out.append('.... to process: %d images (all rewritten)' % len(tasks))
    else:
        out.append('.... to process: %d of %d images, frames %d-%d (rewritten; the others stay)'
                   % (frameRange[2], len(tasks), frameRange[0], frameRange[1]))
    for sc in scenes:
        where = ('%-24s' % sc['name']) if sc['name'] else ''
        base = sc['base']
        nolut = 'none -- alignment only' if (align or hz) else 'none -- copy only'
        if base is None and not sc['ranges']:
            out.append('.... LUT       : %s%s' % (where, nolut))
            continue
        note = (' (%d frames)' % sc['baseFrames']) if sc['ranges'] else ''
        out.append('.... LUT       : %s%s%s' % (where, base.describe() if base is not None
                                                else nolut, note))
        for a, b, lut, n in sc['ranges']:
            out.append('.... LUT %s: %s%s (%d frames)'
                       % ('%05d-%05d' % (a, b), where, lut.describe(), n)
                       + ('  -- NOTE: no frame in this range' if n == 0 else ''))
    for d in emptyDirs:
        out.append('.... note      : no images in %s (folder mirrored anyway)' % os.path.basename(d))
    if IGNORED_DIRS:
        names = ', '.join(os.path.basename(d) for d in IGNORED_DIRS[:4])
        out.append('.... note      : images at the top level -- %d subfolder(s) ignored (%s%s)'
                   % (len(IGNORED_DIRS), names, ', ...' if len(IGNORED_DIRS) > 4 else ''))
    if stale:
        out.append('.... note      : %d older output file(s) belong to no input and stay as they are,'
                   % len(stale))
        out.append('                 e.g. %s' % stale[0])
    return out

def progressLine(done, total, seconds):
    fps = done / seconds if seconds > 0 else 0.0
    tLeft = (total - done) / fps if fps > 0 else 0
    tTot = total / fps if fps > 0 else 0
    fTime = time.strftime('%H:%M:%S', time.gmtime(tLeft))
    fTot = time.strftime('%H:%M:%S', time.gmtime(tTot))
    return (f'Images left: {total-done:>6} ({total:>6}) | '
            f'{fps:5.1f} fps | Time left: {fTime} ({fTot})')

def writeRunLog(outputDir, inputDir, mode, align, scenes, done, errors, stale, seconds,
                roi=None, frameRange=None, edgeRef=None):
    hz = edgeRef is not None
    doc = {
        'cineAlign': VERSION,
        'time': datetime.datetime.now().isoformat(timespec='seconds'),
        'input': os.path.abspath(inputDir),
        'output': os.path.abspath(outputDir),
        'mode': 'scene folders' if mode == 'mirror' else 'flat directory',
        'sprocket_align': bool(align or hz),
        'vertical_align': bool(align),
        'search_area': ([round(float(v), 4) for v in (roi if roi is not None else ROI_DEFAULT)]
                        if (align or hz) else None),
        'horizontal_align': {'edge_ref_px': round(float(edgeRef), 2)} if hz else None,
        'frame_range': ({'from': frameRange[0], 'to': frameRange[1], 'frames': frameRange[2]}
                        if frameRange is not None else None),
        'processed': done - len(errors),
        'failed': len(errors),
        'seconds': round(seconds, 1),
        'stale_outputs_left': len(stale),
        'scenes': [{'scene': sc['name'] or '(flat)', 'frames': sc['frames'],
                    'lut': sc['base'].record() if sc['base'] is not None else None,
                    'lut_frames': sc['baseFrames'],
                    'range_luts': [{'from': a, 'to': b, 'frames': n, 'lut': lut.record()}
                                   for a, b, lut, n in sc['ranges']]}
                   for sc in scenes],
    }
    if errors:
        doc['errors'] = [{'file': p, 'error': m} for p, m in errors]
    path = os.path.join(outputDir, RUN_LOG)
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
            fh.write('\n')
        print('.... run log   : %s' % path)
    except OSError as e:
        print('.... WARNING   : run log not written (%s)' % e)

if __name__ == '__main__':

    print('***** cineAlign %s' % VERSION)
    print()

    inputDir = input("Set input directory ['%s'] > " % defaultDir)
    inputDir = inputDir or defaultDir

    if not os.path.isdir(inputDir):
        print("***** input directory does not exist. Exiting!")
        exit(0)

    outputDir = input("Select output dir ['%s'] > " % saveDir)
    outputDir = outputDir or saveDir

    outputDir = os.path.join(inputDir, outputDir)

    stored = loadAlign(inputDir)
    yn = 'y' if (stored['vertical'] if stored['vertical'] is not None else alignSprocket) else 'n'
    ans = input("Align vertically to sprocket hole? (y/n) ['%s'] > " % yn).strip().lower() or yn
    if ans not in ('y', 'n'):
        print("***** please answer y or n. Exiting!")
        exit(0)
    align = (ans == 'y')

    print()
    print('.... scanning input')

    try:
        tasks, scenes, emptyDirs, stale, mode = collectTasks(inputDir, outputDir)
    except (RuntimeError, ValueError, OSError) as e:
        print()
        print('***** %s' % e)
        exit(0)

    if not tasks:
        print("***** could not find any images. Exiting!")
        exit(0)

    roi = stored['roi']
    horizontal = stored['horizontal']
    edgeRef = stored['edge_ref'] if horizontal else None
    if horizontal and edgeRef is None:
        print('.... measuring the normal position of the hole edge')
        edgeRef, found, tried = measureEdgeRef([t[0] for t in tasks], roi)
        if edgeRef is None:
            print('.... WARNING   : edge found in %d of %d frames only' % (found, tried))
    for line in summaryLines(tasks, scenes, emptyDirs, stale, mode, align, roi=roi,
                             horizontal=horizontal, edgeRef=edgeRef):
        print(line)
    for t in tasks:
        t += [align, roi, edgeRef]

    print()

    pool    = ThreadPool()
    errors  = []
    done    = 0
    total   = len(tasks)
    pStart  = time_msec()

    for ok, imgPath, msg in pool.imap_unordered(processImage, tasks):

        done += 1
        if not ok:
            errors.append((imgPath, msg))

        delta = (time_msec() - pStart) / 1000
        if delta <= 0:
            continue

        print(progressLine(done, total, delta) + '   \r', end="")

    pool.close()
    pool.join()

    print()
    print()
    print('***** done: %d processed, %d failed' % (done - len(errors), len(errors)))

    writeRunLog(outputDir, inputDir, mode, align, scenes, done, errors, stale,
                (time_msec() - pStart) / 1000, roi=roi, edgeRef=edgeRef)

    if errors:
        print()
        print('***** failed files:')
        for imgPath, msg in errors:
            print('      %s  ->  %s' % (imgPath, msg))
