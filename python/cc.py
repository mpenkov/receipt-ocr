#!/usr/bin/env python
"""Divide a binarized image into individual connected components."""

from __future__ import division
import cv
import cv2
import numpy as np
import math
import copy

MIN_HEIGHT = 32
MIN_ASPECT_RATIO = 0.6
MAX_ASPECT_RATIO = 6
TOL = 4

def is_good_roi(roi):
    """Checks the aspect ratio and minimum size of the ROI.
    Returns True if it's OK."""
    x,y,width,height = roi
    return height >= MIN_HEIGHT and MIN_ASPECT_RATIO <= height/width <= MAX_ASPECT_RATIO

def is_overlap(r1, r2):
    """Returns True if the two ROIs overlap."""
    #
    # r1 is always on the left
    #
    if r2[0] < r1[0]:
        r1, r2 = r2, r1
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    if not x1 <= x2 <= x1+w1:
        return False

    return (y1 <= y2 <= y1+h1) or (y2 <= y1 <= y2+h2)

def merge(r1, r2):
    """Merge two ROIs, assuming they overlap."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    xx1 = min(x1, x2)
    yy1 = min(y1, y2)
    xx2 = max(x1+w1, x2+w2)
    yy2 = max(y1+h1, y2+h2)

    ww = xx2 - xx1
    hh = yy2 - yy1
    return (xx1, yy1, ww, hh)

def merge_overlaps(rois):
    """Perform a single iteration of overlapping ROIs.
    Returns the number of overlaps merged."""
    merged = []
    used = [False]*len(rois)
    for i,r1 in enumerate(rois):
        if used[i]:
            continue
        for j,r2 in enumerate(rois):
            if j <= i or used[j]:
                continue
            if is_overlap(r1, r2):
                merged.append(merge(r1, r2))
                used[i] = used[j] = True
    num_merged = len(merged)
    for i,r in enumerate(rois):
        if used[i]:
            continue
        merged.append(r)

    return merged, num_merged

def get_rois(binarized):
    #
    # findContours uses the input image as scratch space
    #
    tmp = copy.deepcopy(binarized)
    #tmp = cv2.dilate(tmp, None)
    contours, _= cv2.findContours(tmp, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
    rois = map(lambda x: cv2.boundingRect(x), contours)
    while True:
        rois, num_merged = merge_overlaps(rois)
        if num_merged == 0:
            break
    rois = filter(is_good_roi, rois)
    return rois

def main():
    import sys
    im = cv2.imread(sys.argv[1], cv.CV_LOAD_IMAGE_GRAYSCALE)
    cv2.imshow(__file__, im)
    cv2.waitKey()

    rois = get_rois(im)
    color_im = cv2.cvtColor(im, cv.CV_GRAY2BGR)
    for (x,y,width,height) in rois:
        cv2.rectangle(color_im, (x,y), (x+width, y+height), (255,0,0,0), 2)
    cv2.imshow(__file__, color_im)
    cv2.waitKey()

if __name__ == "__main__":
    main()
