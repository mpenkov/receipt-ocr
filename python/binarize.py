#!/usr/bin/env python
"""Binarize the unrotated receipt into foreground and background areas."""
from __future__ import division
import cv
import cv2
import numpy as np
import math

def binarize(im, debug=False):
    ycrcb = cv2.cvtColor(im, cv.CV_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    if debug:
        cv2.imshow(__file__, y)
        cv2.waitKey()

    #
    # TODO: how to reliably determine these parameters?
    #
    bsize = 9
    C = 8
    thresh = cv2.adaptiveThreshold(y, 255, cv.CV_ADAPTIVE_THRESH_MEAN_C, cv.CV_THRESH_BINARY_INV, bsize, C)
    return thresh

def main():
    import sys
    im = cv2.imread(sys.argv[1])
    binarized = binarize(im)
    cv2.imshow(__file__, binarized)
    cv2.waitKey()

if __name__ == "__main__":
    main()
