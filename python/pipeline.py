#!/usr/bin/env python
"""Unrotate, binarize, detect and classify."""
from __future__ import division

import cv
import cv2

from unrotate import unrotate
from binarize import binarize
from cc import get_rois
from collect_data import imshow_large, SCREEN_HEIGHT
from extract_features import extract_features

def main():
    import sys

    classifier = cv2.SVM()
    classifier.load(sys.argv[1])

    im = cv2.imread(sys.argv[2])
    im = cv2.GaussianBlur(im, (3,3), 0)
    imshow_large(__file__, im)
    cv2.waitKey()
    
    unrotated = unrotate(im)
    #
    # Check if image is right way up, correct otherwise
    #
    imshow_large(__file__, unrotated)
    key = cv2.waitKey()
    if key & 0xff == ord("r"):
        unrotated = cv2.flip(cv2.flip(unrotated, 0), 1)
        imshow_large(__file__, unrotated)
        cv2.waitKey()

    binarized = binarize(unrotated)
    rois = get_rois(binarized)
    results = {}

    grayscale = cv2.cvtColor(unrotated, cv.CV_BGR2GRAY)
    
    for (x,y,width,height) in rois:
        roi = grayscale[y:y+height,x:x+width]
        vec = extract_features(roi)
        label = classifier.predict(vec)
        results[(x,y,width,height)] = "01234567890X"[int(label)]

    scale = SCREEN_HEIGHT/unrotated.shape[0]
    unrotated = cv2.cvtColor(grayscale, cv.CV_GRAY2BGR)
    scaled = cv2.resize(unrotated, (0,0), fx=scale, fy=scale)
    
    for roi in rois:
        x = int(roi[0]*scale)
        y = int(roi[1]*scale)
        width = int(roi[2]*scale)
        height = int(roi[3]*scale)            
        cv2.rectangle(scaled, (x,y), (x+width, y+height), (0,255,0,0), 1)
        if results[roi] == "X":
            continue
        cv2.putText(scaled, results[roi], (x, y), cv.CV_FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0, 0))

    cv2.imshow(__file__, scaled)
    cv2.waitKey()

if __name__ == "__main__":
    main()
