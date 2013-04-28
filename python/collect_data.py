#!/usr/bin/env python
"""Extract ROIs from data and save them in separate images."""
from __future__ import division
import os
import os.path as P
import cv
import cv2
import copy

from unrotate import unrotate
from binarize import binarize
from cc import get_rois

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

def imshow_large(window_name, im):
    """Show an image such that it fits on the entire screen."""
    if im.shape[0] < SCREEN_HEIGHT and im.shape[1] < SCREEN_WIDTH:
        cv2.imshow(window_name, im)
        return 

    if im.shape[0] > im.shape[1]:
        scale = SCREEN_HEIGHT/im.shape[0]
    else:
        scale = SCREEN_WIDTH/im.shape[1]

    scaled = cv2.resize(im, (0,0), fx=scale, fy=scale) 
    cv2.imshow(window_name, scaled)
    return

def main():
    import sys
    in_dir = sys.argv[1]
    images = map(lambda x: P.join(in_dir, x), os.listdir(in_dir))
    out_dir = sys.argv[2]
    counters = {}
    for subdir in "0123456789X":
        path = P.join(out_dir, subdir)
        if not P.isdir(path):
            os.makedirs(path)
        files = os.listdir(path)
        indices = map(lambda x: int(P.splitext(x)[0]), files)
        try:
            counters[subdir] = max(indices) + 1
        except ValueError:
            counters[subdir] = 0

    for fpath in images:
        print fpath
        im = cv2.imread(fpath)
        im = cv2.GaussianBlur(im, (3,3), 0)
        imshow_large(__file__, im)
        key = cv2.waitKey()
        if key & 0xff == ord("q"):
            sys.exit(0)
        elif key & 0xff == ord("n"):
            continue

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
        elif key & 0xff == ord("n"):
            continue

        binarized = binarize(unrotated)

        scale = SCREEN_HEIGHT/unrotated.shape[0]
        colorbin = copy.deepcopy(unrotated)
        colorbin = cv2.resize(colorbin, (0,0), fx=scale, fy=scale) 
        rois = get_rois(binarized)
        for (x,y,width,height) in rois:
            x = int(x*scale)
            y = int(y*scale)
            width = int(width*scale)
            height = int(height*scale)
            cv2.rectangle(colorbin, (x,y), (x+width, y+height), (255,0,0,0), 1)
        cv2.imshow(__file__, colorbin)
        key = cv2.waitKey()
        if key & 0xff == ord("q"):
            break
        elif key & 0xff == ord("n"):
            continue

        for (x,y,width,height) in rois:
            colorbin2 = copy.deepcopy(colorbin)
            x_ = int(x*scale)
            y_ = int(y*scale)
            width_ = int(width*scale)
            height_ = int(height*scale)
            cv2.rectangle(colorbin2, (x_,y_), (x_+width_, y_+height_), (0,0,255,0), 1)

            sub = unrotated[y:y+height, x:x+width]
            supers = cv2.resize(sub, (192, 192), interpolation=cv.CV_INTER_NN)

            cv2.imshow(__file__, colorbin2)
            cv2.imshow("supers", supers)
            key = cv2.waitKey()
            if key & 0xff == ord("q"):
                sys.exit(0)
            elif key & 0xff == ord("n"): # move on to the next image
                break
            elif (key & 0xff) in map(ord, "0123456789"):
                digit = chr(key & 0xff)
                subdir = P.join(out_dir, digit)
                out_file = P.join(subdir, str(counters[digit]) + ".png")
                cv2.imwrite(out_file, sub)
                counters[digit] += 1
            else:
                subdir = P.join(out_dir, "X")
                out_file = P.join(subdir, str(counters["X"]) + ".png")
                cv2.imwrite(out_file, sub)
                counters["X"] += 1
                

if __name__ == "__main__":
    main()
