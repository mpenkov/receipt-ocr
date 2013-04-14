#!/usr/bin/env python
"""Locate the paper receipt in an image, and rotate it such that text is the right way up."""
from __future__ import division
import cv
import cv2
import numpy as np
import math

def unrotate(original, debug=False):
    #
    # scale down so that we can use consistent number of iterations for morphological ops
    #
    scale = 512/original.shape[1]
    im = cv2.resize(original, (0,0), fx=scale, fy=scale) 

    ycrcb = cv2.cvtColor(im, cv.CV_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    if debug:
        cv2.imshow(__file__, y)
        cv2.waitKey()

    #
    # TODO: adaptive threshold?  Ohtsu?
    #
    _, thresh = cv2.threshold(y, 92, 255, cv.CV_THRESH_BINARY)
    if debug:
        cv2.imshow(__file__, thresh)
        cv2.waitKey()

    num_iter = 3
    closed = cv2.dilate(cv2.erode(thresh, None, iterations=num_iter), None, iterations=num_iter)
    if debug:
        cv2.imshow(__file__, closed)
        cv2.waitKey()

    contours, _= cv2.findContours(closed, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)

    if debug:
        cv2.drawContours(im, contours, -1, (255, 0, 0))
        cv2.imshow(__file__, im)
        cv2.waitKey()

    #
    # Pick the rectangle with the largest area
    #
    rects = []
    for contour in contours:
        #
        # TODO: check if the contour is rectangular enough
        #
        rect = cv2.minAreaRect(contour)
        area = cv2.contourArea(contour)
        rects.append((area, rect))

    rect = sorted(rects, reverse=True)[0][1]
    (c_x, c_y), (width, height), theta_deg = rect

    alpha = math.atan2(height, width)
    theta = theta_deg*math.pi/180
    diag = math.sqrt(width**2 + height**2)
    a_x = c_x - diag/2 * math.cos(alpha-theta)
    a_y = c_y + diag/2 * math.sin(alpha-theta)

    b_x = c_x - diag/2 * math.cos(alpha+theta)
    b_y = c_y - diag/2 * math.sin(alpha+theta)

    d_x = c_x + diag/2 * math.cos(alpha-theta)
    d_y = c_y - diag/2 * math.sin(alpha-theta)

    e_x = c_x + diag/2 * math.cos(alpha+theta)
    e_y = c_y + diag/2 * math.sin(alpha+theta)

    if debug:
        cv2.circle(im, (int(a_x), int(a_y)), 5, (255, 0, 0), -1)   # blue
        cv2.circle(im, (int(b_x), int(b_y)), 5, (0, 255, 0), -1)   # green
        cv2.circle(im, (int(d_x), int(d_y)), 5, (0, 0, 255), -1)   # red
        cv2.circle(im, (int(e_x), int(e_y)), 5, (255, 255, 0), -1) # cyan
        cv2.imshow(__file__, im)
        cv2.waitKey()

    A_x, A_y, B_x, B_y, D_x, D_y, E_x, E_y = map(lambda x: x/scale, [a_x, a_y, b_x, b_y, d_x, d_y, e_x, e_y])
    Width, Height = width/scale, height/scale

    src = np.array([(A_x, A_y), (B_x, B_y), (D_x, D_y), (E_x, E_y)], dtype="float32")
    dst = np.array([(0, Height), (0, 0), (Width, 0), (Width, Height)], dtype="float32")

    perspective = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(original, perspective, (0, 0))
    warped = warped[:Height, :Width]
    if width > height:
        warped = cv2.flip(cv2.transpose(warped), 1)

    return warped

def main():
    import sys
    original = cv2.imread(sys.argv[1])
    unrotated = unrotate(original)

    cv2.imshow(__file__, unrotated)
    cv2.waitKey()

    
if __name__ == "__main__":
    main()
