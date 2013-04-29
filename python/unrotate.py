#!/usr/bin/env python
"""Locate the paper receipt in an image, and rotate it such that text is the right way up."""
from __future__ import division
import cv
import cv2
import numpy as np
import math

def white_paper_mask(im, debug=False):
    """Locate white regions of the original image."""
    hsv = cv2.cvtColor(im, cv.CV_BGR2HSV)
    _, s, v = cv2.split(hsv)
    _, s = cv2.threshold(s, 64, 255, cv.CV_THRESH_BINARY_INV)
    _, v = cv2.threshold(v, 96, 255, cv.CV_THRESH_BINARY)
    mask = cv2.bitwise_and(s, v)

    if debug:
        cv2.imshow(__file__, s)
        cv2.waitKey()
        cv2.imshow(__file__, v)
        cv2.waitKey()
        cv2.imshow(__file__, mask)
        cv2.waitKey()
    return mask

def unrotate(original, debug=False):
    mask = white_paper_mask(original, debug)
    #
    # scale down so that we can use consistent number of iterations for morphological ops
    #
    scale = 512/original.shape[1]
    thresh = cv2.resize(mask, (0,0), fx=scale, fy=scale)
    scaled = cv2.resize(original, (0,0), fx=scale, fy=scale)

    num_iter = 1
    closed = cv2.dilate(cv2.erode(thresh, None, iterations=num_iter), None, iterations=num_iter)
    if debug:
        cv2.imshow(__file__, closed)
        cv2.waitKey()

    contours, _= cv2.findContours(closed, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)

    if debug:
        cv2.drawContours(scaled, contours, -1, (255, 0, 0))
        cv2.imshow(__file__, scaled)
        cv2.waitKey()

    #
    # Pick the quad with the largest area
    #
    quads = []
    for contour in contours:
        #
        # TODO: check if the contour is rectangular enough
        #
        area = cv2.contourArea(contour)

        #
        # TODO: How to decide this threshold?
        # TODO: order the vertices in the decided polygon in some predictable way
        #
        poly = cv2.approxPolyDP(contour, 50, True)
        if len(poly) == 4:
            quads.append((area, poly))

    assert quads

    #
    # Pick the largest quad, by area
    #
    quad = sorted(quads, reverse=True)[0][1]
    a_x = quad[0][0][0]
    a_y = quad[0][0][1]
    b_x = quad[1][0][0]
    b_y = quad[1][0][1]
    d_x = quad[2][0][0]
    d_y = quad[2][0][1]
    e_x = quad[3][0][0]
    e_y = quad[3][0][1]
    width = math.sqrt((a_x-b_x)**2 + (a_y-b_y)**2)
    height = math.sqrt((b_x-d_x)**2 + (b_y-d_y)**2)

    if debug:
        cv2.circle(scaled, (int(a_x), int(a_y)), 5, (255, 0, 0), -1)   # blue
        cv2.putText(scaled, "A", (int(a_x), int(a_y)), cv.CV_FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.circle(scaled, (int(b_x), int(b_y)), 5, (0, 255, 0), -1)   # green
        cv2.putText(scaled, "B", (int(b_x), int(b_y)), cv.CV_FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.circle(scaled, (int(d_x), int(d_y)), 5, (0, 0, 255), -1)   # red
        cv2.putText(scaled, "D", (int(d_x), int(d_y)), cv.CV_FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.circle(scaled, (int(e_x), int(e_y)), 5, (255, 255, 0), -1) # cyan
        cv2.putText(scaled, "E", (int(e_x), int(e_y)), cv.CV_FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        cv2.imshow(__file__, scaled)
        cv2.waitKey()

    #
    # Scale up to the original image and calculate a perspective transform
    #
    A_x, A_y, B_x, B_y, D_x, D_y, E_x, E_y = map(lambda x: x/scale, [a_x, a_y, b_x, b_y, d_x, d_y, e_x, e_y])
    Width, Height = width/scale, height/scale

    src = np.array([(A_x, A_y), (B_x, B_y), (D_x, D_y), (E_x, E_y)], dtype="float32")
    dst = np.array([(Width, 0), (0, 0), (0, Height), (Width, Height)], dtype="float32")

    perspective = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(original, perspective, (0, 0))
    warped = warped[:Height, :Width]
    if width > height:
        warped = cv2.flip(cv2.transpose(warped), 1)

    return warped

def main():
    import sys
    original = cv2.imread(sys.argv[1])
    assert original is not None
    unrotated = unrotate(original, debug=False)

    cv2.imshow(__file__, unrotated)
    cv2.waitKey()
    cv2.imwrite("unrotated.png", unrotated)

if __name__ == "__main__":
    main()
