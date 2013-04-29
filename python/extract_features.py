#!/usr/bin/env python
"""Extract a feature vector from a training image."""
from __future__ import division
import cv
import cv2
import numpy as np
from collections import Counter

def count_non_zero(binarized, rows=False):
    """Count the number of non-zero elements in each row or column."""
    if rows:
        nonzero = np.zeros((binarized.shape[0],))
        for i in range(binarized.shape[0]):
            nonzero[i] = np.count_nonzero(binarized[i,:])
    else:
        nonzero = np.zeros((binarized.shape[1],))
        for i in range(binarized.shape[1]):
            nonzero[i] = np.count_nonzero(binarized[:,i])
    return nonzero

def scale_and_center(im, dim):
    im = cv2.equalizeHist(im)
    if im.shape[0] > im.shape[1]:
        new_height = dim
        new_width = int(dim/im.shape[0]*im.shape[1])
    else:
        new_width = dim
        new_height = int(dim/im.shape[1]*im.shape[0])

    #
    # center the image in a DIMxDIM tile
    #
    scale = cv2.resize(im, (new_width, new_height), interpolation=cv.CV_INTER_LINEAR)

    tile = np.zeros((dim, dim), dtype=np.uint8)
    tile[:,:] = 255
    if im.shape[0] > im.shape[1]:
        pad = (dim-scale.shape[1])/2
        tile[:,pad:pad+scale.shape[1]] = scale
    else:
        pad = (dim-scale.shape[0])/2
        tile[pad:pad+scale.shape[0],:] = scale

    return tile

def extract_features(im, debug=False):
    dim = 64

    if False:
        #
        # This is sensitive to aspect ratio
        #
        imdim = scale_and_center(im, dim)
    else:
        #
        # This is insensitive to aspect ratio
        #
        imdim = cv2.resize(im, (dim, dim), interpolation=cv.CV_INTER_LINEAR)

    _, binarized = cv2.threshold(imdim, 0, 255, cv.CV_THRESH_OTSU)

    if debug:
        cv2.imshow(__file__, binarized)
        cv2.waitKey(30)

    lowres = cv2.resize(imdim, (5, 5), interpolation=cv.CV_INTER_LINEAR)
    if debug:
        lowres_ = cv2.resize(lowres, (500,500), interpolation=cv.CV_INTER_NN)
        cv2.imshow(__file__, lowres_)
        cv2.waitKey(30)

    lowres = np.reshape(lowres, 25)
    nonzero_rows = count_non_zero(binarized, True)
    nonzero_cols = count_non_zero(binarized, False)

    vec = np.concatenate((lowres, nonzero_rows, nonzero_cols))
    vec32f = np.zeros(vec.shape, dtype=np.float32)
    vec32f[:] = vec
    return vec32f

def equalize_samples(samples):
    """Make sure there are equal numbers of samples for each class."""
    counter = Counter()
    for (y,x) in samples:
        counter[y] += 1

    min_samples = min(counter.values())
    counter = Counter()
    new_samples = []
    for (y,x) in samples:
        if counter[y] >= min_samples:
            continue
        new_samples.append((y,x))
        counter[y] += 1
    return new_samples

def split_samples(samples, ratio=0.7):
    import random
    random.shuffle(samples)
    cutoff = int(len(samples)*0.7)
    training = samples[:cutoff]
    test = samples[cutoff:]
    return training, test

def main():
    import sys
    import os
    import os.path as P
    root = sys.argv[1]
    output = sys.argv[2]
    subdir = os.listdir(root)
    samples = []
    for i, sd in enumerate(sorted(subdir)):
        if not P.isdir(P.join(root, sd)):
            continue
        files = filter(lambda x: x.endswith(".png"), os.listdir(P.join(root, sd)))
        for f in sorted(files):
            im = cv2.imread(P.join(root, sd, f), cv.CV_LOAD_IMAGE_GRAYSCALE)
            assert im is not None
            vec = extract_features(im)
            #print i, sd, f
            samples.append((i, vec))

    dim = len(vec)
    samples = equalize_samples(samples)
    training, test = split_samples(samples)

    print len(training), "training samples,", len(test), "test samples"

    X = np.zeros((len(training), dim), dtype=np.float32)
    Y = np.zeros((len(training), 1), dtype=np.float32)
    for i, (y, x) in enumerate(training):
        X[i,:] = x
        Y[i] = y

    classifier = cv2.SVM()
    classifier.train_auto(X, Y, None, None, None) # selects best parameters

    for (y,x) in test:
        print y, classifier.predict(x)

    classifier.save(output)

if __name__ == "__main__":
    main()
