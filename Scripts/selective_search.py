"""
'Example : performs selective search bounding box identification

Author : Toby Breckon, toby.breckon@durham.ac.uk
Copyright (c) 2018 Department of Computer Science, Durham University, UK

License: MIT License

ackowledgements: based on the code and examples presented at:
https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/'

Original Module Docstring^^^^ I slightly edited the script so that I can use it
more modularly
"""
import cv2
import numpy as np
from utils import *


def perform_selective_search(image, ss_object, max_rects, min_area):

    ss_object.setBaseImage(image)

    # Switch to fast but low recall Selective Search method
    ss_object.switchToSelectiveSearchFast()

    # # Switch to high recall but slow Selective Search method (slower)
    # ss_object.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss_object.process()  # one rect: x1, y1, w, h

    # ignoring overly small regions
    accepted_rects = np.array([rect for rect in rects[:max_rects] if (rect[2] * rect[3] > min_area)])

    # extracting column information
    x1 = accepted_rects[:, 0]
    y1 = accepted_rects[:, 1]
    w = accepted_rects[:, 2]
    h = accepted_rects[:, 3]
    # calculating second point
    x2 = x1 + w
    y2 = y1 + h

    #rebuilding rects in x1, y1, x2, y2 format
    accepted_rects = np.transpose(np.array([x1, y1, x2, y2]))

    # clearing images from selective search object
    ss_object.clearImages()

    # locate indices of boxes to be kept after overlap elimination
    surviving_indeces = non_max_suppression_fast(np.int32(accepted_rects), 0.4)

    # return non-overlapping boxes (to some threshold)
    return accepted_rects[surviving_indeces]
