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
    rects = np.array(rects)[max_rects]

    # get mask for accepted rects
    mask = np.logical_and(
        np.greater(rects[:,3], rects[:,2]), #return only regions where h > width
        np.greater(rects[:,2] * rects[:,3], min_area) # return only sufficiently large regions
    )

    # filtering out results
    accepted_rects = rects[mask]

    return accepted_rects
