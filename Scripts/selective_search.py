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

    return accepted_rects
