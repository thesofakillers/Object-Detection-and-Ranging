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
import os
import sys
import math
import numpy as np


def perform_selective_search(image, ss_object, max_rects, min_area):
    min_width = np.sqrt(min_area / 2)
    min_height = min_area / min_width

    ss_object.setBaseImage(image)

    # Switch to fast but low recall Selective Search method
    ss_object.switchToSelectiveSearchFast()

    # # Switch to high recall but slow Selective Search method (slower)
    # ss_object.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss_object.process()

    rects_to_return = [rect for rect in rects[:max_rects]
                       if (rect[2] > min_width and rect[3] > min_height)]

    ss_object.clearImages()

    return rects_to_return
