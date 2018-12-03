"""
performs selective search bounding box identification
"""
import cv2
import numpy as np


def perform_selective_search(image, ss_object, max_rects, min_area):
    """
    Performs Selective Search on a given image, returning a list of proposed
    regions.

    Input(s):
    -image: numpy array representing an image
    -ss_object: OpenCV selective search object instance
    -max_rects: the maximum number of rectangles to return
    -min_area: the min_area a proposed region can be

    Output(s):
    -accepted_rects: list of region proposals (rects)
    """
    # Set the base image of the object
    ss_object.setBaseImage(image)

    # Switch to fast but low recall Selective Search method
    ss_object.switchToSelectiveSearchFast()

    # run selective search segmentation on image set as Base
    rects = ss_object.process()  # one rect: x1, y1, w, h

    # keep only the first max_rects number of regions
    rects = np.array(rects)[:max_rects]

    # get mask for accepted rects (can be considered a heuristic)
    mask = np.logical_and(
        # return only regions where h > width
        np.greater(rects[:, 3], rects[:, 2]),
        # return only sufficiently large regions
        np.greater(rects[:, 2] * rects[:, 3], min_area)
    )

    # filtering out results
    accepted_rects = rects[mask]

    # and returning
    return accepted_rects
