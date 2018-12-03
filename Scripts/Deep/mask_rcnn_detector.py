"""
Performs MaskRCNN detection on an inputted image and returns detection.
This particular module made by myself.
"""

import numpy as np

def mask_rcnn_detect(image, model, class_names):
    """
    Detects objects in a given image using MaskRCNN and returns them

    Inputs:
    -image: np array representing an image
    -model: Mask RCNN model object as defined in model.py
    -class_names: list of class names, with indices corresponding to their codes

    Returns:
    -rects: rectangles describing detection boxes
    -class_ids: class numerical ids of each detection
    -classes: class names of each detection
    -scores: confidence scores of each detection
    """
    # Run detection on the image
    results = model.detect([image], verbose=0)

    # get results
    r = results[0]

    # extract bounding box info
    rects = r['rois']
    # reshape in format desired by detect_and_range.py
    rects[:, [1, 0]] = rects[:, [0, 1]]  # swap column 0 and column 1
    rects[:, [3, 2]] = rects[:, [2, 3]]  # swap column 2 and column 3

    # getting class ids
    class_ids = r['class_ids']

    # using class ids to index class_names and get detected classes
    classes = class_names[class_ids]

    # get correspinding scores
    scores = r['scores']

    # thresholding heuristic
    mask = np.greater_equal(scores, 0.9)

    return rects[mask], class_ids[mask], classes[mask], scores[mask]
