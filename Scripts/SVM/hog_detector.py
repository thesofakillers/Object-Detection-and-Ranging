"""
functionality: perform detection based on HOG feature descriptor / SVM classification
using a very basic multi-scale, sliding window (exhaustive search) approach

This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
License: MIT License

Minor portions: based on fork from https://github.com/nextgensparx/PyBOW

Original Module Docstring^^^^ I slightly edited the script so that I can use it
more modularly
"""
from utils import *
import SVM.params as params
from SVM.selective_search import *


def hog_detect(image, svm_object, ss_object, disparity_image, focal_length, distance_between_cameras):
    human_height = 1.75  # meters, on average
    human_width = 1.75 / 2  # meters, approximating

    # initialize detections and corresponding detection_classes lists
    detections = []
    detection_classes = []
    detection_depths = []

    # get rid of sky when performing selective search
    roi = select_roi_maintain_size(image, 116)

    # perform selective_search
    region_proposals = perform_selective_search(roi, ss_object, 1000, 3600)

    # loop through region proposals
    for region_proposal_rect in region_proposals:
        # extract information from the rect
        x1, y1, w, h = region_proposal_rect
        x2, y2 = (x1 + w), (y1 + h)

        # calculate distance to observed region
        region_depth = compute_single_depth(
            (x1, y1, x2, y2), disparity_image, focal_length, distance_between_cameras)

        # check the detected area size makes sense
        if not (area_depth_heuristic(human_height, human_width, h, w, region_depth, focal_length, 0.4)):
            continue

        # get the corresponding window
        region_proposal = crop_image(image, y1, y2, x1, x2)

        # create image data object from window
        img_data = ImageData(region_proposal)

        # compute the hog descriptor
        img_data.compute_hog_descriptor()

        # classify each HoG by passing it through the SVM classifier
        if img_data.hog_descriptor is not None:
            # apply svm classification
            retval, [result] = svm_object.predict(
                np.float32([img_data.hog_descriptor]))
            class_number = result[0]

            # if we get a detection, then record it
            if class_number == params.DATA_CLASS_NAMES["pedestrian"]:
                rect = x1, y1, x2, y2
                # append the rect to the list of detections
                detections.append(rect)
                # append class number to list of class numbers
                detection_classes.append(class_number)
                # append detection depth to the list of detection depths
                detection_depths.append(region_depth)
    # converting to numpy.arrays for convenience
    detections = np.array(detections)
    detection_classes = np.array(detection_classes)
    detection_depths = np.array(detection_depths)
    # remove overlapping boxes.
    # get indices of surviving boxes
    surviving_indeces = non_max_suppression_fast(np.int32(detections), 0.4)
    # keep only surviving detections
    detections = detections[surviving_indeces].astype("int")
    detection_classes = detection_classes[surviving_indeces]
    detection_depths = detection_depths[surviving_indeces]
    # return detection rects and respective detection_classes
    return detections, detection_classes, detection_depths
