"""
functionality: perform detection based on HOG feature descriptor / SVM classification
using selective search region proposal
"""
import utils
import SVM.params as params
import SVM.selective_search as selective_search
import numpy as np


def hog_detect(image, svm_object, ss_object, disparity_image, focal_length, distance_between_cameras):
    """
    Performs detection on an image via an SVM classifier trained on HoG
    descriptors. Returns detected object rectangles, their class codes, and their
    depths in the image

    Inputs:
    -image: numpy array representing an image
    -svm_object: cv2.ml.SVM_load(<trained_xml_file>) SVM object
    -disparity_image: numpy array corresponding to the disparity of "image"
    -focal_length: the focal length in pixels of the cameras used
    -distance_between_cameras: the baseline distance in meters between the cameras

    Outputs:
    -detections: list of rects, where rect = x1, y1, x2, y2
    -detection_classes: list of class codes corresponding to rects
    -detection_depths: list of depths (meters) of each detected rect
    """
    human_height = 1.75  # meters, on average
    human_width = 1.75 / 2  # meters, approximating

    # initialize detections and corresponding detection_classes lists
    detections = []
    detection_classes = []
    detection_depths = []

    # get rid of sky when performing selective search (heuristic)
    roi = utils.select_roi_maintain_size(image, 116)

    # perform selective_search, returns list of region proposals
    region_proposals = selective_search.perform_selective_search(
        roi, ss_object, 1000, 3600)

    # loop through region proposals
    for region_proposal_rect in region_proposals:
        # extract information from the region proposal
        x1, y1, w, h = region_proposal_rect
        x2, y2 = (x1 + w), (y1 + h)

        # calculate distance to observed region
        region_depth = utils.compute_single_depth(
            (x1, y1, x2, y2), disparity_image, focal_length, distance_between_cameras)

        # check the detected area size makes sense (heuristic)
        if not (utils.area_depth_heuristic(human_height, human_width, h, w, region_depth, focal_length, 0.4)):
            continue

        # get the corresponding window
        region_proposal = utils.crop_image(image, y1, y2, x1, x2)

        # create image data object from window
        img_data = utils.ImageData(region_proposal)

        # compute the hog descriptor
        img_data.compute_hog_descriptor()

        # classify each HoG by passing it through the SVM classifier
        if img_data.hog_descriptor is not None:
            # apply svm classification
            retval, [result] = svm_object.predict(
                np.float32([img_data.hog_descriptor]))
            class_number = result[0]

            # if we get a detection, then record it
            if class_number == params.DATA_CLASS_NAMES["person"]:
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
    surviving_indeces = utils.non_max_suppression_fast(
        np.int32(detections), 0.4)
    # keep only surviving detections
    detections = detections[surviving_indeces].astype("int")
    detection_classes = detection_classes[surviving_indeces]
    detection_depths = detection_depths[surviving_indeces]

    # return detection rects and respective detection_classes and depths
    return detections, detection_classes, detection_depths
