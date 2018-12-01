
def mask_rcnn_detect(image, model, class_names):
    """
    Inputs:
    -image: np array representing an image
    -model: Mask RCNN model object as defined in model.py
    -class_names: list of class names, with indices corresponding to their codes
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

    return rects, class_ids, classes, scores
