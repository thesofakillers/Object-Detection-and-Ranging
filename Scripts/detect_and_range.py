"""
Detects pedestrians and estimates the distance to them using HoG, SVM and SGBM.

Heavily Based off of scripts shown in /T_Breckon/ made by
Prof Toby Breckon of Durham University

by 2018/2019 Durham Uni CS candidate dzgf42
"""
# <section>~~~~~~~~~~~~~~~~~~~~~~~~~~~Imports~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import cv2
import os
import sys
import numpy as np
from utils import *
# </section>End of Imports


# <section>~~~~~~~~~~~~~~~~~~~~~~~~OpenCV settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#optimize when possible
cv2.setUseOptimized(True);
#try using multithreading when possible
cv2.setNumThreads(4);
# </section>End of Disparity Settings


# <section>~~~~~~~~~~~~~~~~~~~~~~~Directory Settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
master_path_to_dataset = "../Data/TTBB-durham-02-10-17-sub10"  # where is the data
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed

# set this to a file timestamp to start from (empty is first example - outside lab)
# set to timestamp to skip forward to
skip_forward_file_pattern = ""

# resolve full directory location of data set for left / right images
full_path_directory_left = os.path.join(
    master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(
    master_path_to_dataset, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left))

# </section>End of Directory Settings


# <section>~~~~~~~~~~~~~~~~~~~~~~~Disparity Settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# setup the disparity stereo processor to find a maximum of 128 disparity values
max_disparity = 128

# create stereo processor from OpenCv
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)
# </section>End of Disparity Settings


# <section>~~~~~~~~~~~~~~~~~~~~~~~~Model Settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# specify classifier used as a string. Options include:
# -"SVM"
# -"Foo" (haven't done anything else yet)
model = "MRCNN"

if model == "SVM":
    from SVM.hog_detector import hog_detect
    import SVM.params as params
    try:
        # load SVM object once, outside of loop
        svm = cv2.ml.SVM_load(params.HOG_SVM_PATH)
    except:
        print("Missing files - SVM!")
        exit()
elif model =="MRCNN":
    import Deep.model as modellib
    from Deep.mask_rcnn_detector import mask_rcnn_detect
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Root directory of DeepL things, assuming the script is run from Scripts
    ROOT_DIR = os.path.abspath("./Deep/")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library

    import coco # Import COCO config
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # see coco.CocoConfig and config.py for more details
    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    # create config object
    config = InferenceConfig()
    # Create model object in inference mode. Pass config object from earlier
    # Model dir here is arbitrary since we are not training
    mask_rcnn = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # Load weights trained on MS-COCO.
    mask_rcnn.load_weights(COCO_MODEL_PATH, by_name=True)
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    deep_class_names = np.array(['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush'])
# </section>


# <section>~~~~~~~~~~~~~~~~~~~Selective Search Settings~~~~~~~~~~~~~~~~~~~~~~~~~~
if model == "SVM":
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

# </section>End of Disparity Settings


# <section>~~~~~~~~~~~~~~~~~~~~~~~Camera Settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
camera_focal_length_px = 399.9745178222656         # focal length in pixels
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
# </section>End of Camera Settings


# <section>~~~~~~~~~~~~~~~~~~~~~~~~~~Functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def check_skip(timestamp, filename):
    """
    Checks if a timestamp has been given and whether the timestamp corresponds
    to the given filename.

    Returns True if this condition is met and False Otherwise"
    """
    if ((len(timestamp) > 0) and not(timestamp in filename)):
        return True
    elif ((len(timestamp) > 0) and (timestamp in filename)):
        return False


def join_paths_both_sides(directory_left, filename_left, directory_right, filename_right):
    "Given directory and filename, joins them into the complete path to the file"
    full_path_filename_left = os.path.join(
        full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(
        full_path_directory_right, filename_right)
    return full_path_filename_left, full_path_filename_right


def convert_to_grayscale(color_images):
    "Given an array of color images, returns an array of grayscale images"
    gray_images = []
    for image in color_images:
        gray_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    return gray_images


def compute_disparity(left_image, right_image, maximum_disparity, noise_filter, width):
    """
    Input: Grayscale Left & Right Images, Maximum Disparity Value
    -Noise filter: increase to be more aggressive
    Output: Disparity between images, scaled appropriately
    """
    # remember to convert to grayscale (as the disparity matching works on grayscale)
    # N.B. need to do for both as both are 3-channel images
    grayL, grayR = convert_to_grayscale([left_image, right_image])

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEED TO IMPROVE/LOOK INTO!!!!!!!!!!!!!!!!!!!!!!
    # perform preprocessing - raise to the power, as this subjectively appears
    # to improve subsequent disparity calculation
    grayL = np.power(grayL, 0.75).astype('uint8')
    grayR = np.power(grayR, 0.75).astype('uint8')
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # compute disparity image from undistorted and rectified stereo images
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)
    disparity = stereoProcessor.compute(grayL, grayR)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!NEEDS MORE INVESTIGATING!!!!!!!!!!!!!!!!!!!!!!!!!
    # filter out noise and speckles (adjust parameters as needed)
    cv2.filterSpeckles(disparity, 0, 4000, maximum_disparity - noise_filter)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # threshold the disparity so that it goes from 0 to max disparity
    _, disparity = cv2.threshold(
        disparity, 0, maximum_disparity * 16, cv2.THRESH_TOZERO)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    # crop area not seen by *both* cameras and and area with car bonnet
    disparity_scaled = crop_image(disparity_scaled, 0, 390, 135, width)

    return disparity_scaled


def click_event(event, x, y, flags, param):
    """
    listens for mouse clicks and prints the depth at that mouse click location
    """
    depth = param
    if event == cv2.EVENT_LBUTTONDOWN:
        print(depth[y, x])


def compute_depth(disparity, focal_length, distance_between_cameras):
    """
    Computes depth in meters
    Input:
    -Disparity in pixels
    -Focal Length in pixels
    -Distance between cameras in meters
    Output:
    -Depth in meters
    """
    with np.errstate(divide='ignore'):
        depth = (focal_length * distance_between_cameras) / disparity
    return depth


def compute_single_depth(rectangle, disparity_image, focal_length, distance_between_cameras):
    """
    Given a rectangular area and a disparity image, estimates the general Depth
    of that rectangular ROI
    """
    # extracting corners from rectangle (top left and bottom right)
    x1, y1, x2, y2 = rectangle
    # cropping and flattening disparity image so that we are only dealing with ROI values
    rectangle_disparity = crop_image(disparity, y1, y2, x1, x2)
    # sorting the disparity ROI by ascending disparity
    rectangle_disparity = np.sort(rectangle_disparity, axis=None)
    # keeping only the final third of the pixels (which we believe correspond to the detected object)
    rectangle_disparity = rectangle_disparity[(
        2 * len(rectangle_disparity)) // 3:]
    # compute corresponding depths
    rectangle_depths = compute_depth(
        rectangle_disparity, focal_length, distance_between_cameras)
    # return the average depth
    return np.average(rectangle_depths)


# </section>End of Functions Section


# <section>~~~~~~~~~~~~~~~~~~~~~~~~~~~~Main~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Colors = gen_N_colors(81)

for filename_left in left_file_list:
    # <section>---------------Directory Checks---------------
    # skipping if requested
    if check_skip(skip_forward_file_pattern, filename_left):
        continue
    else:
        skip_forward_file_pattern = ""

    # from the left image filename get the correspondoning right image
    filename_right = filename_left.replace("_L", "_R")
    full_path_filenames = join_paths_both_sides(full_path_directory_left, filename_left,
                                                full_path_directory_right, filename_right)
    full_path_filename_left, full_path_filename_right = full_path_filenames

    # # for sanity print out these filenames
    # print(full_path_filename_left)
    # print(full_path_filename_right)
    # </section>-----------End of Directory Checks-----------
    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):
        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        #print("-- files loaded successfully\n")

        # compute image width
        original_width = np.size(imgL, 1)

        # compute disparity
        disparity = compute_disparity(
            imgL, imgR, max_disparity, 5, original_width)

        # cropping left image to match disparity & depth sizes
        imgL = crop_image(imgL, 0, 390, 135, original_width)

        # get detections as rectangles and their respective classes
        if model =="SVM":
            detection_rects, detection_classes = hog_detect(imgL, svm, ss)
        elif model =="MRCNN":
            detection_rects, detection_classes, detection_class_names, confidences = mask_rcnn_detect(imgL, mask_rcnn, deep_class_names)

        # get a single depth estimation for each detected object
        detection_depths = np.fromiter((compute_single_depth(
            rect, disparity, camera_focal_length_px, stereo_camera_baseline_m) for rect in detection_rects), float)

        # <section>-------------------Display-----------
        min_depth = 1000
        min_depth_class = "No Detections"
        units = " meters"
        # draw detections onto imgL
        for i in range(len(detection_classes)):
            # get rect
            det_rect = detection_rects[i]
            x1, y1, x2, y2 = det_rect

            if model == "SVM":
                # get class number
                det_class = int(detection_classes[i])
                # get class name based on class number
                det_class_name = get_class_name(det_class)
                # get color based on class number
                color = Colors[det_class]
            elif model == "MRCNN":
                # get class name
                det_class_name = detection_class_names[i]
                # get color based on class number
                color = Colors[detection_classes[i]]
                # get confidence
                confidence = str(round(confidences[i], 2))


            # get depth
            det_depth = round(detection_depths[i], 1)
            # draw colored rectangle where detected object is
            cv2.rectangle(imgL, (x1, y1),
                          (x2, y2), color, 2)
            # label rectangle
            cv2.putText(imgL, "{}: {} m".format(det_class_name,
                                                det_depth), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
            if model == "MRCNN":
                # add confidence label
                cv2.putText(imgL, "{}".format(confidence), (x1+4, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
            if det_depth < min_depth:
                min_depth = det_depth
                min_depth_class = det_class_name
        #requested standard out
        if min_depth >= 1000:
            min_depth = "Depth Irrelevant"
            units = ""
        print(filename_left)
        print("{}: {} ({}{})\n".format(
            filename_right, min_depth_class, min_depth, units))

        # show left color image
        cv2.imshow('detected objects', imgL)

        # show disparity image (scaling it to the full 0->255 range)
        cv2.imshow("disparity", (disparity
                                 * (256 / max_disparity)).astype(np.uint8))

        # #listen for mouse clicks and print depth where clicked
        # cv2.setMouseCallback("disparity", click_event, param = depth)

        # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        cv2.waitKey(16) & 0xFF
        # </section>-------End of Display Section--------
    else:
        print("-- files skipped (perhaps one is missing or not PNG)\n")

# close all windows
cv2.destroyAllWindows()
# </section>
