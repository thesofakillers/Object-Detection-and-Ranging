#####################################################################
"""
Computes SGBM disparity and equivalent depth of incoming stereo images

Heavily Based off of stereo_disparity.py and stereo_to_3d.py by
Prof Toby Breckon of Durham University

by 2018/2019 Durham Uni CS candidate dzgf42
"""
#############################Imports#################################
import cv2
import os
import numpy as np

###########################Directory Settings########################
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

#######################Disparity Settings##########################
# setup the disparity stereo processor to find a maximum of 128 disparity values
max_disparity = 128

crop_disparity = True  # display full or cropped disparity image
pause_playback = False  # pause until key press after each image

# create stereo processor from OpenCv
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

###########################Camera Settings########################
camera_focal_length_px = 399.9745178222656         # focal length in pixels
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

###############################Functions##########################


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
    return ((focal_length * distance_between_cameras) / disparity)


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


def compute_disparity(gray_left_image, gray_right_image, maximum_disparity, noise_filter, crop_boolean):
    """
    Input: Grayscale Left & Right Images, Maximum Disparity Value
    -Noise filter: increase to be more aggressive
    -Crop_Boolean: set to True if you wish to crop unnescessary areas
    Output: Disparity between images, scaled appropriately
    """
    # compute disparity image from undistorted and rectified stereo images
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)
    disparity = stereoProcessor.compute(gray_left_image, gray_right_image)

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
    if (crop_boolean):
        width = np.size(disparity_scaled, 1)
        disparity_scaled = disparity_scaled[0:390, 135:width]

    return disparity_scaled


def click_event(event, x, y, flags, param):
    """
    listens for mouse clicks and prints the depth at that mouse click location
    """
    depth = param
    if event == cv2.EVENT_LBUTTONDOWN:
        print(depth[y, x])


#################################Main############################
for filename_left in left_file_list:
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

    # for sanity print out these filenames
    print(full_path_filename_left)
    print(full_path_filename_right)

    # check the file is a PNG file (left) and check a correspondoning right image
    # actually exists
    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)):
        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such
        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        cv2.imshow('left image', imgL)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        cv2.imshow('right image', imgR)

        print("-- files loaded successfully\n")

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images
        grayL, grayR = convert_to_grayscale([imgL, imgR])

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEED TO IMPROVE/LOOK INTO!!!!!!!!!!!!!!!!!!!!!!
        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation
        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # compute disparity
        disparity = compute_disparity(
            grayL, grayR, max_disparity, 5, crop_disparity)

        # display image (scaling it to the full 0->255 range)
        cv2.imshow("disparity", (disparity *
                                 (256 / max_disparity)).astype(np.uint8))

        # compute depth from disparity
        depth = compute_depth(
            disparity, camera_focal_length_px, stereo_camera_baseline_m)

        # #listen for mouse clicks and print depth where clicked
        # cv2.setMouseCallback("disparity", click_event, param = depth)


        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # crop - c
        # pause - space

        # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF
        if (key == ord('x')):       # exit
            break  # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled)
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity)
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback)
    else:
        print("-- files skipped (perhaps one is missing or not PNG)")
        print()

# close all windows
cv2.destroyAllWindows()

#####################################################################
