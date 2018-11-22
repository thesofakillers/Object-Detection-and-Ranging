################################################################################

# functionality: parameter settings for detection algorithm training/testing

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Origin acknowledgements: forked from https://github.com/nextgensparx/PyBOW

################################################################################

import cv2
import os
import sys

################################################################################
# settings for datsets in general

master_path_to_dataset = "../Data/Training"  # ** need to edit this **

# data location - training examples

DATA_training_path_neg = os.path.join(
    master_path_to_dataset, "INRIAPerson/Train/neg/")
DATA_training_path_pos = os.path.join(
    master_path_to_dataset, "INRIAPerson/train_64x128_H96/pos/")

# data location - testing examples

DATA_testing_path_neg = os.path.join(
    master_path_to_dataset, "INRIAPerson/Test/neg/")
DATA_testing_path_pos = os.path.join(
    master_path_to_dataset, "INRIAPerson/test_64x128_H96/pos/")

# size of the sliding window patch / image patch to be used for classification
# (for larger windows sizes, for example from selective search - resize the
# window to this size before feature descriptor extraction / classification)

DATA_WINDOW_SIZE = [64, 128]

# the maximum left/right, up/down offset to use when generating samples for training
# that are centred around the centre of the image

DATA_WINDOW_OFFSET_FOR_TRAINING_SAMPLES = 3

# number of sample patches to extract from each negative training example

DATA_training_sample_count_neg = 10

# number of sample patches to extract from each positive training example

DATA_training_sample_count_pos = 5

# class names - N.B. ordering of 0, 1 for neg/pos = order of paths

DATA_CLASS_NAMES = {
    "other": 0,
    "pedestrian": 1
}

################################################################################
# settings for HOG approaches

HOG_SVM_PATH = "../Write/"+sys.argv[1]

HOG_SVM_kernel = cv2.ml.SVM_LINEAR  # see opencv manual for other options
HOG_SVM_max_training_iterations = 500  # stop training after max iterations
HOG_SVM_DEGREE = 3
################################################################################

COLORS = [(0, 0, 0),
          (255, 100, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 255)]
