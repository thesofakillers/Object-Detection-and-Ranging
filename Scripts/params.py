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

DATA_training_sample_count_neg = 45

# number of sample patches to extract from each positive training example

DATA_training_sample_count_pos = 15

# class names - N.B. ordering of 0, 1 for neg/pos = order of paths

DATA_CLASS_NAMES = {
    "other": 0,
    "pedestrian": 1
}

#<section>~~~~~~~~~~~~~~~~~~~~~~~~~~~HoG Settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #<section>~~~~~~~~~~~~~~~~~HoG Descriptor Settings~~~~~~~~~~~~~~~~~~~~~~~~~~
HOG_DESC_winSize =(64, 128) #window Size
HOG_DESC_blockSize = (16,16) # block size
HOG_DESC_blockStride = (8,8) # block stride
HOG_DESC_cellSize = (8,8) # cell size
HOG_DESC_nbins = 9 # number of bins
HOG_DESC_derivAperture = 1 # not documented
HOG_DESC_winSigma = -1 #gaussian window smoothing parameter
HOG_DESC_histogramNormType = 0 #equivalent to L2
HOG_DESC_L2HysThreshold = 0.2 # L2-Hys normalization method shrinkage.
HOG_DESC_gammaCorrection = True # whether or not to employ gamma correction
    #</section>

    #<section>~~~~~~~~~~~~~~~~~~~HoG SVM Settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HOG_SVM_PATH = "../Write/"+sys.argv[1]

HOG_SVM_kernel = cv2.ml.SVM_RBF  # see opencv manual for other options
HOG_SVM_max_training_iterations = 500  # stop training after max iterations
HOG_SVM_DEGREE = 3
    #</section>

#</section>

COLORS = [(0, 0, 0),
          (255, 100, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 255)]
