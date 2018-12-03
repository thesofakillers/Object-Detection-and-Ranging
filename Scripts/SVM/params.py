"""
functionality: parameter settings for detection algorithm training/testing (Mostly for SVM)

This version: Modified version on work by (c) 2018 Toby Breckon,
Dept. Computer Science, Durham University, UK
License: MIT License

Origin acknowledgements: forked from https://github.com/nextgensparx/PyBOW
"""

import cv2
import os
import sys

################################################################################
# settings for datasets in general (path to directory containing Train and Test)
master_path_to_dataset = "../Data/Training"

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

# size of the image patch to be used for classification
DATA_WINDOW_SIZE = [64, 128]

# the maximum left/right, up/down offset to use when generating samples for training
# that are centred around the centre of the image
DATA_WINDOW_OFFSET_FOR_TRAINING_SAMPLES = 3

# number of sample patches to extract from each negative training example
DATA_training_sample_count_neg = 15

# number of sample patches to extract from each positive training example
DATA_training_sample_count_pos = 5

# class names - N.B. ordering of 0, 1 for neg/pos = order of paths
DATA_CLASS_NAMES = {
    "other": 0,
    "person": 1
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
HOG_SVM_PATH_SAVED = "../Write/svm_hog_3to1_RBF.xml" # for testing/usage
HOG_SVM_PATH_TRAIN = "../Write/train.xml" # for saving trained files
HOG_SVM_kernel = cv2.ml.SVM_RBF  # kernel type
HOG_SVM_max_training_iterations = 500  # stop training after max iterations
HOG_SVM_DEGREE = 3 #if poly kernel used
    #</section>

#</section>
