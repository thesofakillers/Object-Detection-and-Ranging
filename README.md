# CompVis_ObjDetection-DistanceRanging

Originally made for Durham University's Department of Computer Science's course _Software, Systems & Applications_ under the sub-module _Computer Vision_, as part of the coursework in 2018/2019.

This project aims to "develop an object detection system that correctly detects one or more types of dynamic objects within the scene in-front of the vehicle and estimates the range (distance in metres) to those objects.", given a set of stereo images taken from a moving car in real world traffic.

A few directories/files are not committed to this repository due to their unwieldy sizes, their availability on the web, or  their proprietary status. These include the training data, which is a cut-down copy of the original INRIA pedestrain data set from <http://pascal.inrialpes.fr/data/human/>. The stereo images themselves, which, due to their large amount, occupy a large amount of space, and are copyright of Durham University, are also part of the uncommitted files.

## Quick Brief

This repository offers two options of object detection.

The first utilizes [HoG descriptors](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) fed into a Support Vector Machine Classifier, utilizing [Selective Search](https://koen.me/research/selectivesearch/) for region proposals. This is where the bulk of the work went into and can detect persons in the images. [This repo](https://github.com/tobybreckon/python-bow-hog-object-detection) had great influence on this section of the work.

The second utilizes a [MaskRCNN](https://arxiv.org/abs/1703.06870) implementation taken from [this repo](https://github.com/matterport/Mask_RCNN), which can detect 80 different classes of objects via weights pre-trained on the [COCO dataset](http://cocodataset.org/#home). Most of the work with regards to this second option was in integrating it into the main script, [detect_and_range.py](Scripts/detect_and_range.py).

Both options utilize the same method for estimating the depth to the detections, namely [SGBM](https://ieeexplore.ieee.org/document/4359315). This section of the work borrows a lot from [this repo](https://github.com/tobybreckon/stereo-disparity).

## Getting Set up

This project was built in [Python 3.5](https://www.python.org/downloads/release/python-350/) and hence will require a version of Python 3.x to work.

It should be noted that if one only wishes to run the SVM implementation, then they only need to ensure that OpenCV 3.4.x is installed. To try out the MaskRCNN implementation, please proceed with the following steps

1.  Ensure that the following modules are installed:
    -   [OpenCV 3.4.x](https://opencv.org/opencv-3-4.html)
    -   [tensorflow-gpu](https://www.tensorflow.org/)
    -   [keras](https://keras.io/)
On Durham University DUDE machines, running `opencv3-4.init` and `tensorflow.init` before proceeding with the rest of the installation should cover it.

2.  Then `cd` into [Scripts/Deep/](Scripts/Deep/) and run `pip3 install --user -r requirements.txt`.
3.  In the same directory, run `python3 setup.py install --user`.
4.  Finally, in any directory, run `pip3 install --user pycocotools`.

## Usage

Before running any script after setup is complete, ensure you are in the [Scripts/](Scripts/) directory.

### Object Detection

This is done via the main script, [detect_and_range.py](Scripts/detect_and_range.py). To use it, type `python3 detect_and_range <model> <start_image>` into the terminal.

-   Here, `<model>` is to be replaced either by:
    -   `SVM` if one wishes to utilize the SVM implementation
    -   `MRCNN` if one wishes to utilize the MaskRCNN implementation
-   Furthermore, `<start_image>` is to be replaced with:
    -   `start` if one wishes to cycle from the start of the images directory
    -   Or simply the filename of the desired image to start from.  

#### Additional Notes

-   Set the path to the directory containing the stereo images in line 44 of [detect_and_range.py](Scripts/detect_and_range.py) under the variable name `master_path_to_dataset`
-   For SVM, there are different trained models saved in [Write/](Write/). Currently what we consider the best from our training is set for usage. To change the model to be used, set it in line 66 of [params.py](Scripts/SVM/params.py) under the variable name `HOG_SVM_PATH_SAVED`.

### Training
Custom training can be performed for the SVM implementation.

This is done via [hog_train.py](Scripts/SVM/hog_train.py). To train a model, set the training parameters in [params.py](Scripts/SVM/params.py) and then run `python3 SVM/hog_train.py` in the terminal. Ensure to change the `HOG_SVM_PATH_TRAIN` variable after successive trainings to avoid overwriting.
