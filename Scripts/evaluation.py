"""
Boilerplate code to copy paste into various versions (to be accessed via git tag)
for evaluating.
"""

# <section>~~~~~~~~~~~~~~~~~~~~~~~~~~Timing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Repeat and record this at least 5 times for each version and record

Might actually just squeeze this into master branch anyway
"""
from utils import *
import cv2
time_to_setup = cv2.getTickCount()
# Imports
# Functions
# Various settings etc
# just before for loop
print_duration(time_to_setup)

np.random.shuffle(left_file_list) # to get random pictures

for filename_left in left_file_list:
    time_per_frame = cv2.getTickCount()
    # .
    # .
    #   comment out cv2.imshow
    # . comment out cv2.keywait
    print_duration(time_per_frame)
    # . END OF 1 ITERATION
# </section>  End of Timing


# <section>~~~~~~~~~~~~~~~~~~~~~~~~~~Accuracy~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
Loops through 63 random pics and saves them to a folder in the desktop with
the number of detections in the file title. A Human is then to go through these
63 pictures and record how many of them are false positives
"""
np.shuffle(left_file_list) # to get random pictures
counter = 0 #for titiling the images we write
for filename_left in left_file_list[::23]: #every 23 pictures (total 63 pics)
    print("{}/{}".format(str(counter), str(63))) # to see progress
    # do the usual detecting
    # .
    # .
    # .
    # instead of imshow("detected objects", imgL)
    n_of_detections = len(detection_classes)
    cv2.imwrite("~/Desktop/CV_Evaluation/{}counter_{}det.png".format(str(counter), str(n_of_detections)), imgL)
    counter += 1
# </section>  End of Accuracy
