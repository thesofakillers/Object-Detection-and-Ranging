################################################################################

# functionality: functions for multi-scale sliding window (exhaustive) search

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Origin acknowledgements: forked from https://github.com/nextgensparx/PyBOW

################################################################################

import numpy as np
import cv2

################################################################################

# re-size an image with respect to its aspect ratio if needed.
# used in the multi-scale image pyramid approach

def resize_img(img, width=-1, height=-1):
    if height == -1 and width == -1:
        raise TypeError("Invalid arguments. Width or height must be provided.")
    h = img.shape[0]
    w = img.shape[1]
    if height == -1:
        aspect_ratio = float(w) / h
        new_height = int(width / aspect_ratio)
        return cv2.resize(img, (width, new_height))
    elif width == -1:
        aspect_ratio = h / float(w)
        new_width = int(height / aspect_ratio)
        return cv2.resize(img, (new_width, height))

################################################################################

# a very basic approach to produce an image at multi-scales (i.e. variant
# re-sized resolutions)

def pyramid(img, scale=1.5, min_size=(30, 30)):
    # yield the original image
    yield img

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(img.shape[1] / scale)
        img = resize_img(img, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield img

################################################################################

# generate a set of sliding window locations across the image

def sliding_window(image, window_size, step_size=8):
    # slide a window across the image
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            # yield the current window
            window = image[y:y + window_size[1], x:x + window_size[0]]
            if not (window.shape[0] != window_size[1] or window.shape[1] != window_size[0]):
                yield (x, y, window)
