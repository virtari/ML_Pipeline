# This file includes all the functions that are used for the bottle augmentation.

# Import the necessary packages
import numpy as np
import cv2
import datetime
import random


# Create a fold on label
# This function creates a fold like line to label, that goes horizontaly across the label
# Image of the label is given as an input
def label_horizontal_fold(img, amount=None, debug=None):
    overlay = img.copy()
    opacity = 0.2  # level of the tansparency
    height = img.shape[0]
    width = img.shape[1]
    thickness = random.randint(2, 3)

    start = random.randint(int(height * 0.2), int(height * 0.9))
    end = start - random.randint(int(height * 0.05 * -1), int(height * 0.2))
    overlay = cv2.line(overlay, (0, start), (width, end), (255, 255, 255), thickness, cv2.LINE_AA)
    overlay = cv2.line(overlay, (0, start + 1), (width, end + 1), (100, 100, 100), thickness - 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    return img


# Create a scratch in the cork
# This function creates a vertical scratch in the cork
def cork_vertical_scratch(img):
    overlay = img.copy()
    opacity = 0.7  # level of the tansparency
    height = img.shape[0]
    width = img.shape[1]

    start = random.randint(int(width * 0.3), int(width * 0.8))
    end = start

    overlay = cv2.line(overlay, (start, int(height * 0.4)), (end, int(width * 0.49)), (200, 200, 200), 1, cv2.LINE_AA)
    overlay = cv2.line(overlay, (start + 1, int(height * 0.4)), (end + 1, int(width * 0.49)), (230, 230, 230), 1,
                       cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    return img


def label_wringle(img):
    overlay = img.copy()
    opacity = 0.2  # level of the tansparency
    height = img.shape[0]
    width = img.shape[1]
    length = random.randint(int(width * 0.1), int(width * 0.3))
    thickness = random.randint(2, 3)
    startx = random.randint(0, int(width / 2))
    i = 0
    start = random.randint(int(height * 0.2), int(height * 0.9))
    end = start + random.randint(-7, 7)

    for i in range(length):
        img[start, length - i - 1] = img[start - 1, length - i]
        img[start - 1, length - i - 1] = img[start - 2, length - i]
        img[start - 2, length - i - 1] = img[start - 3, length - i]

    overlay = cv2.line(overlay, (startx, start - 1), (startx + length, end - 1), (255, 255, 255), thickness,
                       cv2.LINE_AA)
    overlay = cv2.line(overlay, (startx, start), (startx + length, end), (255, 255, 255), thickness + 1, cv2.LINE_AA)
    overlay = cv2.line(overlay, (startx, start + 1), (startx + length, end + 1), (150, 150, 150), thickness,
                       cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    return img


# Take part of the label off.
def label_tear(img, edgex=None, edgey=None, orientation=0):
    overlay1 = img.copy()
    overlay2 = img.copy()
    overlay3 = img.copy()
    alignment = orientation
    opacity1 = 0.7  # level of the tansparency for Layer1
    opacity2 = 0.6  # level of the tansparency for Layer2
    opacity3 = 0.5  # level of the tansparency for LÖayer3
    h = img.shape[0]
    w = img.shape[1]
    dice = random.randint(1, 4)
    if dice == 1:
        alignment = 0
    if dice == 2:
        alignment = 180
    if dice == 3:
        alignment = 0
    else:
        alignment = 180

    # Add some randomness
    x = random.randint(int(w * 0.1), int(w * 0.9))
    y = random.randint(int(h * 0.1), int(h * 0.9))
    sizex = random.randint(int(w * 0.015), int(w * 0.08))
    sizey = random.randint(int(h * 0.03), int(h * 0.08))

    # Layer 1
    img = cv2.ellipse(img, (x, y + 1), (sizex, sizey), alignment, 0, 180, (0, 0, 0), -1)
    cv2.addWeighted(overlay1, opacity1, overlay2, 1 - opacity1, 0, overlay2)
    # Layer 2
    overlay2 = cv2.ellipse(overlay2, (x, y), (sizex, sizey), alignment, 0, 180, (255, 255, 255), -1)
    cv2.addWeighted(overlay2, opacity2, overlay3, 1 - opacity2, 0, overlay3)
    # Layer 3
    overlay3 = cv2.ellipse(overlay3, (x, y), (sizex - 1, sizey - 3), alignment, 0, 180, (24, 88, 172), -1)
    cv2.addWeighted(overlay3, opacity3, img, 1 - opacity3, 0, img)

    return img
