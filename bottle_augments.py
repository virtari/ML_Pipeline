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
    end = start - random.randint(int(height * 0.05), int(height * 0.1))
    overlay = cv2.line(overlay, (0, start), (width, end), (255, 255, 255), thickness, cv2.LINE_AA)
    overlay = cv2.line(overlay, (0, start + 1), (width, end + 1), (100, 100, 100), thickness-1, cv2.LINE_AA)
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
    length = random.randint(15, 25)
    thickness = random.randint(2,3)
    i = 0
    start = random.randint(int(height * 0.2), int(height * 0.9))
    end = start + 5

    for i in range(length):
        img[start, length - i - 1] = img[start - 1, length - i]
        img[start - 1, length - i - 1] = img[start - 2, length - i]
        img[start - 2, length - i - 1] = img[start - 3, length - i]

    overlay = cv2.line(overlay, (0, start - 1), (length, end - 1), (255, 255, 255), thickness, cv2.LINE_AA)
    overlay = cv2.line(overlay, (0, start), (length, end), (255, 255, 255), thickness+1, cv2.LINE_AA)
    overlay = cv2.line(overlay, (0, start + 1), (length, end + 1), (150, 150, 150), thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    return img


def label_part_off(img):
    overlay = img.copy()
    opacity = 0.2  # level of the transparency
    height = img.shape[0]
    width = img.shape[1]
    length = int(height / 4)
    start = random.randint(int(width * 0.1), int(width * 0.9))
    position = random.randint(0, 1)

    for i in range(length):
        if position == 0:
            start_point = img[0, start]
            img = cv2.line(img, (start+i, 0), (start, length-i), (int(start_point[0]), int(start_point[1]), int(start_point[2])), 2)
        if position == 1:
            start_point = img[height-1, start]
            img = cv2.line(img, (start - i, height-1), (start, height - length - i),
                           (int(start_point[0]), int(start_point[1]), int(start_point[2])), 2)
    return img
