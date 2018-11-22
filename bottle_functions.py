# This file includes all the functions that are needed for the bottle quality inspection. It is split to 3 stages: 1) Framegrabbing 2)Pre-processing 3)Quality inspection

# Import the necessary packages
import numpy as np
import cv2
import datetime


# STAGE 1: Camera and frame grabbing

# STAGE 2: Pre-Processing

# - Detect the edge of the bottle for cropping. side is string "left" or "right" to define side to detect.
#  Margin for leaving space to edges (pixels)
def detect_edge(image, side, margin=None, debug=None):
    timer = datetime.datetime.now()  # Log the start time
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # defines the % of height where the edge is detected. 0.9 is 90% from top of image
    detect_height = np.size(image, 0) * 0.8
    max_width = np.size(image, 1)
    color = 0  # initialize color
    variance = 0.1  # how many percent variation from detected background color
    treshold = int(
        image[int(detect_height), int(max_width - 3)] * (1 - variance))  # detect background color and add variance
    i = 1
    if side == str.lower("right"):  # Detect right edge
        while i in range(max_width):
            color = image[int(detect_height), int(max_width - i)]
            if color <= treshold:
                if debug is not None:
                    print(str(side) + " edge found at: " + str(i) + " / " + str(max_width) + " color: " + str(
                        color) + " / " + str(treshold) + " Time: " + str(datetime.datetime.now() - timer))
                return max_width - i + margin
            i += 1
    elif side == str.lower("left"):  # Detect left edge
        while i in range(max_width):
            color = image[int(detect_height), 0 + i]
            if color <= treshold:
                if debug is not None:
                    print(str(side) + " edge found at: " + str(i) + " / " + str(max_width) + " color: " + str(
                        color) + " / " + str(treshold) + " Time: " + str(datetime.datetime.now() - timer))
                return i - margin
            i += 1
    else:
        print("Please input either Left or Right as a side for detection")


# Create a array of images from input image
def create_boxes(img, size=128, dimx=2, dimy=10):
    w = np.size(img, 1)  # get width
    h = np.size(img, 0)  # get height
    if (size * dimx) >= w and (size * dimy) >= h:  # check if input image is possible to split in boxes
        print("Image too small. Only :" + str(w) + " x " + str(h) + " but should be: " + str(size * dimx) + " x " + str(
            size * dimy))
        return None
    else:
        loose_w = int((w - (size * dimx)) / 2)  # calculate the extra pixels to crop from left side
        cropped_img = crop(img, 0, h, loose_w, loose_w + (size * dimx))  # crop the image before boxing
        M = np.size(cropped_img, 0) // dimy
        N = np.size(cropped_img, 1) // dimx
        array = [cropped_img[x:x + M, y:y + N] for x in range(0, cropped_img.shape[0], M) for y in
                 range(0, cropped_img.shape[1], N)]
    return array


# - Crop the bottle image
def crop(image, top, bottom, left, right, debug=None):
    timer = datetime.datetime.now()  # Log the start time

    if debug is not None:
        print("crop done in: " + str(datetime.datetime.now() - timer))
    return image[int(top):int(bottom), int(left):int(right)]


# - Resize a image
def resize(image, width=None, height=None, inter=cv2.INTER_AREA, debug=None):
    timer = datetime.datetime.now()  # Log the start time
    dim = None  # initialize the dimensions of the image to be resized and
    (h, w) = image.shape[:2]  # grab the image size

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        r = height / float(h)  # calculate the ratio of the height and construct the dimensions
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        r = width / float(w)  # calculate the ratio of the width and construct the dimensions
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)  # resize the image
    if debug is not None:  # debug message
        print("resize done in: " + str(datetime.datetime.now() - timer))

    return resized  # return the resized image


# - create templates for label and cork matching
def create_template(image, side, left_edge, right_edge, debug=None):
    w = np.size(image, 1)
    h = np.size(image, 0)
    template = 0
    if side == "label":
        template = crop(image, h * 0.42, h * 0.87, left_edge, right_edge)
    if side == "cork":
        template = crop(image, h * 0.02, h * 0.35, left_edge * 1.2, right_edge * 0.87)
    if debug is not None:  # debug message
        print("temp: w:" + str(w) + " / h:" + str(h))
        print("new: w:" + str(np.size(template, 1)) + " / h:" + str(np.size(template, 0)))
    return template


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def rotate(image, angle, center=None, scale=1.0):
    # Grab the dimensions of the image
    (h, w) = image.shape[:2]

    # If the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
    return rotated


def create_hue(image, amount=5, debug=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h += amount
    s += amount
    v += amount
    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    if debug is not None:  # debug message
        print("temp: w:" + str(w) + " / h:" + str(h))
        print("new: w:" + str(np.size(template, 1)) + " / h:" + str(np.size(template, 0)))
    return image


# STAGE 3: Quality inspection

# - Match the front label with OpenCV Template matching
def match_template(image, part, template, treshold, debug=None):
    timer = datetime.datetime.now()  # Log the start time
    res = cv2.matchTemplate(image, template,
                            method=cv2.TM_CCOEFF_NORMED)  # Perform match operations with selected method
    loc = np.where(res >= treshold)  # Store the coordinates of matched area in a numpy array
    for pt in zip(*loc[::-1]):
        # print(str(part) + " found!")
        if debug is not None:  # debug message
            print("template matching done in: " + str(datetime.datetime.now() - timer))
        return "Found!"

    if debug is not None:  # debug message
        print("label matching ended in: " + str(datetime.datetime.now() - timer))
    return str(part) + " NOT found"
