import cv2
import os
import numpy as np
import bottle_functions as bf
import bottle_augments as ba
import datetime
from tqdm import tqdm_gui
import random

# Parameters for pre-processing data
iterations = 1  # How many rounds of random iterations per image (besides the original)
target_size = 100  # target image size in % of original image
make_blur = 10  # probability of blur
set_blur = 5  # max blur to apply
make_rotate = 0  # probability of blur
set_rotate = 1  # degrees of rotation from 0
make_contrast = 0  # probability of contrast adjustment
set_contrast = 8  # amount of adjustment in %
slice_size = 10  # What is the size of the final image in percents

test_set = 15  # how many % of test set data (roughly
validate_set = 30  # validate set value - test_set

log = None  # None for no logging, and 1 for logging
number = 10
start_time = datetime.datetime.now()
dataset_folder = "Dataset2"
input_folder = ['OK', 'NOK']
f = open("output_log.txt", "w")
f.write("number;filename;classification\n")
count_train = 0
count_test = 0
count_validate = 0

for folder in range(len(input_folder)):
    output = dataset_folder + "/" + input_folder[folder] + "/"
    images = [img for img in os.listdir(output) if img.endswith(".JPG")]
    pbar = tqdm_gui(total=len(images) * (iterations + 1), desc='processing ' + str(input_folder[folder]))

    for image in images:
        frame = cv2.imread(os.path.join(output, image))
        if make_rotate > random.randint(0, 100):
            frame = bf.rotate(frame, random.randint(set_rotate * -1, set_rotate), center=None, scale=1.0)
            # print("frame number " + str(number) + " / " + image + " done")

        if make_contrast > random.randint(0, 100):
            frame = bf.adjust_gamma(frame, gamma=random.randint(100 - set_contrast, 100 + set_contrast) / 100)

        if make_blur > random.randint(0, 100):
            blur = random.randint(1, set_blur)
            frame = cv2.blur(frame, (blur, blur))

        array = bf.create_boxes(frame, 256, 1, 5)
        for z in range(0, 5):
            w = np.size(frame, 1)
            h = np.size(frame, 0)
            # cv2.imwrite("output/" + str(input_folder[folder]) + "/" + "org_" + str(image), frame)
            f.write(str(number) + ";" + str(image) + ";" + str(input_folder[folder]) + "\n")
            number += 1
            pbar.update(1)
            for i in range(iterations):
                result = array[z].copy()
                # Resize image
                if target_size < 100:
                    result = bf.resize(result, width=None, height=int(h * (target_size / 100)), inter=cv2.INTER_AREA,
                                       debug=None)

                if input_folder[folder] == 'NOK':
                    defect = random.randint(1, 3)
                    if defect == 1:
                        result = ba.label_horizontal_fold(result)
                    if defect == 2:
                        result = ba.label_wringle(result)
                    if defect == 3:
                        result = ba.label_part_off(result)

                # random transformations
                """

                """

                # Write Output and end


                roll = random.randint(0, 100)
                if roll < test_set:
                    output_folder = dataset_folder + "/TEST/"
                    count_test += 1
                elif roll < validate_set:
                    output_folder = dataset_folder + "/VALIDATE/"
                    count_validate += 1
                else:
                    output_folder = dataset_folder + "/TRAIN/"
                    count_train += 1

                #output_folder = "Dataset2/"
                cv2.imwrite("output/" + str(output_folder) + str(input_folder[folder]) + "/i" + str(i) + "_b" + str(
                    z) + "_" + str(image),
                            result)
                f.write(str(number) + ";" + str(image) + ";" + str(input_folder[folder]) + "\n")
                number += 1
                pbar.update(1)
        pbar.close()

pbar.close()
f.close()
print("train=" + str(count_train) + " test=" + str(count_test) + " validate=" + str(count_validate))
end_time = datetime.datetime.now()
print("execution time: " + str(end_time - start_time))
