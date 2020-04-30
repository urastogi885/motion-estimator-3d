import cv2
from sys import argv
from utils.data_prep import *

script, dataset_location = argv


if __name__ == '__main__':
    filenames = extract_locations(str(dataset_location))
    for file in filenames:
        img_frame = cv2.imread(file, 0)
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerGR2BGR)
        cv2.imshow("Image", img_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
