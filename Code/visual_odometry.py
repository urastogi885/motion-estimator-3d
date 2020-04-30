import cv2
from sys import argv
from utils.data_prep import *

script, dataset_location, model_location = argv


if __name__ == '__main__':
    cam_params = read_camera_model(str(model_location))
    filenames = extract_locations(str(dataset_location))
    for file in filenames:
        img_frame = cv2.imread(file, 0)
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerGR2BGR)
        undistorted_img = undistort_image(img_frame, cam_params[-1])
        cv2.imshow("Image", undistorted_img)
        key = cv2.waitKey(1)
        if key == 27:
            break
