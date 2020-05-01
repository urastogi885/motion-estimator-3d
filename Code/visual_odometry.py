import os
import cv2
from sys import argv
from utils.data_prep import *

script, dataset_location, model_location = argv


if __name__ == '__main__':
    INPUT_IMAGES = 'undistorted_images/'
    save_path = '../' + INPUT_IMAGES
    current_dir = os.path.abspath(os.getcwd())[:-4]
    cam_params = read_camera_model(str(model_location))
    filenames = extract_locations(str(dataset_location))
    # Create object to define video format and writer
    video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # video_output = cv2.VideoWriter(str(output_location), video_format, 30.0, (1280, 960))
    # Begin pre-processing pipeline
    # If input video does not exist, create it
    if not os.path.exists(current_dir + INPUT_IMAGES):
        print('Input images not found...Generating input images...')
        os.makedirs(current_dir + INPUT_IMAGES[:-1])
        count = 1399381444704913
        for file in filenames:
            img_frame = cv2.imread(file, 0)
            img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerGR2BGR)
            undistorted_img = undistort_image(img_frame, cam_params[-1])
            cv2.imwrite(save_path + str(count) + '.png', undistorted_img)
            count += 1
        print('Generated')
    else:
        print('Input video found!')
    print('Processing...')
    images = extract_locations(save_path)
    for image in images:
        img = cv2.imread(image)
        cv2.imshow('Frame', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    print('Done!')
