import os
import cv2
from sys import argv
from utils.data_prep import *

script, dataset_location, model_location = argv


if __name__ == '__main__':
    INPUT_VIDEO = 'undistorted_input.avi'
    current_dir = os.path.abspath(os.getcwd())[:-4]
    cam_params = read_camera_model(str(model_location))
    filenames = extract_locations(str(dataset_location))
    # Create object to define video format and writer
    video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    # video_output = cv2.VideoWriter(str(output_location), video_format, 30.0, (1280, 960))
    # If input video does not exist, create it
    if not os.path.exists(current_dir + INPUT_VIDEO):
        print('Input video not found...Generating input video...')
        video_input = cv2.VideoWriter('../' + INPUT_VIDEO, video_format, 30.0, (1280, 960))
        for file in filenames:
            img_frame = cv2.imread(file, 0)
            img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerGR2BGR)
            undistorted_img = undistort_image(img_frame, cam_params[-1])
            video_input.write(undistorted_img)
        video_input.release()
        print('Generated')
    else:
        print('Input video found!')
    print('Processing...')
    video_capture = cv2.VideoCapture('../' + INPUT_VIDEO)
    while True:
        img_exists, img = video_capture.read()
        if not img_exists:
            break
        cv2.imshow('Video frame', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    # video_output.release()
    cv2.destroyAllWindows()
