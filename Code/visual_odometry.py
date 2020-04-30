import cv2
from sys import argv
from utils.data_prep import *

script, dataset_location, model_location, output_location = argv


if __name__ == '__main__':
    video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_output = cv2.VideoWriter(str(output_location), video_format, 30.0, (1280, 960))
    cam_params = read_camera_model(str(model_location))
    filenames = extract_locations(str(dataset_location))
    for file in filenames:
        img_frame = cv2.imread(file, 0)
        img_frame = cv2.cvtColor(img_frame, cv2.COLOR_BayerGR2BGR)
        undistorted_img = undistort_image(img_frame, cam_params[-1])
        video_output.write(undistorted_img)
    video_output.release()
    cv2.destroyAllWindows()
