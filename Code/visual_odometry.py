import os
import cv2
from sys import argv
from matplotlib.pyplot import scatter, show
from utils.data_prep import *
from utils.motion_estimator import MotionEstimator

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
    # Begin basic pipeline
    print('Processing...')
    # Extract paths of all undistorted images
    images = extract_locations(save_path)
    # Create object of motion estimator class
    motion_estimator = MotionEstimator((cam_params[0], cam_params[1]), (cam_params[2], cam_params[3]))
    last_h_mat = motion_estimator.original_h
    for i in range(len(images) - 1):
        # Read current and next frame
        current_img = cv2.imread(images[i], cv2.COLOR_BGR2GRAY)
        next_img = cv2.imread(images[i+1], cv2.COLOR_BGR2GRAY)
        # Crop current and next frame to retain only ROI
        current_img = current_img[150:650, 0:1280]
        next_img = next_img[150:650, 0:1280]
        # Extract key features using SIFT
        features_curr, features_next = motion_estimator.extract_features(current_img, next_img)
        # Get the final fundamental matrix and inliers for current and next image frame
        fundamental_mat, inliers_curr, inliers_next = motion_estimator.ransac_with_8_point(features_curr, features_next)
        # Evaluate the essential matrix from the fundamental matrix
        essential_mat = motion_estimator.calc_essential_matrix(fundamental_mat)
        # Get all possible camera poses using the essential matrix
        cam_centers, rotation_mats = motion_estimator.estimate_camera_pose(essential_mat)
        # Get the final camera pose
        final_cam_center, final_r_mat = motion_estimator.disambiguate_camera_pose(cam_centers, rotation_mats,
                                                                                  inliers_curr, inliers_next)
        # Get the homogeneous transform using the final camera pose
        last_h_mat = last_h_mat @ motion_estimator.get_homogeneous_matrix(final_r_mat, final_cam_center)
        transform = last_h_mat @ motion_estimator.h_mat_last_row.T
        print(transform[0], -transform[2])
        scatter(transform[0], -transform[2], color='r')

        # cv2.imshow('Frame', curr_img)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
    show()
    print('Done!')
