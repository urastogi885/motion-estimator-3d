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
    # Begin basic pipeline
    print('Processing...')
    images = extract_locations(save_path)
    for i in range(len(images) - 1):
        # Read current and next frame
        curr_img = cv2.imread(images[i], cv2.COLOR_BGR2GRAY)
        next_img = cv2.imread(images[i+1], cv2.COLOR_BGR2GRAY)
        # Crop current and next frame to retain only ROI
        curr_img = curr_img[150:650, 0:1280]
        next_img = next_img[150:650, 0:1280]
        # Create SIFT detector object
        sift = cv2.xfeatures2d.SIFT_create()
        # Get key-points and descriptors for both frames
        key_points_curr, descriptor_curr = sift.detectAndCompute(curr_img, None)
        key_points_next, descriptor_next = sift.detectAndCompute(next_img, None)
        # Define parameters for Flann-based matcher
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        # Create object of Flann-based matcher
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # Get matches between the current and next frame
        matches = matcher.knnMatch(descriptor_curr, descriptor_next, k=2)
        # Define empty list to store features of current and next frame
        features_curr, features_next = [], []
        # Employ ratio test
        for _, (m, n) in enumerate(matches):
            if m.distance < 0.5 * n.distance:
                features_curr.append(key_points_curr[m.queryIdx].pt)
                features_next.append(key_points_next[m.trainIdx].pt)
        # cv2.imshow('Frame', curr_img)
        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
    print('Done!')
