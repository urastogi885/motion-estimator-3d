from sys import argv
import matplotlib.pyplot as plt
from utils.data_prep import *
from utils.motion_estimator import MotionEstimator

"""
:param dataset_location: location of the dataset relative to the project directory
:param model_location: location of the model text files relative to the project directory
"""
script, dataset_location, model_location = argv


if __name__ == '__main__':
    INPUT_IMAGES = 'undistorted_images/'
    save_path = '../' + INPUT_IMAGES
    cam_params = read_camera_model(str(model_location))
    # Begin pre-processing pipeline
    generate_undistorted_images(str(dataset_location), save_path, cam_params[-1])
    # Begin basic pipeline
    print('Processing...')
    # Extract paths of all undistorted images
    images = extract_locations(save_path)
    # Create object of motion estimator class
    motion_estimator = MotionEstimator((cam_params[0], cam_params[1]), (cam_params[2], cam_params[3]))
    # Get homogeneous transform
    last_h_mat = motion_estimator.original_h
    # Save copy of previous homogeneous transform
    h_mat_copy = last_h_mat.copy()
    origin = np.array([[0, 0, 0, 1]]).T
    for i in range(18, len(images) - 1):
        # Read current and next frame
        current_img = cv2.imread(images[i], cv2.COLOR_BGR2GRAY)
        next_img = cv2.imread(images[i+1], cv2.COLOR_BGR2GRAY)
        # Crop current and next frame to retain only ROI
        current_img = current_img[150:650, 0:1280]
        next_img = next_img[150:650, 0:1280]
        # Extract key features using ORB
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
        # Retrieve position of camera from homogeneous matrix
        transform = last_h_mat @ origin
        # Plot position of camera
        plt.scatter(transform[0][0], -transform[2][0], color='r')
        # Begin pipeline using opencv in-built methods
        # Convert features of current and next frames into numpy arrays
        f_curr, f_next = np.array(features_curr), np.array(features_next)
        # Evaluate the essential matrix using in-built opencv function
        e_mat, mask = cv2.findEssentialMat(f_curr, f_next, motion_estimator.k_mat)
        f_curr, f_next = f_curr[mask.ravel() == 1], f_next[mask.ravel() == 1]
        # Get final camera pose using in-built opencv function
        _, r_mat, t_mat, _ = cv2.recoverPose(e_mat, f_curr, f_next, motion_estimator.k_mat)
        # Get homogeneous matrix using rotation and translation matrices
        h_mat_copy = h_mat_copy @ motion_estimator.get_homogeneous_matrix(r_mat, t_mat)
        # Retrieve position of camera from homogeneous matrix
        pose_in_built = h_mat_copy @ origin
        # Plot position of camera recovered using opencv in-built functions
        plt.scatter(pose_in_built[0][0], -pose_in_built[2][0], color='b')
    # Add labels
    plt.title('Camera Pose')
    plt.ylabel('x-coordinate')
    plt.xlabel('z-coordinate')
    # Show final plot
    plt.show()
    print('Done!')
