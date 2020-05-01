import cv2
import numpy as np
from random import randint


def get_8_points(len_features):
    """
    Function to get 8 indices of random points
    Implements the 8-point algorithm
    :param len_features: total no. of features retrieved from feature extractor
    :return: a list containing indices of 8 random feature points
    """
    eight_points = []
    # Iterate until 8 points are not stored in the list
    while len(eight_points) != 8:
        # Get a index of a random point
        index = randint(0, len_features - 1)
        # Add only distinct points to the list
        if index not in eight_points:
            eight_points.append(index)
    return eight_points


class MotionEstimator:
    """
    A class to estimate 3D motion of a camera
    """
    def __init__(self, focal_lengths, principal_pts):
        # Define K-matrix using camera parameters
        self.k_mat = np.array([[focal_lengths[0], 0, principal_pts[0]],
                               [0, focal_lengths[1], principal_pts[1]],
                               [0, 0, 1]])
        # Get inverse of K-matrix
        self.k_mat_inv = np.linalg.inv(self.k_mat)
        # Define no. of iterations for RANSAC
        self.iterations = 50
        # Define threshold for outlier rejection in RANSAC
        self.epsilon = 0.01
        # Create SIFT detector object
        self.sift = cv2.xfeatures2d.SIFT_create()
        # Define parameters for Flann-based matcher
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        # Create object of Flann-based matcher
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_features(self, curr_img, next_img):
        """
        Method to extract matching key features from 2 images
        :param curr_img: current image frame from the dataset
        :param next_img: next image frame from the dataset
        :return: a tuple of lists containing matching key features from both images
        """
        # Get key-points and descriptors for both frames
        key_points_curr, descriptor_curr = self.sift.detectAndCompute(curr_img, None)
        key_points_next, descriptor_next = self.sift.detectAndCompute(next_img, None)
        # Get matches between the current and next frame
        matches = self.matcher.knnMatch(descriptor_curr, descriptor_next, k=2)
        # Define empty list to store features of current and next frame
        features_curr, features_next = [], []
        # Employ ratio test
        for _, (m, n) in enumerate(matches):
            if m.distance < 0.5 * n.distance:
                features_curr.append(key_points_curr[m.queryIdx].pt)
                features_next.append(key_points_next[m.trainIdx].pt)
        return features_curr, features_next

    def ransac_with_8_point(self, features_curr, features_next):
        """
        Method to ransac on features of current and next image frame using 8-point algorithm
        :param features_curr: a list of feature points in the current image frame
        :param features_next: a list of feature points in the next image frame
        :return: a tuple containing the 3x3 fundamental matrix, inliers in the current and next image frame
        """
        count_inliers = 0
        final_fundamental_mat = np.zeros((3, 3))
        # Define list to store inliers from current and next image frame
        inliers_curr, inliers_next = [], []
        for _ in range(self.iterations):
            count = 0
            good_features_curr, good_features_next = [], []
            temp_features_curr, temp_features_next = [], []
            # Get 8 random points and extract features at those points
            for pt in get_8_points(len(features_curr)):
                good_features_curr.append([features_curr[pt][0], features_curr[pt][1]])
                good_features_next.append([features_next[pt][0], features_next[pt][1]])
            # Calculate fundamental matrix using these 8 features
            fundamental_mat = self.calc_fundamental_matrix(good_features_curr, good_features_next)
            # Employ outlier rejection
            for i in range(len(features_curr)):
                if self.check_fundamental_matrix(features_curr[i], features_next[i], fundamental_mat):
                    count += 1
                    temp_features_curr.append(features_curr[i])
                    temp_features_next.append(features_next[i])
            if count > count_inliers:
                count_inliers = count
                final_fundamental_mat = fundamental_mat
                inliers_curr, inliers_next = temp_features_curr, temp_features_next
        return final_fundamental_mat, inliers_curr, inliers_next


    @staticmethod
    def calc_fundamental_matrix(features_curr, features_next):
        """
        Method to estimate the fundamental matrix using 8 points
        :param features_curr: a list of 8 feature points in the current image frame
        :param features_next: a list of 8 feature points in the next image frame
        :return: a 3x3 fundamental matrix
        """
        a_mat = np.empty((8, 9))
        # Iterate over all features
        for i in range(len(features_curr)):
            # Get positions of features in current and next frame
            x_curr, y_curr = features_curr[i][0], features_curr[i][1]
            x_next, y_next = features_next[i][0], features_next[i][1]
            # Fill ith column of A matrix
            a_mat[i] = np.array([x_next * x_curr, x_next * y_curr, x_next,
                                 y_next * x_curr, y_next * y_curr, y_next,
                                 x_curr, y_curr, 1])
        # Get SVD of A matrix
        _, _, v = np.linalg.svd(a_mat, full_matrices=True)
        # Get SVD of last column of V matrix
        u, s, v_new = np.linalg.svd(v[-1].reshape(3, 3))
        # Restrain fundamental matrix to a rank of 2
        s_new = np.array([[s[0], 0, 0],
                          [0, s[1], 0],
                          [0, 0, 0]])
        f_mat = u @ s_new @ v_new
        return f_mat

    def check_fundamental_matrix(self, feature_curr, feature_next, fundamental_mat):
        """
        Method to calculate transpose(x2).F.x1 and check if it satisfies the desired threshold
        :param feature_curr: a tuple of feature in current image frame
        :param feature_next: a tuple of corresponding feature in next image frame
        :param fundamental_mat: a 3x3 array of fundamental matrix
        :return: true if estimate is less than threshold
        """
        # Get transpose of current features
        fc_new = np.array([feature_curr[0], feature_curr[1], 1]).T
        fn_new = np.array([feature_next[0], feature_next[1], 1])
        # Estimate transpose(x2).F.x1
        est = abs(np.squeeze(np.matmul(np.matmul(fn_new, fundamental_mat), fc_new)))
        return est < self.epsilon

    def calc_essential_matrix(self, fundamental_mat):
        """
        Method to calculate the essential matrix
        :param fundamental_mat: a numpy 3x3 array of fundamental matrix
        :return: a numpy 3x3 array of essential matrix
        """
        temp = np.matmul(np.matmul(self.k_mat.T, fundamental_mat), self.k_mat)
        u, _, v = np.linalg.svd(temp, full_matrices=True)
        sigma = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        essential_mat = np.matmul(np.matmul(u, sigma), v)
        return essential_mat

    @staticmethod
    def estimate_camera_pose(essential_mat):
        """
        Estimate the pose of the camera
        :param essential_mat: a numpy 3x3 array of essential matrix
        :return: a tuple containing camera centers and
        """
        camera_centers, rotation_matrices = [], []
        u, _, v = np.linalg.svd(essential_mat)
        w_mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        # Get the 4 pose configurations of the camera
        for i in range(4):
            # Evaluate center of camera for camera pose
            cam_center = u[:, 2]
            # Check if additive inverse needs to be taken
            if i % 2 == 1:
                cam_center = -cam_center
            # Evaluate rotation matrix for camera pose
            # Check whether transpose of W matrix needs to be taken
            if i < 2:
                rotation_mat = u @ w_mat @ v
            else:
                rotation_mat = u @ w_mat.T @ v
            # Check for negative determinant condition
            if np.linalg.det(rotation_mat) < 0:
                cam_center, rotation_mat = -cam_center, -rotation_mat
            camera_centers.append(cam_center)
            rotation_matrices.append(rotation_mat)
        return camera_centers, rotation_matrices
