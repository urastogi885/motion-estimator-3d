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
        self.k_mat = np.array([[focal_lengths[0], 0, principal_pts[0]],
                               [0, focal_lengths[1], principal_pts[1]],
                               [0, 0, 1]])
        self.k_mat_inv = np.linalg.inv(self.k_mat)
        self.iterations = 50
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
            for i in range(len(features_curr)):
                if self.check_fundamental_matrix(features_curr[i], features_next[i], fundamental_mat) < 0.01:
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
        return u @ s_new @ v_new

    @staticmethod
    def check_fundamental_matrix(feature_curr, feature_next, fundamental_mat):
        fc_new = np.array([feature_curr[0], feature_curr[1], 1]).T
        fn_new = np.array([feature_next[0], feature_next[1], 1])
        est = abs(np.squeeze(np.matmul(np.matmul(fn_new, fundamental_mat), fc_new)))
        return est
