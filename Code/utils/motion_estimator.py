import cv2
import numpy as np


def get_cells_list():
    cells = []
    for i in range(8):
        for j in range(8):
            cells.append((i, j))
    return cells


class MotionEstimator:
    """
    A class to estimate 3D motion of a camera
    """
    def __init__(self, focal_lengths, principal_pts):
        self.k_mat = np.array([[focal_lengths[0], 0, principal_pts[0]],
                               [0, focal_lengths[1], principal_pts[1]],
                               [0, 0, 1]])
        self.k_mat_inv = np.linalg.inv(self.k_mat)
        self.cells = get_cells_list()
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
