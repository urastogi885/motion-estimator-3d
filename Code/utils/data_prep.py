import numpy as np
from glob import glob
from scipy.ndimage import map_coordinates as interp2


def undistort_image(image, look_up_table):
    """
    Function to undistort an image using a look-up table
    :param image:
    :param look_up_table:
    :return:
    """
    ################################################################################
    # Copyright (c) 2019 University of Maryland
    # Authors:
    #  Kanishka Ganguly (kganguly@cs.umd.edu)
    #
    # This work is licensed under the Creative Commons
    # Attribution-NonCommercial-ShareAlike 4.0 International License.
    # To view a copy of this license, visit
    # http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
    # Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
    #
    ################################################################################
    reshaped_lut = look_up_table[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
    undistorted = np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)
                                        for channel in range(0, image.shape[2])]), 0, 3)
    return undistorted.astype(image.dtype)


def read_camera_model(models_dir):
    """
    Function to load camera intrinsics and undistortion look-up table from disk
    :param models_dir: directory containing camera models
    :return: a tuple containing focal length & principal point of camera, transformation matrix and a look-up table for
    undistortion
    """
    ################################################################################
    # Copyright (c) 2019 University of Maryland
    # Authors:
    #  Kanishka Ganguly (kganguly@cs.umd.edu)
    #
    # This work is licensed under the Creative Commons
    # Attribution-NonCommercial-ShareAlike 4.0 International License.
    # To view a copy of this license, visit
    # http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
    # Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
    #
    ################################################################################
    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    lut_path = models_dir + "/stereo_narrow_left_distortion_lut.bin"

    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0, 0]
    fy = intrinsics[0, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[0, 3]
    # 4x4 matrix that transforms x-forward coordinate frame at camera origin and image frame for specific lens
    g_camera_image = intrinsics[1:5, 0:4]
    # Look-up table for undistortion
    # Look-up table consists of (u,v) pair for each pixel)
    lut = np.fromfile(lut_path, np.double)
    lut = lut.reshape([2, lut.size // 2])
    look_up_table = lut.transpose()

    return fx, fy, cx, cy, g_camera_image, look_up_table


def extract_locations(dataset_location):
    """
    Get locations of all JPEG images from the given folder
    :param dataset_location: location of the folder
    :return: a list containing proper location of all JPEG images
    """
    # Define an empty list to store file locations
    filename_list = []
    # Add file locations to the list
    for filename in glob(str(dataset_location) + '*.png'):
        filename_list.append(filename)
    # Sort the list and return it
    filename_list.sort()
    return filename_list
