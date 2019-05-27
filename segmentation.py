import cv2 as cv
import numpy as np
from drawgridpoint import reference_matrix
# import the predefined extern reference variable


def Segementation(src, mag, ang, divide):
    shape = mag.shape
    height_list = np.arange(0, shape[0], divide)
    width_list = np.arange(0, shape[1], divide)

    '''moving object segmentation and optical flow filter'''
    filter_matrix = np.ones_like(mag)

    # use reference for checking the camera go forward or backward
    reference_frontback = np.zeros_like(ang)
    # reference for checking turn left or right
    reference_leftright = np.zeros_like(ang)

    # setup the forward and back reference matrix
    top_height = shape[0] // 4
    bottom_height = shape[0] // 2 - top_height
    left_width = shape[1] // 6
    right_width = shape[1] // 3 - left_width
    reference_frontback[:shape[0] // 2, shape[1] // 3: 2 * (shape[1] // 3)] = reference_matrix[501 - top_height: 501 + bottom_height, 501 - left_width: 501 + right_width]
    reference_frontback[:shape[0] // 2, :shape[1] // 3] = np.pi
    reference_frontback[:shape[0] // 2, 2 * (shape[1] // 3):] = 0

    # setup the left and right reference matrix
    # using 180 degree reference use to detect if the camera turning right
    reference_leftright[:shape[0]//2, :] = np.pi

    for heightindex in height_list:
        for widthindex in width_list:
            mag_temp = mag[heightindex][widthindex]
            if mag_temp > 5:
                length = mag_temp
                arg = ang[heightindex][widthindex]
                deltaX = int(length * np.cos(arg))
                deltaY = int(length * np.sin(arg))
                cv.arrowedLine(src, (widthindex, heightindex), (widthindex + deltaX, heightindex + deltaY), (0, 255, 0),
                               1, 8, 0, 0.3)
