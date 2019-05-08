import cv2 as cv
import numpy as np


def Draw_gridpoint(frame, divide, size):
    output = np.copy(frame)
    shape = frame.shape
    height_list = np.arange(0, shape[0], divide)
    width_list = np.arange(0, shape[1], divide)
    for heightindex in height_list:
        for widthindex in width_list:
            cv.circle(output, (widthindex, heightindex), size, (0, 255, 0), size)
    return output


def Draw_vector(src, mag, ang, divide):
    shape = mag.shape
    height_list = np.arange(0, shape[0], divide)
    width_list = np.arange(0, shape[1], divide)
    for heightindex in height_list:
        for widthindex in width_list:
            length = mag[heightindex][widthindex]
            arg = ang[heightindex][widthindex]
            deltaX = int(length*np.cos(arg))
            deltaY = int(length*np.sin(arg))
            cv.arrowedLine(src, (widthindex, heightindex), (widthindex + deltaX, heightindex + deltaY), (0, 255, 0), 1, 8, 0, 0.3)


def Example_function():
    frame = np.ones((480, 640, 3),  np.uint8)
    frame.fill(255)
    frame = Draw_gridpoint(frame, 20, 1)
    '''show the result'''
    cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imshow('grid result', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
