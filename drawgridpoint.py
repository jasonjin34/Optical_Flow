import cv2 as cv
import numpy as np
import threading
from imutils.object_detection import non_max_suppression

def Draw_gridpoint(frame, mag, divide, size, pointdistance):
    output = np.copy(frame)
    shape = frame.shape
    height_list = np.arange(0, shape[0], divide)
    width_list = np.arange(0, shape[1], divide)
    for heightindex in height_list:
        for widthindex in width_list:
            if mag[heightindex][widthindex] > pointdistance / 4:
                cv.circle(output, (widthindex, heightindex), size, (0, 255, 0), size)
    return output


def Draw_vector(src, mag, ang, divide):
    shape = mag.shape
    height_list = np.arange(0, shape[0], divide)
    width_list = np.arange(0, shape[1], divide)

    '''human detection'''
    # car cascade for car detections
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(src_gray, 1.1, 1)
    # draw the original bounding boxes
    for (x, y, w, h) in faces:
        cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)

    reference_matrix = np.ones_like(ang)
    face_region_matrix = np.ones_like(ang)

    center_x = int(shape[1] / 2)
    center_y = int(shape[0] / 2)

    for heightindex in height_list:
        for widthindex in width_list:
            if heightindex != center_y and widthindex != center_x:
                x_diff = widthindex - center_x
                y_diff = heightindex - center_y
                temp_angle = np.arctan(y_diff / x_diff)
            output_angle = 0
            if temp_angle >= 0:
                if widthindex >= center_x:
                    output_angle = temp_angle
                else:
                    output_angle = np.pi + temp_angle
            else:
                if widthindex >= center_x:
                    output_angle = np.pi * 2 - temp_angle
                else:
                    output_angle = np.pi - temp_angle
            reference_matrix[heightindex][widthindex] = output_angle

    # draw the optical flow vector
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

    filter_matrix = np.where(mag > 10, 1, 0)
    frontback_matrix = np.cos(ang - reference_matrix) * filter_matrix
    sum_frontback = np.sum(frontback_matrix)

    leftright_matrix = np.cos(ang) * filter_matrix
    sum_leftright = np.sum(leftright_matrix)

    status1 = ''
    status2 = ''
    if abs(sum_frontback) > 10000:
        if sum_frontback > 0:
            status1 = 'Away'
        else:
            status1 = 'Near'

    if abs(sum_leftright) > 10000:
        if sum_leftright > 0:
            status2 = 'Right'
        else:
            status2 = 'Left'

    return status1, status2


# draw point thread function
def draw_point(img, start_height, end_height, start_width, end_width):
    for heightindex in range(start_height, end_height):
        for widthindex in range(start_width, end_width):
            cv.circle(img, (widthindex, heightindex), 1, (0, 0, 255), 1)


class myThread(threading.Thread):
    def __int__(self, threadID, name, input_tuple_list, src_image):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.input_tuple = input_tuple_list
        self.src_image = src_image

    def run(self):
        for index in self.input_tuple:
            cv.circle(self.src_image, (index[0], index[1]), 1, (0, 0, 255), 1)


def Draw_moving_object(dst, mag, ang, pointdistance, imagetype):
    shape = mag.shape
    height = shape[0]
    width = shape[1]
    height_range = np.arange(0, height, pointdistance)
    width_range = np.arange(0, width, pointdistance)

    point_list = []

    if imagetype == 1:
        # discrete image
        for heightindex in height_range:
            for widthindex in width_range:
                if mag[heightindex][widthindex] >= 10:
                    cv.circle(dst, (widthindex, heightindex), 1, (0, 0, 255), 1)
                    point_list.append((heightindex, widthindex))

    elif imagetype == 0:
        # sparese image
        for heightindex in range(height):
            for widthindex in range(width):
                if mag[heightindex][widthindex] >= 5:
                    angle = ang[heightindex][widthindex]
                    if angle > np.pi / 2  :#  and angle < np.pi / 2:
                        cv.circle(dst, (widthindex, heightindex), 1, (0, 0, 255), 1)
                    else:
                        cv.circle(dst, (widthindex, heightindex), 1, (255, 0, 0), 1)


def Example_function():
    frame = np.ones((480, 640, 3),  np.uint8)
    frame.fill(255)
    frame = Draw_gridpoint(frame, 20, 1)
    '''show the result'''
    cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.imshow('grid result', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()
