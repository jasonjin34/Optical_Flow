import cv2 as cv
import numpy as np
import threading
from imutils.object_detection import non_max_suppression

# globacl variable
car_cascade = cv.CascadeClassifier('trainedcar_small/data/cascade.xml')

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


'''direction detection for camera detection'''
# define reference matrix in global variable
temp_matrix = np.arange(1001 * 1001).reshape(1001, 1001)
temp_matrix_x = np.where(temp_matrix % 1001 - 500 != 0, temp_matrix % 1001 - 500, 0.001)
temp_matrix_y = 500 - temp_matrix // 1001
# reference matrix for direction detection
reference_matrix = np.arctan(temp_matrix_y / temp_matrix_x)
reference_matrix[501:, 501:] = reference_matrix[501:, 501:] + 2 * np.pi
reference_matrix[501:, :500] = reference_matrix[501:, :500] + np.pi
reference_matrix[:500, :500] = reference_matrix[:500, :500] + 2 * np.pi


# front back status data smoothing
sum_frontback_list = []

def Direction_detect(src, mag, ang, divide):
    shape = mag.shape
    height_list = np.arange(0, shape[0], divide)
    width_list = np.arange(0, shape[1], divide)
    reference = np.ones_like(ang)

    top_height = shape[0] // 2
    bottom_height = shape[0] - top_height
    left_width = shape[1] // 2
    right_width = shape[1] - left_width

    reference = reference_matrix[501 - top_height: 501 + bottom_height, 501 - left_width: 501 + right_width]

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
    frontback_matrix = np.cos(ang - reference) * filter_matrix
    sum_frontback = np.sum(frontback_matrix)
    average_sum_frontback = 0

    # smooth front back value
    sum_list_length = len(sum_frontback_list)
    if sum_list_length <= 3:
        sum_frontback_list.append(sum_frontback)
    else:
        sum_frontback_list.append(sum_frontback)
        del sum_frontback_list[0]

    average_sum_frontback = sum(sum_frontback_list) / len(sum_frontback_list)

    leftright_matrix = np.cos(ang) * filter_matrix
    sum_leftright = np.sum(leftright_matrix)

    status1 = ''
    status2 = ''
    if abs(average_sum_frontback) > 2000:
        if average_sum_frontback > 0:
            status1 = 'Near'
        else:
            status1 = 'Away'

    if abs(sum_leftright) > 10000:
        if sum_leftright > 0:
            status2 = 'Right'
        else:
            status2 = 'Left'

    return status1, status2


def Draw_vector(src, mag, ang, divide):
    shape = mag.shape
    height_list = np.arange(0, shape[0], divide)
    width_list = np.arange(0, shape[1], divide)

    '''human detection and optical flow filter'''
    filter_matrix = np.ones_like(mag)

    # car cascade for car detections
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.GaussianBlur(src_gray, (13, 13), 0)
    cars = car_cascade.detectMultiScale(src_gray, 1.2, 3, minSize=(60, 60))

    for (x, y, w, h) in cars:
        if not y > shape[0] / 2 and x < shape[1] / 3 or not y > shape[0] / 2 and x > shape[1] * 2 / 3:
            cv.rectangle(src, (x, y), (x + w, y + h), (0, 0, 255), 2)
            filter_matrix[y: y+h, x: x+w].fill(0)
    mag = mag*filter_matrix

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

    filter_matrix = np.where(mag > 10, 1, 0)
    frontback_matrix = np.cos(ang - reference_frontback) * filter_matrix
    sum_frontback = np.sum(frontback_matrix)

    # smooth front back value
    sum_list_length = len(sum_frontback_list)
    if sum_list_length <= 3:
        sum_frontback_list.append(sum_frontback)
    else:
        sum_frontback_list.append(sum_frontback)
        del sum_frontback_list[0]
    # calculate the front back value and determine if camera go forward or backward
    average_sum_frontback = sum(sum_frontback_list) / len(sum_frontback_list)

    # check for turning left or right, forward or backward
    leftright_matrix = np.cos(ang - reference_leftright) * filter_matrix
    sum_topleft_matrix = np.sum(leftright_matrix[:shape[0]//2, :shape[1] // 3])
    sum_topmiddle_matrix = np.sum(leftright_matrix[:shape[0] // 2, shape[1] // 3: 2 * (shape[1] // 3)])
    sum_topright_matrix = np.sum(leftright_matrix[:shape[0] // 2, 2 * (shape[1] // 3):])

    if sum_topright_matrix > 0 and sum_topleft_matrix > 0 and sum_topmiddle_matrix > 0:
        status = 'Turning Right'
    elif sum_topright_matrix < 0 and sum_topleft_matrix < 0 and sum_topmiddle_matrix < 0:
        status = 'Turning Left'
    else:
        if abs(average_sum_frontback) > 300:
            if average_sum_frontback > 0:
                status = 'Forward: ' + str(round(average_sum_frontback))
            else:
                status = 'Backward: ' + str(round(average_sum_frontback))
        else:
            status = 'Not Moving: ' + str(round(average_sum_frontback))

    return status


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
