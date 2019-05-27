import cv2 as cv
import numpy as np

'''define global variable for pose estimation -- reference matrix generation'''
# define reference matrix in global variable
temp_matrix = np.arange(1001 * 1001).reshape(1001, 1001)
temp_matrix_x = np.where(temp_matrix % 1001 - 500 != 0, temp_matrix % 1001 - 500, 0.001)
temp_matrix_y = 500 - temp_matrix // 1001
# reference matrix for direction detection
reference_matrix = np.arctan(temp_matrix_y / temp_matrix_x)
reference_matrix[501:, 501:] = reference_matrix[501:, 501:] + 2 * np.pi
reference_matrix[501:, :500] = reference_matrix[501:, :500] + np.pi
reference_matrix[:500, :500] = reference_matrix[:500, :500] + np.pi

'''video input specification is height 476 width 712 channels 3
   front image edge 238
   define a list of matrix with size 
'''
reference_matrix_list = []
edge_length = 238
distance_range_list = np.arange(23, edge_length + 23, int(edge_length / 10.0))
for top_height_index in distance_range_list:
    for left_width_index in distance_range_list:
        print(top_height_index, left_width_index)
        reference_matrix_list.append(reference_matrix[501 - top_height_index: 501 + edge_length - top_height_index, 501 - left_width_index: 501 + edge_length - left_width_index])
print(reference_matrix_list[113], reference_matrix_list[113].shape)
print('next')
print(reference_matrix)

def Pose_estimation(src, mag, ang, divide):
    shape = mag.shape
    height_list = np.arange(0, shape[0], divide)
    width_list = np.arange(0, shape[1], divide)

    for heightindex in height_list:
        for widthindex in width_list:
            mag_temp = mag[heightindex][widthindex]
            if mag_temp > 5:
                length = mag_temp
                arg = ang[heightindex][widthindex]
                deltaX = int(length * np.cos(arg))
                deltaY = int(length * np.sin(arg))
                cv.arrowedLine(src, (widthindex, heightindex), (widthindex + deltaX, heightindex + deltaY), (0, 255, 0),
                   1, 8, 0, 0.2)

    '''pose estimation - front frame'''
    ang_forward = ang[:shape[0] // 2, shape[1] // 3: 2 * shape[1] // 3 + 1]
    mag_forward_temp = mag[:shape[0] // 2, shape[1] // 3: 2 * shape[1] // 3 + 1]
    mag_forward = np.where(mag_forward_temp > 5, 1, 0)

    count = 0
    reference_max_sum = 0
    reference_position = 0
    for matrix_index in reference_matrix_list:
        temp_matrix = np.where(ang_forward - matrix_index < np.pi/10, 1*mag_forward, 0)
        temp_sum = np.sum(temp_matrix)
        if reference_max_sum < abs(temp_sum):
            reference_max_sum = abs(temp_sum)
            reference_position = count
        count += 1
    Pose_x = int(reference_position % 11) * 23 + 238
    Pose_y = int(reference_position / 11) * 23
    print(Pose_x, Pose_y, reference_position)
    if Pose_x != 238:
        cv.circle(src, (Pose_x, Pose_y), 5, (255, 0, 0), 3)
