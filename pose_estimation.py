import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

'''define global variable for pose estimation -- reference matrix generation'''
# define reference matrix in global variable
temp_matrix = np.arange(1001 * 1001).reshape(1001, 1001)
temp_matrix_x = np.where(temp_matrix % 1001 - 500 != 0, temp_matrix % 1001 - 500, 0.001)
temp_matrix_y = 500 - temp_matrix // 1001
# front camera reference matrix for direction detection
reference_matrix = np.arctan(temp_matrix_y / temp_matrix_x)
reference_back_matrix = reference_matrix.copy()
reference_matrix[501:, 501:] = reference_matrix[501:, 501:] + 2 * np.pi
reference_matrix[501:, :500] = reference_matrix[501:, :500] + np.pi
reference_matrix[:500, :500] = reference_matrix[:500, :500] + np.pi
# back camera reference matrix
reference_back_matrix[:500, 501:] = reference_back_matrix[:500, 501:] + np.pi
reference_back_matrix[:500, :500] = reference_back_matrix[:500, :500] + 2 * np.pi
reference_back_matrix[501:, :500] = reference_back_matrix[501:, :500] - np.pi
reference_back_matrix[501:, 501:] = reference_back_matrix[501:, 501:] + np.pi

'''video input specification is height 476 width 712 channels 3
   front image edge 238
   define a list of matrix with size 
'''
reference_matrix_list = []
reference_matrix_back_list = []
edge_length = 238
distance_range_list = np.arange(23, edge_length + 23, int(edge_length / 10.0))
for top_height_index in distance_range_list:
    for left_width_index in distance_range_list:
        # print(top_height_index, left_width_index)
        reference_matrix_list.append(reference_matrix[501 - top_height_index: 501 + edge_length - top_height_index,
                                     501 - left_width_index: 501 + edge_length - left_width_index])
        reference_matrix_back_list.append(reference_matrix[501 - top_height_index: 501 + edge_length - top_height_index,
                                          501 - left_width_index: 501 + edge_length - left_width_index])


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
        temp_matrix = np.where(ang_forward - matrix_index < np.pi/10, 1 * mag_forward, 0)
        temp_sum = np.sum(temp_matrix)
        if reference_max_sum < abs(temp_sum):
            reference_max_sum = abs(temp_sum)
            reference_position = count
        count += 1
    Pose_x = int(reference_position % 11) * 23 + 238
    Pose_y = int(reference_position / 11) * 23
    cv.circle(src, (Pose_x, Pose_y), 5, (255, 0, 0), 3)
    print('front frame: ' + str(reference_position) + ' ' + str(reference_max_sum))

    '''pose estimation - back frame'''
    ang_backward = ang[shape[0] // 2:, shape[1] // 3: 2 * shape[1] // 3 + 1]
    mag_backward_temp = mag[shape[0] // 2:, shape[1] // 3: 2 * shape[1] // 3 + 1]
    mag_backward = np.where(mag_backward_temp > 5, 1, 0)

    '''backframe pose'''
    count = 0
    reference_max_sum = 0
    reference_position = 0
    for matrix_index in reference_matrix_back_list:
        temp_matrix = np.where(abs(ang_backward - matrix_index) < np.pi / 10, 1 * mag_backward, 0)
        temp_sum = np.sum(temp_matrix)
        if reference_max_sum < temp_sum:
            reference_max_sum = temp_sum
            reference_position = count
        count += 1
    Pose_x_back = int(reference_position % 11) * 23 + 238
    Pose_y_back = int(reference_position / 11) * 23 + 238

    if reference_max_sum > 5:
        cv.circle(src, (Pose_x_back, Pose_y_back), 5, (0, 0, 255), 3)
    print('back frame: ' + str(reference_position) + ' ' + str(reference_max_sum))


def Pose_estimation_point_projection(ang, mag, output_image_index, filelocation):
    # check the size of mag and ang
    size = ang.shape[1] // 3
    # the shape of the optial flow results are 476 height, 712 width, we need to extract the front frame
    # which is width [238, 474] height [0, 238]
    mag_front_temp = mag[0:size:10, size:size*2:10]# mag[0:236:10, 238:474:10]
    mag_forward = np.where(mag_front_temp > 5, 1, 0)
    ang_front = ang[0:size:10, size:size*2:10]# ang[0:236:10, 238:474:10]
    ang_front = mag_forward * ang_front
    edge_size = ang_front.shape[0]
    size = np.power(edge_size, 2)
    delta_x = - np.cos(ang_front)
    delta_y = + np.sin(ang_front)

    # generate position matrix
    x_temp = np.arange(0, 236, 10)
    y_temp = np.arange(230, -10, -10)
    position_x, position_y = np.meshgrid(x_temp, y_temp)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title('Pose Estimation of Intersection Points', fontsize=16)

    # set up the filter
    sample_size = position_y.shape[0]
    ang_front = ang_front.reshape(sample_size * sample_size)
    position_x = position_x.reshape(sample_size * sample_size)
    position_y = position_y.reshape(sample_size * sample_size)
    delta_x = delta_x.reshape(sample_size * sample_size)
    delta_y = delta_y.reshape(sample_size * sample_size)
    mag_forward = mag_forward.reshape(sample_size * sample_size)
    filter_list = []

    # filter
    count = 0
    for index in mag_forward:
        if index == 0:
            filter_list.append(count)
        count += 1
    position_x = np.delete(position_x, filter_list)
    position_y = np.delete(position_y, filter_list)
    delta_x = np.delete(delta_x, filter_list)
    delta_y = np.delete(delta_y, filter_list)
    ang_front = np.delete(ang_front, filter_list)

    ax.quiver(position_x, position_y, delta_x, delta_y)
    ax.axis([0, 240, 0, 480])
    ax.set_aspect('equal')

    # calculate the intersection point
    x_intersect_array = np.array([])
    y_intersect_array = np.array([])
    count = 0
    input_data_length = position_x.size
    for index1 in range(input_data_length):
        for index2 in range(index1, input_data_length):
            if (abs(delta_y[index1] - delta_y[index2]) > 0.1 and abs(delta_x[index1] != delta_x[index2]) > 0.1) and (
                    - ang_front[index1] + ang_front[index2] < 0):  # not signular
                A = np.array(([delta_x[index1], -delta_x[index2]], [delta_y[index1], -delta_y[index2]]))
                B = np.array(
                    ([[position_x[index2] - position_x[index1]], [position_y[index2] - position_y[index1]]]))
                inv_A = np.linalg.inv(A)
                mag = inv_A.dot(B)
                if mag[0][0] > 0 and mag[1][0] > 0:
                    x_intersect_array = np.append(x_intersect_array, position_x[index1] + mag[0][0] * delta_x[index1])
                    y_intersect_array = np.append(y_intersect_array, position_y[index1] + mag[0][0] * delta_y[index1])
                count += 1

    '''implemented ransac algorithm to find the '''
    size = x_intersect_array.size
    if size == 0 or y_intersect_array.size == 0:
        return
    x_intersect_array = x_intersect_array.reshape(size, 1)
    y_intersect_array = y_intersect_array.reshape(size, 1)
    ransac = linear_model.RANSACRegressor(residual_threshold=20)
    ransac.fit(x_intersect_array,y_intersect_array)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(x_intersect_array.min(), x_intersect_array.max())[:, np.newaxis]
    line_y_ransac = ransac.predict(line_X)
    plt.plot(line_X, line_y_ransac, color='cornflowerblue',
             label='RANSAC regressor')

    plt.scatter(x_intersect_array[inlier_mask], y_intersect_array[inlier_mask], color='yellowgreen', marker='.',
                label='Inliers')
    plt.scatter(x_intersect_array[outlier_mask], y_intersect_array[outlier_mask], color='gold', marker='.',
                label='Outliers')

    ''' find the most possible middle point as pose '''
    # before using ransac
    x_differ_array = np.array([])
    y_differ_array = np.array([])
    for temp in x_intersect_array:
        temp_array = np.power(x_intersect_array - temp, 2)
        x_differ_array = np.append(x_differ_array, np.sum(temp_array))

    for temp in y_intersect_array:
        temp_array = np.power(x_intersect_array - temp, 2)
        y_differ_array = np.append(y_differ_array, np.sum(temp_array))

    error_array = np.sqrt(x_differ_array + y_differ_array)
    min = error_array[0]
    count = 0
    position = 0
    for index in error_array:
        if index < min:
            min = index
            position = count
        count += 1
    plt.plot(x_intersect_array[position], y_intersect_array[position], 'bo', label='Pose before Ransac')

    # after using ransac
    x_differ_array = np.array([])
    y_differ_array = np.array([])
    x_intersect_array = x_intersect_array[inlier_mask]
    y_intersect_array = y_intersect_array[inlier_mask]
    ransac_score = ransac.score(x_intersect_array, y_intersect_array)

    for temp in x_intersect_array:
        temp_array = np.power(x_intersect_array - temp[0], 2)
        x_differ_array = np.append(x_differ_array, np.sum(temp_array))

    for temp in y_intersect_array:
        temp_array = np.power(x_intersect_array - temp[0], 2)
        y_differ_array = np.append(y_differ_array, np.sum(temp_array))
    error_array = np.sqrt(x_differ_array + y_differ_array)
    position = np.argmin(error_array)
    print('Distance Method ', x_intersect_array[position], y_intersect_array[position], output_image_index, ' ransac score:'+str(ransac_score))

    # using error circle method
    # filtered data
    intersection_count_array = np.array([])
    for index1 in range(x_intersect_array.size):
        temp_count = 0
        for index2 in range(x_intersect_array.size):
            if index1 != index2:
                if np.sqrt(np.power(x_intersect_array[index1] - x_intersect_array[index2], 2) + np.power(
                        y_intersect_array[index1] - y_intersect_array[index2], 2)) < 10:
                    temp_count += 1
        intersection_count_array = np.append(intersection_count_array, temp_count)
    circle_method_point_x = x_intersect_array[np.argmax(intersection_count_array)]
    circle_method_point_y = y_intersect_array[np.argmax(intersection_count_array)]

    print('Circle Method ', circle_method_point_x, circle_method_point_y, output_image_index)

    plt.plot(circle_method_point_x, circle_method_point_y, 'go', label='Pose based on Circle Error')
    plt.plot(x_intersect_array[position], y_intersect_array[position], 'ro', label='Pose Distance between Points')
    plt.xlabel("X Value", fontsize=16)
    plt.ylabel("Y Value", fontsize=16)
    plt.legend(loc='best')
    plt.savefig(filelocation + str(output_image_index) + '.png')
    plt.close('all')

    return circle_method_point_x, circle_method_point_y, x_intersect_array[position], y_intersect_array[position]