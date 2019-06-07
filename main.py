import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from drawgridpoint import Draw_vector, Direction_detect, Optical_flow_result
from pose_estimation import Pose_estimation
import time

# TO DO
# 1. Setup input and set up grid !check!!
# 2. Test dense optical flow algorithm farneback optical flow !!check
# 3. output image listcmpare !check!
# 4. Time check
# 5. Noise detection, car detection for our input sampple !!check
# 6. Set up the reference matrix !!check
# 7. Test rotated video !!check
# 8. revise the new method based on two image
# 9. generate the optical flow data based on two frame
# 10.write the algorithm of intersection vector
# 11.plot the intersection points
# 12.implement rasac to find the optimized center point
# 13.k-mean 

def main():
    static_bool = True;
    if not static_bool:
        video_index = 1
        # set up frame for optical flow
        if video_index == 0:
            cap = cv.VideoCapture(0)
            ret, frame1 = cap.read()
            prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        if video_index == 1:
            filestring = 'src_video/testvideo2.mp4'
            cap = cv.VideoCapture(filestring)
            ret, frame1 = cap.read()
            prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        if video_index == 2:
            filestring = 'rotatedvideo/rotatedVideo.mp4'
            cap = cv.VideoCapture(filestring)
            ret, frame1 = cap.read()
            prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

        # set up the reference for object detection
        reference_frame = np.ones_like(frame1)
        reference_frame.fill(255)

        print(frame1.shape)  # height, width, colorspace layer
        width = frame1.shape[1]
        height = frame1.shape[0]

        '''predefine the mask image'''
        denseImage = np.zeros_like(frame1)
        denseImage.fill(255)  # empty image

        '''output video'''
        fourcc_out = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        result_video = cv.VideoWriter('result_video/resultvideo2.mp4', fourcc_out, 5, (width, height))

        # define number of the frame
        frame_num = 0
        while frame_num < 200:
            start = time.time()
            reference_frame.fill(255)
            ret, frame2 = cap.read()

            if video_index == 1:
                # dense optical flow algorithm farneback optical flow
                # next is the gray image
                next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
                cv.normalize(mag, mag, 0, 20, cv.NORM_MINMAX)
                point_distance = 10

                # draw the optical flow vector
                camera_status = Draw_vector(frame2, mag, ang, point_distance)
                cv.putText(frame2, camera_status, (30, 30), 5, 1, (0, 0, 255), 1, cv.LINE_8)
                cv.imshow('dense optical flow', frame2)
                # record for the video
                result_video.write(frame2)
                k = cv.waitKey(10) & 0xff
                if k == 27:
                    break
                elif k == ord('s'):
                    cv.imwrite('result_image/opticalfb.png', frame2)

            # camear video test functions
            elif video_index == 0:
                # dense optical flow algorithm farneback optical flow
                # next is the gray image
                next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
                cv.normalize(mag, mag, 0, 20, cv.NORM_MINMAX)
                point_distance = 10

                # draw the optical flow vector
                frontback, leftright = Direction_detect(frame2, mag, ang, point_distance)
                if len(frontback):
                    cv.putText(frame2, frontback, (30, 30), 3, 1, (0, 0, 255), 1, cv.LINE_8)
                if len(leftright):
                    cv.putText(frame2, leftright, (450, 30), 3, 1, (0, 0, 255), 1, cv.LINE_8)
                if len(frontback) == 0 and len(leftright) == 0:
                    cv.putText(frame2, 'No Movement', (30, 30), 3, 1, (255, 0, 0), 1, cv.LINE_8)

                cv.imshow('dense optical flow', frame2)
                result_video.write(frame2)
                k = cv.waitKey(5) & 0xff
                if k == 27:
                    break
                elif k == ord('s'):
                    cv.imwrite('result_image/optical.png', frame2)

            elif video_index == 2:
                # init dense optical flow result
                next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
                flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
                cv.normalize(mag, mag, 0, 20, cv.NORM_MINMAX)
                point_distance = 10
                Pose_estimation(frame2, mag, ang, point_distance)
                cv.imshow('dense optical flow', frame2)
                cv.imwrite('result_image_rotated/' + str(frame_num) + '.jpg', frame2)
                k = cv.waitKey(5) & 0xff
                if k == 27:
                    break

            prvs = np.copy(next)
            frame_num = frame_num + 1

        cap.release()
        result_video.release()
        cv.destroyAllWindows()
    else:
        frame_old = cv.imread('frames/old_frame.jpg', 0)
        frame_new = cv.imread('frames/new_frame.jpg', 0)

        flow = cv.calcOpticalFlowFarneback(frame_old, frame_new, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        cv.normalize(mag, mag, 0, 20, cv.NORM_MINMAX)
        Pose_estimation(frame_old, mag, ang, 10)
        cv.imshow('dense optical flow', frame_old)

        # check the size of mag and ang
        size = ang.shape[1]*ang.shape[0]
        # the shape of the optial flow results are 476 height, 712 width, we need to extract the front frame
        # which is width [238, 474] height [0, 238]
        mag_front_temp = mag[0:236:10, 238:474:10]
        mag_forward = np.where(mag_front_temp > 5, 1, 0)
        ang_front = ang[0:236:10, 238:474:10]
        ang_front = mag_forward * ang_front
        edge_size = ang_front.shape[0]
        size = np.power(edge_size, 2)
        delta_x = - np.cos(ang_front)
        delta_y = + np.sin(ang_front)

        # generate position matrix
        x_temp = np.arange(0, 236, 10)
        y_temp = np.arange(230, -10, -10)
        position_x, position_y = np.meshgrid(x_temp, y_temp)

        fig, ax = plt.subplots(figsize=(7, 7))

        # set up the filter
        sample_size = position_y.shape[0]
        ang_front = ang_front.reshape(sample_size*sample_size)
        position_x = position_x.reshape(sample_size*sample_size)
        position_y = position_y.reshape(sample_size*sample_size)
        delta_x = delta_x.reshape(sample_size*sample_size)
        delta_y = delta_y.reshape(sample_size*sample_size)
        mag_forward = mag_forward.reshape(sample_size*sample_size)
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

        q = ax.quiver(position_x, position_y, delta_x, delta_y)
        ax.axis([0, 240, 0, 480])
        ax.set_aspect('equal')
        ax.quiverkey(q, X=0.3, Y=1.1, U=1, label='Optical Flow Result, length = 10', labelpos='E')


        # calculate the intersection point
        x_intersect_array = np.array([])
        y_intersect_array = np.array([])
        count = 0
        input_data_length = position_x.size
        for index1 in range(input_data_length):
            for index2 in range(index1, input_data_length):
                if (delta_y[index1] != delta_y[index2] and delta_x[index1] != delta_x[index2]) and (- ang_front[index1] + ang_front[index2] < 0): # not signular
                    A = np.array(([delta_x[index1], -delta_x[index2]], [delta_y[index1], -delta_y[index2]]))
                    B = np.array(([[position_x[index2] - position_x[index1]], [position_y[index2] - position_y[index1]]]))
                    inv_A = np.linalg.inv(A)
                    mag = inv_A.dot(B)
                    x_intersect_array = np.append(x_intersect_array, position_x[index1] + mag[0][0] * delta_x[index1])
                    y_intersect_array = np.append(y_intersect_array, position_y[index1] + mag[0][0] * delta_y[index1])
                    count += 1

        # print(x_intersect_array.shape)
        x_differ_array = np.array([])
        y_differ_array = np.array([])

        for temp in x_intersect_array:
            temp_array = np.power(x_intersect_array - temp, 2)
            x_differ_array = np.append(x_differ_array, np.sum(temp_array))

        for temp in y_intersect_array:
            temp_array = np.power(x_intersect_array - temp, 2)
            y_differ_array = np.append(y_differ_array, np.sum(temp_array))

        error_array = x_differ_array + y_differ_array
        min = error_array[0]
        count = 0
        position = 0
        for index in error_array:
            if index < min:
                min = index
                position = count
            count += 1
        print(position, x_intersect_array[position], y_intersect_array[position])

        plt.plot(x_intersect_array[position], y_intersect_array[position], 'ro')
        plt.show()
        cv.waitKey()
        cv.destroyAllWindows()


if __name__ == '__main__':
    main()