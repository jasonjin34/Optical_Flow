import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from drawgridpoint import Draw_vector, Direction_detect, Optical_flow_result
from pose_estimation import Pose_estimation, Pose_estimation_point_projection
import time

# TO DO
# 1. Setup input and set up grid !check!!
# 2. Test dense optical flow algorithm farneback optical flow !!check
# 3. output image listcmpare !check!
# 4. Time check
# 5. Noise detection, car detection for our input sampple !!check
# 6. Set up the reference matrix !!check
# 7. Test rotated video !!check
# 8. revise the new method based on two image !! check
# 9. generate the optical flow data based on two frame !! check
# 10.write the algorithm of intersection vector !! check
# 11.plot the intersection points !! check
# 12.implement rasac to find the optimized center point !! check
# 13.k-mean !! check

def main():
    static_bool = False;
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
        while frame_num < 100:
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
                Pose_estimation_point_projection(ang, mag, frame_num, 'frame_preset/result')
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
        dataX_circle = np.array([])
        dataY_circle = np.array([])
        dataX_distance = np.array([])
        dataY_distance = np.array([])

        for index in range(1, 8):
            frame_old = cv.imread('frames/old_frame' + str(index) + '.jpg', 0)
            frame_new = cv.imread('frames/new_frame' + str(index) + '.jpg', 0)
            flow = cv.calcOpticalFlowFarneback(frame_old, frame_new, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            cv.normalize(mag, mag, 0, 20, cv.NORM_MINMAX)
            X_c, Y_c, X_d, Y_d =Pose_estimation_point_projection(ang, mag, index, 'frames/result')
            dataY_distance = np.append(dataY_distance, Y_d)
            dataX_distance = np.append(dataX_distance, X_d)
            dataY_circle = np.append(dataY_circle, Y_c)
            dataX_circle = np.append(dataX_circle, X_c)

        print('standard deviation of circle error method are: X: ' + str(
            np.std(dataX_circle)) + ' Y: ' + str(np.std(dataY_circle)))
        print('standard deviation of distance error method are: X: ' + str(
            np.std(dataX_distance)) + ' Y: ' + str(np.std(dataY_distance)))


if __name__ == '__main__':
    main()