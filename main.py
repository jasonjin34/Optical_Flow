import cv2 as cv
import numpy as np
from drawgridpoint import Draw_gridpoint, Draw_vector
import time

# TO DO
# 1. Setup input and set up grid
# 2. Test dense optical flow algorithm farneback optical flow

def main():
    video_index = 1
    if video_index == 0:
        cap = cv.VideoCapture(0)
        ret, frame1 = cap.read()
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    if video_index == 1:
        filestring = 'testvideo2.mp4'
        cap = cv.VideoCapture(filestring)
        ret, frame1 = cap.read()
        print(ret)
        prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

    print(frame1.shape) # height, width, colorspace layer
    width = frame1.shape[1]
    height = frame1.shape[0]

    '''predefine the mask image'''
    denseImage = np.zeros_like(frame1)
    denseImage.fill(255)

    '''output video'''
    fourcc_out = cv.VideoWriter_fourcc('D','I','V','X')
    result_video = cv.VideoWriter('resultvideo.mp4', fourcc_out, 10, (width, height))

    frame_num = 0
    while(frame_num < 50):
        ret, frame2 = cap.read()
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        start = time.time()
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        end = time.time()
        print('Optical Flow Farneback time: ' + str(end - start))
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        cv.normalize(mag, mag, 0, 20, cv.NORM_MINMAX)

        Draw_vector(frame2, mag, ang, 20)

        cv.imshow('dense optical flow', frame2)
        result_video.write(frame2)
        k = cv.waitKey(20) & 0xff
        if k == 27:
            break
        #elif k == ord('s'):
            # cv.imwrite('opticalfb.png', frame2)
            # cv.imwrite('opticalhsv.png', denseImage)
        prvs = np.copy(next)
        frame_num = frame_num + 1

    cap.release()
    result_video.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()