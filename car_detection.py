import cv2 as cv


videoinput = 1

if videoinput == 1:
    cap = cv.VideoCapture('src_video/testvideo.mp4')
else:
    cap = cv.VideoCapture(0)

car_cascade_filename = 'trainedcar_small/data/cascade.xml'
car_cascade_big_filename = 'Training-HaarCascade/trainedcar/data/cascade.xml'

car_cascade = cv.CascadeClassifier(car_cascade_filename)

'''output video'''
fourcc_out = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
car_detection_video = cv.VideoWriter('car_detection.mp4', fourcc_out, 10, (712, 472))

frame_count = 0
while frame_count < 300:
    ret, frames = cap.read()

    gray = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (15, 15), 0, 0)
    #  gray = cv.medianBlur(gray, 5)
    cars = car_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    height, width, channels = frames.shape
    print(height, width)

    optimize = 1
    if optimize:
        for (x, y, w, h) in cars:
            if not y > height / 2 and x < width / 3 or not y > height / 2 and x > width * 2 / 3:
                cv.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
    else:
        for (x, y, w, h) in cars:
            cv.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow('video2', frames)
    car_detection_video.write(frames)
    frame_count += 1

    if cv.waitKey(33) == 27:
        break


car_detection_video.release()
cv.destroyAllWindows()