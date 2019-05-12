import cv2 as cv

videoinput = 1

if videoinput == 1:
    cap = cv.VideoCapture('testvideo2.mp4')
else:
    cap = cv.VideoCapture(0)

car_cascade = cv.CascadeClassifier('cascade.xml')

while True:
    ret, frames = cap.read()

    gray = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.3, 3, minSize=(30, 30))

    for (x, y, w, h) in cars:
        cv.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv.imshow('video2', frames)

    if cv.waitKey(33) == 27:
        break

cv.destroyAllWindows()