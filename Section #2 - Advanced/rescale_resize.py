#pylint:disable=no-member

import cv2 as cv

from utils import rescaleFrame


# img = cv.imread('../Resources/Photos/cat.jpg')
# cv.imshow('Cat', img)

def changeRes(width,height):
    # Live video
    capture.set(3,width)
    capture.set(4,height)
    
# Reading Videos
capture = cv.VideoCapture('../Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break

    frame_resized = rescaleFrame(frame, scale=.2)
    
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()