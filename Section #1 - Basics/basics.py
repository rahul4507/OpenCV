import cv2 as cv
import os

import numpy as np

from utils import rescaleFrame

folder_path = '../Output/Section #1/'

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Reading images
# img_with_color = cv.imread("../Resources/photos/cat.jpg", cv.IMREAD_COLOR)
# img_with_unchanged = cv.imread("../Resources/photos/cat.jpg", cv.IMREAD_UNCHANGED)
# img_with_grayscale = cv.imread("../Resources/photos/cat.jpg", cv.IMREAD_GRAYSCALE)
#
# cv.imshow('cat1',img_with_color)
# cv.imwrite(os.path.join(folder_path, "cat_img_with_color.jpg"), img_with_color)
#
# cv.imshow('cat2',img_with_unchanged)
# cv.imwrite(os.path.join(folder_path, "cat_img_with_unchanged.jpg"), img_with_unchanged)
#
# cv.imshow('cat3',img_with_grayscale)
# cv.imwrite(os.path.join(folder_path, "cat_img_with_grayscale.jpg"), img_with_grayscale)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Reading Videos
vid = cv.VideoCapture("../Resources/Videos/dog.mp4")
# vid = cv.VideoCapture("D:\Movies\Interstellar.2014.2014.1080p.BluRay.x264.YIFY.mp4")
i=0
while True:
    isTrue, frame = vid.read()
    if not isTrue:
        break
    frame = rescaleFrame(frame,0.3)
    cv.imshow("video",frame)
    if cv.waitKey(10) & 0xFF == ord('d'):
        break

vid.release()
cv.destroyAllWindows()

