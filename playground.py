import cv2 as cv
import numpy as np
from keras.src.backend.jax.numpy import identity

img = cv.imread('./Resources/Photos/park.jpg',cv.IMREAD_UNCHANGED)

cv.imshow('original group1 image BGR',img)

kernel = np.array([[-1, -1, -1],  [-1, 8, -1],   [-1, -1, -1]])
filter_applied = cv.filter2D(img, -1, kernel)
cv.imshow("After Applying identity kernel", filter_applied)

# # converting to GrayScale
# img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('grayscale img', img_grayscale)
#
# # converting to HSV
# img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSV img', img_grayscale)
#
# # converting to LAB
# img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LAB img', img_grayscale)
#
# # converting to RGB
# img_grayscale = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGB img', img_grayscale)

# Blur the image
img_blur = cv.GaussianBlur(img, (5,5),cv.BORDER_DEFAULT)
cv.imshow("blur image", img_blur)

# Edge Cascade
canny = cv.Canny(img_blur, threshold1=125, threshold2=175)
cv.imshow("canny", canny)

# Dilate Image
kernel = np.ones((3,3),np.uint8)
dilated = cv.dilate(canny, kernel, iterations=1)
cv.imshow("Dilated", dilated)

# Dilate Image
kernel = np.ones((3,3),np.uint8)
dilated = cv.erode(dilated, kernel, iterations=1)
cv.imshow("eroded", dilated)



cv.waitKey(0)
cv.destroyAllWindows()

del img_blur
del img
del canny
del dilated
del filter_applied