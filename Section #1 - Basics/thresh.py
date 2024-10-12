#pylint:disable=no-member

import cv2 as cv

img = cv.imread('../Resources/Photos/nums.jpg')
img = cv.resize(img, (500,500), cv.INTER_CUBIC)
cv.imshow('nums', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)


# Thresholding is a technique in OpenCV, which is the assignment of pixel values in relation to
# the threshold value provided. In thresholding, each pixel value is compared with the threshold value.
# thresholding is only performed on grayscale images as it will have
# only one pixel value ranging from 0 to 255
# Thresholding is a very popular segmentation technique, used for separating an object considered as a foreground
# from its background. A threshold is a value which has two regions on its either side i.e. below the threshold
# or above the threshold.

# Simple Thresholding
threshold, thresh = cv.threshold(gray, 35, 255, cv.THRESH_BINARY )
cv.imshow('Simple Thresholded', thresh)

# let's apply opening operation on it to see whether it's going to get clean or not
kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
cleaned = cv.morphologyEx(thresh, cv.MORPH_OPEN,kernel)
# Optional: Apply closing after opening to further clean up any holes
cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)
cv.imshow("eroded image", cleaned)

threshold1, thresh_inv = cv.threshold(gray, 35, 255, cv.THRESH_BINARY_INV )
cv.imshow('Simple Thresholded Inverse', thresh_inv)

# Adaptive Thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9)
cv.imshow('Adaptive Thresholding', adaptive_thresh)

cv.waitKey(0)
cv.destroyAllWindows()

del adaptive_thresh
del thresh