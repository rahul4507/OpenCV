import cv2 as cv
import numpy as np
from keras.src.backend.jax.numpy import identity

img1 = cv.imread('./Resources/Photos/park.jpg',cv.IMREAD_UNCHANGED)
img2 = cv.imread('./Resources/Photos/lady.jpg',cv.IMREAD_UNCHANGED)

cv.imshow('img1',img1)
cv.imshow('img2',img2)

kernel = np.array([[-1, -1, -1],  [-1, 8, -1],   [-1, -1, -1]])
filter_applied = cv.filter2D(img1, -1, kernel)
cv.imshow("After Applying identity kernel", filter_applied)

# converting to GrayScale
img_grayscale = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imshow('grayscale img', img_grayscale)

# converting to HSV
img_grayscale = cv.cvtColor(img1, cv.COLOR_BGR2HSV)
cv.imshow('HSV img', img_grayscale)

# converting to LAB
img_grayscale = cv.cvtColor(img1, cv.COLOR_BGR2LAB)
cv.imshow('LAB img', img_grayscale)

# converting to RGB
img_grayscale = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
cv.imshow('RGB img', img_grayscale)

# Blur the image
img_blur = cv.GaussianBlur(img1, (5,5),cv.BORDER_DEFAULT)
cv.imshow("blur image", img_blur)

# Edge Cascade
canny = cv.Canny(img_blur, threshold1=125, threshold2=175)
cv.imshow("canny", canny)

# Dilate Image
kernel = np.ones((3,3),np.uint8)
dilated = cv.dilate(canny, kernel, iterations=1)
cv.imshow("Dilated", dilated)

# Erode Image
kernel = np.ones((3,3),np.uint8)
eroded = cv.erode(dilated, kernel, iterations=1)
cv.imshow("eroded", eroded)

# let's perform arithmatic operations on image
img1 = cv.resize(img1, (img2.shape[1], img2.shape[0]))
added = cv.addWeighted(img1, 0.2, img2, 0.8, 0)

image = cv.bitwise_xor(img1,img2)
cv.imshow("blended images",added)
cv.imshow("bitwise or",image)


# thresholding
ret1,thresh1 = cv.threshold(img1,127,255,cv.THRESH_BINARY)
ret2,thresh2 = cv.threshold(img1,127,255,cv.THRESH_BINARY_INV)
ret3,thresh3 = cv.threshold(img1,127,255,cv.THRESH_TRUNC)
ret4,thresh4 = cv.threshold(img1,127,255,cv.THRESH_TOZERO)
ret5,thresh5 = cv.threshold(img1,127,255,cv.THRESH_TOZERO_INV)

cv.imshow("thresh1", thresh1)
cv.imshow("thresh2", thresh2)
cv.imshow("thresh3", thresh3)
cv.imshow("thresh4", thresh4)
cv.imshow("thresh5", thresh5)


cv.waitKey(0)
cv.destroyAllWindows()

del img_blur
del img1
del img2
del canny
del dilated
del filter_applied
del thresh5
del thresh1
del thresh2
del thresh3
del thresh4
