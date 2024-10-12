import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/cats.jpg', cv.IMREAD_UNCHANGED)
cv.imshow('img', img)

# 1. Gaussian Filtering that is GaussianBlur
gaussian_blur_img  = cv.GaussianBlur(img, (11,11), cv.BORDER_DEFAULT)
cv.imshow('gaussian blur image', gaussian_blur_img)

#2. Median Blur
median_blur = cv.medianBlur(img, 11)
cv.imshow('median blur', median_blur)

#3. Bilateral Filtering
bilateral_filter = cv.bilateralFilter(img, 10,75,75)
cv.imshow('bilateral filter', bilateral_filter)

# Sharpening an Image Using Custom 2D-Convolution Kernels
kernel3 = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
sharp_img = cv.filter2D(src=img, ddepth=-1, kernel=kernel3)
cv.imshow('Sharpened', sharp_img)

# Apply Identity kernel --> it will not have any effect on the image at all
identity_kernel = np.array([[0, 0,  0],
                   [0,  1, 0],
                    [0, 0 , 0]])
img_after_identity_kernel = cv.filter2D(src=img, ddepth=-1, kernel=identity_kernel)
cv.imshow('img_after_identity_kernel', img_after_identity_kernel)

# Apply Edge Detection kernel
edge_kernel = np.array([[-1, -1,  -1],
                   [-1,  8, -1],
                    [-1, -1 , -1]])
img_after_edge_kernel = cv.filter2D(src=img, ddepth=-1, kernel=edge_kernel)
cv.imshow('img_after_edge_kernel', img_after_edge_kernel)

# Apply Box Blur kernel
box_blur_kernel =(np.ones((5,5),np.float32))/25
img_after_box_blur_kernel = cv.filter2D(src=img, ddepth=-1, kernel=box_blur_kernel)
cv.imshow('img_after_box_blur_kernel', img_after_box_blur_kernel)

# Apply cv2's Blur function
img_after_blur_fun = cv.blur(img,(5,5))
cv.imshow('img_after_blur_fun', img_after_blur_fun)

cv.waitKey(0)
cv.destroyAllWindows()