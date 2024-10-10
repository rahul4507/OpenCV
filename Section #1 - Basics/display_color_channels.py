import cv2 as cv
import numpy as np

#Print each color channels from an image
img = cv.imread('../Resources/Photos/lady.jpg',cv.IMREAD_UNCHANGED)

(b,g,r) = cv.split(img)
zeros = np.zeros_like(b)

r_img = cv.merge([zeros, zeros, r])  # Only R channel
g_img = cv.merge([zeros, g, zeros])  # Only G channel
b_img = cv.merge([b, zeros, zeros])  # Only B channel

# Display the images
cv.imshow('R-RGB', r_img)
cv.imshow('G-RGB', g_img)
cv.imshow('B-RGB', b_img)

cv.waitKey(0)
cv.destroyAllWindows()
