import cv2 as cv


# Cv Question asked in 2024 mse
img = cv.imread('../Resources/Photos/cats.jpg',cv.IMREAD_UNCHANGED)
cv.imshow("input-image",img)

(r,g,b) = cv.split(img)

img_merged = cv.merge((b,g,r))

cv.imshow("output-image",img_merged)

cv.waitKey(0)
cv.destroyAllWindows()