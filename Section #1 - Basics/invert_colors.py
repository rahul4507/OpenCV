import cv2 as cv


# Cv Question asked in 2024 mse
img = cv.imread('../Resources/Photos/lady.jpg',cv.IMREAD_UNCHANGED)
cv.imshow("input-image",img)

(r,g,b) = cv.split(img)

img_merged = cv.merge((b,g,r))

cv.imshow("output-image",img_merged)

output_image = cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow("output using cvtcolor", output_image)

cv.waitKey(0)
cv.destroyAllWindows()