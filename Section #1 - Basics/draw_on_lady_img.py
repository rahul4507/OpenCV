import cv2 as cv

lady = cv.imread('../Resources/Photos/lady.jpg')
cv.imshow('Beautiful Lady', lady)

# let's draw a rectangle on her
lady_img = cv.rectangle(lady, (0,0), (lady.shape[0]//2, lady.shape[1]//2),(0,0,255), 2)
cv.imshow('lady_with_rectangle', lady_img)

# let's draw a circle on her
lady_img = cv.circle(lady, (200,200), 25,(255,0,0), 2)
cv.imshow('lady_with_rectangle_and_circle', lady_img)

# let's draw a line on her
lady_img = cv.line(lady, (200,200), (637,685),(255,0,0), 2)
cv.imshow('lady_with_rectangle_and_circle_and_line', lady_img)

# let's write something on her
lady_img = (
    cv.putText(
    lady, "Hello, My Name is Rahul!!!", (0,200),cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,255,153),2
))
cv.imshow('lady_img', lady_img)

# let's change contrast and brightness of image
lady_img= (
    cv.convertScaleAbs(lady_img,alpha=1.5,beta=10)
)

cv.imshow('lady_img', lady_img)

cv.waitKey(0)
cv.destroyAllWindows()
