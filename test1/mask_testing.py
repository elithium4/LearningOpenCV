import cv2 as cv
import numpy as np

#найти и оставить на картинке только красные фрагменты

img = cv.imread("sample.jpg")
img = cv.addWeighted(img, 1.5, img, -0.2, 0)

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)


lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv.inRange(img_hsv, lower_red, upper_red)

lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv.inRange(img_hsv, lower_red, upper_red)

mask = mask0+mask1

kernel = np.ones((20,20),np.uint8)
mask = cv.dilate(mask, kernel, iterations = 1)
mask = cv.erode(mask, kernel, iterations = 1)


new_img = cv.bitwise_and(img, img, mask = mask)


cv.imwrite("mask.jpg", mask)
cv.imwrite("img_with_mask.jpg", new_img)
