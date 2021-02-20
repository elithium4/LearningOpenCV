import cv2 as cv
import numpy as np

#проверка, как работает общее выделение контуров
#попытка выделить самый большой контур предмета красного цвета

img = cv.imread("sample2.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, img_thresh = cv.threshold(img_gray, 127, 255, 0)

contours, hierarchy = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

n_img = img.copy()
cv.drawContours(n_img, contours, -1, (0, 0, 0), 2, cv.LINE_AA, hierarchy, 1)

cv.imwrite("cont_v_1.jpg", n_img)


img_better = cv.addWeighted(img, 1.5, img, -0.2, 0)
img_hsv = cv.cvtColor(img_better, cv.COLOR_BGR2HSV)


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


new_img = img.copy()

contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

max_s = 0
main_c = contours[0]

c = max(contours, key=cv.contourArea)

cv.drawContours(new_img, [c], 0, (0, 0, 0), 2)

cv.imwrite("cont_v_2.jpg", new_img)



