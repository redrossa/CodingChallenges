import cv2
import numpy as np
from matplotlib import pyplot as plt


def convex_hull_pointing_up(ch):
    points_above_center, points_below_center = [], []

    x, y, w, h = cv2.boundingRect(ch)
    aspect_ratio = w / h

    if aspect_ratio < 0.9:
        vertical_center = y + h / 2

        for point in ch:
            if point[0][1] < vertical_center:
                points_above_center.append(point)
            elif point[0][1] >= vertical_center:
                points_below_center.append(point)

        left_x = points_below_center[0][0][0]
        right_x = points_below_center[0][0][0]
        for point in points_below_center:
            if point[0][0] < left_x:
                left_x = point[0][0]
            if point[0][0] > right_x:
                right_x = point[0][0]

        for point in points_above_center:
            if (point[0][0] < left_x) or (point[0][0] > right_x):
                return False
    else:
        return False

    return True


def cone_segment(rects):
    max_x = rects[0][0]
    min_x = rects[-1][0]
    max_y = rects[0][1]
    min_y = rects[-1][1]
    slope = (max_y - min_y) / (max_x - min_x)
    intercept = min_y - slope * min_x
    y = lambda x: int(slope * x + intercept)
    x = lambda y: int((y - intercept) / slope)
    e1 = (x(0), 0)
    e2 = (0, y(0)) if slope < 0 else (2000, y(2000))
    return e1, e2


fpath = 'red.png'
img = cv2.imread(fpath)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

img_thresh_low = cv2.inRange(img_hsv, np.array([0, 135, 135]), np.array([15, 255, 255]))
img_thresh_high = cv2.inRange(img_hsv, np.array([159, 135, 135]), np.array([179, 255, 255]))
img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_high)

kernel = np.ones((5, 5))
img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)

img_edges = cv2.Canny(img_thresh_blurred, 80, 160)

contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

approx_contours = []
for c in contours:
    approx = cv2.approxPolyDP(c, 10, closed=True)
    approx_contours.append(approx)

all_convex_hulls = []
for ac in approx_contours:
    all_convex_hulls.append(cv2.convexHull(ac))

convex_hulls_3to10 = []
for ch in all_convex_hulls:
    if 3 <= len(ch) <= 10:
        convex_hulls_3to10.append(cv2.convexHull(ch))

cones = []
bounding_rects = []
for ch in convex_hulls_3to10:
    if convex_hull_pointing_up(ch):
        cones.append(ch)
        rect = cv2.boundingRect(ch)
        bounding_rects.append(rect)

right_bounding_rects = bounding_rects[::2]
right_segment = cone_segment(right_bounding_rects)

left_bounding_rects = bounding_rects[1::2]
left_segment = cone_segment(left_bounding_rects)

img_cones = np.zeros_like(img_edges)
cv2.drawContours(img_cones, cones, -1, (255, 255, 255), 2)
plt.imshow(img_cones)
plt.show()

img = cv2.line(img, right_segment[0], right_segment[1], (0, 0, 255), 3)
img = cv2.line(img, left_segment[0], left_segment[1], (0, 0, 255), 3)
cv2.imwrite('out.png', img)
