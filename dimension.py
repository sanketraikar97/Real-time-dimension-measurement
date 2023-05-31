from scipy.spatial import distance as dist
import cv2
import numpy as np
import argparse
import imutils
from imutils import perspective
from imutils import contours
import math

scale = 3
wPaper = 210 * scale
hPaper = 297 * scale
# Intialize an argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, help="path to the input image")
args = vars(ap.parse_args())


# Helper function to calculate midpoint
def midpoint(pts1, pts2):
    return ((pts1[0] + pts2[0]) * 0.5, (pts1[1] + pts2[1]) * 0.5)


def contourDetect(image, minarea):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    # Apply Canny edge detection to find out edges and then apply dilation and erosion to get rid of irregularities
    kernel = np.ones((5, 5))
    canny = cv2.Canny(gray, 80, 150)
    canny = cv2.dilate(canny, kernel, iterations=2)
    canny = cv2.erode(canny, kernel, iterations=1)
    contour, hierarchy = cv2.findContours(
        canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = imutils.grab_contours((contour, hierarchy))
    (contour, _) = contours.sort_contours(contour)
    for i in contour:
        if cv2.contourArea(i) > minarea:
            # box = cv2.minAreaRect(i)
            perimeter = cv2.arcLength(i, True)
            corners = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            box = cv2.boundingRect(corners)
            # print(corners)

    return (contour, corners, image)


def objectCont(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    # Apply Canny edge detection to find out edges and then apply dilation and erosion to get rid of irregularities
    kernel = np.ones((2, 2))
    canny = cv2.Canny(gray, 80, 150)
    canny = cv2.dilate(canny, kernel, iterations=2)
    canny = cv2.erode(canny, kernel, iterations=1)
    contour, hierarchy = cv2.findContours(
        canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = imutils.grab_contours((contour, hierarchy))
    (contour, _) = contours.sort_contours(contour)
    return (contour, image)


def reorderPoints(points):
    # print(points.shape)
    newPoints = np.zeros_like(points)
    points = points.reshape((points.shape[0], points.shape[2]))
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints


def warpImg(img, points, width, height, pad=20):
    points = reorderPoints(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    WarpImg = cv2.warpPerspective(img, matrix, (width, height))
    WarpImg = WarpImg[pad : WarpImg.shape[0] - pad, pad : WarpImg.shape[1] - pad]
    return WarpImg


def calculatePPM(box, width):
    box = perspective.order_points(box)
    print(box)
    (tl, tr, br, bl) = box
    (mlx, mly) = midpoint(tl, bl)
    (mrx, mry) = midpoint(tr, br)
    # dw = dist.euclidean((mlx, mly), (mrx, mry))
    dw = dist.euclidean(tl, tr)
    ppm = dw / width
    # print(dw, ppm)
    return ppm


# load the image, convert it to grayscale, and blur it slightly
img = cv2.imread(args["image"])
# img = cv2.imread("images/1.jpg")

(contour, obox, img) = contourDetect(img, minarea=50000)
# cv2.imshow("image", img)
# print(obox)
imgWarp = warpImg(img, obox, wPaper, hPaper)
cv2.imshow("image", imgWarp)
cv2.waitKey(0)

(final_cont, image) = objectCont(imgWarp)
obox = obox.reshape((4, 2))
# ppm = calculatePPM(obox, wPaper)

# Loop over the contours
cpy = image.copy()
for cnt in final_cont:
    # if cv2.contourArea(cnt) < 100:
    #     continue

    # Order the points of the box in the following order: top-left,top-right,bottom-right,bottom-left
    # final_box = np.asarray(final_box)
    # print(final_box.shape)
    if cv2.contourArea(cnt) > 1000:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    cv2.drawContours(cpy, [box.astype("int")], -1, (0, 255, 0), 2)

    # draw a circle at the corner points
    for x, y in box:
        cv2.circle(cpy, (int(x), int(y)), 3, (0, 0, 255), -1)

    # Calculate the mid points
    (tl, tr, br, bl) = box
    (mtx, mty) = midpoint(tl, tr)
    (mbx, mby) = midpoint(br, bl)
    (mlx, mly) = midpoint(tl, bl)
    (mrx, mry) = midpoint(tr, br)

    # Draw the midpoints
    cv2.circle(cpy, (int(mtx), int(mty)), 3, (0, 0, 255), -1)
    cv2.circle(cpy, (int(mbx), int(mby)), 3, (0, 0, 255), -1)
    cv2.circle(cpy, (int(mlx), int(mly)), 3, (0, 0, 255), -1)
    cv2.circle(cpy, (int(mrx), int(mry)), 3, (0, 0, 255), -1)

    # Connect the midpoints
    cv2.line(cpy, (int(mtx), int(mty)), (int(mbx), int(mby)), (0, 255, 255), 2)
    cv2.line(cpy, (int(mlx), int(mly)), (int(mrx), int(mry)), (0, 255, 255), 2)

    dtb = dist.euclidean((mtx // scale, mty // scale), (mbx // scale, mby // scale))
    dlr = dist.euclidean((mlx // scale, mly // scale), (mrx // scale, mry // scale))

    width = round((dlr) / 10, 1)
    height = round((dtb) / 10, 1)

    cv2.putText(
        cpy,
        f"width:{width}cm",
        (int(mtx - 15), int(mty - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        cpy,
        f"height:{height}cm",
        (int(mlx - 15), int(mly)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Output", cpy)
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()
