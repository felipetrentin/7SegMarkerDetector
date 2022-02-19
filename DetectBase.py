import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
font = cv2.FONT_HERSHEY_COMPLEX


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [200, 0],
        [200, 200],
        [0, 200]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (200, 200))

    # return the warped image
    return warped


def main():
    # define color black
    black = np.array((
        [0, 0, 0],  #lower
        [30, 30, 30]))#upper
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.imread("Exemplos/Screenshot2.png") #load image
    image = imutils.resize(image, height=1000) #resize the image to 1000x1000
    ratio = image.shape[0] / 1000.0
    orig = image.copy() #create a copy of the image
    mask = cv2.inRange(image, black[0], black[1]) #binary mask
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=10) #filter as to reduce noise
    #closing = cv2.dilate(closing,kernel,iterations = 1)
    #cv2.imshow("mask", closing)
    # find edges in the image
    #edged = cv2.Canny(closing, 75, 200)

    #find and sort contours
    cnts = cv2.findContours(
        closing.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found a square
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt.any() == None:
        screenCnt = np.zeros((4, 2), dtype="float32")
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


    # show the bounding box
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    plt.subplot(121), plt.imshow(image)
    #cv2.imshow("Original", imutils.resize(orig, height = 650))
    binary_nums = cv2.inRange(warped, 150, 255)
    binary_nums = cv2.erode(binary_nums, kernel, iterations=1)
    binary_nums = cv2.dilate(binary_nums, kernel, iterations=2)
    #cv2.imshow("Scanned", binary_nums)
    plt.subplot(222), plt.imshow(warped, cmap='plasma')
    plt.subplot(221), plt.imshow(binary_nums, cmap='plasma')
    print(np.array([[binary_nums[19][46], binary_nums[35][62], binary_nums[70][67],
          binary_nums[86][48], binary_nums[70][30], binary_nums[40][36], binary_nums[52][47]]]))
    print(np.array([[binary_nums[20][100], binary_nums[32][121], binary_nums[71][117],
          binary_nums[85][100], binary_nums[70][85], binary_nums[36][84], binary_nums[20][100]]]))
    plt.show()
    # print(binary_nums)


main()
cv2.waitKey(0)
cv2.destroyAllWindows()
