import sys
from glob import glob

import cv2 as cv
from imutils import paths

use_mask = False
img = None
templ = None
mask = None
image_window = "Source Image"
result_window = "Result window"
match_method = 0
max_Trackbar = 5


def MatchingMethod(param):
    match_method = param

    img_display = img.copy()

    method_accepts_mask = (cv.TM_SQDIFF == match_method or match_method == cv.TM_CCORR_NORMED)
    if use_mask and method_accepts_mask:
        result = cv.matchTemplate(img, templ, match_method, None, mask)
    else:
        result = cv.matchTemplate(img, templ, match_method)

    # cv.imshow(result_window, result)
    cv.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)

    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)

    if match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc

    print(matchLoc)
    print((matchLoc[1] + templ.shape[1], matchLoc[0] + templ.shape[0]))
    # cv.rectangle(img_display, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0, 0, 255), 2, 8,
    #              0)
    cv.rectangle(img_display, (matchLoc[0], matchLoc[1]), (matchLoc[0] + templ.shape[1], matchLoc[1] + templ.shape[0]), (0, 0, 255), 20)
    # cv.rectangle(result, matchLoc, (matchLoc[0] + templ.shape[0], matchLoc[1] + templ.shape[1]), (0, 0, 255), 2, 8, 0)
    img_display = cv.resize(img_display, (0, 0), fx=0.25, fy=0.25)
    cv.imshow(image_window, img_display)
    cv.waitKey(0)


if __name__ == "__main__":
    for folder in sorted(glob("input/*")):
        # folder = "input/280"
        imagePaths = sorted(list(paths.list_images(folder)))
        images = []

        for imagePath in imagePaths:
            image = cv.imread(imagePath)
            # image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
            images.append(image)

        img = images[1]
        templ = images[0][:, int(images[0].shape[1] * 0.85):]

        if use_mask:
            mask = cv.imread("", cv.IMREAD_COLOR)

        cv.namedWindow(image_window)
        cv.namedWindow(result_window)

        cv.imshow(result_window, cv.resize(templ, (0, 0), fx=0.25, fy=0.25))

        trackbar_label = 'Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED'
        cv.createTrackbar(trackbar_label, image_window, match_method, max_Trackbar, MatchingMethod)

        MatchingMethod(match_method)
