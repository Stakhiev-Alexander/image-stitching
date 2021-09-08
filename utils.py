import cv2
import numpy as np


def project(in_pts, out_pts, img, h, w):
    matrix = cv2.getPerspectiveTransform(in_pts, out_pts)
    out_img = cv2.warpPerspective(img, matrix, (w, h), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
    return out_img


def cut_and_project(img, overlap_percent=0.2, warp_percent=0.1):
    """
    Method for slicing a image into two with overlap percentage and projection
    :param img: image in openCV format
    :param overlap_percent: percentage by which one sliced piece of the image will overlap the other
    :return: 2 cropped and projected images in openCV format
    """
    # yet without projection
    h, w = img.shape[0], img.shape[1]
    # print(h, w)
    crop_img1 = img[:, int(w * (0.5 - overlap_percent)):]
    crop_img2 = img[:, :int(w * (0.5 + overlap_percent))]

    h, w = crop_img1.shape[0], crop_img1.shape[1]
    # print(h, w)
    in_pts = np.float32([[0, int(h * warp_percent)], [w - 1, 0], [w - 1, h - 1], [0, int(h * (1 - warp_percent))]])
    out_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    cp_img2 = project(in_pts, out_pts, crop_img2, h, w)

    in_pts = np.float32([[0, 0], [w - 1, int(h * warp_percent)], [w - 1, int(h * (1 - warp_percent))], [0, h - 1]])
    out_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    cp_img1 = project(in_pts, out_pts, crop_img1, h, w)

    # cv2.imshow("1", cv2.resize(cp_img1, (0, 0), fx=0.25, fy=0.25))
    # cv2.imshow("2", cv2.resize(cp_img2, (0, 0), fx=0.25, fy=0.25))
    # cv2.waitKey()
    return cp_img1, cp_img2
