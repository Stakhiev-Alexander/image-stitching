from glob import glob

import cv2
import numpy as np

from opencv_imagestitching import stitch_images
from utils import cut_and_project

plot = False
imgs_wildcard = 'in/all/*.png'
imgs_num = 50

if __name__ == '__main__':
    imgs_list = glob(imgs_wildcard)
    imgs_num = len(imgs_list) if imgs_num is None else imgs_num
    imgs_list = imgs_list[:imgs_num]

    print('Total images: ', imgs_num)
    print("wp ol D1 D2 D3 Dmean")
    for scale in [0.25, 0.5]:
        print(f"scale: {scale}")
        for warp_percent in np.arange(0.02, 0.13, 0.02):
            for overlap_percent in np.arange(0.04, 0.25, 0.02):
        # for warp_percent in [0.15]:
        #     for overlap_percent in [0.15]:
                none_num = 0
                diffs = [0, 0, 0]

                return_none = False
                for img_path in imgs_list:
                    orig_img = cv2.imread(img_path)
                    orig_img = cv2.resize(orig_img, (0, 0), fx=scale, fy=scale)
                    h, w = orig_img.shape[0], orig_img.shape[1]
                    overlap_length = overlap_percent * w

                    w1 = int(w * (0.5 - overlap_percent / 2)) - 1
                    w2 = int(w * (0.5 + overlap_percent / 2)) - 1

                    gt_pts = [(w1, 0),
                              (w2, 0),
                              (w2, int(h / 2)),
                              (w2, h),
                              (w1, h),
                              (w1, int(h / 2))]

                    pts_to_transform = [(0, 0),
                                        (0, int(h / 2)),
                                        (0, h)]

                    img1, img2, matrix = cut_and_project(orig_img, overlap_percent=overlap_percent, warp_percent=warp_percent,
                                                 plot=False)

                    result_img, retval = stitch_images(img1, img2, feature_extractor='fast', feature_matcher='knn',
                                                       nfeatures=10000)

                    if result_img is None:
                        return_none = True
                        continue

                    pts_to_transform = np.float32(pts_to_transform).reshape(-1, 1, 2)
                    f_pts = cv2.perspectiveTransform(pts_to_transform, retval)

                    gt_pts = np.float32(gt_pts).reshape(-1, 1, 2)
                    gt_pts = cv2.perspectiveTransform(gt_pts, matrix)

                    f_pts = np.concatenate((f_pts, np.array([[[w2, 0]], [[w2, int(h / 2)]], [[w2, h]]])), axis=0)

                    if plot:
                        for pt in gt_pts:
                            result_img = cv2.circle(result_img, (int(pt[0][0]), int(pt[0][1])), radius=25,
                                                    color=(0, 255, 0), thickness=-1)

                        for pt in f_pts:
                            result_img = cv2.circle(result_img, (int(pt[0][0]), int(pt[0][1])), radius=15,
                                                    color=(0, 0, 255), thickness=-1)

                        orig_img_half = cv2.resize(orig_img, (0, 0), fx=0.25, fy=0.25)
                        result_half = cv2.resize(result_img, (0, 0), fx=0.25, fy=0.25)

                        cv2.imshow("orig_img", orig_img_half)
                        cv2.imshow("result", result_half)
                        cv2.waitKey()

                    # difference in Euclidean distance between points at the edges of the images
                    diffs[0] += abs(
                        np.linalg.norm(gt_pts[0] - gt_pts[1]) - np.linalg.norm(f_pts[0] - f_pts[3])) / overlap_length
                    diffs[1] += abs(
                        np.linalg.norm(gt_pts[2] - gt_pts[5]) - np.linalg.norm(f_pts[1] - f_pts[4])) / overlap_length
                    diffs[2] += abs(
                        np.linalg.norm(gt_pts[3] - gt_pts[4]) - np.linalg.norm(f_pts[2] - f_pts[5])) / overlap_length

                if return_none:
                    print(f'{warp_percent:.2f} '
                          f'{int(overlap_length)}({(overlap_percent * 100):.0f}%) '
                          f' None return')
                else:
                    print(f'{warp_percent:.2f} '
                          f'{int(overlap_length)}({(overlap_percent*100):.0f}%) '
                          f'{diffs[0] / imgs_num:.2f} '
                          f'{diffs[1] / imgs_num:.2f} '
                          f'{diffs[2] / imgs_num:.2f} '
                          f'{sum(diffs) / len(diffs) / imgs_num:.2f}')
        print()
        print()

