from glob import glob

import cv2
from skimage.metrics import structural_similarity as ssim

from opencv_imagestitching import stitch_images
from utils import cut_and_project

plot = False

if __name__ == '__main__':
    for img_path in glob('in/*.png'):
        orig_img = cv2.imread(img_path)
        img1, img2 = cut_and_project(orig_img)

        result_img = stitch_images(img2, img1, feature_extractor='sift', feature_matching='bf')

        h, w = orig_img.shape[0], orig_img.shape[1]
        result_img = cv2.resize(result_img, (w, h))

        if plot:
            orig_img_half = cv2.resize(orig_img, (0, 0), fx=0.5, fy=0.5)
            result_half = cv2.resize(result_img, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("orig_img", orig_img_half)
            cv2.imshow("result", result_half)
            cv2.waitKey()

        score = ssim(orig_img, result_img, multichannel=True)
        print(img_path.split('\\')[-1], ': ', score)
