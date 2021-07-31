import time
from glob import glob

import cv2
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from opencv_imagestitching import stitch_images
from utils import cut_and_project

plot = False

if __name__ == '__main__':
    imgs_num = len(glob('in/*/*/*.png'))
    print(imgs_num)
    for nfeatures in range(1100, 3001, 200):
        ssim_sum = 0
        none_num = 0
        start_time = time.time()

        for img_path in tqdm(glob('in/*/*/*.png')):

            orig_img = cv2.imread(img_path)
            img1, img2 = cut_and_project(orig_img)

            result_img = stitch_images(img2, img1, feature_extractor='sift', feature_matching='knn',
                                       nfeatures=nfeatures)

            if result_img is None:
                none_num += 1
                continue

            h, w = orig_img.shape[0], orig_img.shape[1]
            result_img = cv2.resize(result_img, (w, h))

            if plot:
                orig_img_half = cv2.resize(orig_img, (0, 0), fx=0.5, fy=0.5)
                result_half = cv2.resize(result_img, (0, 0), fx=0.5, fy=0.5)
                cv2.imshow("orig_img", orig_img_half)
                cv2.imshow("result", result_half)
                cv2.waitKey()

            score = ssim(orig_img, result_img, multichannel=True)
            ssim_sum += score
            # print(img_path.split('\\')[-1], ': ', score)

        print(ssim_sum / (imgs_num - none_num), time.time() - start_time, none_num)
