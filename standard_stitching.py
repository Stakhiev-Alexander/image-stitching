from glob import glob

import cv2
from imutils import paths, is_cv3

for folder in sorted(glob("input/*")):
    # folder = "input/37"
    imagePaths = sorted(list(paths.list_images(folder)))
    images = []

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        # image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        images.append(image)

    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if is_cv3() else cv2.Stitcher_create()
    # mask = np.zeros(images[0].shape, dtype="uint8")
    # cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    # (status, stitched) = stitcher.stitch(images, masks=[mask1, mask1])
    (status, stitched) = stitcher.stitch(images)

    if status == 0:
        stitched = cv2.resize(stitched, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Stitched", stitched)
        cv2.waitKey(0)
    else:
        print("[INFO] image stitching failed ({})".format(status))
