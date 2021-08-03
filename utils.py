def cut_and_project(img, overlap_percent=0.2):
    """
    Method for slicing a image into two with overlap percentage and projection
    :param img: image in openCV format
    :param overlap_percent: percentage by which one sliced piece of the image will overlap the other
    :return: 2 cropped and projected images in openCV format
    """
    # yet without projection
    w = img.shape[1]
    crop_img1 = img[:, int(w * (0.5 - overlap_percent)):]
    crop_img2 = img[:, :int(w * (0.5 + overlap_percent))]

    return crop_img1, crop_img2
