def cut_and_project(img, overlap_percent=0.2):
    w = img.shape[1]
    crop_img1 = img[:, :int(w*(0.5+overlap_percent))]
    crop_img2 = img[:, int(w*(0.5-overlap_percent)):]

    return crop_img1, crop_img2
