import cv2
import imutils
import numpy as np

cv2.ocl.setUseOpenCL(False)

feature_extractors = ('sift', 'kaze', 'akaze', 'brisk', 'orb', 'fast')
feature_matchers = ('bf', 'knn')


def _create_matcher(matching_method, extraction_method):
    """
    Create and return a Matcher Object
    """
    cross_check = True if matching_method == 'bf' else False

    if extraction_method == 'orb' or extraction_method == 'brisk':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    return matcher


def _match_key_points(features1, features2, matching_method, extraction_method, ratio=0.75):
    matcher = _create_matcher(matching_method, extraction_method)

    if matching_method == 'bf':
        matches = matcher.match(features1, features2)

        # Sort the features in order of distance.
        # The points with small distance (more similarity) are ordered first in the vector
        matches = sorted(matches, key=lambda x: x.distance)
    else:
        raw_matches = matcher.knnMatch(features1, features2, 2)
        matches = []

        # loop over the raw matches and ensure the distance is within a certain ratio of each
        for m, n in raw_matches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if m.distance < n.distance * ratio:
                matches.append(m)

    return matches


def _detect_and_describe(image, method, nfeatures=5000):
    """
    Compute key points and feature descriptors using an specific method
    """

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.SIFT_create(nfeatures=nfeatures)
    elif method == 'kaze':
        descriptor = cv2.KAZE_create()
    elif method == 'akaze':
        descriptor = cv2.AKAZE_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    elif method == 'fast':
        descriptor = cv2.FastFeatureDetector_create()
        kps = descriptor.detect(image)

    # get keypoints and descriptors
    if method != 'fast':
        kps, features = descriptor.detectAndCompute(image, None)
    else:
        descriptor = cv2.SIFT_create(nfeatures=nfeatures)
        kps = sorted(kps, key=lambda kp: kp.response, reverse=True)
        kps = kps[:nfeatures]
        kps, features = descriptor.compute(image, kps)

    return kps, features


def _get_homography(kps1, kps2, matches, reprojection_threshold=4):
    # convert the keypoints to numpy arrays
    kps1 = np.float32([kp.pt for kp in kps1])
    kps2 = np.float32([kp.pt for kp in kps2])

    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kps1[m.queryIdx] for m in matches])
        ptsB = np.float32([kps2[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        retval, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                            reprojection_threshold)

        return retval
    else:
        return None


def _perspective_and_size_correction(img1, img2, retval):
    # Apply panorama correction
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0] + img2.shape[0]

    result = cv2.warpPerspective(img1, retval, (width, height))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    # transform the panorama image to grayscale and threshold it
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

    # crop the image to the bbox coordinates
    # result = result[y:y + h, x:x + w]

    return result


def stitch_images(img1, img2, feature_extractor='sift', feature_matcher='bf', nfeatures=5000):
    """
    Function applies keypoint detection, matches these keypoints and stitches them together
    :param img1: image in openCV format
    :param img2: image in openCV format
    :param feature_extractor: method to detect keypoint (one of 'sift', 'kaze', 'akaze', 'brisk', 'orb')
    :param feature_matcher: method to match keypoint (one of 'bf', 'knn')
    :param nfeatures: feature number constraint for SIFT
    :return: resulting image if successful otherwise None
    """
    assert feature_extractor in feature_extractors
    assert feature_matcher in feature_matchers

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    kps1, features1 = _detect_and_describe(img1_gray, method=feature_extractor, nfeatures=nfeatures)
    kps2, features2 = _detect_and_describe(img2_gray, method=feature_extractor, nfeatures=nfeatures)

    matches = _match_key_points(features1, features2, matching_method=feature_matcher,
                                extraction_method=feature_extractor)

    retval = _get_homography(kps1, kps2, matches)
    if retval is None:
        return None

    result = _perspective_and_size_correction(img1, img2, retval)

    return result, retval
