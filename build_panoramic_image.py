import cv2
import numpy as np
from typing import Tuple, List, Optional

# for type hints
RgbImg = np.array                   # shape = (h, w, 3)
Point2dList = np.array              # shape = (N, 2)
DescriptorsList = np.array          # shape = (N, DESCRIPTOR_SIZE)
HMat = np.array                     # shape = (3, 3)
BBox = Tuple[int, int, int, int]    # BoundingBox(x, y, right, top)


def build_panoramic_image(imgs: List[RgbImg]) -> RgbImg:
    used_imgs: List[Tuple[RgbImg, HMat]] = _find_all_homography(imgs)
    result_bbox: BBox = _get_result_panorama_bbox(used_imgs)
    return _join_panoramic_image(used_imgs, result_bbox)


def _detect_sift_points_and_descriptors(img: RgbImg) -> \
        Tuple[Point2dList, DescriptorsList]:
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return np.array([kp.pt for kp in keypoints]), descriptors


def _snn_matching(query_descriptors: DescriptorsList,
                  train_descriptors: DescriptorsList) -> List[cv2.DMatch]:
    matcher = cv2.BFMatcher_create(cv2.NORM_L2, False)
    matches_all = matcher.knnMatch(query_descriptors, train_descriptors, 2)
    return [m1 for m1, m2 in matches_all if m1.distance < 0.8 * m2.distance]


def _find_matches(points1: Point2dList, descriptors1: DescriptorsList,
                  points2: Point2dList, descriptors2: DescriptorsList) -> \
        Tuple[Point2dList, Point2dList]:
    matches = _snn_matching(descriptors1, descriptors2)
    return np.array([points1[m.queryIdx] for m in matches]), \
        np.array([points2[m.trainIdx] for m in matches])


def _find_homography(points1: Point2dList, points2: Point2dList) -> \
        Optional[HMat]:
    _MIN_MATCHES, _MIN_INLIERS = 100, 30
    if len(points1) < _MIN_MATCHES:
        return None
    hmat, inliers_mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    return hmat if np.sum(inliers_mask) >= _MIN_INLIERS else None


def _find_all_homography(imgs: List[RgbImg]) -> List[Tuple[RgbImg, HMat]]:
    first_img = imgs[0]
    result = [(first_img, np.eye(3))]
    points1, descriptors1 = _detect_sift_points_and_descriptors(first_img)
    for img in imgs[1:]:
        points, descriptors = _detect_sift_points_and_descriptors(img)
        matches = _find_matches(points1, descriptors1, points, descriptors)
        hmat = _find_homography(matches[0], matches[1])
        if hmat is not None:
            result.append((img, hmat))
    return result


def _get_transformed_img_bbox(img: RgbImg, hmat: HMat) -> BBox:
    h, w = img.shape[:2]
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    corners = cv2.perspectiveTransform(corners, hmat).reshape(-1, 2)
    return np.min(corners[:, 0]), np.min(corners[:, 1]), \
        np.max(corners[:, 0]), np.max(corners[:, 1])


def _max_panoramic_image_bbox(first_image: RgbImg) -> BBox:
    _MAX_SIZE_DIAMETERS = 2.0
    h, w = first_image.shape[:2]
    sz = int(_MAX_SIZE_DIAMETERS * np.linalg.norm((w, h)))
    return -sz, -sz, w + sz, h + sz


def _get_result_panorama_bbox(used_imgs: List[Tuple[RgbImg, HMat]]) -> BBox:
    x, y, r, t = 0.0, 0.0, 0.0, 0.0
    for img, hmat in used_imgs:
        i_x, i_y, i_r, i_t = _get_transformed_img_bbox(img, hmat)
        x, y, r, t = min(x, i_x), min(y, i_y), max(r, i_r), max(t, i_t)
    x, y, r, t = int(x), int(y), int(r), int(t)

    min_x, min_y, max_r, max_t = _max_panoramic_image_bbox(used_imgs[0][0])
    return max(x, min_x), max(y, min_y), min(r, max_r), min(t, max_t)


def _join_panoramic_image(used_imgs: List[Tuple[RgbImg, HMat]],
                          result_bbox: BBox) -> RgbImg:
    offset_hmat = np.array([[1, 0, -result_bbox[0]],
                            [0, 1, -result_bbox[1]],
                            [0, 0, 1]])

    result_w = result_bbox[2] - result_bbox[0]
    result_h = result_bbox[3] - result_bbox[1]

    sum_rgb = np.zeros((result_h, result_w, 3))
    sum_mask = np.zeros((result_h, result_w))
    for img, hmat in used_imgs:
        warped_img = cv2.warpPerspective(
            img, offset_hmat @ hmat, (result_w, result_h))
        mask = np.sum(warped_img, axis=2) != 0
        mask = cv2.erode(mask.astype(sum_mask.dtype), np.ones((3, 3)))
        sum_rgb[mask != 0] += warped_img[mask != 0]
        sum_mask += mask

    sum_rgb[sum_mask != 0] /= sum_mask[sum_mask != 0][:, None]
    return sum_rgb.astype(np.uint8)
