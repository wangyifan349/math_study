"""pip install opencv-python opencv-python-headless opencv-contrib-python"""

import cv2
import numpy as np
def stitch_images(img1, img2):
    # -------------------------------
    # 将图像转为灰度
    # -------------------------------
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # -------------------------------
    # 使用SIFT特征检测器来检测特征点并计算描述符
    # -------------------------------
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    # -------------------------------
    # 使用FLANN匹配特征
    # -------------------------------
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    # -------------------------------
    # 存储符合条件的优秀匹配点
    # -------------------------------
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    # -------------------------------
    # 如果找到足够的匹配点，进行图像拼接
    # -------------------------------
    if len(good_matches) > 10:
        # 获取匹配点的坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        # 计算单应性矩阵
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # -------------------------------
        # 使用单应性矩阵将图像进行透视变换
        # -------------------------------
        width = img1.shape[#citation-1](citation-1) + img2.shape[#citation-1](citation-1)
        height = img1.shape[0]
        result = cv2.warpPerspective(img1, H, (width, height))
        result[0:img2.shape[0], 0:img2.shape[#citation-1](citation-1)] = img2
        return result
    else:
        print("匹配点不足，无法拼接")
        return None
# -------------------------------
# 读取输入图像
# -------------------------------
img1 = cv2.imread('image1.jpg')  # 第一张图像
img2 = cv2.imread('image2.jpg')  # 第二张图像
# -------------------------------
# 执行图像拼接
# -------------------------------
result = stitch_images(img1, img2)
# -------------------------------
# 显示结果
# -------------------------------
if result is not None:
    cv2.imshow("Stitched Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
