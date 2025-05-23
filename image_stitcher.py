import cv2
import numpy as np

def correct_color_balance(img1, img2):
    """
    对两张图像进行色彩校正，使其颜色更加一致。
    这里采用简单的平均颜色校正方法。
    """
    avg_color1 = np.mean(img1, axis=(0, 1))
    avg_color2 = np.mean(img2, axis=(0, 1))

    # 计算颜色比例因子
    color_ratios = avg_color1 / avg_color2

    # 将比例因子应用于 img2
    corrected_img2 = img2 * color_ratios
    corrected_img2 = np.clip(corrected_img2, 0, 255).astype(np.uint8) # 确保像素值在 0-255 范围内

    return img1,corrected_img2

def detect_and_describe(image, method="sift"):
    """
    检测图像中的特征点和提取特征描述符。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "sift":
        descriptor = cv2.SIFT_create()
    elif method == "orb":
        descriptor = cv2.ORB_create()
    elif method == "brisk":
        descriptor = cv2.BRISK_create()
    elif method == "akaze":
        descriptor = cv2.AKAZE_create()
    else:
        raise ValueError("Invalid feature detection method: {}".format(method))

    (kps, features) = descriptor.detectAndCompute(gray, None)

    # 将关键点从 KeyPoint 对象转换为 NumPy 数组 (不再使用列表表达式)
    keypoints = []
    for kp in kps:
        keypoints.append(kp.pt)
    kps_np = np.float32(keypoints)

    return (kps_np, features)

def match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reproj_thresh=4.0):
    """
    使用 Lowe's ratio test 和 RANSAC 算法来匹配两张图像中的关键点。
    """
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)

    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:
        # 从关键点坐标构建点数组 (不再使用列表表达式)
        ptsA = []
        ptsB = []
        for (trainIdx, queryIdx) in matches:
            ptsA.append(kpsA[queryIdx])
            ptsB.append(kpsB[trainIdx])

        ptsA_np = np.float32(ptsA)
        ptsB_np = np.float32(ptsB)

        (H, status) = cv2.findHomography(ptsA_np, ptsB_np, cv2.RANSAC, reproj_thresh)
        return (matches, H, status)

    return None

def feather_blending(img1, img2, mask):
    """
    使用羽化融合技术平滑图像之间的过渡。
    """
    h, w, _ = img1.shape
    result = np.zeros((h, w, 3), dtype="float32")

    for y in range(0, h):
        for x in range(0, w):
            if mask[y, x] == 0:  # 来自img1
                result[y, x] = img1[y, x]
            elif mask[y, x] == 255:  # 来自img2
                result[y, x] = img2[y, x]
            else: # 融合区域, 使用线性插值
                alpha = mask[y, x] / 255.0
                result[y, x] = (1 - alpha) * img1[y, x] + alpha * img2[y, x]

    return result.astype("uint8")

def create_mask(img1, img2, offset_x):
    """
    为羽化融合创建遮罩。
    """
    h, w, _ = img1.shape
    mask = np.zeros((h, w), dtype="uint8")

    overlap_width = w - offset_x
    start = w - overlap_width    # 确保起始位置不越界 （可以简化为 offset_x）
    end = w

    for x in range(start, end):
        alpha = (x - start) / overlap_width  # 根据像素位置计算权重
        mask[:, x] = alpha * 255  # 将权重映射到 0-255 的范围

    return mask

def stitch_images(images, features_method="sift", blending="feather"):
    """将多张图像拼接成一张全景图, 实现自动图像排序和羽化融合.
    """

    # 1. 自动图像排序 (基于特征匹配关系)
    num_images = len(images)
    if num_images > 2:
        # 构建匹配矩阵，记录每两张图像之间的匹配质量
        match_matrix = np.zeros((num_images, num_images))
        for i in range(num_images):
            for j in range(i + 1, num_images):
                (kpsA, featuresA) = detect_and_describe(images[i], method=features_method)
                (kpsB, featuresB) = detect_and_describe(images[j], method=features_method)
                M = match_keypoints(kpsA, kpsB, featuresA, featuresB)
                if M is not None:
                    matches, _, _ = M
                    match_matrix[i, j] = len(matches)
                    match_matrix[j, i] = len(matches) # 保证矩阵对称

        # 使用图论算法寻找最佳图像顺序 (这里简化为选择匹配最多的图像作为起点)
        start_image_index = np.argmax(np.sum(match_matrix, axis=1))
        image_order = [start_image_index]
        remaining_images = list(range(num_images))
        remaining_images.remove(start_image_index)

        while remaining_images:
            current_image_index = image_order[-1]
            best_match_index = -1
            best_match_count = 0

            for i in remaining_images:
                match_count = match_matrix[current_image_index, i]
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_match_index = i

            if best_match_index != -1:
                image_order.append(best_match_index)
                remaining_images.remove(best_match_index)
            else: # 如果找不到匹配的图像，则随机选择一张
                 image_order.append(remaining_images[0])
                 remaining_images.remove(remaining_images[0])

        # 应用图像顺序 (不再使用列表表达式)
        ordered_images = []
        for i in image_order:
            ordered_images.append(images[i])

    else:
        ordered_images = images # 如果只有两张图片则不需要排序

    # 2. 图像拼接
    base_image = ordered_images[0]  #  选择一张图片作为基准图像.

    for i in range(1, len(ordered_images)):
        # 对基准图和待拼接图做色彩校正，使得颜色尽可能一致
        base_image, ordered_images[i] = correct_color_balance(base_image, ordered_images[i])

        (kpsA, featuresA) = detect_and_describe(base_image, method=features_method)
        (kpsB, featuresB) = detect_and_describe(ordered_images[i], method=features_method)
        M = match_keypoints(kpsA, kpsB, featuresA, featuresB)

        if M is None:
            print("Warning: Feature matching failed between images {} and {}. Skipping.".format(i - 1, i))
            continue  # 如果匹配失败则跳过

        matches, H, status = M
        h, w, _ = base_image.shape
        warped_image = cv2.warpPerspective(ordered_images[i], H, (w * 2, h))  # 扩大画布，避免信息丢失

        # 创建融合 Mask
        offset_x = 0
        for x in range(0, w):
            if (warped_image[h//2, x] != [0,0,0]).any():
                offset_x = x
                break;

        if blending == "feather":
            mask = create_mask(base_image, warped_image[:, 0:w], offset_x)
            blended_result = feather_blending(base_image, warped_image[:, 0:w], mask)
            base_image = blended_result
        else:  # No blending , 直接覆盖
             warped_image[0:h, 0:w] = base_image
             base_image = warped_image[:, 0:w]

    # 3. 裁剪黑边 (可选)
    gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        base_image = base_image[y:y + h, x:x + w]

    return base_image

if __name__ == '__main__':
    # 加载多张需要拼接的图像
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]  # 替换为你的图像路径
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        images.append(img)

    # 执行图像拼接
    result = stitch_images(images, features_method="sift", blending="feather")

    if result is not None:
        cv2.imshow("Stitched Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image stitching failed.")
