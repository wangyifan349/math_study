import cv2
import numpy as np

# 读取图像，转换成灰度图像
image_path = 'test.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("图片读取失败")
else:
    # 创建SURF检测器，hessianThreshold=400为特征点检测灵敏度
    surf = cv2.xfeatures2d.SURF_create(400)
    # 进行关键点检测，检测到的关键点放入keypoints列表
    keypoints = surf.detect(img, None)
    # 根据检测的关键点计算描述子
    keypoints, descriptors = surf.compute(img, keypoints)
    # 用蓝色圆圈绘制关键点到图像上，线宽4像素
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0), flags=4)
    # 显示绘制了关键点的图像
    cv2.imshow('SURF KeyPoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 打印检测的关键点数量和描述子数组形状
    print("SURF检测到关键点数:", len(keypoints))
    if descriptors is not None:
        print("描述子形状:", descriptors.shape)
    else:
        print("未计算到描述子")

---
import cv2
import numpy as np

# 读取图像，转换成灰度图像
image_path = 'test.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("图片读取失败")
else:
    # 创建ORB检测器，最多检测500个关键点
    orb = cv2.ORB_create(nfeatures=500)
    # 进行关键点检测，得到关键点列表
    keypoints = orb.detect(img, None)
    # 计算每个关键点的描述子
    keypoints, descriptors = orb.compute(img, keypoints)
    # 绘制绿色的关键点，线宽2像素
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=2)
    # 显示关键点绘制后的图像
    cv2.imshow('ORB KeyPoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 打印关键点数量和描述子大小
    print("ORB检测到关键点数:", len(keypoints))
    if descriptors is not None:
        print("描述子形状:", descriptors.shape)
    else:
        print("未计算到描述子")
---

import cv2
import numpy as np
# 读取图像，转换成灰度图像
image_path = 'test.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("图片读取失败")
else:
    # 创建SURF检测器，hessianThreshold=400控制检测点数
    surf = cv2.xfeatures2d.SURF_create(400)
    # 检测出图像中的关键点
    keypoints = surf.detect(img, None)
    # 计算检测结果的描述子
    keypoints, descriptors = surf.compute(img, keypoints)
    # 用蓝色绘制检测到的关键点
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(255, 0, 0), flags=4)
    # 弹窗显示绘制关键点后的图像
    cv2.imshow('SURF KeyPoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 打印关键点数量和描述子形状
    print("SURF检测到关键点数:", len(keypoints))
    if descriptors is not None:
        print("描述子形状:", descriptors.shape)
    else:
        print("未计算到描述子")
