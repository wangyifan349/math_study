import cv2
import numpy as np

# 读取图像，并转换为灰度图
image_path = 'test.jpg'  # 替换为你自己的图片路径
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("图片读取失败")
else:
    # 创建SURF检测器对象，hessianThreshold越大，检测的关键点越少
    surf = cv2.xfeatures2d.SURF_create(400)
    # 检测关键点
    keypoints = surf.detect(img, None)
    # 计算关键点描述子
    keypoints, descriptors = surf.compute(img, keypoints)
    # 绘制关键点
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (255,0,0), 4)
    # 显示结果图像
    cv2.imshow('SURF KeyPoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 输出检测到的关键点个数和描述子形状信息
    print("SURF检测到关键点数:", len(keypoints))
    if descriptors is not None:
        print("描述子数组形状:", descriptors.shape)
    else:
        print("未计算到描述子")





import cv2
import numpy as np

# 读取图像，并转换为灰度图
image_path = 'test.jpg'  # 替换为你自己的图片路径
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("图片读取失败")
else:
    # 创建ORB检测器对象，nfeatures设置最多检测多少关键点
    orb = cv2.ORB_create(nfeatures=500)
    # 检测关键点
    keypoints = orb.detect(img, None)
    # 计算关键点描述子
    keypoints, descriptors = orb.compute(img, keypoints)
    # 绘制关键点
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (0,255,0), 2)
    # 显示结果图像
    cv2.imshow('ORB KeyPoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 输出检测到的关键点个数和描述子形状信息
    print("ORB检测到关键点数:", len(keypoints))
    if descriptors is not None:
        print("描述子数组形状:", descriptors.shape)
    else:
        print("未计算到描述子")






import cv2
import numpy as np
# 读取图像，并转换为灰度图
image_path = 'test.jpg'  # 替换为你自己的图片路径
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("图片读取失败")
else:
    # 创建SURF检测器对象，hessianThreshold越大，检测的关键点越少
    surf = cv2.xfeatures2d.SURF_create(400)
    # 检测关键点
    keypoints = surf.detect(img, None)
    # 计算关键点描述子
    keypoints, descriptors = surf.compute(img, keypoints)
    # 绘制关键点
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (255,0,0), 4)
    # 显示结果图像
    cv2.imshow('SURF KeyPoints', img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 输出检测到的关键点个数和描述子形状信息
    print("SURF检测到关键点数:", len(keypoints))
    if descriptors is not None:
        print("描述子数组形状:", descriptors.shape)
    else:
        print("未计算到描述子")



