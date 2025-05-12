"""
这个程序使用`dlib`库进行人脸检测和特征提取，并通过计算特征向量的欧氏距离来比较两张人脸图像的相似度。
"""

import dlib
import cv2
import numpy as np

# 加载预训练模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 加载图像
image1 = cv2.imread("face1.jpg")
image2 = cv2.imread("face2.jpg")

# 图像转换为灰度
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 面部检测
faces1 = detector(image1_gray)
faces2 = detector(image2_gray)

# 图像1的处理
if len(faces1) > 0:
    face1 = faces1[0]
    landmarks1 = predictor(image1_gray, face1)  # 提取特征点
    for n in range(0, 68):
        x = landmarks1.part(n).x
        y = landmarks1.part(n).y
        cv2.circle(image1, (x, y), 1, (0, 255, 0), -1)

    # 提取人脸特征描述符
    descriptor1 = face_recognition_model.compute_face_descriptor(image1_gray, landmarks1)
    descriptor1 = np.array(descriptor1)

# 图像2的处理
if len(faces2) > 0:
    face2 = faces2[0]
    landmarks2 = predictor(image2_gray, face2)  # 提取特征点
    for n in range(0, 68):
        x = landmarks2.part(n).x
        y = landmarks2.part(n).y
        cv2.circle(image2, (x, y), 1, (0, 255, 0), -1)

    # 提取人脸特征描述符
    descriptor2 = face_recognition_model.compute_face_descriptor(image2_gray, landmarks2)
    descriptor2 = np.array(descriptor2)

# 计算欧式距离
distance = np.linalg.norm(descriptor1 - descriptor2)
print(f"欧式距离：{distance}")

# 显示结果图像
cv2.imshow("Image 1 with Landmarks", image1)
cv2.imshow("Image 2 with Landmarks", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
