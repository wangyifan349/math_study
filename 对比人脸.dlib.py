import dlib
import cv2
import numpy as np
from scipy.spatial.distance import cosine

# 加载人脸检测器和模型文件
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
def get_face_landmarks_and_descriptor(image_path):
    """
    从图像中提取人脸关键点和特征向量。
    """
    # 读取图像
    image = cv2.imread(image_path)
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 检测人脸
    faces_detected = face_detector(gray_image, 1)
    for face_rect in faces_detected:
        # 获取人脸的特征点
        shape = shape_predictor(gray_image, face_rect)
        # 提取每个特征点的坐标
        landmarks = []
        for i in range(68):
            part = shape.part(i)
            landmarks.append((part.x, part.y))
            # 标记关键点
            cv2.circle(image, (part.x, part.y), 2, (0, 255, 0), -1)
            cv2.putText(image, str(i), (part.x, part.y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        # 提取人脸特征向量
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
        
        # 显示图像并标注关键点
        cv2.imshow("Landmarks", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return landmarks, np.array(face_descriptor)
    return None, None

def compute_similarity(descriptor1, descriptor2):
    """
    计算两个特征向量的欧氏距离和余弦相似度。
    """
    euclidean_distance = np.linalg.norm(descriptor1 - descriptor2)
    cosine_similarity = 1 - cosine(descriptor1, descriptor2)
    return euclidean_distance, cosine_similarity

# 使用示例
landmarks1, descriptor1 = get_face_landmarks_and_descriptor("image1.jpg")
landmarks2, descriptor2 = get_face_landmarks_and_descriptor("image2.jpg")
if descriptor1 is not None and descriptor2 is not None:
    # 打印每个图像的关键点
    print("Image1 Landmarks:", landmarks1)
    print("Image2 Landmarks:", landmarks2)
    
    # 计算并打印欧氏距离和余弦相似度
    distance, similarity = compute_similarity(descriptor1, descriptor2)
    print(f"Euclidean Distance: {distance}")
    print(f"Cosine Similarity: {similarity}")
else:
    print("未检测到人脸")
