import dlib
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import os

# 配置模型文件路径（请确保路径正确）
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_RECOG_MODEL_PATH = "dlib_face_recognition_resnet_model_v1.dat"

# 检查模型文件是否存在
if not os.path.exists(SHAPE_PREDICTOR_PATH) or not os.path.exists(FACE_RECOG_MODEL_PATH):
    raise FileNotFoundError("模型文件不存在，请确认路径：" + SHAPE_PREDICTOR_PATH + " 和 " + FACE_RECOG_MODEL_PATH)

# 初始化人脸检测器和模型
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)

def get_face_landmarks_and_descriptor(image_path, save_result=False):
    """
    从图像中提取所有检测到的人脸的关键点和特征向量。
    
    参数：
        image_path: 图像文件路径
        save_result: 是否保存标记关键点的图像（默认 False）
    
    返回：
        faces_info: 一个列表，其中每个元素字典格式为：
            {
                "rect": dlib.rectangle,         # 人脸区域
                "landmarks": [(x1,y1), ...],      # 68个关键点坐标
                "descriptor": np.ndarray        # 人脸特征向量
            }
        如果图像读取失败或没有检测到人脸则返回一个空列表 []。
    """
    if not os.path.exists(image_path):
        print(f"图像 {image_path} 不存在！")
        return []
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像 {image_path}")
        return []
    
    # 为绘制结果创建一份复制
    draw_image = image.copy()
    
    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 检测人脸（第二个参数越大，检测越准确，但速度会降低）
    faces_detected = face_detector(gray_image, 1)
    
    faces_info = []
    for face_rect in faces_detected:
        # 获取该人脸的68个关键点
        shape = shape_predictor(gray_image, face_rect)
        landmarks = []
        for i in range(68):
            part = shape.part(i)
            landmarks.append((part.x, part.y))
            # 绘制关键点（可调整颜色、字体等）
            cv2.circle(draw_image, (part.x, part.y), 2, (0, 255, 0), -1)
            cv2.putText(draw_image, str(i), (part.x, part.y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # 提取人脸描述符（特征向量）
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
        faces_info.append({
            "rect": face_rect,
            "landmarks": landmarks,
            "descriptor": np.array(face_descriptor)
        })
    
    # 显示并可选保存图像结果
    if len(faces_detected) > 0:
        cv2.imshow("Face Landmarks - " + os.path.basename(image_path), draw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if save_result:
            result_path = os.path.splitext(image_path)[0] + "_landmarks.jpg"
            cv2.imwrite(result_path, draw_image)
    else:
        print("图像 {} 中未检测到人脸！".format(image_path))
    
    return faces_info

def compute_similarity(descriptor1, descriptor2):
    """
    计算两个特征向量间的欧氏距离和余弦相似度
    
    参数：
        descriptor1, descriptor2: np.ndarray 人脸特征向量
    返回：
        euclidean_distance: 欧氏距离
        cosine_similarity: 余弦相似度
    """
    euclidean_distance = np.linalg.norm(descriptor1 - descriptor2)
    cosine_similarity = 1 - cosine(descriptor1, descriptor2)
    return euclidean_distance, cosine_similarity

def compare_images(image_path1, image_path2):
    """
    提取两张图像中所有人脸信息，并比较两张图像中所有可能的脸对，输出最相似的一对脸部。
    
    参数：
        image_path1, image_path2: 图像文件路径
    """
    faces_info1 = get_face_landmarks_and_descriptor(image_path1, save_result=True)
    faces_info2 = get_face_landmarks_and_descriptor(image_path2, save_result=True)
    
    if not faces_info1:
        print("图像 {} 中没有检测到人脸！".format(image_path1))
        return
    
    if not faces_info2:
        print("图像 {} 中没有检测到人脸！".format(image_path2))
        return

    best_match = {
        "face_idx1": None,
        "face_idx2": None,
        "euclidean_distance": float('inf'),
        "cosine_similarity": 0
    }
    
    # 遍历两张图像中的所有人脸，比较每一对的相似度
    for idx1, face1 in enumerate(faces_info1):
        for idx2, face2 in enumerate(faces_info2):
            distance, similarity = compute_similarity(face1["descriptor"], face2["descriptor"])
            # 这里以欧氏距离作为主要比较指标，也可以根据实际需求综合两者指标来选择最佳匹配
            if distance < best_match["euclidean_distance"]:
                best_match["face_idx1"] = idx1
                best_match["face_idx2"] = idx2
                best_match["euclidean_distance"] = distance
                best_match["cosine_similarity"] = similarity
    
    print("最佳匹配结果：")
    print("图像1中第 {} 个人脸".format(best_match["face_idx1"] + 1))
    print("图像2中第 {} 个人脸".format(best_match["face_idx2"] + 1))
    print("欧氏距离: {:.4f}".format(best_match["euclidean_distance"]))
    print("余弦相似度: {:.4f}".format(best_match["cosine_similarity"]))

# 示例调用：
if __name__ == "__main__":
    image_path1 = "image1.jpg"
    image_path2 = "image2.jpg"
    compare_images(image_path1, image_path2)
