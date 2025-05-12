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









import cv2
import numpy as np
import dlib
# ----------------- 人脸检测和特征点提取 -----------------
def get_landmarks(image, detector, predictor):
    faces = detector(image, 1)
    if len(faces) == 0:
        raise Exception("No face detected in image.")
    landmarks = predictor(image, faces[0])
    landmarks_points = []
    for part in landmarks.parts():
        landmarks_points.append((part.x, part.y))
    return np.array(landmarks_points)
# ----------------- 仿射变换 -----------------
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_matrix = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_matrix, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst
# ----------------- 三角形区域变换 -----------------
def warp_triangle(img_source, img_target, tri_source, tri_target):
    bounding_rect_source = cv2.boundingRect(np.float32([tri_source]))
    bounding_rect_target = cv2.boundingRect(np.float32([tri_target]))
    tri_source_rect = []
    tri_target_rect = []
    tri_target_rect_int = []
    for i in range(3):
        tri_source_rect.append((tri_source[i] [0] - bounding_rect_source[0], tri_source[i] [#citation-1](citation-1) - bounding_rect_source[#citation-1](citation-1)))
        tri_target_rect.append((tri_target[i] [0] - bounding_rect_target[0], tri_target[i] [#citation-1](citation-1) - bounding_rect_target[#citation-1](citation-1)))
        tri_target_rect_int.append((int(tri_target[i] [0] - bounding_rect_target[0]), int(tri_target[i] [#citation-1](citation-1) - bounding_rect_target[#citation-1](citation-1))))
    mask = np.zeros((bounding_rect_target[#citation-3](citation-3), bounding_rect_target[#citation-2](citation-2), 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_target_rect_int), (1.0, 1.0, 1.0), 16, 0)
    img_source_rect = img_source[bounding_rect_source[#citation-1](citation-1):bounding_rect_source[#citation-1](citation-1) + bounding_rect_source[#citation-3](citation-3), bounding_rect_source[0]:bounding_rect_source[0] + bounding_rect_source[#citation-2](citation-2)]
    size = (bounding_rect_target[#citation-2](citation-2), bounding_rect_target[#citation-3](citation-3))
    img_target_rect = apply_affine_transform(img_source_rect, tri_source_rect, tri_target_rect, size)
    img_target_rect = img_target_rect * mask
    img_target[bounding_rect_target[#citation-1](citation-1):bounding_rect_target[#citation-1](citation-1) + bounding_rect_target[#citation-3](citation-3), 
               bounding_rect_target[0]:bounding_rect_target[0] + bounding_rect_target[#citation-2](citation-2)] = img_target_rect + img_target[bounding_rect_target[#citation-1](citation-1):bounding_rect_target[#citation-1](citation-1) + bounding_rect_target[#citation-3](citation-3), bounding_rect_target[0]:bounding_rect_target[0] + bounding_rect_target[#citation-2](citation-2)] * (1 - mask)
# ----------------- 面部交换 -----------------
def swap_faces(image_source, image_target, landmarks_source, landmarks_target):
    hull_index = cv2.convexHull(np.array(landmarks_target), returnPoints=False)
    hull_source = [landmarks_source[int(idx)] for idx in hull_index.squeeze()]
    hull_target = [landmarks_target[int(idx)] for idx in hull_index.squeeze()]
    rect = cv2.boundingRect(np.float32([hull_target]))
    subdivide = cv2.Subdiv2D(rect)
    subdivide.insert(hull_target)
    triangles = subdivide.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    for triangle in triangles:
        indices = []
        for i in range(0, 6, 2):
            for j in range(len(landmarks_target)):
                if (landmarks_target[j] [0] == triangle[i]) and (landmarks_target[j] [#citation-1](citation-1) == triangle[i + 1]):
                    indices.append(j)
        if len(indices) == 3:
            tri_src = [hull_source[i] for i in indices]
            tri_tgt = [hull_target[i] for i in indices]
            warp_triangle(image_source, image_target, tri_src, tri_tgt)
    mask = np.zeros_like(image_target, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(hull_target), (255, 255, 255))
    bounding_rect = cv2.boundingRect(np.float32([hull_target]))
    center = ((bounding_rect[0] + bounding_rect[0] + bounding_rect[#citation-2](citation-2)) // 2, (bounding_rect[#citation-1](citation-1) + bounding_rect[#citation-1](citation-1) + bounding_rect[#citation-3](citation-3)) // 2)
    # 使用泊松融合进行脸部合成
    output_image = cv2.seamlessClone(np.uint8(image_target), image_target, mask, center, cv2.NORMAL_CLONE)
    return output_image
# ----------------- 主流程 -----------------
# 加载模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 加载图像
image1 = cv2.imread("face1.jpg")
image2 = cv2.imread("face2.jpg")
# 获取特征点
landmarks1 = get_landmarks(image1, detector, predictor)
landmarks2 = get_landmarks(image2, detector, predictor)
# 进行面部交换
output_image = swap_faces(image1, image2, landmarks1, landmarks2)
# 显示换脸结果
cv2.imshow("Swapped Face with Blending", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
    面部轮廓：1-17（共17个点）
    左眉毛：18-22（共5个点）
    右眉毛：23-27（共5个点）
    鼻梁：28-31（共4个点）
    鼻子底部：32-36（共5个点）
    左眼：37-42（共6个点）
    右眼：43-48（共6个点）
    上唇：49-55（共7个点）
    下唇：55-60（共6个点）
    内唇：61-68（共8个点）
dlib 的 68 个面部特征点模型来为人脸检测进行特征点提取。
"""
              
