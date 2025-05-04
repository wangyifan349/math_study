"""pip install opencv-contrib-python""""
"""
- 该程序实现了两张图片中的人脸检测和人脸对比。  
- 使用OpenCV的Haar级联分类器完成人脸的定位，提取图像中的人脸区域。  
- 读取两张图片，先转换成灰度图，再用人脸检测器找到人脸坐标。  
- 对检测到的人脸区域进行裁剪并统一调整为100×100大小，便于后续识别处理。  
- 使用LBPH人脸识别器训练第一张图片的人脸特征，作为基准样本。  
- 将第二张图片中的人脸特征与第一张进行比对，计算相似度得分。  
- 相似度得分越低表示两张人脸越相似，设定阈值判断是否为同一人。  
- 程序会在图片上绘制红色和绿色矩形框，标示检测到的人脸区域，便于观察。  
- 包含了基本的异常处理，如图片读取失败或未检测到人脸时的提示。  
- 该代码简单明了，无嵌套结构，便于理解和修改，适合作为入门人脸识别示例。
"""
import cv2
import numpy as np

# 加载Haar人脸检测器
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
# 读取第一张图片，转换为灰度图
image1_path = 'face1.jpg'  # 替换为你的图片路径
img1_color = cv2.imread(image1_path)
if img1_color is None:
    print("第一张图片读取失败")
else:
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    # 检测人脸，检测参数：scaleFactor=1.1, minNeighbors=5
    faces1 = face_cascade.detectMultiScale(img1_gray, 1.1, 5)
    if len(faces1) == 0:
        print("第一张图片没有检测到人脸")
    else:
        # 取第一张人脸的坐标和大小
        x1, y1, w1, h1 = faces1[0]
        # 裁剪人脸区域并调整大小到100x100
        face_roi_1 = img1_gray[y1:y1+h1, x1:x1+w1]
        face_roi_1 = cv2.resize(face_roi_1, (100, 100))
# 读取第二张图片，转换为灰度图
image2_path = 'face2.jpg'  # 替换为你的图片路径
img2_color = cv2.imread(image2_path)
if img2_color is None:
    print("第二张图片读取失败")
else:
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    faces2 = face_cascade.detectMultiScale(img2_gray, 1.1, 5)
    if len(faces2) == 0:
        print("第二张图片没有检测到人脸")
    else:
        x2, y2, w2, h2 = faces2[0]
        face_roi_2 = img2_gray[y2:y2+h2, x2:x2+w2]
        face_roi_2 = cv2.resize(face_roi_2, (100, 100))
# 判断两张图片的人脸是否都成功检测到了
if ('face_roi_1' in locals()) and ('face_roi_2' in locals()):
    # 创建LBPH人脸识别器
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 准备训练数据：list单元素加上标签1
    faces = []
    faces.append(face_roi_1)
    labels = []
    labels.append(1)
    # 训练识别器
    recognizer.train(faces, np.array(labels))
    # 预测第二张人脸与训练样本的距离和标签
    label_predicted, confidence = recognizer.predict(face_roi_2)
    print("预测标签:", label_predicted)
    print("相似度得分 (数值越小越相似):", confidence)
    threshold = 85.0
    if confidence < threshold:
        print("判断为同一个人")
    else:
        print("判断为不同人")
    # 绘制人脸框
    cv2.rectangle(img1_color, (x1, y1), (x1+w1, y1+h1), (255,0,0), 2)
    cv2.rectangle(img2_color, (x2, y2), (x2+w2, y2+h2), (0,255,0), 2)
    # 显示图像
    cv2.imshow('第一张图片人脸', img1_color)
    cv2.imshow('第二张图片人脸', img2_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("至少一张图片没有成功检测到人脸，无法进行比对")
