import face_recognition  # 导入人脸识别库
import cv2  # 导入 OpenCV 库，用于摄像头操作和图像处理
import os  # 导入 os 库，用于文件和目录操作
import pickle  # 导入 pickle 库，用于保存和加载人脸编码数据

def load_known_faces(database_path, encoding_file="known_faces_encoding.pickle"):
    """
    从指定目录加载已知人脸图像，提取人脸特征，并将特征保存到文件中，以便下次快速加载。
    如果存在已经保存的人脸编码文件，则直接加载该文件，跳过特征提取步骤。

    参数：
        database_path: 存放已知人脸图像的目录路径。
        encoding_file: (可选) 保存人脸编码的文件名，默认为 "known_faces_encoding.pickle"。

    返回：
        known_face_encodings: 已知人脸特征编码列表。
        known_face_names: 已知人脸姓名列表。
    """
    known_face_encodings = []  # 初始化已知人脸编码列表
    known_face_names = []  # 初始化已知人脸姓名列表

    if os.path.exists(encoding_file):
        # 如果存在编码文件，则直接加载，避免重复计算人脸特征
        print("从文件中加载已知人脸编码...")
        with open(encoding_file, "rb") as f:  # 以二进制读取模式打开文件
            known_face_encodings, known_face_names = pickle.load(f)  # 从文件中加载数据
        return known_face_encodings, known_face_names  # 返回加载的人脸编码和姓名

    else:
        # 如果不存在编码文件，则从图像文件提取编码
        print("从已知人脸图像中提取特征...")
        for filename in os.listdir(database_path):  # 遍历数据库目录中的所有文件
            if not filename.endswith((".jpg", ".jpeg", ".png")):  # 忽略非图像文件
                continue  # 跳过本次循环，处理下一个文件

            image_path = os.path.join(database_path, filename)  # 构建图像文件的完整路径
            image = face_recognition.load_image_file(image_path)  # 加载图像文件

            face_locations = face_recognition.face_locations(image)  # 检测图像中的所有人脸位置
            face_encodings = face_recognition.face_encodings(image, face_locations)  # 提取人脸的特征编码

            if face_encodings:  # 确保检测到人脸，如果 face_encodings 不为空
                known_face_encodings.append(face_encodings[0])  # 将第一个检测到的人脸的编码添加到列表中(假定每张图只有一张人脸)
                known_face_names.append(os.path.splitext(filename)[0])  # 使用文件名（不包含扩展名）作为人名
            else:
                print(f"警告: 未在 {filename} 中检测到人脸。")  # 打印警告信息，提示未检测到人脸

        # 保存人脸编码到文件，方便下次加载
        print("将已知人脸编码保存到文件中...")
        with open(encoding_file, "wb") as f:  # 以二进制写入模式打开文件
            pickle.dump((known_face_encodings, known_face_names), f)  # 将数据保存到文件中

        return known_face_encodings, known_face_names  # 返回提取和保存的人脸编码和姓名

def recognize_face_from_camera(known_face_encodings, known_face_names, tolerance=0.6):
    """
    从摄像头捕获图像，并与已知人脸进行对比，实现实时人脸识别。

    参数：
        known_face_encodings: 已知人脸特征编码列表。
        known_face_names: 已知人脸姓名列表。
        tolerance: 容差值，用于控制人脸相似度的阈值，值越小越严格，默认为 0.6。
    """
    video_capture = cv2.VideoCapture(0)  # 使用 OpenCV 打开摄像头，0 表示默认摄像头

    if not video_capture.isOpened():  # 检查摄像头是否成功打开
        print("无法打开摄像头")  # 如果摄像头没有成功打开，打印错误信息
        return  # 退出函数

    face_locations = []  # 初始化人脸位置列表，用于存储检测到的人脸的位置
    face_encodings = []  # 初始化人脸编码列表，用于存储检测到的人脸的特征编码
    face_names = []  # 初始化人名列表，用于存储识别出的人名
    process_this_frame = True  # 控制帧处理频率的标志位，True 表示当前帧需要处理

    while True:  # 无限循环，直到按下 "q" 键退出
        # 从摄像头读取单帧图像
        ret, frame = video_capture.read()  # ret 表示是否成功读取帧，frame 是读取到的图像

        # 为了节省计算资源，每隔一帧处理一次
        if process_this_frame:
            # 缩放图像，减少计算量。将图像缩小到原始尺寸的 1/4
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # OpenCV 使用 BGR 颜色格式，而 face_recognition 使用 RGB 颜色格式，所以需要转换
            rgb_small_frame = small_frame[:, :, ::-1]  # 通过切片操作，反转颜色通道的顺序

            # 在当前帧中查找所有人脸和人脸编码
            face_locations = face_recognition.face_locations(rgb_small_frame)  # 检测人脸位置
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # 提取人脸编码

            face_names = []  # 清空人名列表，准备存储当前帧的识别结果
            for face_encoding in face_encodings:  # 遍历当前帧中检测到的所有人脸编码
                # 尝试将当前人脸与已知人脸进行匹配
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)  # 比较人脸

                name = "Unknown"  # 默认设置为 "Unknown"，表示未知人物

                # 如果找到匹配
                if True in matches:
                    first_match_index = matches.index(True)  # 找到第一个匹配的人脸的索引
                    name = known_face_names[first_match_index]  # 使用已知人脸的姓名

                face_names.append(name)  # 将识别出的人名添加到人名列表中

        process_this_frame = not process_this_frame  # 反转标志位，控制帧处理频率

        # 显示结果
        for (top, right, bottom, left), name in zip(face_locations, face_names):  # 遍历所有人脸位置和人名
            # 因为之前将图像缩小了 1/4，所以现在需要将人脸位置坐标放大 4 倍
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 在人脸周围绘制矩形框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # (0, 0, 255) 表示红色

            # 在人脸框下方绘制包含人名的标签
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)  # 填充矩形
            font = cv2.FONT_HERSHEY_DUPLEX  # 选择字体
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)  # 绘制文本

        # 显示结果图像
        cv2.imshow('Video', frame)  # 在名为 "Video" 的窗口中显示图像

        # 按下 "q" 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 等待 1 毫秒，检测键盘输入
            break  # 退出循环

    # 释放摄像头资源
    video_capture.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

if __name__ == "__main__":
    # 定义已知人脸图像的目录和编码文件
    known_faces_dir = "face_database"  # 已知人脸图像所在的目录
    encoding_file = "known_faces_encoding.pickle"  # 保存人脸编码的文件名

    # 加载已知人脸数据
    known_face_encodings, known_face_names = load_known_faces(known_faces_dir, encoding_file)  # 加载人脸数据

    # 开始实时人脸识别
    recognize_face_from_camera(known_face_encodings, known_face_names)  # 启动摄像头并进行人脸识别
