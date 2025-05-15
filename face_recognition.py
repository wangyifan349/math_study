import face_recognition
import cv2

# 加载图片并获取脸部编码和位置信息
def load_face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        print(f"未能在图像 {image_path} 中检测到人脸")
        return None, None
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    return face_encoding, face_locations[0]

# 给图片中的脸部画矩形框并显示
def show_face_box(image_path, face_location, window_name):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法使用 OpenCV 加载图像 {image_path}")
        return
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 加载图片并编码
einstein_encoding, einstein_location = load_face_encoding("einstein.jpg")
unknown1_encoding, unknown1_location = load_face_encoding("unknown1.jpg")
my_encoding, my_location = load_face_encoding("me.jpg")
unknown2_encoding, unknown2_location = load_face_encoding("unknown2.jpg")
print("---------------------------")
# 比较Unknown1是否为Einstein
if einstein_encoding is not None and unknown1_encoding is not None:
    matches = face_recognition.compare_faces([einstein_encoding], unknown1_encoding)
    distance = face_recognition.face_distance([einstein_encoding], unknown1_encoding)[0]
    if matches[0]:
        print(f"Unknown image 1 matches Einstein! Distance: {distance:.4f}")
        if unknown1_location:
            show_face_box("unknown1.jpg", unknown1_location, "Unknown1 matched Einstein")
    else:
        print(f"Unknown image 1 does not match Einstein! Distance: {distance:.4f}")
else:
    print("Einstein 或 Unknown1 的人脸编码为空，无法比较。")
print("---------------------------")
# 比较Unknown2是否为“我”
if my_encoding is not None and unknown2_encoding is not None:
    matches = face_recognition.compare_faces([my_encoding], unknown2_encoding)
    distance = face_recognition.face_distance([my_encoding], unknown2_encoding)[0]
    if matches[0]:
        print(f"Unknown image 2 is a picture of me! Distance: {distance:.4f}")
        if unknown2_location:
            show_face_box("unknown2.jpg", unknown2_location, "Unknown2 matched Me")
    else:
        print(f"Unknown image 2 is not a picture of me! Distance: {distance:.4f}")
else:
    print("我的人脸编码或 Unknown2 的人脸编码为空，无法比较。")
print("---------------------------")
