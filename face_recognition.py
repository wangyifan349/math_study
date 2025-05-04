import face_recognition
# 加载已知和未知的图像文件，并进行面部编码
known_image = face_recognition.load_image_file("einstein.jpg")  # 使用爱因斯坦的图片
unknown_image_1 = face_recognition.load_image_file("unknown1.jpg")
picture_of_me = face_recognition.load_image_file("me.jpg")
unknown_image_2 = face_recognition.load_image_file("unknown2.jpg")
# 提取已知和未知图片中的脸部编码
einstein_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding_1 = face_recognition.face_encodings(unknown_image_1)[0]
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]
unknown_encoding_2 = face_recognition.face_encodings(unknown_image_2)[0]
# 比较已知人脸与未知人脸的相似性
results_1 = face_recognition.compare_faces([einstein_encoding], unknown_encoding_1)
results_2 = face_recognition.compare_faces([my_face_encoding], unknown_encoding_2)
# 输出比较结果
print("---------------------------")
if results_1[0]:
    print("Unknown image 1 matches Einstein!")
else:
    print("Unknown image 1 does not match Einstein!")
print("---------------------------")
if results_2[0]:
    print("Unknown image 2 is a picture of me!")
else:
    print("Unknown image 2 is not a picture of me!")
print("---------------------------")
