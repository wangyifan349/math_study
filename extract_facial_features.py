import dlib
import cv2
import numpy as np

# Load the pre-trained models for face detection and facial and face recognition
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Define the facial landmarks indices according to dlib's 68 points model
FACIAL_LANDMARKS_IDXS = {
    "mouth": range(48, 68),
    "right_eyebrow": range(17, 22),
    "left_eyebrow": range(22, 27),
    "right_eye": range(36, 42),
    "left_eye": range(42, 48),
    "nose": range(27, 36),
    "jaw": range(0, 17)
}

def shape_to_np(shape):
    coords = []
    for i in range(68):
        coords.append((shape.part(i).x, shape.part(i).y))
    return coords

def draw_face_regions(image, landmarks):
    for part_name, indices in FACIAL_LANDMARKS_IDXS.items():
        points = []
        for i in indices:
            points.append(landmarks[i])
        color = (0, 255, 0) if part_name != "jaw" else (255, 0, 0)
        for i in range(len(points)):
            (x, y) = points[i]
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
            if part_name != "jaw" and i < len(points) - 1:
                next_point = points[i + 1]
                cv2.line(image, (x, y), next_point, color, 1)
        if part_name == "jaw":
            for i in range(1, len(points)):
                cv2.line(image, points[i - 1], points[i], color, 1)

def extract_facial_features_and_embeddings(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray, 1)

    for rect in faces:
        shape = shape_predictor(gray, rect)
        landmarks = shape_to_np(shape)
        
        draw_face_regions(image, landmarks)
        
        # Extract face descriptor (128-D embedding)
        face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
        face_descriptor_np = np.array(face_descriptor)

        print("\n128-D Face Descriptor:\n", face_descriptor_np)

    # Display the output with annotations
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to your image
extract_facial_features_and_embeddings('path/to/your/image.jpg')
