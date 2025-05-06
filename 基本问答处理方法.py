from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import random

# 使用经过微调的句子编码器模型
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)
# 示例问答数据库，包括多个详细回答
qa_pairs = [
    {
        "question": "What is the capital of France?", 
        "answers": [
            "The capital of France is Paris. It is known for its rich history and cultural landmarks such as the Eiffel Tower and the Louvre Museum.",
            "Paris is the capital city of France, famous for its fashion, art, and gastronomy. It is often termed as 'The City of Light'."
        ]
    },
    {
        "question": "What is the largest planet?", 
        "answers": [
            "Jupiter is the largest planet in our solar system. It is a gas giant and has a very strong magnetic field.",
            "The largest planet is Jupiter, which has more than 70 moons, including the largest one named Ganymede."
        ]
    },
    {
        "question": "What are the benefits of Niacinamide?", 
        "answers": [
            "Niacinamide, a form of vitamin B3, is effective in improving skin texture and tone. It can help reduce the appearance of pores, fine lines, and wrinkles.",
            "The benefits of Niacinamide include reducing inflammation, helping with acne, and improving the skin barrier by boosting ceramide production."
        ]
    },
    {
        "question": "How does Vitamin C benefit the skin?", 
        "answers": [
            "Vitamin C is a powerful antioxidant that can help brighten the skin and reduce hyperpigmentation. It's crucial for collagen synthesis, aiding in skin firmness.",
            "The main benefits of Vitamin C include its ability to protect the skin from oxidative damage, lighten dark spots, and support skin repair mechanisms."
        ]
    }
]

# 对所有问题进行编码
questions = []
for pair in qa_pairs:
    questions.append(pair["question"])
question_embeddings = []
for question in questions:
    embedding = model.encode(question)
    question_embeddings.append(embedding)
question_embeddings = np.array(question_embeddings)
# 创建FAISS索引
dimension = question_embeddings.shape[#citation-1](citation-1)
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(question_embeddings)
def find_best_answer(input_question, top_k=1):
    input_embedding = model.encode(input_question)
    distances, indices = faiss_index.search(np.array([input_embedding]), top_k)
    results = []
    for index in indices[0]:
        if index != -1:
            answers = qa_pairs[index] ["answers"]
            chosen_answer = random.choice(answers)
            results.append(chosen_answer)
    return results
def continuous_qa_session():
    print("Welcome to the QA system! Type 'exit' to quit.")
    while True:
        user_question = input("\nEnter your question: ")
        if user_question.lower() == 'exit':
            print("Exiting the QA system. Goodbye!")
            break
        answers = find_best_answer(user_question)
        if answers:
            print(f"Answer: {answers[0]}")
        else:
            print("Sorry, I don't have an answer for that question.")
# 启动持续问答会话
continuous_qa_session()


"""
import numpy as np
def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度."""
    # 确保输入是numpy数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # 计算点积和范数
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    # 计算余弦相似度
    if norm_vec1 == 0 or norm_vec2 == 0:
        # 如果有一个向量是零向量，余弦相似度未定义，这里返回0
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

# 示例
vec_a = [1, 2, 3]
vec_b = [4, 5, 6]
print("Cosine Similarity:", cosine_similarity(vec_a, vec_b))
"""

"""
def l2_distance(vec1, vec2):
    """计算两个向量之间的欧几里得距离（L2距离）."""
    # 确保输入是numpy数组
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # 计算元素差平方的和，再取平方根
    return np.linalg.norm(vec1 - vec2)
# 示例
vec_a = [1, 2, 3]
vec_b = [4, 5, 6]
print("L2 Distance:", l2_distance(vec_a, vec_b))
"""




import face_recognition
# Load the images
image1 = face_recognition.load_image_file("path/to/your/first/image.jpg")
image2 = face_recognition.load_image_file("path/to/your/second/image.jpg")
# Get the face encodings for the images
encodings1 = face_recognition.face_encodings(image1)
encodings2 = face_recognition.face_encodings(image2)
# Ensure there are encodings available for both images
if len(encodings1) > 0 and len(encodings2) > 0:
    face_encoding1 = encodings1[0]
    face_encoding2 = encodings2[0]
    # Compare faces
    results = face_recognition.compare_faces([face_encoding1], face_encoding2)
    print(f"Faces match: {results[0]}")
else:
    print("Could not find faces in one or both of the images.")



import face_recognition
import numpy as np
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def l2_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)
# Load the images
image1 = face_recognition.load_image_file("path/to/your/first/image.jpg")
image2 = face_recognition.load_image_file("path/to/your/second/image.jpg")
# Get the face encodings for the images
encodings1 = face_recognition.face_encodings(image1)
encodings2 = face_recognition.face_encodings(image2)
# Ensure there are encodings available for both images
if len(encodings1) > 0 and len(encodings2) > 0:
    face_encoding1 = encodings1[0]
    face_encoding2 = encodings2[0]
    # Compare faces
    results = face_recognition.compare_faces([face_encoding1], face_encoding2)
    print(f"Faces match: {results[0]}")
    # Compute additional similarity measures
    alg2_dist = l2_distance(face_encoding1, face_encoding2)
    cos_sim = cosine_similarity(face_encoding1, face_encoding2)
    print(f"Al2 Distance: {alg2_dist}")
    print(f"Cosine Similarity: {cos_sim}")
else:
    print("Could not find faces in one or both of the images.")
