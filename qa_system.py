import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 问答对，key是问题，value是答案内容
qa_pairs = {
    "What is a paramecium?": (
        "A paramecium is a single-celled organism found primarily in freshwater environments. "
        "It is a member of the group of organisms known as ciliates, named for the tiny hair-like structures "
        "called cilia that cover their surfaces. These cilia beat rhythmically to propel the paramecium through the water. "
        "Paramecia are often studied in biological laboratories due to their complex and specialized cell organelles, including a mouth pore, "
        "contractile vacuoles for expelling excess water, and a macronucleus and micronucleus that handle different cellular functions."
    ),
    "Describe the structure of DNA.": (
        "DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions used in the growth, development, "
        "functioning, and reproduction of all known living organisms and many viruses. DNA is composed of two long strands forming "
        "a double helix with a backbone made of alternating sugar (deoxyribose) and phosphate groups. The strands are connected by "
        "four types of nitrogen bases: adenine (A), thymine (T), cytosine (C), and guanine (G). Adenine pairs with thymine and cytosine pairs "
        "with guanine, allowing DNA to hold the instructions for building proteins in the form of genetic sequences."
    ),
    "What is photosynthesis?": (
        "Photosynthesis is a biochemical process by which green plants, algae, and certain bacteria convert light energy into chemical energy. "
        "During photosynthesis, chlorophyll pigments in plant chloroplasts capture sunlight, which is then used to transform carbon dioxide "
        "from the atmosphere and water from the soil into glucose and oxygen. The overall chemical equation for photosynthesis can be written as: "
        "6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. This process is essential for life on Earth, providing the primary source of energy for nearly "
        "all organisms and driving the planet's carbon cycle."
    ),
    "What does RNA do?": (
        "RNA, or ribonucleic acid, plays several important roles in the biological processes inside cells. Made of nucleotides similar to DNA but "
        "with ribose sugar and uracil instead of thymine, RNA's primary role is to act as a messenger between DNA and the ribosomes where proteins "
        "are synthesized—in a process called translation. This messenger RNA (mRNA) is transcribed from the DNA template and carries the genetic code "
        "to ribosomes. There are also other forms of RNA such as transfer RNA (tRNA) and ribosomal RNA (rRNA) that assist in the assembly of proteins "
        "by interpreting the mRNA sequence and linking amino acids together in the correct order."
    ),
    "What is an ecosystem?": (
        "An ecosystem is a dynamic complex of plant, animal, and microorganism communities, and the non-living environment interacting as a functional unit. "
        "Ecosystems vary greatly in size and can be as large as a desert or as small as a puddle. They provide a wide array of services essential to human "
        "survival and quality of life, including food production, fresh water, climate regulation, disease regulation, and recreational benefits. "
        "The interaction between different biotic (living) and abiotic (non-living) components involves nutrient cycles, energy flows, and food webs, "
        "maintaining the resilience and sustainability of the entire system."
    )
}

# 保存文件路径
vectorizer_filepath = "vectorizer.pkl"
tf_matrix_filepath = "tf_matrix.pkl"

# 提取问题列表（顺序一定，确保对应tf_matrix行顺序不变）
questions = list(qa_pairs.keys())

# 载入CountVectorizer和tf_matrix，如果有就加载，没有就新建后保存
if os.path.exists(vectorizer_filepath) and os.path.exists(tf_matrix_filepath):
    # 加载已保存的vectorizer
    with open(vectorizer_filepath, "rb") as f:
        vectorizer = pickle.load(f)
    # 加载已保存的tf_matrix（稀疏矩阵）
    with open(tf_matrix_filepath, "rb") as f:
        tf_matrix = pickle.load(f)
else:
    # 没有保存文件，训练新的vectorizer和tf_matrix
    vectorizer = CountVectorizer()
    tf_matrix = vectorizer.fit_transform(questions)
    # 保存vectorizer和tf_matrix
    with open(vectorizer_filepath, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(tf_matrix_filepath, "wb") as f:
        pickle.dump(tf_matrix, f)

print("这是一个简单的问答系统。输入 'exit' 退出。")

while True:
    query = input("你：").strip()
    if query.lower() == "exit":
        print("退出问答系统。再见！")
        break

    if not query:
        print("请输入有效的问题。")
        continue

    # 使用加载或训练好的vectorizer直接转换query
    query_vector = vectorizer.transform([query])

    # 计算query与所有问题的余弦相似度
    cosine_similarities = cosine_similarity(query_vector, tf_matrix).flatten()

    max_sim_index = cosine_similarities.argmax()
    max_sim_value = cosine_similarities[max_sim_index]

    similarity_threshold = 0.2  # 阈值可根据需求调整

    print("\n相似度得分：")
    for i, question in enumerate(questions):
        print(f"  '{question}': {cosine_similarities[i]:.4f}")

    if max_sim_value < similarity_threshold:
        print("\n抱歉，未能找到合适的答案。请尝试换个问法。")
    else:
        matched_question = questions[max_sim_index]
        answer = qa_pairs[matched_question]
        print(f"\n机器人（最相似问题：'{matched_question}'，相似度：{max_sim_value:.4f}）：")
        print(answer)

    print("\n" + "-"*60)
