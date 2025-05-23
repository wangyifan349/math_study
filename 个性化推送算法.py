import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 数据准备
# 物品信息
items = {
    'item1': 'action adventure science fiction superhero',  #科幻，超人
    'item2': 'science fiction space travel adventure',     #科幻，太空旅行
    'item3': 'comedy romance drama',                    #喜剧爱情剧
    'item4': 'action thriller crime mystery',           #动作惊悚犯罪
    'item5': 'fantasy adventure magic',                  # 奇幻魔法
    'item6': 'documentary nature wildlife'              # 纪录片，野生动物
}

# 用户-物品交互矩阵 （喜好程度，0表示未交互）
user_item_matrix = np.array([
    [5, 4, 0, 0, 0, 0],  # 用户1：喜欢科幻相关，不喜欢喜剧
    [0, 5, 3, 0, 0, 0],  # 用户2：喜欢科幻和一些喜剧
    [0, 0, 0, 4, 5, 0],  # 用户3：喜欢犯罪，和奇幻
    [5, 0, 0, 0, 0, 3],  # 用户4：喜欢科幻， 和纪录片
    [0, 0, 5, 4, 0, 0]   # 用户5：喜欢喜剧和动作
])

item_names = list(items.keys())
user_names = ['User 1', 'User 2', 'User 3', 'User 4', 'User 5']

# 2. 基于内容的推荐函数
def content_based_recommendation(items, user_liked_items, top_n=2):
    """
    基于内容的推荐： 根据用户喜欢的物品，推荐相似物品
    """
    item_ids = list(items.keys())
    descriptions = list(items.values())

    # 1. 将物品描述转换为TF-IDF向量
    vectorizer = TfidfVectorizer()
    item_vectors = vectorizer.fit_transform(descriptions)

    # 2. 创建用户画像： 取用户喜欢物品向量的平均值
    user_profile = np.zeros(item_vectors.shape[1])
    for item_id in user_liked_items:
        index = item_ids.index(item_id)
        user_profile += item_vectors[index].toarray().flatten()
    user_profile /= len(user_liked_items) # 平均值

    # 3. 计算用户画像与所有物品的相似度 (余弦相似度)
    similarities = cosine_similarity(user_profile.reshape(1, -1), item_vectors).flatten()

    # 4. 排除用户已经喜欢的物品 (避免重复推荐)
    for item_id in user_liked_items:
        index = item_ids.index(item_id)
        similarities[index] = -1  # 将相似度设置为-1，确保不被推荐

    # 5. 推荐最相似的物品
    top_indices = np.argsort(similarities)[::-1][:top_n] # 从大到小排列，选取top N
    recommendations = [item_ids[i] for i in top_indices] # 获取推荐的物品ID

    return recommendations

# 3. 基于物品的协同过滤函数
def item_based_collaborative_filtering(user_item_matrix, item_names, user_index, top_n=2):
    """
    基于物品的协同过滤: 根据用户已喜欢的物品，推荐相似的物品（基于其他用户的行为）
    """
    # 1. 计算物品之间的相似度 (余弦相似度)
    item_similarity_matrix = cosine_similarity(user_item_matrix.T)

    # 2. 获取目标用户评价过的物品 (包括喜欢和不喜欢)
    rated_items_indices = np.where(user_item_matrix[user_index] > 0)[0]

    # 3. 计算每个物品的推荐分数
    scores = np.zeros(len(item_names))
    for liked_item_index in rated_items_indices:
        scores += item_similarity_matrix[liked_item_index] * user_item_matrix[user_index, liked_item_index] #  用户对一个物品的得分作为权重

    # 4. 排除用户已经评价过的物品
    for rated_item_index in rated_items_indices:
        scores[rated_item_index] = -1

    # 5. 推荐分数最高的物品
    top_indices = np.argsort(scores)[::-1][:top_n]
    recommendations = [item_names[i] for i in top_indices]

    return recommendations

# 4. 混合推荐函数 （简单加权平均，可根据实际情况调整权重）
def hybrid_recommendation(items, user_item_matrix, item_names, user_index, top_n=2,
                          content_weight=0.5, cf_weight=0.5):
    """
    混合推荐：结合基于内容的推荐和协同过滤
    """
    # 1. 获取用户喜欢的物品 (用于内容推荐)
    liked_items_indices = np.where(user_item_matrix[user_index] > 3)[0]  # 评分大于3视为喜欢
    user_liked_items = [item_names[i] for i in liked_items_indices]

    # 2. 执行基于内容的推荐
    content_recommendations = content_based_recommendation(items, user_liked_items, top_n=top_n)

    # 3. 执行基于物品的协同过滤
    cf_recommendations = item_based_collaborative_filtering(user_item_matrix, item_names, user_index, top_n=top_n)

    # 4. 合并推荐结果 (可以根据实际需要调整合并策略)
    # 这里简单地将两种推荐结果合并，并去除重复项
    hybrid_recommendations = list(set(content_recommendations + cf_recommendations))

    return hybrid_recommendations[:top_n]

# 5. 演示用户界面
def display_gui():
    """
    简单的用户界面，用于演示推荐结果
    """
    print("---------------------------------------------")
    print("        个性化推荐系统演示")
    print("---------------------------------------------")

    while True:
        print("\n请选择用户：")
        for i, user_name in enumerate(user_names):
            print(f"{i+1}. {user_name}")
        print("0. 退出")

        try:
            choice = int(input("请输入用户编号："))

            if choice == 0:
                break
            elif 1 <= choice <= len(user_names):
                user_index = choice - 1
                user_name = user_names[user_index]

                # 获取推荐结果
                recommendations = hybrid_recommendation(items, user_item_matrix, item_names, user_index, top_n=3)

                print(f"\n为 {user_name} 推荐的物品：")
                for recommendation in recommendations:
                    print(f"- {recommendation}: {items[recommendation]}") # 显示物品信息

            else:
                print("无效的输入，请重新选择。")

        except ValueError:
            print("无效的输入，请输入数字。")

# 6. 运行演示
display_gui()
