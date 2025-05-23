import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats

# 1. 数据准备
items = {
    'item1': 'action adventure science fiction superhero',
    'item2': 'science fiction space travel adventure',
    'item3': 'comedy romance drama',
    'item4': 'action thriller crime mystery',
    'item5': 'fantasy adventure magic',
    'item6': 'documentary nature wildlife'
}

# 用户-物品交互矩阵 (喜好程度，0表示未交互)
user_item_matrix = np.array([
    [5, 4, 0, 0, 0, 0],  # 用户1：喜欢科幻相关，不喜欢喜剧
    [0, 5, 3, 0, 0, 0],  # 用户2：喜欢科幻和一些喜剧
    [0, 0, 0, 4, 5, 0],  # 用户3：喜欢犯罪，和奇幻
    [5, 0, 0, 0, 0, 3],  # 用户4：喜欢科幻， 和纪录片
    [0, 0, 5, 4, 0, 0]   # 用户5：喜欢喜剧和动作
])

item_names = list(items.keys())
user_names = ['User 1', 'User 2', 'User 3', 'User 4', 'User 5']

# ------------------------  相似度计算方法  ------------------------
def pearson_correlation(user1_ratings, user2_ratings):
    """
    计算两个用户的 Pearson 相关系数，使用 scipy。
    处理评分缺失的情况（评分都为0）时，返回0.
    """
    # 只保留共同评分的items
    rated_indices1 = np.where(user1_ratings > 0)[0]
    rated_indices2 = np.where(user2_ratings > 0)[0]

    common_items = np.intersect1d(rated_indices1, rated_indices2) #获取共同评分的items

    if len(common_items) == 0: # 没有共同评分的物品
      return 0.0

    user1_common_ratings = user1_ratings[common_items]
    user2_common_ratings = user2_ratings[common_items]

    correlation, _ = scipy.stats.pearsonr(user1_common_ratings, user2_common_ratings)
    # 如果存在nan,返回0
    if np.isnan(correlation):
      return 0.0

    return correlation

def jaccard_index(item1_users, item2_users):
    """
    计算两个物品的 Jaccard 指数。
    item1_users: 喜欢物品1，不包括不喜欢的用户。
    item2_users: 喜欢物品2的用户。

    """
    intersection = np.intersect1d(item1_users, item2_users)
    union = np.union1d(item1_users, item2_users)

    if len(union) == 0:
      return 0.0
    return len(intersection) / len(union)
# ---------------------------------------------------------------
# 2. 基于内容的推荐函数 (与之前相同)
def content_based_recommendation(items, user_liked_items, top_n):
   # 代码省略 (与前面的代码相同)
    item_ids = items.keys()
    descriptions = items.values()
    item_ids_list = list(item_ids)

    # 1. 将物品描述转换为TF-IDF向量
    vectorizer = TfidfVectorizer()
    item_vectors = vectorizer.fit_transform(descriptions)

    # 2. 创建用户画像： 取用户喜欢物品向量的平均值
    user_profile = np.zeros(item_vectors.shape[1])
    for item_id in user_liked_items:
        index = item_ids_list.index(item_id)
        item_vector = item_vectors[index]
        user_profile += item_vector.toarray().flatten()

    if len(user_liked_items) > 0:
        user_profile /= len(user_liked_items) # 平均值

    # 3. 计算用户画像与所有物品的相似度 (余弦相似度)
    similarities = cosine_similarity(user_profile.reshape(1, -1), item_vectors).flatten()

    # 4. 排除用户已经喜欢的物品 (避免重复推荐)
    for item_id in user_liked_items:
        index = item_ids_list.index(item_id)
        similarities[index] = -1  # 将相似度设置为-1，确保不被推荐

    # 5. 推荐最相似的物品
    sorted_indices = np.argsort(similarities)[::-1] # 从大到小排列
    top_indices = sorted_indices[:top_n] #选取top N

    recommendations = []
    for i in top_indices:
        recommendation = items.keys().__getitem__(i)
        recommendations.append(recommendation) # 获取推荐的物品ID

    return recommendations

# 3. 基于物品的协同过滤函数 (修改后)
def item_based_collaborative_filtering(user_item_matrix, item_names, user_index, top_n, similarity_metric="cosine"):
    """
    基于物品的协同过滤: 根据用户已喜欢的物品，推荐相似的物品（基于其他用户的行为）
    similarity_metric: 指定使用的相似度计算方法 (cosine, pearson, jaccard)
    """
    num_items = user_item_matrix.shape[1] #物品总数

    # 1. 计算物品之间的相似度
    item_similarity_matrix = np.zeros((num_items, num_items)) #初始化矩阵
    for i in range(num_items):
        for j in range(num_items):
            if similarity_metric == "cosine":
                item_similarity_matrix[i, j] = cosine_similarity(user_item_matrix[:, i].reshape(1, -1), user_item_matrix[:, j].reshape(1, -1))[0, 0]
            elif similarity_metric == "pearson":
                item_similarity_matrix[i, j] = pearson_correlation(user_item_matrix[:, i], user_item_matrix[:, j])
            elif similarity_metric == "jaccard":
                #查找喜欢物品i的用户
                user_like_item1 = np.where(user_item_matrix[:,i] > 0)[0]
                #查找喜欢物品j的用户
                user_like_item2 = np.where(user_item_matrix[:,j] > 0)[0]
                item_similarity_matrix[i, j] = jaccard_index(user_like_item1, user_like_item2)
            else:
                raise ValueError("无效的相似度计算方法。")

    # 2. 获取目标用户评价过的物品 (包括喜欢和不喜欢)
    rated_items_indices = np.where(user_item_matrix[user_index] > 0)[0]

    # 3. 计算每个物品的推荐分数
    scores = np.zeros(len(item_names))
    for liked_item_index in rated_items_indices:
        scores += item_similarity_matrix[liked_item_index] * user_item_matrix[user_index, liked_item_index]  # 用户对一个物品的得分作为权重

    # 4. 排除用户已经评价过的物品
    for rated_item_index in rated_items_indices:
        scores[rated_item_index] = -1

    # 5. 推荐分数最高的物品
    sorted_indices = np.argsort(scores)[::-1]
    top_indices = sorted_indices[:top_n]

    recommendations = []
    for i in top_indices:
        recommendation = item_names[i]  # 获取推荐的物品名称
        recommendations.append(recommendation)

    return recommendations

# 4. 混合推荐函数 （简单加权平均）
def hybrid_recommendation(items, user_item_matrix, item_names, user_index, top_n,
                          content_weight=0.5, cf_weight=0.5, cf_similarity="cosine"):
    """
    混合推荐:结合基于内容的推荐和协同过滤.
    cf_similarity: 指定协同过滤中使用的相似度计算方法.
    """
    # 1. 获取用户喜欢的物品
    liked_items_indices = np.where(user_item_matrix[user_index] > 3)[0]  # 评分大于3视为喜欢
    user_liked_items = []
    for i in liked_items_indices:
        item = item_names[i]
        user_liked_items.append(item)

    # 2. 执行基于内容的推荐
    content_recommendations = content_based_recommendation(items, user_liked_items, top_n=top_n)

    # 3. 执行基于物品的协同过滤
    cf_recommendations = item_based_collaborative_filtering(user_item_matrix, item_names, user_index, top_n=top_n, similarity_metric=cf_similarity)

    # 4. 合并推荐结果
    hybrid_recommendations = content_recommendations + cf_recommendations

    # Remove duplicate items
    seen = set()
    unique_recommendations = []
    for item in hybrid_recommendations:
        if item not in seen:
            unique_recommendations.append(item)
            seen.add(item)

    final_recommendations = unique_recommendations[:top_n]

    return final_recommendations

# 5. 演示用户界面
def display_gui():
    """
    用户界面演示
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

                # 获取混合推荐，可以指定CF的相似度计算方法
                top_n = 3
                cf_similarity_metric = input("请选择协同过滤的相似度计算方法 (cosine, pearson, jaccard, 默认为cosine): ") or "cosine"
                recommendations = hybrid_recommendation(items, user_item_matrix, item_names, user_index, top_n=top_n, cf_similarity=cf_similarity_metric)

                print(f"\n为 {user_name} 推荐的物品 (使用 {cf_similarity_metric})：")
                for recommendation in recommendations:
                    print(f"- {recommendation}: {items[recommendation]}")

            else:
                print("无效的输入，请重新选择。")

        except ValueError:
            print("无效的输入，请输入数字。")

# 6. 运行演示
display_gui()
